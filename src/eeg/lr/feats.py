"""Признаки для скрининга «лево/право» и прогон within/pool моделей по онлайн-протоколу.

Для каждого набора признаков считаются (для всех окон всех сессий, блоки 2–7):
  * pw3 / pp3 — лог-вероятности 3 классов within-sLDA / пул-LR (LOSubjO);
  * rw / rp   — бинарные лог-шансы log P(L)/P(R) within-sLDA / пул-LR, обученных только на L/R окнах.
Результат кэшируется в scratch: <SCR>/res_<feat>.pkl.

python -m src.eeg.lr.feats <feat> [<feat> ...]
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_eeg_cache
from NIRS_IMC_EEG_fNIRS.src.eeg.common import RefState
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import (EAR, FRONTAL, MIRROR, TRUE_NAMES, apply_spatial, feature_mirror, idx,
                             mirror_signed_perm, spatial_matrix, swap_lr)
from NIRS_IMC_EEG_fNIRS.src.eeg.online import SLDA

SCR = Path(os.environ.get("EEGLR_SCR", "/place/vartmp/claude-740548/-home-vkborisov16-projects-hackatons-medic/"
                          "62fd6f60-50fc-45a1-9add-e0b5b7dac57f/scratchpad/eeg_lr"))
NJ = int(os.environ.get("EEGLR_NJ", "8"))
SM = ["c3", "c4", "cz", "f3", "f4", "fz", "p3", "p4", "pz", "c7", "c8"]
NOEAR = [c for c in TRUE_NAMES if c not in EAR]
NOEARFRONT = [c for c in TRUE_NAMES if c not in EAR + FRONTAL]

# feat -> (как получить ковариации (n,B,k,k) из 21-канальных, тип признака, полосы)
FEATS = {
    "lvlap": ("lap", "logvar", None),
    "lvbip": ("bip", "logvar", None),
    "lvall": ("all", "logvar", None),
    "tsall": ("all", "ts", None),
    "tslap": ("lap", "ts", None),
    "tssm": ("sm", "ts", None),
    "tsnoear": ("noear", "ts", None),
    "tsnoearfront": ("noearfront", "ts", None),
    "tslapcore": ("lap_core", "ts", None),
    "tsbip": ("bip", "ts", None),
    # узкополосные логдисперсии из data/features/eeg_lr_nb (13 полос)
    "nbbip": ("nb_bip", "nblogvar", None),
    "nblap": ("nb_lap", "nblogvar", None),
    "nbboth": ("nb_both", "nblogvar", None),
    "nbbipmu": ("nb_bip", "nblogvar", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
}
NB_SL = {"nb_bip": slice(0, 12), "nb_lap": slice(12, 23), "nb_both": slice(0, 23)}


def nb_W(space):
    return np.vstack([spatial_matrix("bip"), spatial_matrix("lap")])[NB_SL[space]]


def nb_logvar(v_ref, v):
    """Аналог RefState.logvar для готовых дисперсий: log(v / медиана v по референсным окнам)."""
    return np.log(np.maximum(v, 1e-30) / np.maximum(np.median(v_ref, axis=0), 1e-30)).reshape(len(v), -1)


def space_mirror(space):
    """(perm, sign) по «каналам» пространства признаков."""
    if space.startswith("nb_"):
        return mirror_signed_perm(nb_W(space))
    if space in ("all", "sm", "noear", "noearfront"):
        names = dict(all=TRUE_NAMES, sm=SM, noear=NOEAR, noearfront=NOEARFRONT)[space]
        pos = {c: i for i, c in enumerate(names)}
        return np.array([pos[MIRROR[c]] for c in names]), np.ones(len(names))
    return mirror_signed_perm(spatial_matrix(space))


def mirror_fn(base_feat, n_bands=7):
    space, kind, _ = FEATS[base_feat]
    if space.startswith("nb_"):
        n_bands = 13 if FEATS[base_feat][2] is None else len(FEATS[base_feat][2])
    fi, fs = feature_mirror(*space_mirror(space), "logvar" if kind == "nblogvar" else kind, n_bands)
    return lambda X: X[:, fi] * fs


def transform(c, space):
    if space == "all":
        return c
    if space in ("sm", "noear", "noearfront"):
        ix = idx(dict(sm=SM, noear=NOEAR, noearfront=NOEARFRONT)[space])
        return c[..., ix[:, None], ix[None, :]]
    return apply_spatial(c, spatial_matrix(space))


_G = {}


def strides(n_train):
    """(ref_stride, train_stride): 4/2 по умолчанию; с опцией +capN обучающих окон не больше N
    (train_stride = ceil(n/N), ref_stride = 2·train_stride) — ограничение времени fit_block."""
    cap = next((int(o[3:]) for o in _G.get("opts", []) if o.startswith("cap")), 0)
    if not cap:
        return 4, 2
    ts_ = max(2, int(np.ceil(n_train / cap)))
    return 2 * ts_, ts_


def feats_session(n):
    space, kind, bands = FEATS[_G["feat"].split("+")[0]]
    d = _G["data"][n]
    if kind == "nblogvar":
        v = _G["nb"][n][:, :, NB_SL[space]].astype(np.float64)
        if bands is not None:
            v = v[:, bands]
        blk = d["block"]
        out = {"pool": nb_logvar(v, v).astype(np.float32)}
        for b in range(2, 8):
            rs, _ = strides((blk < b).sum())
            out[b] = nb_logvar(v[blk < b][::rs], v[blk <= b]).astype(np.float32)
        return n, out
    c = d["covs"].astype(np.float64)
    if bands is not None:
        c = c[:, bands]
    c = transform(c, space)
    blk = d["block"]
    out = {"pool": getattr(RefState().fit(c), kind)(c).astype(np.float32)}
    for b in range(2, 8):
        tr = blk < b
        rs, _ = strides(tr.sum())
        out[b] = getattr(RefState().fit(c[tr][::rs]), kind)(c[blk <= b]).astype(np.float32)
    return n, out


def fit_pool(s, C):
    F, data = _G["F"], _G["data"]
    tr = [n for n in data if data[n]["subject"] != s]
    X = np.vstack([F[n]["pool"] for n in tr]).astype(np.float64)
    y = np.concatenate([data[n]["y"] for n in tr])
    if "mp" in _G["opts"]:
        X = np.vstack([X, _G["mir"](X)]); y = np.concatenate([y, swap_lr(y)])
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    m3 = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(Xs, y)
    t = y > 1
    m2 = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(Xs[t], y[t])
    return s, (sc, m3, m2)


def lr_logodds_slda(X, y, Xte, shrink):
    t = y > 1
    m = SLDA(shrink).fit(X[t], y[t])
    lp = m.predict_log_proba(Xte)
    return lp[:, 1] - lp[:, 2]


def within_session(n):
    F, data, pools, shrink = _G["F"], _G["data"], _G["pools"], _G["shrink"]
    d = data[n]
    N = len(d["y"])
    res = {k: np.full((N, 3), np.nan) for k in ("pw3", "pp3")}
    res.update(rw=np.full(N, np.nan), rp=np.full(N, np.nan))
    pool = pools.get(d["subject"])
    for b in range(2, 8):
        X = F[n][b].astype(np.float64)
        blk = d["block"][d["block"] <= b]; y = d["y"][d["block"] <= b]
        tr, te = blk < b, blk == b
        gi = d["block"] == b
        _, ts_ = strides(tr.sum())
        Xtr = X[tr][::ts_]; ytr = y[tr][::ts_]
        if "mw" in _G["opts"]:
            Xtr = np.vstack([Xtr, _G["mir"](Xtr)]); ytr = np.concatenate([ytr, swap_lr(ytr)])
        res["pw3"][gi] = SLDA(shrink).fit(Xtr, ytr).predict_log_proba(X[te])
        res["rw"][gi] = lr_logodds_slda(Xtr, ytr, X[te], shrink)
        if pool is None:                    # +np: пул не обучается (берётся из baseline)
            continue
        sc, m3, m2 = pool
        Xs = sc.transform(X[te])
        res["pp3"][gi] = m3.predict_log_proba(Xs)
        lp2 = m2.predict_log_proba(Xs)
        res["rp"][gi] = lp2[:, 0] - lp2[:, 1]
    return n, res


def run(feat, data, C=0.01, shrink=0.8, win=2.0):
    out_f = SCR / (f"res_{feat}.pkl" if win == 2.0 else f"res_{feat}_w{win:g}.pkl")
    if out_f.exists():
        return pickle.load(open(out_f, "rb"))
    ctx = mp.get_context("fork")
    base = feat.split("+")[0]
    for o in feat.split("+")[1:]:          # +s0.5 — shrinkage within, +C0.1 — C пула
        if o.startswith("s"):
            shrink = float(o[1:])
        elif o.startswith("C"):
            C = float(o[1:])
    _G.update(data=data, feat=feat, shrink=shrink, opts=feat.split("+")[1:], mir=mirror_fn(base))
    if FEATS[base][1] == "nblogvar":
        from NIRS_IMC_EEG_fNIRS.src.eeg.lr.nbcache import OUT as NB_OUT
        _G["nb"] = {n: np.load(NB_OUT / f"{n}.npz")[f"var_w{win:g}"] for n in data}
        for n in data:
            assert len(_G["nb"][n]) == len(data[n]["y"])
    with ProcessPoolExecutor(NJ, mp_context=ctx) as ex:
        F = dict(ex.map(feats_session, list(data)))
    _G["F"] = F
    subjects = sorted({d["subject"] for d in data.values()})
    pools = {}
    if "np" not in _G["opts"]:
        with ProcessPoolExecutor(NJ, mp_context=ctx) as ex:
            pools = dict(ex.map(fit_pool, subjects, [C] * len(subjects)))
    _G["pools"] = pools
    with ProcessPoolExecutor(NJ, mp_context=ctx) as ex:
        res = dict(ex.map(within_session, list(data)))
    SCR.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(out_f, "wb"), protocol=4)
    print(f"{feat} w{win:g}: dim={F[next(iter(F))]['pool'].shape[1]}", flush=True)
    return res


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = sys.argv[1:]
    win = 2.0
    if args[0].startswith("--win="):
        win = float(args.pop(0).split("=")[1])
    data = load_eeg_cache(win, f64=True)
    for f in args:
        run(f, data, win=win)
