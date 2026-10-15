"""Эксперимент 2: кросс-субъектный перенос через рецентрирование TS + within-модель, смешение.

Для целевой сессии и блока b: референс (масштабы каналов + LE-среднее по полосам) — по окнам
блоков < b этой сессии; пул — все сессии ДРУГИХ участников, каждая рецентрирована по своему
собственному референсу (оффлайн, всё, т.к. это чужие данные). Пул-модель — LR на TS.
Вероятности: within-sLDA, pool-LR и смесь лог-вероятностей с весом alpha.

python -m src.eeg.exp_cross <win> <chans: all|nofront|sm> [save_tag]
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eeg.cache import load_eeg_cache
from src.eeg.common import ALL, NO_FRONTAL, SENSORIMOTOR, RefState, report, save_probs, score, subcov

CHSETS = dict(all=ALL, nofront=NO_FRONTAL, sm=SENSORIMOTOR)


def feats_session(args):
    """TS-признаки: 'pool' — ref по всей сессии; b=2..7 — ref по блокам < b."""
    c, block, feat = args
    out = {"pool": getattr(RefState().fit(c), feat)(c)}
    for b in range(2, 8):
        tr = block < b
        out[b] = getattr(RefState().fit(c[tr]), feat)(c[block <= b])
    return out


_P = {}


def fit_pool(s, C=0.01):
    F, data, names = _P["F"], _P["data"], _P["names"]
    tr = [n for n in names if data[n]["subject"] != s]
    X = np.vstack([F[n]["pool"] for n in tr]); y = np.concatenate([data[n]["y"] for n in tr])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(sc.transform(X), y)
    return sc, m


def main(win=2.0, chn="nofront", feat="ts", save=None):
    data = load_eeg_cache(win)
    names = list(data)
    chans = CHSETS[chn]
    with ProcessPoolExecutor(10) as ex:
        F = dict(zip(names, ex.map(feats_session, [(subcov(data[n]["covs"], chans).astype(np.float64),
                                                   data[n]["block"], feat) for n in names])))
    subjects = sorted({data[n]["subject"] for n in names})

    # пул-модели leave-subject-out (параллельно по участникам)
    import multiprocessing as mp
    _P.update(F=F, data=data, names=names)
    with ProcessPoolExecutor(10, mp_context=mp.get_context("fork")) as ex:
        pools = dict(zip(subjects, ex.map(fit_pool, subjects)))
    Pw, Pp = {}, {}
    for n in names:
        d = data[n]; sc, pm = pools[d["subject"]]
        Pw[n] = np.full((len(d["y"]), 3), np.nan); Pp[n] = Pw[n].copy()
        for b in range(2, 8):
            Xb = F[n][b]; blk = d["block"][d["block"] <= b]; yb = d["y"][d["block"] <= b]
            tr, te = blk < b, blk == b
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xb[tr], yb[tr])
            Pw[n][d["block"] == b] = m.predict_proba(Xb[te])
            Pp[n][d["block"] == b] = pm.predict_proba(sc.transform(Xb[te]))
    tag = f"w{win:g} {feat} {chn}"
    report(f"{tag} within sLDA", score(data, Pw))
    report(f"{tag} pool LR (LOSubjO)", score(data, Pp))
    best = None
    for a in [0.2, 0.35, 0.5, 0.65]:
        Pm = {n: np.exp((1 - a) * np.log(np.clip(Pw[n], 1e-9, 1)) + a * np.log(np.clip(Pp[n], 1e-9, 1))) for n in names}
        Pm = {n: p / p.sum(1, keepdims=True) for n, p in Pm.items()}
        sc_ = score(data, Pm); report(f"{tag} mix a={a}", sc_)
        if best is None or sc_["pts"] > best[0]:
            best = (sc_["pts"], a, Pm)
    if save:
        save_probs(save, best[2]); save_probs(save + "_within", Pw); save_probs(save + "_pool", Pp)
    return Pw, Pp


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    a = sys.argv[1:]
    main(float(a[0]) if a else 2.0, a[1] if len(a) > 1 else "nofront", a[2] if len(a) > 2 else "ts",
         a[3] if len(a) > 3 else None)
