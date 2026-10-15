"""Какие признаки слияния опираются на порядок проб (чередование покой↔задача, баланс классов)?

Вход — npz прогона compliance_async --predict judge (вероятности ЭЭГ/NIRS окон судьи, хронологический
протокол). Для каждого набора признаков FEAT мета-LR обучается хронологически (сессии раньше S,
в ИСХОДНОМ порядке проб, как в жизни) и проверяется на сессии S с переставленными внутри блоков
сегментами (вероятности базовых моделей фиксированы):

  orig     — исходный порядок;
  alt      — покой ↔ покой, задача ↔ задача (чередование сохранено);
  grouped  — все покои блока, затем все задачи (чередования нет, баланс классов в 60 с нарушен);
  reset    — исходный порядок, но быстрые EMA (τ-признаки и выходное сглаживание) обнуляются в
             начале каждого сегмента (центрирование 60 с — без изменений): вклад памяти между пробами.

Оценка — только сессии, у которых есть хронологическая мета-модель (≥ min-prev ранних сессий).

    python -m src.scripts.compliance_order --runs <judge npz dir> --out <json>
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from src.fusion.replay import fit_meta, hier_seq, meta_logp, sess_fp, loglinear  # noqa: E402
from src.fusion.stacked import FLOOR, _clean, _Ema, _ratio  # noqa: E402
from src.scripts.compliance_meta import load_runs  # noqa: E402

FEATS = {
    "F0_current": dict(eeg_taus=[0.0, 0.5, 1.0, 2.0, 4.0], nirs_taus=[0.0, 1.0, 2.0, 4.0], center_T=[60.0]),
    "F1_nocenter": dict(eeg_taus=[0.0, 0.5, 1.0, 2.0, 4.0], nirs_taus=[0.0, 1.0, 2.0, 4.0], center_T=[]),
    "F2_short": dict(eeg_taus=[0.0, 0.5, 1.0], nirs_taus=[0.0, 1.0], center_T=[60.0]),
    "F3_short_nocenter": dict(eeg_taus=[0.0, 0.5, 1.0], nirs_taus=[0.0, 1.0], center_T=[]),
}
HIER = (0.75, 2.0)
_G = {}


def features(P, t, seg_start, taus, center_T, reset):
    """Как ModalityFeatures.step (ref = taus[1]), но быстрые EMA можно обнулять на старте сегмента."""
    n = len(P)
    ref_tau = taus[min(1, len(taus) - 1)]
    dim = 2 * (len(taus) + len(center_T))
    X = np.full((n, dim), np.nan)
    emas = [_Ema(v) for v in taus]
    cemas = [_Ema(v) for v in center_T]
    for i in range(n):
        p = _clean(P[i])
        if p is None:
            continue
        if reset and seg_start[i]:
            emas = [_Ema(v) for v in taus]
        out = [_ratio(e(p, t[i])) for e in emas]
        if center_T:
            ref = out[taus.index(ref_tau)]
            x0 = _ratio(p)
            out += [ref - c(x0, t[i]) for c in cemas]
        X[i] = np.concatenate(out)
    return X


def seg_starts(e):
    return np.r_[True, np.diff(e) != 62]


def permute(r, kind, rng):
    """Переставить сегменты окон внутри блоков; время слотов сохраняется."""
    e = r["e"]
    new = np.r_[True, (np.diff(e) != 62) | (np.diff(r["y"]) != 0)]
    st = np.flatnonzero(new)
    segs = list(zip(st, np.r_[st[1:], len(e)]))
    blk = r["blk"]
    order_rows, t_new = [], []
    for b in np.unique(blk):
        ss = [s for s in segs if blk[s[0]] == b]
        labs = np.array([r["y"][s0] for s0, _ in ss])
        idx = np.arange(len(ss))
        if kind == "alt":
            o = idx.copy()
            for g in (labs == 1, labs > 1):
                o[idx[g]] = rng.permutation(idx[g])
        elif kind == "grouped":
            o = np.r_[idx[labs == 1], idx[labs > 1]]
        else:
            o = idx
        for k, j in enumerate(o):
            s0, s1 = ss[j]
            slot0 = e[ss[k][0]]
            order_rows += list(range(s0, s1))
            t_new += list(slot0 + (e[s0:s1] - e[s0]))
    rows = np.array(order_rows)
    q = {k: (r[k][rows] if isinstance(r[k], np.ndarray) and len(r[k]) == len(e) else r[k]) for k in r}
    q["e"] = np.array(t_new)
    q["t"] = q["e"] / 250.0
    return q


def prepare(r, feat, reset=False):
    ss = seg_starts(r["e"])
    Xe = features(r["pe"], r["t"], ss, feat["eeg_taus"], feat["center_T"], reset)
    Xn = features(r["pn"], r["t"], ss, feat["nirs_taus"], feat["center_T"], reset)
    return dict(r, Xe=Xe, Xn=Xn)


def smooth_eval(r, L, reset):
    P = np.exp(L)
    if not reset:
        S = hier_seq(P, r["t"], *HIER)
    else:
        ss = seg_starts(r["e"])
        S = np.full_like(P, np.nan)
        st = np.flatnonzero(ss)
        for a, b in zip(st, np.r_[st[1:], len(P)]):
            S[a:b] = hier_seq(P[a:b], r["t"][a:b], *HIER)
    out = np.log(np.clip(S, FLOOR, None))
    out[~np.isfinite(out).all(1)] = [0, -1, -1]
    return sess_fp(r, out)


def job(args):
    fname, n, prev = args
    R = _G["R"]
    feat = FEATS[fname]
    Rf = {k: prepare(R[k], feat) for k in prev + [n]}
    M = fit_meta(Rf, prev)
    rng = np.random.default_rng(0)
    res = {}
    for kind in ("orig", "alt", "grouped", "reset"):
        reps = 5 if kind == "alt" else 1
        vals, vll = [], []
        for _ in range(reps):
            q = permute(R[n], kind if kind != "reset" else "orig", rng)
            qf = prepare(q, feat, reset=(kind == "reset"))
            vals.append(smooth_eval(qf, meta_logp(qf, M), kind == "reset"))
            # LogLinearFusion сдачи (0.75 / приор 3.5 / иерархич. 0.75-2) — без стекинга
            Pll = loglinear(qf, 0.55, [0.0, 3.5, 3.5], hier=None, tau=0.0)
            vll.append(smooth_eval(qf, np.log(np.clip(Pll, FLOOR, None)), kind == "reset"))
        res[kind] = np.mean(vals, 0).tolist()
        res[kind + "_loglinear"] = np.mean(vll, 0).tolist()
    return fname, n, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-prev", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--feats", default=",".join(FEATS))
    a = ap.parse_args()
    from src.dataset.io import RAW_DIR, load_session
    R = load_runs(a.runs)
    for n, r in R.items():
        r["blk"] = load_session(RAW_DIR / "Data" / f"{n}.mat").blocks.astype(int)[r["e"] - 1]
    _G["R"] = R
    names = sorted(R)
    jobs = [(f, n, names[:i]) for f in a.feats.split(",") for i, n in enumerate(names) if i >= a.min_prev]
    per = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for f, n, res in ex.map(job, jobs):
            per.setdefault(f, {})[n] = res
    agg = {}
    for f, d in per.items():
        agg[f] = {}
        for kind in next(iter(d.values())):
            v = np.array([d[n][kind] for n in d])
            F, P = v[:, 0].mean(), v[:, 1].mean()
            agg[f][kind] = {"min_f1": round(F, 4), "recall": round(P, 4), "pts": round(60 * F + 30 * P, 2)}
    Path(a.out).write_text(json.dumps({"agg": agg, "per": per}, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
