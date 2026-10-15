"""Каскадный NIRS-декодер: ансамбль sLDA «покой/задача» × ансамбль sLDA «левая/правая».

P(покой) = 1 − p_t,  P(левая) = p_t·(1 − p_r),  P(правая) = p_t·p_r,
где p_t — среднее логитов 8 бинарных sLDA «задача vs покой», p_r — среднее логитов 4 бинарных
sLDA «правая vs левая» (обучаются только на окнах задачи, начиная с 3 с от начала сегмента —
выбор обучающих окон по меткам прошлых блоков, на тесте используются все окна).

Оффлайн-прогон (онлайн-порядок: блок b предсказывается моделью по блокам < b):
    python -m src.nirs.cascade            # → data/probs/nirs_*.pkl + метрики
"""
from __future__ import annotations

import pickle
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.dataset.io import PROJECT_DIR
from src.nirs.evaluate import ch_select, load_bank, make_X, report, run_sessions

# имя банка → параметры полосового фильтра (lo, hi, order); см. build_bank.py
BANKS = {
    "bp005_10": (0.005, 0.10, 2),
    "bp01_20": (0.01, 0.20, 2),
    "bp02_30": (0.02, 0.30, 2),
    "bp01_50": (0.01, 0.50, 3),
}
# (банк, набор каналов, лаги [с], режим)
RT_MEMBERS = [
    ("bp01_50", "chan", (0, 4, 8, 12), "diff"),
    ("bp01_50", "all", (0, 4, 8, 12), "diff"),
    ("bp01_20", "chan", (0, 4, 8, 12), "diff"),
    ("bp02_30", "chan", (0, 4, 8, 12), "diff"),
    ("bp005_10", "chan", (0, 4, 8, 12), "diff"),
    ("bp02_30", "all", (0, 4, 8, 12), "diff"),
    ("bp01_20", "chan", (0, 3, 6, 9, 12), "level"),
    ("bp01_50", "chan", (0, 1, 2, 4, 6), "diff"),
]
LR_MEMBERS = [
    ("bp01_20", "all", (0,), "level"),
    ("bp02_30", "all", (0,), "level"),
    ("bp005_10", "all", (0, 3), "diff"),
    ("bp01_20", "chan", (0,), "level"),
]
LR_TMIN = 3.0          # с от начала сегмента — минимальная позиция обучающего окна для L/R


def pos_in_segment(ends, y):
    """Время (с) от начала однородного сегмента до конца окна; только для отбора обучающих окон."""
    new = np.r_[True, (np.diff(ends) > 70) | (np.diff(y) != 0)]
    first = np.maximum.accumulate(np.where(new, np.arange(len(ends)), 0))
    return (ends - ends[first]) / 250.0


def slda():
    return LDA(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def combine(p_task, p_right):
    return np.c_[1 - p_task, p_task * (1 - p_right), p_task * p_right]


def fit_member(X, y, pos, task, train_mask):
    if task == "RT":
        sel = train_mask
        yb = y > 1
    else:
        sel = train_mask & (y > 1) & (pos >= LR_TMIN)
        yb = y == 3
    return slda().fit(X[sel], yb[sel])


def _session_cascade(dsets, rt_members, lr_members):
    """dsets: {bank: dict сессии}. Возвращает (P (n,3), время fit по блокам)."""
    d0 = next(iter(dsets.values()))
    y, blk = d0["y"], d0["block"]
    pos = pos_in_segment(d0["ends"], y)
    feats = lambda m: make_X(dsets[m[0]], ch_select(m[1]), m[2], m[3])
    Xrt = [feats(m) for m in rt_members]
    Xlr = [feats(m) for m in lr_members]
    P = np.full((len(y), 3), np.nan)
    fit_t = []
    for b in range(2, 8):
        tr, te = blk < b, blk == b
        t0 = time.perf_counter()
        mrt = [fit_member(X, y, pos, "RT", tr) for X in Xrt]
        mlr = [fit_member(X, y, pos, "LR", tr) for X in Xlr]
        fit_t.append(time.perf_counter() - t0)
        zt = np.mean([logit(m.predict_proba(X[te])[:, 1]) for m, X in zip(mrt, Xrt)], 0)
        zr = np.mean([logit(m.predict_proba(X[te])[:, 1]) for m, X in zip(mlr, Xlr)], 0)
        P[te] = combine(sigmoid(zt), sigmoid(zr))
    return P, fit_t


def run_cascade(banks, rt_members, lr_members):
    names = list(banks["bp01_20"])

    def fn(n):
        return _session_cascade({k: banks[k][n] for k in banks}, rt_members, lr_members)
    out = run_sessions(fn, names)
    return {n: o[0] for n, o in out.items()}, np.concatenate([o[1] for o in out.values()])


def run_within3(bank, member=("bp01_50", "chan", (0, 1, 2, 3, 4, 6, 8), "diff")):
    """Альтернатива: одна 3-классовая sLDA с равными priors."""
    def fn(n):
        d = bank[n]
        X = make_X(d, ch_select(member[1]), member[2], member[3])
        P = np.full((len(X), 3), np.nan)
        for b in range(2, 8):
            tr, te = d["block"] < b, d["block"] == b
            P[te] = LDA(solver="lsqr", shrinkage="auto", priors=[1 / 3] * 3).fit(X[tr], d["y"][tr]).predict_proba(X[te])
        return P
    return run_sessions(fn, list(bank))


def smooth(probs, banks_any, tau_win=4.0):
    """Причинное EMA вероятностей по предыдущим окнам той же сессии (только для справки)."""
    a = 1.0 / tau_win
    out = {}
    for n, P in probs.items():
        Q = P.copy()
        e = banks_any[n]["ends"]
        for i in range(1, len(P)):
            if np.isnan(P[i]).any() or np.isnan(Q[i - 1]).any() or e[i] - e[i - 1] > 70:
                continue
            Q[i] = (1 - a) * Q[i - 1] + a * P[i]
        out[n] = Q
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    banks = {k: load_bank(f"nirs_bank_{k}") for k in BANKS}
    ref = banks["bp01_20"]
    out_dir = PROJECT_DIR / "data" / "probs"
    out_dir.mkdir(parents=True, exist_ok=True)

    P_ens, ft = run_cascade(banks, RT_MEMBERS, LR_MEMBERS)
    report("cascade_ens (8 RT × 4 LR sLDA)", ref, P_ens)
    print(f"  fit_block (все 12 sLDA) median={np.median(ft):.3f}s max={ft.max():.3f}s")
    P_one, _ = run_cascade(banks, RT_MEMBERS[:1], LR_MEMBERS[:1])
    report("cascade_single (1 RT × 1 LR sLDA)", ref, P_one)
    P_w3 = run_within3(banks["bp01_50"])
    report("within3_slda (3-кл. sLDA, равные priors)", ref, P_w3)
    for tau in (4, 8):
        report(f"  [справка] cascade_ens + EMA τ={tau} окон", ref, smooth(P_ens, ref, tau))

    for nm, P in [("nirs_cascade_ens", P_ens), ("nirs_cascade_single", P_one), ("nirs_within3_slda", P_w3)]:
        with open(out_dir / f"{nm}.pkl", "wb") as f:
            pickle.dump({k: v.astype(np.float32) for k, v in P.items()}, f)
    print("saved →", out_dir)
