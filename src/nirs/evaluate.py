"""Оффлайн-оценка NIRS-декодеров в онлайн-порядке (блоки 2–7, обучение только на прошлом).

Данные — кэш тензоров лагов `data/features/nirs_bank_*` (см. build_bank.py).
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from src.dataset.io import PROJECT_DIR
from src.dataset.windows import confusion, metrics

FEAT_DIR = PROJECT_DIR / "data" / "features"
N_CH_HALF = 29          # o0..o17, olat0..8, oglob, olatm


def load_bank(tag):
    out = {}
    for p in sorted((FEAT_DIR / tag).glob("*.npz")):
        z = np.load(p)
        if "T" not in z.files:
            continue
        d = {k: z[k] for k in z.files}
        d["subject"] = str(z["subject"])
        out[p.stem] = d
    return out


def mirror_perm():
    """Перестановка/знаки производных каналов при зеркалировании полушарий (левое ↔ правое)."""
    perm = np.arange(2 * N_CH_HALF)
    sign = np.ones(2 * N_CH_HALF)
    for h in (0, N_CH_HALF):
        for i in range(8):
            perm[h + i], perm[h + i + 8] = h + i + 8, h + i
        perm[h + 16], perm[h + 17] = h + 17, h + 16
        sign[h + 18:h + 27] = -1          # lat
        sign[h + 28] = -1                 # latm
    return perm, sign


def ch_select(kind):
    o = np.arange(N_CH_HALF)
    r = o + N_CH_HALF
    sets = {
        "all": np.r_[o, r],
        "hbo": o,
        "chan": np.r_[o[:18], r[:18]],
        "latglob": np.r_[o[18:], r[18:]],
        "chan_glob": np.r_[o[:18], o[27], r[:18], r[27]],
        "hbo_latglob": o[18:],
    }
    return sets[kind]


def make_X(d, chans, lags, mode="level", scale=True):
    """Признаки из тензора T (n, D, L). mode: level — значения на лагах;
    diff — значение сейчас + разности (сейчас − лаг)."""
    T = d["T"]
    if scale:
        T = T / d["scale"][None, :, None]
    li = [int(np.flatnonzero(d["lags"] == l)[0]) for l in lags]
    S = T[:, chans][:, :, li]
    if mode == "diff":
        S = np.concatenate([S[:, :, :1], S[:, :, :1] - S[:, :, 1:]], axis=2)
    return S.reshape(len(S), -1)


def score(bank, probs):
    res = {}
    for name, P in probs.items():
        d = bank[name]
        te = d["block"] >= 2
        res[name] = metrics(confusion(d["y"][te], P[te].argmax(1) + 1))
    f = float(np.mean([r["min_f1"] for r in res.values()]))
    r = float(np.mean([r["recall"] for r in res.values()]))
    return f, r, res


def run_sessions(fn, names, n_jobs=10):
    out = Parallel(n_jobs=n_jobs)(delayed(fn)(n) for n in names)
    return dict(zip(names, out))


def report(label, bank, probs):
    f, r, _ = score(bank, probs)
    print(f"{label:55s} minF1={f:.3f} recall={r:.3f} nirs_pts={5 + 15 * f:.2f}", flush=True)
    return f, r
