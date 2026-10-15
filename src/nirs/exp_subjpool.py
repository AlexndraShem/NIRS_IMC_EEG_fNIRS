"""Эксперимент subject_pool (NIRS): кросс-сессионный пул того же участника для каскада sLDA.

Для целевой сессии n (участник s) и каждого из 12 членов каскада (`src.nirs.cascade`):
  * within  — sLDA на блоках < b этой сессии (как в сдаче; эталон без прореживания);
  * pool    — sLDA на ВСЕХ окнах сессий пула (признаки каждой сессии нормированы своим std блока 1);
  * aug     — sLDA на объединении «пул + блоки < b текущей сессии» (переобучается на каждом блоке).
Пулы: SA — другие сессии участника (протокол A), SB — более ранние сессии участника (B),
GA — все остальные сессии (A, справочно), G0 — другие участники (справочно, как прежний LOSO).

Сохраняются средние логиты RT (задача/покой) и LR (правая/левая) по членам — смеси считаются
в `src/eeg/exp_subjpool_eval.py`.

    python -m src.nirs.exp_subjpool <out.pkl> [--jobs 8] [--kinds SA,SB,GA]
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

from src.nirs.cascade import BANKS, LR_MEMBERS, LR_TMIN, RT_MEMBERS, logit, pos_in_segment, slda
from src.nirs.evaluate import ch_select, load_bank, make_X

MEMBERS = [("RT", m) for m in RT_MEMBERS] + [("LR", m) for m in LR_MEMBERS]


def _sel(task, y, pos, mask):
    if task == "RT":
        return mask, y > 1
    return mask & (y > 1) & (pos >= LR_TMIN), y == 3


def _feats(banks, n):
    d0 = banks["bp01_20"][n]
    X = [make_X(banks[m[0]][n], ch_select(m[1]), m[2], m[3]) for _, m in MEMBERS]
    return X, d0["y"], d0["block"], pos_in_segment(d0["ends"], d0["y"])


def _pool_names(n, kind, names, subj):
    s = subj[n]
    return {"SA": [m for m in names if m != n and subj[m] == s],
            "SB": [m for m in names if m < n and subj[m] == s],
            "GA": [m for m in names if m != n],
            "G0": [m for m in names if subj[m] != s]}[kind]


def run_one(n, banks, names, subj, kinds, do_aug):
    warnings.filterwarnings("ignore")
    X, y, blk, pos = _feats(banks, n)
    out = {"y": y, "block": blk}
    nm = len(MEMBERS)
    te_all = blk >= 2
    # within (эталон, без прореживания)
    Z = np.full((nm, len(y)), np.nan)
    for b in range(2, 8):
        tr, te = blk < b, blk == b
        for i, (task, _) in enumerate(MEMBERS):
            sel, yb = _sel(task, y, pos, tr)
            Z[i, te] = logit(slda().fit(X[i][sel], yb[sel]).predict_proba(X[i][te])[:, 1])
    out["within"] = Z
    t_aug = []
    for kind in kinds:
        pn = _pool_names(n, kind, names, subj)
        if not pn:
            continue
        PF = [_feats(banks, m) for m in pn]
        Xp = [np.vstack([f[0][i] for f in PF]) for i in range(nm)]
        yp = np.concatenate([f[1] for f in PF]); pp = np.concatenate([f[3] for f in PF])
        Zp = np.full((nm, len(y)), np.nan)
        for i, (task, _) in enumerate(MEMBERS):
            sel, yb = _sel(task, yp, pp, np.ones(len(yp), bool))
            Zp[i, te_all] = logit(slda().fit(Xp[i][sel], yb[sel]).predict_proba(X[i][te_all])[:, 1])
        out[kind] = Zp
        if do_aug and kind in ("SA", "SB"):
            Za = np.full((nm, len(y)), np.nan)
            for b in range(2, 8):
                tr, te = blk < b, blk == b
                t0 = time.perf_counter()
                for i, (task, _) in enumerate(MEMBERS):
                    s1, y1 = _sel(task, yp, pp, np.ones(len(yp), bool))
                    s2, y2 = _sel(task, y, pos, tr)
                    m = slda().fit(np.vstack([Xp[i][s1], X[i][s2]]), np.r_[y1[s1], y2[s2]])
                    Za[i, te] = logit(m.predict_proba(X[i][te])[:, 1])
                t_aug.append(time.perf_counter() - t0)
            out[kind + "_aug"] = Za
    out["t_aug"] = t_aug
    return n, out


def main(out_path, jobs=8, kinds=("SA", "SB", "GA"), do_aug=True):
    banks = {k: load_bank(f"nirs_bank_{k}") for k in BANKS}
    names = sorted(banks["bp01_20"])
    subj = {n: banks["bp01_20"][n]["subject"] for n in names}
    res = Parallel(n_jobs=jobs)(delayed(run_one)(n, banks, names, subj, kinds, do_aug) for n in names)
    pickle.dump(dict(res), open(out_path, "wb"), protocol=4)
    print("saved", out_path, len(res))


if __name__ == "__main__":
    a = sys.argv[1:]
    jobs = int(a[a.index("--jobs") + 1]) if "--jobs" in a else 8
    kinds = a[a.index("--kinds") + 1].split(",") if "--kinds" in a else ("SA", "SB", "GA")
    main(a[0], jobs, kinds, "--noaug" not in a)
