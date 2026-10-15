"""Артефакты подучастниковых NIRS-пулов (эксперимент subject_pool).

Для каждого из 12 членов каскада (`src.nirs.cascade.RT_MEMBERS + LR_MEMBERS`) — бинарная sLDA
(lsqr, Ledoit–Wolf, priors 0.5/0.5) на всех окнах сессий пула (признаки каждой сессии нормированы
её std блока 1; L/R — только окна задачи с позицией ≥ 3 с). Сохраняются coef_k / intercept (лог-шансы),
онлайн: z_pool = clip(X·coef + b, ±logit(1 − 1e-4)), смешивается с within-логитами в
`solution/nirs_model.py::NirsCascadeEns` (веса pool_weight_rt / pool_weight_lr).

  * сдача:   <art_dir>/nirs_subj/<участник>.npz — все сессии участника с NIRS;
  * dev LOSessO: <loseso_dir>/<сессия>/nirs_subj/<участник>.npz — остальные сессии участника.

    python -m src.nirs.build_subjpool [--loseso-dir data/features/subjpool_loseso]
                                     [--art-dir solution/artifacts] [--jobs 8]
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src.dataset.io import PROJECT_DIR
from src.nirs.cascade import BANKS
from src.nirs.exp_subjpool import MEMBERS, _feats, _sel
from src.nirs.evaluate import load_bank
from src.nirs.cascade import slda


def fit_save(sessions, banks, dst):
    warnings.filterwarnings("ignore")
    PF = [_feats(banks, m) for m in sessions]
    yp = np.concatenate([f[1] for f in PF]); pp = np.concatenate([f[3] for f in PF])
    out = {"members": np.array(json.dumps([[t, list(m[0:2]) + [list(m[2]), m[3]]] for t, m in MEMBERS])),
           "sessions": np.array(sessions)}
    icpt = []
    for i, (task, _) in enumerate(MEMBERS):
        X = np.vstack([f[0][i] for f in PF])
        sel, yb = _sel(task, yp, pp, np.ones(len(yp), bool))
        m = slda().fit(X[sel], yb[sel])
        out[f"coef_{i}"] = m.coef_[0].astype(np.float64)
        icpt.append(float(m.intercept_[0]))
    out["intercept"] = np.array(icpt)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    return str(dst)


def main(loseso_dir=None, art_dir=None, jobs=8):
    loseso_dir = Path(loseso_dir or PROJECT_DIR / "data/features/subjpool_loseso")
    art_dir = Path(art_dir or PROJECT_DIR / "solution/artifacts")
    banks = {k: load_bank(f"nirs_bank_{k}") for k in BANKS}
    names = sorted(banks["bp01_20"])                   # только сессии с NIRS
    subj = {n: banks["bp01_20"][n]["subject"] for n in names}
    all_names = sorted(p.stem for p in (PROJECT_DIR / "data/features/v1_win1").glob("*.npz"))
    all_subj = {n: n.split("_")[-1] for n in all_names}
    tasks = []
    for s in sorted(set(subj.values())):
        tasks.append(([m for m in names if subj[m] == s], art_dir / "nirs_subj" / f"{s}.npz"))
    for n in all_names:                                  # и сессиям без NIRS (на случай деградации)
        pool = [m for m in names if m != n and subj[m] == all_subj[n]]
        if pool:
            tasks.append((pool, loseso_dir / n / "nirs_subj" / f"{all_subj[n]}.npz"))
    res = Parallel(n_jobs=jobs)(delayed(fit_save)(p, banks, d) for p, d in tasks)
    print("saved", len(res))


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, dflt=None: a[a.index(k) + 1] if k in a else dflt
    main(opt("--loseso-dir"), opt("--art-dir"), int(opt("--jobs", 8)))
