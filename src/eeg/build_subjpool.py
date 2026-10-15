"""Артефакты подучастниковых ЭЭГ-пулов (эксперимент subject_pool).

  * сдача:   solution/artifacts/eeg_subj/<участник>.npz — LR (C=0.1) на ВСЕХ сессиях участника;
             общий пул solution/artifacts/eeg_pool.npz = pool_ts2s_all_all.npz (все 56 сессий, без изменений);
  * dev LOSessO: <loseso_dir>/<сессия>/eeg_subj/<участник>.npz — LR на остальных сессиях участника,
             <loseso_dir>/<сессия>/eeg_pool.npz — общий пул на всех сессиях, кроме этой (GA; берётся из
             --ga-dir, если уже обучен в src.eeg.exp_subjpool, иначе обучается здесь с тёплым стартом).

    python -m src.eeg.build_subjpool <tsF64_w2_all.pkl> [--ga-dir <scratch>/loseso_eeg]
           [--loseso-dir data/features/subjpool_loseso] [--art-dir solution/artifacts] [--jobs 8]
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.dataset.io import PROJECT_DIR
from src.eeg.cache import load_eeg_cache
from src.eeg.final import MODEL_DIR
from src.eeg.online import PoolLR

C_SUBJ = 0.1
C_GEN = 0.01
_P = {}


def _fit(args):
    sessions, C, dst, warm = args
    F, data = _P["F"], _P["data"]
    X = np.vstack([F[m]["pool"] for m in sessions]).astype(np.float64)
    y = np.concatenate([data[m]["y"] for m in sessions])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=300, class_weight="balanced")
    if warm:
        z = np.load(MODEL_DIR / "pool_ts2s_all_all.npz")
        lr.set_params(warm_start=True)
        lr.coef_, lr.intercept_ = z["coef"].astype(np.float64), z["intercept"].astype(np.float64)
    lr.fit(sc.transform(X), y)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    PoolLR.from_sklearn(sc, lr).save(dst)
    return str(dst)


def main(fpath, ga_dir=None, loseso_dir=None, art_dir=None, jobs=8):
    loseso_dir = Path(loseso_dir or PROJECT_DIR / "data/features/subjpool_loseso")
    art_dir = Path(art_dir or PROJECT_DIR / "solution/artifacts")
    data = load_eeg_cache(2.0, f64=True)
    names = sorted(data)
    subj = {n: data[n]["subject"] for n in names}
    _P.update(F=pickle.load(open(fpath, "rb")), data=data)
    tasks = []
    for s in sorted(set(subj.values())):
        tasks.append(([m for m in names if subj[m] == s], C_SUBJ, art_dir / "eeg_subj" / f"{s}.npz", False))
    for n in names:
        tasks.append(([m for m in names if m != n and subj[m] == subj[n]], C_SUBJ,
                      loseso_dir / n / "eeg_subj" / f"{subj[n]}.npz", False))
        dst = loseso_dir / n / "eeg_pool.npz"
        src = Path(ga_dir) / n / "eeg_pool.npz" if ga_dir else None
        if src is not None and src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            tasks.append(([m for m in names if m != n], C_GEN, dst, True))
    tasks.sort(key=lambda t: -len(t[0]))            # большие пулы — первыми
    print(f"задач: {len(tasks)}", flush=True)
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        for i, d in enumerate(ex.map(_fit, tasks)):
            print(i + 1, d, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0].startswith("-") or not Path(a[0]).is_file():
        sys.exit(__doc__ + "\nнужен путь к tsF64_w2_all.pkl — кэш TS-признаков, его пишет "
                 "`python -m src.eeg.final 2 all 0.8 0.8 ts2s_all <scratch>`")
    opt = lambda k, dflt=None: a[a.index(k) + 1] if k in a else dflt
    main(a[0], opt("--ga-dir"), opt("--loseso-dir"), opt("--art-dir"), int(opt("--jobs", 8)))
