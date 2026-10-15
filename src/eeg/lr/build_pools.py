"""Пул-артефакты экспертов eeg_lr_nbsm4 («nb» и «sm4») для сдачи и dev-протоколов.

  * сдача: solution/artifacts/eeg_lrnb_pool.npz, eeg_lrsm4_pool.npz — общий пул на всех 56 сессиях
    (LR C=0.01), eeg_lr{nb,sm4}_subj/<участник>.npz — пул участника на всех его сессиях (C=0.1);
  * LOSubjO: data/features/eeg_models/pool_lr{nb,sm4}_<участник>.npz (уже строит src.eeg.lr.final --pools);
  * LOSessO: data/features/subjpool_loseso/<S>/eeg_lr{nb,sm4}_pool.npz (все сессии, кроме S) и
    eeg_lr{nb,sm4}_subj/<участник>.npz (остальные сессии участника);
  * хронологический: data/features/chrono/<S>/... — только сессии раньше S (общий пул — цепочками
    с тёплым стартом от пула предыдущей сессии, как src/scripts/build_chrono.py).

Существующие файлы в каталогах не трогаются — только добавляются новые.

    python -m src.eeg.lr.build_pools [--jobs 8] [--only art,loseso,chrono]
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from NIRS_IMC_EEG_fNIRS.src.dataset.io import PROJECT_DIR
from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_eeg_cache
from NIRS_IMC_EEG_fNIRS.src.eeg.lr import feats as FT
from NIRS_IMC_EEG_fNIRS.src.eeg.online import PoolLR

EXPERTS = {"nb": ("nbboth", 2.0), "sm4": ("tssm", 4.0)}
ART = PROJECT_DIR / "solution" / "artifacts"
LOSESO = PROJECT_DIR / "data" / "features" / "subjpool_loseso"
CHRONO = PROJECT_DIR / "data" / "features" / "chrono"
_P = {}


def pool_features(feat, win, jobs):
    data = load_eeg_cache(win, f64=True)
    FT._G.clear()
    FT._G.update(data=data, feat=feat, opts=[], mir=FT.mirror_fn(feat))
    if FT.FEATS[feat][1] == "nblogvar":
        from NIRS_IMC_EEG_fNIRS.src.eeg.lr.nbcache import OUT as NB_OUT
        FT._G["nb"] = {n: np.load(NB_OUT / f"{n}.npz")[f"var_w{win:g}"] for n in data}
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        F = dict(ex.map(FT.feats_session, list(data)))
    X = {n: F[n]["pool"].astype(np.float64) for n in data}
    y = {n: data[n]["y"] for n in data}
    subj = {n: data[n]["subject"] for n in data}
    return X, y, subj


def _lr(sessions, C, warm=None):
    X = np.vstack([_P["X"][m] for m in sessions]); y = np.concatenate([_P["y"][m] for m in sessions])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=300, class_weight="balanced")
    if warm is not None:
        lr.set_params(warm_start=True)
        lr.coef_, lr.intercept_ = warm[0].copy(), warm[1].copy()
    lr.fit(sc.transform(X), y)
    return sc, lr


def _save(sc, lr, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    PoolLR.from_sklearn(sc, lr).save(dst)


def task(args):
    sessions, C, dst, warm_all = args
    warm = None
    if warm_all:
        z = np.load(_P["all_file"])
        warm = (z["coef"].astype(np.float64), z["intercept"].astype(np.float64))
    t0 = time.perf_counter()
    sc, lr = _lr(sessions, C, warm)
    _save(sc, lr, dst)
    return f"{dst} ({len(sessions)} сессий, {time.perf_counter() - t0:.0f} с)"


def chain(idx):
    names, pref = _P["names"], _P["prefix"]
    warm, log = None, []
    for i in idx:
        n = names[i]
        prev = names[:i]
        if not prev:
            continue
        sc, lr = _lr(prev, 0.01, warm)
        warm = (lr.coef_, lr.intercept_)
        _save(sc, lr, CHRONO / n / f"{pref}_pool.npz")
        log.append(f"chrono {n}: {len(prev)}")
    return log


def main(jobs=8, only=("art", "loseso", "chrono")):
    ctx = mp.get_context("fork")
    for name, (feat, win) in EXPERTS.items():
        X, y, subj = pool_features(feat, win, jobs)
        names = sorted(X)
        pref = f"eeg_lr{name}"
        _P.update(X=X, y=y, names=names, prefix=pref, all_file=ART / f"{pref}_pool.npz")
        tasks = []
        if "art" in only:
            sc, lr = _lr(names, 0.01)
            _save(sc, lr, ART / f"{pref}_pool.npz")
            print(f"{name}: общий пул сдачи", flush=True)
            for s in sorted(set(subj.values())):
                tasks.append(([m for m in names if subj[m] == s], 0.1, ART / f"{pref}_subj" / f"{s}.npz", False))
        for n in names:
            same = [m for m in names if subj[m] == subj[n]]
            if "loseso" in only:
                tasks.append(([m for m in names if m != n], 0.01, LOSESO / n / f"{pref}_pool.npz", True))
                oth = [m for m in same if m != n]
                if oth:
                    tasks.append((oth, 0.1, LOSESO / n / f"{pref}_subj" / f"{subj[n]}.npz", False))
            if "chrono" in only:
                prev = [m for m in same if m < n]
                if prev:
                    tasks.append((prev, 0.1, CHRONO / n / f"{pref}_subj" / f"{subj[n]}.npz", False))
        tasks.sort(key=lambda t: -len(t[0]))
        with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
            for r in ex.map(task, tasks):
                print(r, flush=True)
        if "chrono" in only:
            chains = [list(c) for c in np.array_split(np.arange(len(names)), jobs)]
            with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
                for log in ex.map(chain, chains):
                    print("\n".join(log), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    main(int(opt("--jobs", 8)), tuple(opt("--only", "art,loseso,chrono").split(",")))
