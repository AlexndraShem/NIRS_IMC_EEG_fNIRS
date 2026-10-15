"""Эксперимент subject_pool (ЭЭГ): пул из прошлых сессий ТОГО ЖЕ участника против общего пула.

Все пулы — логистическая регрессия (стандартизация + LR, class_weight="balanced") на TS-признаках
сессий пула, каждая сессия рецентрирована по собственному референсу (ключ "pool" в кэше
`tsF64_w2_all.pkl`, см. src.eeg.final). Целевая сессия n для блока b — признаки F[n][b]
(референс по блокам < b этой сессии), предсказываются окна блока b.

Варианты пулов для сессии n участника s (дата — префикс имени файла):
  G0  — другие участники (leave-one-subject-out, текущая сдача; LR C=0.01);
  GA  — все остальные сессии (протокол A, leave-one-session-out; C=0.01);
  SA  — только другие сессии участника s (протокол A; C ∈ {0.01, 0.1});
  GB  — все более ранние сессии (протокол B, хронологический; C=0.01);
  SB  — более ранние сессии участника s (протокол B; C ∈ {0.01, 0.1});
  G0B — другие участники + более ранние сессии s («смешанный» хронологический).

    python -m src.eeg.exp_subjpool <tsF64_w2_all.pkl> <out.pkl> [--jobs 8] [--kinds GA,SA]
                                   [--save-dir data/features/subjpool_loseso] [--force]

--save-dir сохраняет модели GA/SA (C из задачи) как dev-артефакты leave-one-session-out.

Выход: {вариант: {сессия: P (n_окон, 3)}}, блок 1 — NaN (строки как в data/features/eeg_fb64).
Анализ/смеси/слияние с NIRS — `src/eeg/exp_subjpool_eval.py`.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eeg.cache import load_eeg_cache
from src.eeg.final import MODEL_DIR

_P = {}


def _pool_sessions(n, kind):
    names, subj = _P["names"], _P["subj"]
    s = subj[n]
    if kind == "G0":
        return [m for m in names if subj[m] != s]
    if kind == "GA":
        return [m for m in names if m != n]
    if kind == "SA":
        return [m for m in names if m != n and subj[m] == s]
    if kind == "GB":
        return [m for m in names if m < n]
    if kind == "SB":
        return [m for m in names if m < n and subj[m] == s]
    if kind == "G0B":
        return [m for m in names if subj[m] != s or m < n]
    raise ValueError(kind)


def _fit_predict(args):
    n, kind, C = args
    F, data = _P["F"], _P["data"]
    tr = _pool_sessions(n, kind)
    d = data[n]
    P = np.full((len(d["y"]), 3), np.nan)
    if not tr:
        return n, kind, C, None, 0.0
    t0 = time.perf_counter()
    X = np.vstack([F[m]["pool"] for m in tr]).astype(np.float64)
    y = np.concatenate([data[m]["y"] for m in tr])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=300, class_weight="balanced")
    w0 = _P.get("warm")
    if w0 is not None and kind not in ("SA", "SB"):
        # тёплый старт от пула на всех 56 сессиях: задача выпуклая, оптимум тот же, итераций меньше
        lr.set_params(warm_start=True)
        lr.coef_, lr.intercept_ = w0[0].copy(), w0[1].copy()
    lr.fit(sc.transform(X), y)
    dt = time.perf_counter() - t0
    sd = _P.get("save_dir")
    if sd and kind in ("GA", "SA"):
        # dev-артефакты leave-one-session-out: <save_dir>/<сессия>/{eeg_pool.npz | eeg_subj/<участник>.npz}
        from src.eeg.online import PoolLR
        dst = Path(sd) / n / ("eeg_pool.npz" if kind == "GA" else f"eeg_subj/{_P['subj'][n]}.npz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        PoolLR.from_sklearn(sc, lr).save(dst)
    for b in range(2, 8):
        blk = d["block"][d["block"] <= b]
        Xb = F[n][b][blk == b]
        P[d["block"] == b] = lr.predict_proba(sc.transform(Xb))
    return n, kind, C, P.astype(np.float32), dt


def main(fpath, out, jobs=8, kinds=("SA", "SB", "GA", "GB", "G0B", "G0"), save_dir=None, force=False):
    data = load_eeg_cache(2.0, f64=True)
    names = sorted(data)
    F = pickle.load(open(fpath, "rb"))
    _P.update(F=F, data=data, names=names, subj={n: data[n]["subject"] for n in names}, save_dir=save_dir)
    wp = MODEL_DIR / "pool_ts2s_all_all.npz"
    if wp.exists():
        z = np.load(wp)
        _P["warm"] = (z["coef"].astype(np.float64), z["intercept"].astype(np.float64))
    tasks = []
    for k in kinds:
        for C in ((0.01, 0.1) if k in ("SA", "SB") else (0.01,)):
            tasks += [(n, k, C) for n in names]
    res = pickle.load(open(out, "rb")) if os.path.exists(out) else {}
    tasks = [t for t in tasks if force or f"{t[1]}_C{t[2]:g}" not in res or t[0] not in res[f"{t[1]}_C{t[2]:g}"]]
    print(f"задач: {len(tasks)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        for i, (n, k, C, P, dt) in enumerate(ex.map(_fit_predict, tasks)):
            key = f"{k}_C{C:g}"
            res.setdefault(key, {})[n] = P
            if (i + 1) % 8 == 0 or i + 1 == len(tasks):
                print(f"{i + 1}/{len(tasks)} {key} {n} fit={dt:.1f}s elapsed={time.time() - t0:.0f}s", flush=True)
                pickle.dump(res, open(out, "wb"), protocol=4)
    pickle.dump(res, open(out, "wb"), protocol=4)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    a = sys.argv[1:]
    jobs = int(a[a.index("--jobs") + 1]) if "--jobs" in a else 8
    kinds = a[a.index("--kinds") + 1].split(",") if "--kinds" in a else ("SA", "SB", "GA", "GB", "G0B", "G0")
    main(a[0], a[1], jobs, kinds, a[a.index("--save-dir") + 1] if "--save-dir" in a else None,
         "--force" in a)
