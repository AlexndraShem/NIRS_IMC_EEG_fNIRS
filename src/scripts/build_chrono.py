"""Dev-артефакты хронологического протокола (правило организаторов для метрик на train):
для сессии S — только модели, обученные на сессиях, записанных РАНЬШЕ S (полная метка времени
`YYYY.MM.DD.hh.mm.ss` в имени файла; сессии одного дня упорядочены по времени).

    data/features/chrono/<S>/eeg_pool.npz            — общий ЭЭГ-пул (LR C=0.01) на всех сессиях < S;
    data/features/chrono/<S>/eeg_subj/<участник>.npz — ЭЭГ-пул участника (LR C=0.1) на его сессиях < S;
    data/features/chrono/<S>/nirs_subj/<участник>.npz — NIRS-пул участника (12 sLDA) на его сессиях < S с NIRS.

Нет более ранних сессий (всех / участника / участника с NIRS) → соответствующего файла нет, и
онлайн-модель откатывается так же, как в сдаче (без общего пула — только within; без пула
участника — прежняя смесь и прежние константы слияния). Каталог создаётся для каждой S
(у самой первой он пуст), `solution/run.py::loseso_artifacts` сначала убирает из копии артефактов
все пулы сдачи, поэтому модели, видевшие S или более поздние сессии, не попадают в прогон.

Общий пул обучается цепочками с тёплым стартом от пула ПРЕДЫДУЩЕЙ сессии (он видел только
подмножество тех же более ранних данных, утечки нет): 8 цепочек параллельно, первая LR
каждой цепочки — холодный старт.

    python -m src.scripts.build_chrono <tsF64_w2_all.pkl> [--out data/features/chrono] [--jobs 8]
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.dataset.io import PROJECT_DIR
from src.eeg.cache import load_eeg_cache
from src.eeg.online import PoolLR

_P = {}


def _lr(sessions, C, warm=None):
    F, data = _P["F"], _P["data"]
    X = np.vstack([F[m]["pool"] for m in sessions]).astype(np.float64)
    y = np.concatenate([data[m]["y"] for m in sessions])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=300, class_weight="balanced")
    if warm is not None and len(np.unique(y)) == 3:
        lr.set_params(warm_start=True)
        lr.coef_, lr.intercept_ = warm[0].copy(), warm[1].copy()
    lr.fit(sc.transform(X), y)
    return sc, lr


def chain(idx):
    """Общие пулы для сессий names[idx] (подряд по времени), тёплый старт по цепочке."""
    names, out = _P["names"], Path(_P["out"])
    warm, log = None, []
    for i in idx:
        n = names[i]
        prev = names[:i]
        if not prev:
            continue
        t0 = time.perf_counter()
        sc, lr = _lr(prev, 0.01, warm)
        warm = (lr.coef_, lr.intercept_)
        PoolLR.from_sklearn(sc, lr).save(out / n / "eeg_pool.npz")
        log.append(f"{n}: общий пул из {len(prev)} сессий, {time.perf_counter() - t0:.0f} с")
    return log


def subj_eeg(n):
    names, subj, out = _P["names"], _P["subj"], Path(_P["out"])
    prev = [m for m in names if m < n and subj[m] == subj[n]]
    if not prev:
        return f"{n}: нет более ранних сессий участника (ЭЭГ)"
    sc, lr = _lr(prev, 0.1)
    dst = out / n / "eeg_subj" / f"{subj[n]}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    PoolLR.from_sklearn(sc, lr).save(dst)
    return f"{n}: пул участника ЭЭГ из {len(prev)}"


def main(fpath, out, jobs=8):
    out = Path(out)
    data = load_eeg_cache(2.0, f64=True)
    names = sorted(data)
    subj = {n: data[n]["subject"] for n in names}
    for n in names:
        (out / n).mkdir(parents=True, exist_ok=True)
    _P.update(F=pickle.load(open(fpath, "rb")), data=data, names=names, subj=subj, out=str(out))
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
        for r in ex.map(subj_eeg, names):
            print(r, flush=True)
    chains = [list(c) for c in np.array_split(np.arange(len(names)), jobs)]
    with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
        for log in ex.map(chain, chains):
            print("\n".join(log), flush=True)
    # NIRS-пулы участника
    from joblib import Parallel, delayed
    from src.nirs.build_subjpool import fit_save
    from src.nirs.cascade import BANKS
    from src.nirs.evaluate import load_bank
    banks = {k: load_bank(f"nirs_bank_{k}") for k in BANKS}
    nn = sorted(banks["bp01_20"])
    tasks = []
    for n in names:
        prev = [m for m in nn if m < n and subj.get(m) == subj[n]]
        if prev:
            tasks.append((prev, out / n / "nirs_subj" / f"{subj[n]}.npz"))
        else:
            print(f"{n}: нет более ранних сессий участника с NIRS")
    Parallel(n_jobs=jobs)(delayed(fit_save)(p, banks, d) for p, d in tasks)
    print("NIRS-пулов:", len(tasks))


USAGE = """нужен позиционный аргумент — кэш TS-признаков tsF64_w2_all.pkl (~3 ГБ):
    python -m src.scripts.build_chrono <scratch>/tsF64_w2_all.pkl [--out data/features/chrono] [--jobs 8]
Кэш пишет `python -m src.eeg.final 2 all 0.8 0.8 ts2s_all <scratch>` (в <scratch>/tsF64_w2_all.pkl).
Ещё нужны: data/features/eeg_fb64/ (`python -m src.eeg.cache --f64`) и
data/features/nirs_bank_{bp005_10,bp01_20,bp02_30,bp01_50}/ (`python -m src.nirs.build_bank`)."""


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0].startswith("-") or not Path(a[0]).is_file():
        sys.exit(USAGE if not a or a[0].startswith("-") else f"нет файла {a[0]}\n{USAGE}")
    opt = lambda k, dflt=None: a[a.index(k) + 1] if k in a else dflt
    main(a[0], opt("--out", str(PROJECT_DIR / "data/features/chrono")), int(opt("--jobs", 8)))
