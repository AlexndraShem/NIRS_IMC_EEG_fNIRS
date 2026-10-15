"""Финальная оценка ЭЭГ-декодера (TS фильтр-банка + within sLDA + пул-LR) на float64-кэше,
сохранение вероятностей и пул-артефактов.

  python -m src.eeg.final <win> <chans> <shrinkage> <alpha> <name> <scratch_dir> [--nosave]

Протокол: для блока b сессии — референс и within-sLDA по блокам < b этой сессии; пул-LR
обучен на всех сессиях ДРУГИХ участников (leave-subject-out), каждая рецентрирована по
собственному референсу. Выход: data/probs/eeg_<name>.pkl (+ _within, _pool),
data/features/eeg_models/pool_<name>_<subj>.npz (LOSO) и pool_<name>_all.npz (для сдачи).
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

from NIRS_IMC_EEG_fNIRS.src.dataset.io import PROJECT_DIR
from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_eeg_cache
from NIRS_IMC_EEG_fNIRS.src.eeg.common import report, save_probs, score, subcov
from NIRS_IMC_EEG_fNIRS.src.eeg.exp_cross import CHSETS, feats_session
from NIRS_IMC_EEG_fNIRS.src.eeg.exp_cross2 import mix
from NIRS_IMC_EEG_fNIRS.src.eeg.online import PoolLR, fit_within

MODEL_DIR = PROJECT_DIR / "data" / "features" / "eeg_models"
_P = {}


def fit_pool(s, C=0.01):
    F, data, names = _P["F"], _P["data"], _P["names"]
    tr = [n for n in names if data[n]["subject"] != s]
    X = np.vstack([F[n]["pool"] for n in tr]); y = np.concatenate([data[n]["y"] for n in tr])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(sc.transform(X), y)
    return PoolLR.from_sklearn(sc, m)


def within_one(n):
    """Точная эмуляция OnlineEEG.fit_block/predict на кэше ковариаций (та же fit_within)."""
    data, shrink, pools, chn = _P["data"], _P["shrink"], _P["pools"], _P["chn"]
    d = data[n]
    c = subcov(d["covs"], CHSETS[chn]).astype(np.float64)
    Pw = np.full((len(d["y"]), 3), np.nan); Pp = Pw.copy()
    times = []
    for b in range(2, 8):
        tr, te = d["block"] < b, d["block"] == b
        t0 = time.perf_counter()
        ref, w = fit_within(c[tr], d["y"][tr], shrink)
        times.append(time.perf_counter() - t0)
        X = ref.ts(c[te])
        Pw[te] = np.exp(w.predict_log_proba(X))
        Pp[te] = np.exp(pools[d["subject"]].predict_log_proba(X))
    return n, Pw, Pp, times


def main(win, chn, shrink, alpha, name, sdir, save=True):
    import NIRS_IMC_EEG_fNIRS.src.eeg.common as cm
    cm.EIG_THREADS = 1          # оценка параллелится процессами; потоки — только в онлайне
    data = load_eeg_cache(win, f64=True)
    names = list(data)
    fcache = Path(sdir) / f"tsF64_w{win:g}_{chn}.pkl"
    if fcache.exists():
        F = pickle.load(open(fcache, "rb"))
    else:
        with ProcessPoolExecutor(10) as ex:
            F = dict(zip(names, ex.map(feats_session, [(subcov(data[n]["covs"], CHSETS[chn]).astype(np.float64),
                                                       data[n]["block"], "ts") for n in names])))
        F = {n: {k: v.astype(np.float32) for k, v in d.items()} for n, d in F.items()}
        pickle.dump(F, open(fcache, "wb"), protocol=4)
    subjects = sorted({data[n]["subject"] for n in names})
    _P.update(F=F, data=data, names=names, shrink=shrink, chn=chn)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(10, mp_context=ctx) as ex:
        pools = dict(zip(subjects + ["all"], ex.map(fit_pool, subjects + ["__none__"])))
    if save:
        for s, p in pools.items():
            p.save(MODEL_DIR / f"pool_{name}_{s}.npz")
    _P["pools"] = pools
    with ProcessPoolExecutor(10, mp_context=ctx) as ex:
        out = list(ex.map(within_one, names))
    Pw = {x[0]: x[1] for x in out}; Pp = {x[0]: x[2] for x in out}
    ft = np.concatenate([x[3] for x in out])
    print(f"fit_within (эмуляция fit_block) mean={ft.mean():.3f}s max={ft.max():.3f}s", flush=True)
    report(f"{name} within sLDA({shrink})", score(data, Pw))
    report(f"{name} pool LR (LOSubjO)", score(data, Pp))
    for a in [0.5, 0.6, 0.7, 0.8]:
        report(f"{name} mix a={a}", score(data, mix(Pw, Pp, a)))
    Pm = mix(Pw, Pp, alpha)
    sc = score(data, Pm)
    report(f"{name} FINAL a={alpha}", sc)
    if save:
        save_probs(name, Pm); save_probs(name + "_within", Pw); save_probs(name + "_pool", Pp)
    per = {n: (r["min_f1"], r["recall"], data[n]["subject"]) for n, r in sc["per"].items()}
    pickle.dump(per, open(Path(sdir) / f"final_per_{name}.pkl", "wb"))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    a = sys.argv[1:]
    main(float(a[0]), a[1], float(a[2]), float(a[3]), a[4],
         a[5] if len(a) > 5 else "/tmp", save="--nosave" not in a)
