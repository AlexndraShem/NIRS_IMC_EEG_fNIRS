"""Потоковая симуляция OnlineEEG на сырых float64-сессиях: чанки по 25 отсчётов (100 мс),
predict в каждой точке сетки 250 мс, fit_block по окончании блока. Замер реальных времён и
сверка вероятностей с оффлайн-эмуляцией (data/probs/eeg_<name>.pkl).

  python -m src.eeg.simulate <name> <win> <chans: nofront|all> <n_sessions>
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

from src.dataset.io import list_sessions, load_session
from src.dataset.windows import window_ends
from src.eeg.cache import load_raw64
from src.eeg.common import PROBS_DIR
from src.eeg.final import MODEL_DIR
from src.eeg.online import OnlineEEG, PoolLR


def simulate(path, name, win, chans, alpha=0.8, chunk=25):
    s = load_session(path)
    x = load_raw64(path)
    pool = PoolLR.load(MODEL_DIR / f"pool_{name}_{s.subject}.npz")
    m = OnlineEEG(pool, win_s=win, chans=chans, alpha=alpha)
    ends, y, blk = window_ends(s.states, s.blocks, s.eeg_fs)
    want = dict(zip(ends.tolist(), range(len(ends))))
    P = np.full((len(ends), 3), np.nan)
    last_of_block = {b: int(np.flatnonzero(s.blocks == b)[-1]) + 1 for b in range(1, 8)}
    tp, tf = [], []
    n, fitted_upto, cur = 0, 0, 1
    while n < len(x):
        k = min(chunk, len(x) - n, last_of_block.get(cur, len(x)) - n) or min(chunk, len(x) - n)
        # предсказание в точках сетки: режем чанк так, чтобы попасть в каждую нужную точку
        nxt = [e for e in ends[(ends > n) & (ends <= n + k)]]
        stops = sorted(set(nxt + [n + k]))
        for st in stops:
            m.push(x[n:st]); n = st
            if n in want and blk[want[n]] >= 2:
                t0 = time.perf_counter(); P[want[n]] = m.predict_proba(); tp.append(time.perf_counter() - t0)
        if cur <= 7 and n == last_of_block[cur]:
            t0 = time.perf_counter(); m.fit_block(cur, s.states[fitted_upto:n]); tf.append(time.perf_counter() - t0)
            fitted_upto = n; cur += 1
    return s.name, P, np.array(tp), np.array(tf)


if __name__ == "__main__":
    from src.eeg.exp_cross import CHSETS
    name, win, chn, k = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    ref = pickle.load(open(PROBS_DIR / f"eeg_{name}.pkl", "rb"))
    paths = list_sessions()[:: max(1, len(list_sessions()) // k)][:k]
    for p in paths:
        nm, P, tp, tf = simulate(p, name, win, CHSETS[chn])
        R = ref[nm]; ok = ~np.isnan(R[:, 0])
        print(f"{nm}: |P_stream - P_offline| max={np.nanmax(np.abs(P[ok] - R[ok])):.2e} "
              f"argmax agree={np.mean(P[ok].argmax(1) == R[ok].argmax(1)):.4f} | predict mean={tp.mean()*1e3:.1f}ms "
              f"max={tp.max()*1e3:.1f}ms | fit_block s={np.round(tf, 3).tolist()}", flush=True)
