"""Проверка онлайн-реализации: потоковый прогон сессии, сверка с оффлайн-вероятностями и замер времени.

OMP_NUM_THREADS=1 python -m src.nirs.simulate_online [сессия ...]
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings

import numpy as np

from src.dataset.io import PROJECT_DIR, RAW_DIR, load_session
from src.nirs.online import OnlineNirsDecoder

warnings.filterwarnings("ignore")


def simulate(name):
    s = load_session(RAW_DIR / "Data" / f"{name}.mat")
    v1 = np.load(PROJECT_DIR / "data" / "features" / "v1_win1" / f"{name}.npz")
    ends, blk = v1["ends"], v1["block"]
    dec = OnlineNirsDecoder(s.nirs_fs)
    P = np.full((len(ends), 3), np.nan)
    tp, tf = [], []
    pushed, cur = 0, 1
    bstart = {b: int(np.flatnonzero(s.blocks == b)[0]) for b in range(1, 8)}
    bend = {b: int(np.flatnonzero(s.blocks == b)[-1]) + 1 for b in range(1, 8)}
    for i, e in enumerate(ends):
        if blk[i] > cur:                       # блок(и) завершены → fit_block
            for b in range(cur, blk[i]):
                k = int(np.searchsorted(s.nirs_frame, bend[b], side="right"))
                dec.push(s.hbo[pushed:k], s.hbr[pushed:k], s.nirs_frame[pushed:k]); pushed = k
                t0 = time.perf_counter()
                dec.fit_block(s.states[bstart[b]:bend[b]], bstart[b])
                tf.append(time.perf_counter() - t0)
            cur = blk[i]
        k = int(np.searchsorted(s.nirs_frame, e, side="right"))
        dec.push(s.hbo[pushed:k], s.hbr[pushed:k], s.nirs_frame[pushed:k]); pushed = k
        t0 = time.perf_counter()
        p = dec.predict(int(e))
        tp.append(time.perf_counter() - t0)
        if p is not None and blk[i] >= 2:
            P[i] = p
    return P, np.array(tp), np.array(tf)


if __name__ == "__main__":
    ref = pickle.load(open(PROJECT_DIR / "data" / "probs" / "nirs_cascade_ens.pkl", "rb"))
    for name in sys.argv[1:] or ["2025.12.01.16.04.01_N01", "2025.12.09.12.52.43_N01"]:
        P, tp, tf = simulate(name)
        R = ref[name]
        m = ~np.isnan(R[:, 0])
        print(f"{name}: max|P_online−P_offline|={np.nanmax(np.abs(P[m] - R[m])):.2e} "
              f"argmax agree={np.mean(P[m].argmax(1) == R[m].argmax(1)):.4f} | "
              f"predict median={1e3 * np.median(tp):.2f} ms p99={1e3 * np.percentile(tp, 99):.2f} ms max={1e3 * tp.max():.1f} ms | "
              f"fit_block median={np.median(tf):.3f} s max={tf.max():.3f} s")
