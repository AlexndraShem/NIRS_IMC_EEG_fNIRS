"""Потоковая симуляция OnlineEEGLR (чанки 100 мс, predict на сетке 250 мс, fit_block по концу блока):
сверка с data/probs/eeg_lr_nbsm4.pkl и замер времени.

python -m src.eeg.lr.simulate <n_sessions> [threads]
python -m src.eeg.lr.simulate --all [n_jobs]   # все 56 сессий потоком → data/probs/eeg_lr_nbsm4.pkl
                                               # (эталон = ровно то, что выдаст онлайн-модель)
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.dataset.io import list_sessions, load_session
from NIRS_IMC_EEG_fNIRS.src.dataset.windows import window_ends
from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_raw64
from NIRS_IMC_EEG_fNIRS.src.eeg.common import PROBS_DIR
from NIRS_IMC_EEG_fNIRS.src.eeg.final import MODEL_DIR
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.online import OnlineEEGLR
from NIRS_IMC_EEG_fNIRS.src.eeg.online import PoolLR


def simulate(path, chunk=25, subj=None):
    s = load_session(path)
    x = load_raw64(path)
    sj = subj or s.subject
    m = OnlineEEGLR(PoolLR.load(MODEL_DIR / f"pool_ts2s_all_{sj}.npz"),
                    PoolLR.load(MODEL_DIR / f"pool_lrnb_{sj}.npz"),
                    PoolLR.load(MODEL_DIR / f"pool_lrsm4_{sj}.npz"))
    ends, y, blk = window_ends(s.states, s.blocks, s.eeg_fs)
    want = dict(zip(ends.tolist(), range(len(ends))))
    P = np.full((len(ends), 3), np.nan)
    last_of_block = {b: int(np.flatnonzero(s.blocks == b)[-1]) + 1 for b in range(1, 8)}
    tp, tf, tpush = [], [], []
    n, fitted_upto, cur = 0, 0, 1
    while n < len(x):
        k = min(chunk, len(x) - n, last_of_block.get(cur, len(x)) - n) or min(chunk, len(x) - n)
        nxt = [e for e in ends[(ends > n) & (ends <= n + k)]]
        for st in sorted(set(nxt + [n + k])):
            t0 = time.perf_counter(); m.push(x[n:st]); tpush.append(time.perf_counter() - t0); n = st
            if n in want and blk[want[n]] >= 2:
                t0 = time.perf_counter(); P[want[n]] = m.predict_proba(); tp.append(time.perf_counter() - t0)
        if cur <= 7 and n == last_of_block[cur]:
            t0 = time.perf_counter(); m.fit_block(cur, s.states[fitted_upto:n]); tf.append(time.perf_counter() - t0)
            fitted_upto = n; cur += 1
    return s.name, P, np.array(tp), np.array(tf), np.array(tpush)


def _one(p):
    nm, P, tp, tf, tpu = simulate(p)
    return nm, P, tp, tf


def run_all(nj=8):
    import os
    from concurrent.futures import ProcessPoolExecutor
    from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_eeg_cache
    from NIRS_IMC_EEG_fNIRS.src.eeg.common import save_probs
    from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import detail, fmt
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(nj) as ex:
        out = list(ex.map(_one, list_sessions()))
    probs = {nm: P for nm, P, _, _ in out}
    save_probs("lr_nbsm4", probs)
    tp = np.concatenate([o[2] for o in out]); tf = np.array([o[3] for o in out])
    data = load_eeg_cache(2.0, f64=True)
    print(fmt("eeg_lr_nbsm4 (поток)", detail(data, probs), detail(data, probs, tau=1.5)))
    print(f"predict: mean={tp.mean()*1e3:.1f}ms p99={np.percentile(tp, 99)*1e3:.1f}ms max={tp.max()*1e3:.1f}ms")
    print("fit_block по блокам 1..7: mean", np.round(tf.mean(0), 3).tolist(), "max", np.round(tf.max(0), 3).tolist())


if __name__ == "__main__":
    import NIRS_IMC_EEG_fNIRS.src.eeg.common as cm
    if sys.argv[1] == "--all":
        cm.EIG_THREADS = 1
        run_all(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
        sys.exit()
    k = int(sys.argv[1])
    cm.EIG_THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ref = pickle.load(open(PROBS_DIR / "eeg_lr_nbsm4.pkl", "rb"))
    ls = list_sessions()
    paths = ls[:: max(1, len(ls) // k)][:k]
    for p in paths:
        nm, P, tp, tf, tpu = simulate(p)
        R = ref[nm]; ok = ~np.isnan(R[:, 0])
        print(f"{nm}: |P_stream - P_offline| max={np.nanmax(np.abs(P[ok] - R[ok])):.2e} "
              f"argmax agree={np.mean(P[ok].argmax(1) == R[ok].argmax(1)):.4f} | predict mean={tp.mean()*1e3:.1f}ms "
              f"max={tp.max()*1e3:.1f}ms | push(100мс) mean={tpu.mean()*1e3:.2f}ms | "
              f"fit_block s={np.round(tf, 3).tolist()}", flush=True)
