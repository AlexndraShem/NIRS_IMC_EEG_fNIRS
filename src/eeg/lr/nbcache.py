"""Кэш логдисперсий пространственно отфильтрованных каналов (12 биполяр + 11 лапласианов)
в узких полосах (причинная фильтрация, как в src/eeg/cache.py).

Полосы: 4-8, затем 2-Гц 8-10 … 28-30 (11 полос), плюс 8-30. Окна 1, 2, 4 с на той же сетке концов.
python -m src.eeg.lr.nbcache  ->  data/features/eeg_lr_nb/<сессия>.npz: var_w{1,2,4} (n, B, 23)
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.signal as ss

from NIRS_IMC_EEG_fNIRS.src.dataset.io import PROJECT_DIR, list_sessions, load_session
from NIRS_IMC_EEG_fNIRS.src.dataset.windows import window_ends
from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_raw64
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import spatial_matrix
from NIRS_IMC_EEG_fNIRS.src.features.extract import eeg_prefilter_sos

NB_BANDS = [(4, 8)] + [(f, f + 2) for f in range(8, 30, 2)] + [(8, 30)]
WINS = (1.0, 2.0, 4.0)
OUT = PROJECT_DIR / "data" / "features" / "eeg_lr_nb"


def build_one(path):
    s = load_session(path)
    ends, y, blk = window_ends(s.states, s.blocks, s.eeg_fs)
    fs = s.eeg_fs
    x = load_raw64(path)
    x = ss.sosfilt(eeg_prefilter_sos(fs), x, axis=0)
    W = np.vstack([spatial_matrix("bip"), spatial_matrix("lap")])
    xs = x @ W.T                                             # (T, 23)
    out = {w: np.full((len(ends), len(NB_BANDS), W.shape[0]), np.nan, np.float32) for w in WINS}
    for bi, band in enumerate(NB_BANDS):
        sos = ss.butter(4, band, "bandpass", fs=fs, output="sos")
        xf = ss.sosfilt(sos, xs, axis=0)
        cs = np.vstack([np.zeros((1, xf.shape[1])), np.cumsum(xf ** 2, axis=0)])
        for w in WINS:
            L = int(round(w * fs))
            ok = ends >= L
            e = ends[ok]
            out[w][ok, bi] = (cs[e] - cs[e - L]) / L
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{s.name}.npz", ends=ends, y=y, block=blk, subject=s.subject,
             **{f"var_w{w:g}": v for w, v in out.items()})
    return s.name


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import sys
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(nj) as ex:
        for n in ex.map(build_one, list_sessions()):
            print(n, flush=True)
