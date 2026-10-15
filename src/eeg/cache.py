"""Кэш ковариаций ЭЭГ для экспериментов src/eeg: фильтр-банк × несколько длин окна истории.

Всё причинно: prefilter (HP 1 Гц + notch 50 Гц) и полосовые фильтры — sosfilt (однонаправленные),
ковариация окна с концом `end` считается по отсчётам [end - L, end). Концы окон — те же, что в
data/features/v1_win1 (window_ends), т.е. строки выровнены с baseline-кэшем.

python -m src.eeg.cache [--f64]    # -> data/features/eeg_fb[64]/<сессия>.npz
Поля: ends, y, block, subject, covs_w{1,2,4} (n, B, C, C) float32, bands.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.io as sio
import scipy.signal as ss

from NIRS_IMC_EEG_fNIRS.src.dataset.io import PROJECT_DIR, list_sessions, load_session
from NIRS_IMC_EEG_fNIRS.src.dataset.windows import window_ends
from NIRS_IMC_EEG_fNIRS.src.features.extract import eeg_prefilter_sos

BANDS = [(4, 8), (8, 10), (10, 13), (13, 18), (18, 24), (24, 30), (8, 30)]
WINS = (1.0, 2.0, 4.0)
OUT_DIR = PROJECT_DIR / "data" / "features" / "eeg_fb"        # float32-кэш io.py (эксперименты)
OUT_DIR64 = PROJECT_DIR / "data" / "features" / "eeg_fb64"    # float64 из .mat (финальные признаки)


def band_sos(fs=250.0, bands=BANDS, order=4):
    return [ss.butter(order, b, "bandpass", fs=fs, output="sos") for b in bands]


def session_covs(x, ends, fs=250.0, wins=WINS, bands=BANDS):
    x = x.astype(np.float64)
    x = x - x[:1]
    x = ss.sosfilt(eeg_prefilter_sos(fs), x, axis=0)
    C = x.shape[1]
    out = {w: np.full((len(ends), len(bands), C, C), np.nan, np.float32) for w in wins}
    for bi, sos in enumerate(band_sos(fs, bands)):
        xf = ss.sosfilt(sos, x, axis=0)
        for w in wins:
            L = int(round(w * fs))
            ok = ends >= L
            for i0 in range(0, len(ends), 256):          # чанками — экономия памяти
                e = ends[i0:i0 + 256]
                idx = np.clip(e[:, None] + np.arange(-L, 0)[None, :], 0, None)
                seg = xf[idx]
                cv = np.einsum("wtc,wtd->wcd", seg, seg) / L
                cv[~ok[i0:i0 + 256]] = np.nan
                out[w][i0:i0 + 256, bi] = cv
    return out


def load_raw64(path):
    """Сырое ЭЭГ во float64 прямо из .mat (кэш io.py хранит float32: при DC до ~7e9 шаг
    квантования ~512 сопоставим с амплитудой слабых каналов). Вычитается первый отсчёт (причинно)."""
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["EEG"])
    x = np.asarray(m["EEG"].Raw, np.float64)
    return x - x[:1]


def build_one(path, out_dir=None):
    out_dir = out_dir or OUT_DIR
    s = load_session(path)
    ends, y, blk = window_ends(s.states, s.blocks, s.eeg_fs)
    x = load_raw64(path) if out_dir != OUT_DIR else s.eeg
    cv = session_covs(x, ends, s.eeg_fs)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{s.name}.npz", ends=ends, y=y, block=blk, subject=s.subject,
             bands=np.array(BANDS), **{f"covs_w{w:g}": v for w, v in cv.items()})
    return s.name


def load_eeg_cache(win=1.0, bands_idx=None, f64=False):
    out = {}
    for p in sorted((OUT_DIR64 if f64 else OUT_DIR).glob("*.npz")):
        z = np.load(p)
        c = z[f"covs_w{win:g}"]
        if bands_idx is not None:
            c = c[:, bands_idx]
        out[p.stem] = dict(ends=z["ends"], y=z["y"], block=z["block"], subject=str(z["subject"]), covs=c)
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import sys
    from functools import partial
    out = OUT_DIR64 if "--f64" in sys.argv else OUT_DIR
    with ProcessPoolExecutor(10) as ex:
        for n in ex.map(partial(build_one, out_dir=out), list_sessions()):
            print(n, flush=True)
