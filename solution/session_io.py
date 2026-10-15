"""Чтение .mat сессии для каркаса (run.py / score.py). Классификатор этот модуль НЕ использует."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_v5(path):
    import scipy.io as sio
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    E = m["EEG"]
    d = dict(eeg=np.asarray(E.Raw, np.float64), states=np.asarray(E.States).astype(np.int64).ravel(),
             blocks=np.asarray(E.Blocks).astype(np.int64).ravel(), eeg_fs=float(E.Frq))
    N = m.get("NIRS")
    if N is not None and hasattr(N, "HbO") and np.size(N.HbO) > 0:
        d.update(hbo=np.asarray(N.HbO, np.float64), hbr=np.asarray(N.HbR, np.float64),
                 nirs_fs=float(N.Frq), nirs_frame=np.asarray(N.Frame).astype(np.int64).ravel())
    return d


def _load_v73(path):  # MATLAB v7.3 (HDF5) — на случай другого экспорта закрытого набора
    import h5py
    with h5py.File(path, "r") as f:
        E = f["EEG"]
        raw = np.asarray(E["Raw"], np.float64)
        raw = raw.T if raw.shape[0] < raw.shape[1] else raw
        d = dict(eeg=raw, states=np.asarray(E["States"]).astype(np.int64).ravel(),
                 blocks=np.asarray(E["Blocks"]).astype(np.int64).ravel(),
                 eeg_fs=float(np.asarray(E["Frq"]).ravel()[0]))
        if "NIRS" in f and "HbO" in f["NIRS"] and np.size(f["NIRS"]["HbO"]) > 2:
            N = f["NIRS"]
            hbo, hbr = np.asarray(N["HbO"], np.float64), np.asarray(N["HbR"], np.float64)
            if hbo.shape[0] < hbo.shape[1]:
                hbo, hbr = hbo.T, hbr.T
            d.update(hbo=hbo, hbr=hbr, nirs_fs=float(np.asarray(N["Frq"]).ravel()[0]),
                     nirs_frame=np.asarray(N["Frame"]).astype(np.int64).ravel())
    return d


def load_session(path) -> dict:
    path = Path(path)
    try:
        d = _load_v5(path)
    except NotImplementedError:
        d = _load_v73(path)
    d["name"] = path.stem
    d.setdefault("hbo", None)
    d.setdefault("hbr", None)
    d.setdefault("nirs_fs", None)
    d.setdefault("nirs_frame", None)
    return d


def window_ends(states, blocks, fs=250.0, win_s=1.0):
    """Окна 1 с / шаг 250 мс на глобальной сетке, целиком в однородном сегменте States∈{1,2,3}.
    Совпадает с src/dataset/windows.py::window_ends. Возвращает (ends, y, block)."""
    win = int(round(win_s * fs))
    step = int(round(fs / 4))              # как io_utils.iter_scored_windows судьи: 62 отсчёта
    n = len(states)
    ends = np.arange(win, n + 1, step).astype(int)
    st = np.asarray(states).astype(int)
    seg_id = np.r_[0, np.cumsum(np.diff(st) != 0)]
    ok = (seg_id[ends - win] == seg_id[ends - 1]) & (st[ends - 1] > 0)
    ends = ends[ok]
    return ends, st[ends - 1], np.asarray(blocks)[ends - 1].astype(int)


def list_sessions(inp) -> list[Path]:
    p = Path(inp)
    if p.is_dir():
        files = sorted(p.glob("*.mat"))
        if not files:
            files = sorted(p.rglob("*.mat"))
        return [f for f in files if f.name.lower() != "montage.mat"]
    return [p]
