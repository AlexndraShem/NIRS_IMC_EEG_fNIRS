"""Чтение сессий хакатона (.mat) и кэш в .npz."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
CACHE_DIR = PROJECT_DIR / "data" / "cache"

EEG_LABELS = ['fp1', 'fp2', 'f3', 'f4', 'c3', 'c4', 'p3', 'p4', 'o1', 'o2', 'f7', 'f8',
              'c7', 'c8', 'p7', 'p8', 'lpa', 'rpa', 'fz', 'cz', 'pz']


@dataclass
class Session:
    name: str
    subject: str
    eeg: np.ndarray          # (n, 21) float32
    states: np.ndarray       # (n,) uint8
    blocks: np.ndarray       # (n,) uint8
    eeg_fs: float
    hbo: np.ndarray | None   # (m, 18)
    hbr: np.ndarray | None
    nirs_states: np.ndarray | None
    nirs_blocks: np.ndarray | None
    nirs_fs: float | None
    nirs_frame: np.ndarray | None  # 1-based индекс отсчёта ЭЭГ


def list_sessions(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted((raw_dir / "Data").glob("*.mat"))


def subject_of(name: str) -> str:
    return re.search(r"_(N\d+)", name).group(1)


def _load_mat(path: Path) -> dict:
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    E = m["EEG"]
    d = dict(eeg=np.asarray(E.Raw, np.float32), states=np.asarray(E.States, np.uint8),
             blocks=np.asarray(E.Blocks, np.uint8), eeg_fs=float(E.Frq))
    N = m.get("NIRS")
    if N is not None and hasattr(N, "HbO") and np.size(N.HbO) > 0:
        d.update(hbo=np.asarray(N.HbO, np.float32), hbr=np.asarray(N.HbR, np.float32),
                 nirs_states=np.asarray(N.States, np.uint8), nirs_blocks=np.asarray(N.Blocks, np.uint8),
                 nirs_fs=float(N.Frq), nirs_frame=np.asarray(N.Frame, np.int64))
    return d


def load_session(path: str | Path, use_cache: bool = True) -> Session:
    path = Path(path)
    cache = CACHE_DIR / (path.stem + ".npz")
    if use_cache and cache.exists():
        z = np.load(cache)
        d = {k: z[k] for k in z.files}
        d = {k: (float(v) if v.ndim == 0 else v) for k, v in d.items()}
    else:
        d = _load_mat(path)
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **d)
    return Session(name=path.stem, subject=subject_of(path.stem),
                   eeg=d["eeg"], states=d["states"], blocks=d["blocks"], eeg_fs=d["eeg_fs"],
                   hbo=d.get("hbo"), hbr=d.get("hbr"), nirs_states=d.get("nirs_states"),
                   nirs_blocks=d.get("nirs_blocks"), nirs_fs=d.get("nirs_fs"),
                   nirs_frame=d.get("nirs_frame"))
