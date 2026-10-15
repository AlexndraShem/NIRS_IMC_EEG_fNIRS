"""Считает и кэширует признаки по всем сессиям.

EEG: ковариации в полосах → поканальная нормировка и рецентрирование по блоку 1
(калибровочный, не классифицируется) → касательное пространство.
Всё, что используется для окна из блока b ≥ 2, получено из данных ≤ конца окна.

python -m src.features.build_cache --win 1.0
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.dataset.io import PROJECT_DIR, list_sessions, load_session
from NIRS_IMC_EEG_fNIRS.src.features.extract import extract_session
from NIRS_IMC_EEG_fNIRS.src.utils.riemann import log_euclid_mean, regularize, tangent_space

FEAT_DIR = PROJECT_DIR / "data" / "features"


def ts_block1_ref(covs, block, eps=1e-2):
    """covs (n, B, C, C). Референс — среднее по окнам блока 1."""
    n, B, C, _ = covs.shape
    out = np.zeros((n, B, C * (C + 1) // 2), np.float32)
    m1 = block == 1
    for b in range(B):
        cb = covs[:, b].astype(np.float64)
        d = np.sqrt(np.median(np.diagonal(cb[m1], axis1=1, axis2=2), axis=0))
        cb = cb / d[None, :, None] / d[None, None, :]
        cb = regularize(cb, eps)
        ref = log_euclid_mean(cb[m1])
        out[:, b] = tangent_space(cb, ref)
    return out.reshape(n, -1)


def build_one(args):
    path, win_s, tag = args
    s = load_session(path)
    f = extract_session(s, win_s)
    ts = ts_block1_ref(f["covs"], f["block"])
    out = FEAT_DIR / tag / (s.name + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    d = dict(ends=f["ends"], y=f["y"], block=f["block"], ts=ts, subject=s.subject,
             logvar=np.log(np.maximum(np.diagonal(f["covs"], axis1=2, axis2=3), 1e-30)).reshape(len(ts), -1).astype(np.float32))
    if f["nirs"] is not None:
        d["nirs"] = f["nirs"]
    np.savez(out, **d)
    return s.name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"v1_win{a.win:g}"
    with ProcessPoolExecutor(16) as ex:
        for name in ex.map(build_one, [(p, a.win, tag) for p in list_sessions()]):
            print(name, flush=True)
