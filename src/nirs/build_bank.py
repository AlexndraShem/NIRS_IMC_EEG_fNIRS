"""Кэш тензоров лагов NIRS, выровненных с окнами data/features/v1_win1.

python -m src.nirs.build_bank --lo 0.01 --hi 0.2 --tag nirs_bank_bp01_20
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.dataset.io import PROJECT_DIR, RAW_DIR, load_session
from src.nirs.features import lag_tensor, scale_from_block1

FEAT_DIR = PROJECT_DIR / "data" / "features"
LAGS = tuple(np.arange(0, 12.5, 1.0))


def build_one(args):
    name, lo, hi, order, car, tag = args
    s = load_session(RAW_DIR / "Data" / f"{name}.mat")
    v1 = np.load(FEAT_DIR / "v1_win1" / f"{name}.npz")
    d = dict(ends=v1["ends"], y=v1["y"], block=v1["block"], subject=str(v1["subject"]))
    if s.hbo is not None:
        T, sig, names = lag_tensor(s, v1["ends"], LAGS, lo, hi, order, car)
        sc = scale_from_block1(sig, s.nirs_blocks, fs=s.nirs_fs)
        d.update(T=T, scale=sc.astype(np.float32), names=np.array(names), lags=np.array(LAGS))
    out = FEAT_DIR / tag / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.01)
    ap.add_argument("--hi", type=float, default=0.2)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--car", action="store_true")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    names = sorted(p.stem for p in (FEAT_DIR / "v1_win1").glob("*.npz"))
    with ProcessPoolExecutor(8) as ex:
        for n in ex.map(build_one, [(n, a.lo, a.hi, a.order, a.car, a.tag) for n in names]):
            pass
    print("done", len(names))
