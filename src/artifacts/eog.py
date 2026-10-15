"""Окновые «глазные» признаки для аудита (оффлайн, НЕ часть решения): среднее и размах
псевдо-ЭОГ отведений по окнам судьи. Предобработка причинная: x − x[0], ВЧ 0.1 Гц.

  heog = f8 − f7 (горизонтальные движения глаз), veog = (fp1 + fp2)/2 − (o1 + o2)/2,
  плюс fp1, fp2, f7, f8 по отдельности (относительно общего референса записи).

    python -m src.artifacts.eog --jobs 12
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.signal as ss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "solution"))
from session_io import load_session, window_ends  # noqa: E402
from src.artifacts.extract import EEG_LABELS, FS  # noqa: E402


def process(p: Path, out: Path):
    d = load_session(p)
    ix = {c: i for i, c in enumerate(EEG_LABELS)}
    x = d["eeg"] - d["eeg"][:1]
    x = ss.sosfilt(ss.butter(1, 0.1, "highpass", fs=FS, output="sos"), x, axis=0)
    x = ss.sosfilt(ss.butter(4, 8.0, "lowpass", fs=FS, output="sos"), x, axis=0)
    sig = np.stack([x[:, ix["f8"]] - x[:, ix["f7"]],
                    (x[:, ix["fp1"]] + x[:, ix["fp2"]]) / 2 - (x[:, ix["o1"]] + x[:, ix["o2"]]) / 2,
                    x[:, ix["fp1"]], x[:, ix["fp2"]], x[:, ix["f7"]], x[:, ix["f8"]]], 1)
    ends, y, wb = window_ends(d["states"], d["blocks"], FS, 1.0)
    W = np.stack([sig[e - 250:e] for e in ends])
    np.savez_compressed(out / f"{p.stem}_eog.npz", mean=W.mean(1), ptp=np.ptp(W, 1), y=y, block=wb, ends=ends)
    return p.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/Data"))
    ap.add_argument("--out", default=str(ROOT / "data/features/artifacts_audit"))
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    ps = sorted(Path(a.data).glob("*.mat"))
    with ProcessPoolExecutor(a.jobs) as ex:
        list(ex.map(process, ps, [Path(a.out)] * len(ps)))


if __name__ == "__main__":
    main()
