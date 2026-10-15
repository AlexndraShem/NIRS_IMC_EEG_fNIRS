"""Извлечение спектральных признаков для аудита артефактов (оффлайн-анализ, НЕ часть решения).

Для каждой сессии:
  * окна судьи (1 с, шаг 62 отсчёта, однородный сегмент States∈{1,2,3}) → лог-мощность
    по 1-Гц бинам 1..124 Гц (Hann, 250 отсчётов), 21 канал, float16;
  * пробы (сегменты States 1/2/3 целиком, ≈9 с) → Welch 0.25 Гц (nperseg 1000);
  * «сырые» проверки: доля повторяющихся значений / насыщений по каналам.
Предобработка причинная: x − x[0], ВЧ 1 Гц (Баттерворт 2, sosfilt).

    python -m src.artifacts.extract --out data/features/artifacts_audit --jobs 12
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

EEG_LABELS = ['c4', 'rpa', 'f8', 'p8', 'f4', 'p4', 'fp2', 'o2', 'cz', 'pz',
              'fz', 'o1', 'fp1', 'p3', 'f3', 'p7', 'f7', 'lpa', 'c3', 'c7', 'c8']
FS = 250.0


def process(p: Path, out: Path):
    d = load_session(p)
    raw = d["eeg"]
    st, bl = d["states"].astype(int), d["blocks"].astype(int)
    # проверки сырого сигнала
    dif = np.diff(raw, axis=0)
    flat_frac = (dif == 0).mean(0)
    mx = np.abs(raw).max(0)
    at_max = (np.abs(raw) >= 0.999 * mx).mean(0)
    x = raw - raw[:1]
    x = ss.sosfilt(ss.butter(2, 1.0, "highpass", fs=FS, output="sos"), x, axis=0)
    # окна судьи
    ends, y, wb = window_ends(st, bl, FS, 1.0)
    seg_start = np.r_[0, np.flatnonzero(np.diff(st)) + 1]
    seg_of = np.searchsorted(seg_start, ends - 1, side="right") - 1
    t_in = (ends - seg_start[seg_of]) / FS              # время от начала сегмента до конца окна
    trial_idx = seg_of
    win = np.hanning(250)
    W = np.stack([x[e - 250:e] for e in ends])           # (n, 250, C)
    W = W - W.mean(1, keepdims=True)
    F = np.fft.rfft(W * win[None, :, None], axis=1)      # (n, 126, C)
    P = (np.abs(F) ** 2)[:, 1:125]                       # 1..124 Гц
    lp = np.log10(P + 1e-12).astype(np.float16).transpose(0, 2, 1)  # (n, C, 124)
    # пробы целиком
    seg_end = np.r_[seg_start[1:], len(st)]
    tr = [(a, b) for a, b in zip(seg_start, seg_end) if st[a] > 0 and b - a >= 1000]
    TP = np.stack([ss.welch(x[a:b], FS, nperseg=1000, axis=0)[1] for a, b in tr])  # (T, 501, C)
    np.savez_compressed(out / f"{p.stem}.npz", lp=lp, y=y, block=wb, t_in=t_in.astype(np.float32),
                        trial=trial_idx, ends=ends,
                        trial_psd=np.log10(TP + 1e-12).astype(np.float32).transpose(0, 2, 1),
                        trial_y=np.array([st[a] for a, _ in tr]), trial_block=np.array([bl[a] for a, _ in tr]),
                        trial_bounds=np.array(tr), flat_frac=flat_frac, at_max=at_max,
                        has_nirs=d["hbo"] is not None)
    return p.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/Data"))
    ap.add_argument("--out", default=str(ROOT / "data/features/artifacts_audit"))
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ps = sorted(Path(a.data).glob("*.mat"))
    with ProcessPoolExecutor(a.jobs) as ex:
        for s in ex.map(process, ps, [out] * len(ps)):
            print(s, flush=True)


if __name__ == "__main__":
    main()
