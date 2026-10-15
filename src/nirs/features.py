"""Причинные NIRS-признаки для 3-классового декодера.

Конвейер (всё — на частоте NIRS, однонаправленные фильтры с начальным состоянием по первому
отсчёту, т.е. эквивалентно потоковой обработке):

1. x = [HbO | HbR] (18 + 18 каналов).
2. Полосовой Баттерворт (по умолчанию 0.01–0.2 Гц, sosfilt, zi = steady-state по x[0]).
3. «Производные каналы»:
   * 36 исходных каналов;
   * 9 латерализационных пар (левое − правое полушарие, канал i+8 − канал i) для HbO и HbR;
   * глобальное среднее HbO / HbR (системные колебания);
   * средняя латерализация группы (mean left − mean right) HbO / HbR;
   * опционально CAR: вычитание глобального среднего из каждого канала.
4. Для окна с концом `end` (exclusive, отсчёты ЭЭГ) берётся последний доступный отсчёт NIRS
   (nirs_frame ≤ end) и значения сигнала на лагах 0, 1, …, K с назад → тензор (n_win, n_ch, n_lag).

Нормировка масштаба — отдельно (см. `scale_from_block1`): делим каждый производный канал на
его std по отсчётам блока 1 (блок 1 не классифицируется и к моменту окон блока ≥ 2 уже в прошлом).

Монтаж (Montage.mat, NIRS_positions): каналы 0–7 и 16 — x > 0 (правое полушарие),
8–15 и 17 — x < 0 (левое); пары симметричны: i ↔ i + 8, 16 ↔ 17.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from src.dataset.io import Session

RIGHT = list(range(0, 8)) + [16]
LEFT = list(range(8, 16)) + [17]


def bandpass_sos(fs: float, lo: float = 0.01, hi: float = 0.2, order: int = 2):
    hi = min(hi, 0.45 * fs)
    if lo and lo > 0:
        return ss.butter(order, [lo, hi], "bandpass", fs=fs, output="sos")
    return ss.butter(order, hi, "lowpass", fs=fs, output="sos")


def causal_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """sosfilt с начальным состоянием, как будто сигнал до старта был постоянен = x[0]."""
    zi = ss.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]  # (n_sec, 2, C)
    y, _ = ss.sosfilt(sos, x, axis=0, zi=zi)
    return y


def derived_channels(hbo: np.ndarray, hbr: np.ndarray, car: bool = False) -> tuple[np.ndarray, list[str]]:
    """hbo/hbr: (m, 18) уже отфильтрованные. Возвращает (m, D) и имена."""
    cols, names = [], []
    for tag, x in (("o", hbo), ("r", hbr)):
        g = x.mean(1, keepdims=True)
        xc = x - g if car else x
        cols.append(xc); names += [f"{tag}{i}" for i in range(18)]
        lat = x[:, LEFT] - x[:, RIGHT]                          # (m, 9) левое − правое
        cols.append(lat); names += [f"{tag}lat{i}" for i in range(9)]
        cols.append(g); names.append(f"{tag}glob")
        cols.append(lat.mean(1, keepdims=True)); names.append(f"{tag}latm")
    return np.hstack(cols), names


def nirs_index(nirs_frame: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Индекс последнего отсчёта NIRS, доступного к концу окна (Frame 1-based ≤ end)."""
    return np.clip(np.searchsorted(nirs_frame, ends, side="right") - 1, 0, None)


def lag_tensor(s: Session, ends: np.ndarray, lags_s=tuple(range(0, 11)), lo=0.01, hi=0.2,
               order=2, car=False):
    """(n_win, D, n_lag) значения производных каналов на лагах назад от конца окна +
    (m, D) сигнал целиком (для статистик блока 1) + имена."""
    fs = s.nirs_fs
    sos = bandpass_sos(fs, lo, hi, order)
    ho = causal_filter(s.hbo.astype(np.float64), sos)
    hr = causal_filter(s.hbr.astype(np.float64), sos)
    sig, names = derived_channels(ho, hr, car)
    ni = nirs_index(s.nirs_frame, ends)
    idx = ni[:, None] - np.round(np.asarray(lags_s) * fs).astype(int)[None, :]
    idx = np.clip(idx, 0, None)
    T = sig[idx]                                   # (n_win, n_lag, D)
    return T.transpose(0, 2, 1).astype(np.float32), sig, names


def scale_from_block1(sig: np.ndarray, nirs_blocks: np.ndarray, skip_s: float = 20.0, fs: float = 12.5):
    """std каждого производного канала по блоку 1 (без первых skip_s секунд — переходный процесс)."""
    m = np.flatnonzero(nirs_blocks == 1)
    m = m[m >= int(skip_s * fs)] if len(m) > int(skip_s * fs) + 50 else m
    sd = sig[m].std(0)
    return np.where(sd > 0, sd, 1.0)
