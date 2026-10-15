"""Причинное извлечение признаков по всей сессии.

Все фильтры — однонаправленные (sosfilt), поэтому прогон по всей записи эквивалентен
потоковой обработке чанками с сохранением состояния фильтра: значение в момент t зависит
только от отсчётов ≤ t. Признаки берутся строго по окнам [end - L, end).
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from NIRS_IMC_EEG_fNIRS.src.dataset.io import Session
from NIRS_IMC_EEG_fNIRS.src.dataset.windows import window_ends

EEG_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 26), (26, 32)]


def eeg_prefilter_sos(fs=250.0):
    hp = ss.butter(2, 1.0, "highpass", fs=fs, output="sos")
    b, a = ss.iirnotch(50.0, 30.0, fs=fs)
    return np.vstack([hp, ss.tf2sos(b, a)])


def eeg_band_sos(fs=250.0, bands=EEG_BANDS, order=4):
    return [ss.butter(order, bnd, "bandpass", fs=fs, output="sos") for bnd in bands]


def eeg_covs(x: np.ndarray, ends: np.ndarray, fs=250.0, win_s=1.0, bands=EEG_BANDS):
    """x: (n, C) сырое ЭЭГ. Возвращает (n_win, n_bands, C, C) ковариации в окнах."""
    x = x.astype(np.float64)
    x = x - x[:1]                                  # убрать гигантский DC до фильтрации (только первый отсчёт — причинно)
    x = ss.sosfilt(eeg_prefilter_sos(fs), x, axis=0)
    L = int(round(win_s * fs))
    idx = ends[:, None] + np.arange(-L, 0)[None, :]
    valid = ends >= L
    idx = np.clip(idx, 0, None)
    out = np.zeros((len(ends), len(bands), x.shape[1], x.shape[1]), np.float32)
    for bi, sos in enumerate(eeg_band_sos(fs, bands)):
        xf = ss.sosfilt(sos, x, axis=0)
        seg = xf[idx]                              # (n_win, L, C)
        out[:, bi] = np.einsum("wtc,wtd->wcd", seg, seg) / L
    out[~valid] = np.nan
    return out


def nirs_features(s: Session, ends: np.ndarray, lags_s=(1.0, 2.0, 4.0, 6.0)):
    """Признаки NIRS в момент конца окна: разности низкочастотно отфильтрованных HbO/HbR
    между «сейчас» и «сейчас − lag» + уровень после медленного high-pass.
    Возвращает (n_win, F) или None, если NIRS нет."""
    if s.hbo is None:
        return None
    fs = s.nirs_fs
    x = np.hstack([s.hbo, s.hbr]).astype(np.float64)
    x = x - x[:1]
    lp = ss.butter(3, min(0.5, 0.4 * fs), "lowpass", fs=fs, output="sos")
    hp = ss.butter(1, 0.01, "highpass", fs=fs, output="sos")
    xl = ss.sosfilt(lp, x, axis=0)
    xh = ss.sosfilt(hp, xl, axis=0)
    # индекс последнего доступного отсчёта NIRS: Frame (1-based) ≤ end
    ni = np.searchsorted(s.nirs_frame, ends, side="right") - 1
    ni = np.clip(ni, 0, None)
    feats = [xh[ni]]
    for lag in lags_s:
        k = max(1, int(round(lag * fs)))
        feats.append(xl[ni] - xl[np.clip(ni - k, 0, None)])
    return np.hstack(feats).astype(np.float32)


def extract_session(s: Session, win_s=1.0):
    ends, y, blk = window_ends(s.states, s.blocks, s.eeg_fs)
    covs = eeg_covs(s.eeg, ends, s.eeg_fs, win_s)
    nf = nirs_features(s, ends)
    return dict(ends=ends, y=y, block=blk, covs=covs, nirs=nf)
