"""Общие потоковые утилиты решения: сетка окон, разбор meta и входных чанков.

Модуль самодостаточен (только numpy): судья запускает только каталог solution/.
"""
from __future__ import annotations

from typing import Any

import numpy as np

N_CLASSES = 3
CLASSES = np.array([1, 2, 3])


# ----------------------------------------------------------------------------- сетка окон
def grid_ends_between(lo: int, hi: int, fs: float, win: int) -> np.ndarray:
    """Концы окон (exclusive, в отсчётах ЭЭГ) глобальной сетки в диапазоне (lo, hi].

    Сетка судьи (io_utils.iter_scored_windows): end = win + j * int(round(fs/4)), j = 0, 1, ...
    (при 250 Гц шаг 62 отсчёта, а не 62.5).
    """
    step = int(round(fs / 4.0))
    j0 = max(0, (lo - win) // step)
    j1 = max(0, (hi - win) // step + 1)
    e = win + np.arange(j0, j1, dtype=np.int64) * step
    return e[(e > lo) & (e <= hi)]


def homogeneous_window_ends(labels: np.ndarray, cand_ends: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Из кандидатов выбрать окна [e-win, e), целиком лежащие в однородном сегменте меток 1..3.

    labels — метки отсчётов с индекса 0 (−1 = неизвестно). Возвращает (ends, y).
    """
    lab = np.asarray(labels).astype(np.int64)
    e = np.asarray(cand_ends, np.int64)
    e = e[(e >= win) & (e <= len(lab))]
    if len(e) == 0:
        return e, e
    seg_id = np.r_[0, np.cumsum(np.diff(lab) != 0)]
    ok = (seg_id[e - win] == seg_id[e - 1]) & (lab[e - 1] >= 1) & (lab[e - 1] <= 3)
    return e[ok], lab[e[ok] - 1]


# ----------------------------------------------------------------------------- meta
def meta_get(meta: Any, *names: str, default=None):
    """Терпимый поиск значения в (возможно вложенном) dict meta без учёта регистра.

    Имя может быть путём через точку: "EEG.Frq". Возвращает первое найденное.
    """
    if not isinstance(meta, dict):
        return default
    low = {str(k).lower(): v for k, v in meta.items()}
    for name in names:
        parts = name.lower().split(".")
        cur: Any = low
        for p in parts:
            if isinstance(cur, dict):
                cur = {str(k).lower(): v for k, v in cur.items()}.get(p)
            else:
                cur = getattr(cur, p, None) if cur is not None else None
            if cur is None:
                break
        if cur is not None:
            return cur
    return default


def as_float(x, default=None):
    try:
        v = float(np.asarray(x).ravel()[0])
        return v if np.isfinite(v) and v > 0 else default
    except Exception:
        return default


def as_int(x, default=None):
    v = as_float(x, None)
    return int(round(v)) if v is not None else default


# ----------------------------------------------------------------------------- чанки
def _first_key(d: dict, *names):
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] is not None:
            return low[n.lower()]
    return None


def parse_eeg_chunk(chunk, n_ch: int | None) -> np.ndarray:
    """→ (n, C) float64. Принимает (n, C), (C, n), (C,) (один отсчёт), dict {'Raw'|'data'|'eeg': ...}."""
    if chunk is None:
        return np.zeros((0, n_ch or 0))
    if isinstance(chunk, dict):
        chunk = _first_key(chunk, "raw", "data", "eeg", "x", "signal", "samples")
        if chunk is None:
            return np.zeros((0, n_ch or 0))
    x = np.asarray(chunk, dtype=np.float64)
    if x.size == 0:
        return np.zeros((0, n_ch or (x.shape[-1] if x.ndim == 2 else 0)))
    if x.ndim == 1:
        if n_ch and x.size % n_ch == 0:
            return x.reshape(-1, n_ch)
        return x[None, :]
    x = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
    if n_ch:
        if x.shape[1] == n_ch:
            return x
        if x.shape[0] == n_ch:
            return x.T
    # число каналов неизвестно: каналов обычно меньше, чем отсчётов в чанке ≥ 1 с,
    # но чанк 250 мс = 62 отсчёта > 21 канала — считаем, что меньшая ось = каналы, если она ≤ 64
    if x.shape[0] < x.shape[1] and x.shape[0] <= 64:
        return x.T
    return x


def parse_nirs_chunk(chunk, n_ch: int | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """→ (values (m, 2*n_ch) = [HbO | HbR] float64, frames (m,) int64 1-based или None).

    Поддерживается: None / пустой массив; dict {'HbO','HbR'[, 'Frame']} (регистр не важен);
    кортеж/список (HbO, HbR[, Frame]); массив (m, 2*n_ch) или (2*n_ch, m);
    массив (m, 2*n_ch + 1) — один столбец Frame (первый или последний, тот что целочисленный
    и неубывающий).
    """
    if chunk is None:
        return None, None
    frames = None
    if isinstance(chunk, dict):
        hbo = _first_key(chunk, "hbo", "oxy", "o2hb")
        hbr = _first_key(chunk, "hbr", "deoxy", "hhb")
        fr = _first_key(chunk, "frame", "frames", "eeg_frame")
        if hbo is None or hbr is None:
            vals = _first_key(chunk, "data", "values", "x", "nirs")
            if vals is None:
                return None, None
            return parse_nirs_chunk(vals if fr is None else (vals, fr), n_ch)
        hbo = _orient(np.asarray(hbo, np.float64), n_ch)
        hbr = _orient(np.asarray(hbr, np.float64), n_ch)
        if hbo.size == 0:
            return None, None
        vals = np.hstack([hbo, hbr])
        frames = None if fr is None else np.asarray(fr, np.int64).ravel()
        return vals, frames
    if isinstance(chunk, (tuple, list)):
        if len(chunk) == 0:
            return None, None
        if len(chunk) in (2, 3) and all(np.ndim(c) >= 1 for c in chunk[:2]) and \
                np.shape(chunk[0]) == np.shape(chunk[1]):
            d = {"HbO": chunk[0], "HbR": chunk[1]}
            if len(chunk) == 3:
                d["Frame"] = chunk[2]
            return parse_nirs_chunk(d, n_ch)
        if len(chunk) == 2 and np.ndim(chunk[1]) == 1:          # (values, frames)
            v, _ = parse_nirs_chunk(np.asarray(chunk[0]), n_ch)
            return v, np.asarray(chunk[1], np.int64).ravel()
    x = np.asarray(chunk, np.float64)
    if x.size == 0:
        return None, None
    if x.ndim == 1:
        x = x[None, :]
    nc2 = 2 * n_ch if n_ch else None
    if nc2:
        if x.shape[1] not in (nc2, nc2 + 1) and x.shape[0] in (nc2, nc2 + 1):
            x = x.T
        if x.shape[1] == nc2 + 1:
            for col in (0, -1):
                c = x[:, col]
                if np.all(np.abs(c - np.round(c)) < 1e-9) and np.all(np.diff(c) >= 0) and c.min() >= 1:
                    frames = c.astype(np.int64)
                    x = np.delete(x, col if col == 0 else x.shape[1] - 1, axis=1)
                    break
            else:
                x = x[:, :nc2]
    elif x.shape[0] < x.shape[1] and x.shape[0] <= 128 and x.shape[1] > 128:
        x = x.T
    return x, frames


def _orient(a: np.ndarray, n_ch: int | None) -> np.ndarray:
    if a.ndim == 1:
        return a[None, :] if (n_ch is None or a.size == n_ch) else a.reshape(-1, n_ch)
    if n_ch and a.shape[1] != n_ch and a.shape[0] == n_ch:
        return a.T
    return a


def parse_labels(labels) -> tuple[str, np.ndarray]:
    """Разбор аргумента fit_block. Возвращает (kind, array):

    * ('samples', (n,)) — метки States отсчётов ЭЭГ (основной вариант);
    * ('windows', (k, 2)) — пары (end_sample, y) для окон;
    * ('indexed', (k, 2)) — пары (индекс отсчёта ЭЭГ, States) — официальный dict каркаса.
    """
    if isinstance(labels, dict):
        idx = _first_key(labels, "eeg_indices", "indices", "eeg_index", "idx")
        st = _first_key(labels, "states", "state", "y", "labels")
        if idx is not None and st is not None:
            # официальный формат каркаса: {"states": States[mask], "eeg_indices": flatnonzero(mask),
            # "block": b}, mask = (Blocks == b) & (States > 0) — индексы 0-based, с разрывами
            return "indexed", np.c_[np.asarray(idx, np.int64).ravel(), np.asarray(st, np.int64).ravel()]
        ends = _first_key(labels, "end_sample", "end_samples", "ends", "end")
        y = _first_key(labels, "y", "labels", "true_y", "states", "state")
        if ends is not None and y is not None:
            return "windows", np.c_[np.asarray(ends, np.int64).ravel(), np.asarray(y, np.int64).ravel()]
        if y is not None:
            return "samples", np.asarray(y).ravel()
        raise ValueError("fit_block: не удалось разобрать dict с метками")
    try:  # pandas.DataFrame (без импорта pandas — он стоит ~1 с на первом вызове)
        if hasattr(labels, "columns") and hasattr(labels, "to_numpy"):
            cols = {c.lower(): c for c in labels.columns}
            ec = next((cols[c] for c in ("end_sample", "end", "ends") if c in cols), None)
            yc = next((cols[c] for c in ("y", "true_y", "label", "labels", "states", "state") if c in cols), None)
            if ec is not None and yc is not None:
                return "windows", np.c_[labels[ec].to_numpy(np.int64), labels[yc].to_numpy(np.int64)]
            return "samples", labels[yc if yc is not None else labels.columns[-1]].to_numpy()
    except Exception:
        pass
    a = np.asarray(labels)
    if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] > 2:
        return "windows", a.astype(np.int64)
    if a.ndim == 2 and a.shape[0] == 2 and a.shape[1] > 2:
        return "windows", a.T.astype(np.int64)
    return "samples", a.ravel()


def proba_to3(model, P: np.ndarray) -> np.ndarray:
    """Привести predict_proba sklearn-модели к столбцам классов 1, 2, 3."""
    P = np.atleast_2d(P)
    out = np.zeros((P.shape[0], N_CLASSES))
    for j, c in enumerate(model.classes_):
        if 1 <= int(c) <= 3:
            out[:, int(c) - 1] = P[:, j]
    s = out.sum(1, keepdims=True)
    return np.where(s > 0, out / np.maximum(s, 1e-12), 1.0 / N_CLASSES)
