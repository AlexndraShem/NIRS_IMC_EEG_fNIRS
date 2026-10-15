"""Сетка окон 1 с / шаг 250 мс, как в онлайн-каркасе, и метрики score.py (по ТЗ §5.1)."""
from __future__ import annotations

import numpy as np

WIN = 250          # 1 с при 250 Гц
STEP = 62.5        # 250 мс


def segments(states: np.ndarray):
    """Однородные сегменты: (start, end, state), end — исключительно."""
    st = states.astype(int)
    ch = np.flatnonzero(np.diff(st)) + 1
    b = np.r_[0, ch]
    e = np.r_[ch, len(st)]
    return [(int(i), int(j), int(st[i])) for i, j in zip(b, e)]


def window_ends(states: np.ndarray, blocks: np.ndarray, fs: float = 250.0):
    """Концы окон (exclusive, в отсчётах ЭЭГ) на глобальной сетке с шагом 250 мс,
    целиком лежащих в однородном сегменте States ∈ {1,2,3}.

    Возвращает ends, y, block. Блок 1 тоже включён (нужен для обучения),
    при оценке его отбрасывают.
    """
    win = int(round(fs))
    step = fs / 4
    n = len(states)
    ends = np.round(np.arange(win, n + 1e-9, step)).astype(int)
    st = states.astype(int)
    # окно [e-win, e) однородно <=> первый и последний отсчёт в одном сегменте
    seg_id = np.r_[0, np.cumsum(np.diff(st) != 0)]
    ok = (seg_id[ends - win] == seg_id[ends - 1]) & (st[ends - 1] > 0)
    ends = ends[ok]
    return ends, st[ends - 1], blocks[ends - 1].astype(int)


def confusion(y_true, y_pred, k: int = 3) -> np.ndarray:
    """G[i, j]: предсказан класс i+1, истинный j+1 (строки — ответы, столбцы — истина)."""
    G = np.zeros((k, k), int)
    for t, p in zip(y_true, y_pred):
        G[int(p) - 1, int(t) - 1] += 1
    return G


def metrics(G: np.ndarray) -> dict:
    diag = np.diag(G).astype(float)
    f1 = 2 * diag / np.maximum(G.sum(1) + G.sum(0), 1)
    rec = diag / np.maximum(G.sum(0), 1)
    return dict(min_f1=float(f1.min()), recall=float(rec.mean()), f1=f1.tolist(), recall_k=rec.tolist())


def score_points(min_f1: float, recall: float, nirs_min_f1: float | None = None) -> float:
    """Числовая часть балла (без дизайна/отчёта и штрафов)."""
    s = 60 * min_f1 + 30 * recall
    if nirs_min_f1 is not None:
        s += 5 + 15 * nirs_min_f1
    return s
