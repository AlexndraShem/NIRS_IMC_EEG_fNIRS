"""Интерфейс модальностных моделей. Любая новая ЭЭГ/NIRS-модель реализует его и подключается
через solution/artifacts/config.json (поле "eeg_model"/"nirs_model": "module:Class").

Жизненный цикл внутри OnlineClassifier:

    m = Model(meta, cfg)                 # meta — нормализованный dict (см. classifier.py)
    m.load(artifacts_dir)                # опционально: предобученные веса / кросс-сессионный пул
    m.update(...)                        # на каждый push: потоковая причинная обработка
    m.fit(ends, y)                       # в fit_block: все размеченные окна завершённых блоков
    p = m.predict_proba(end)             # в predict: (3,) вероятности классов 1..3 или None

`ends` — концы окон в абсолютных отсчётах ЭЭГ (exclusive): окно = [end - win, end).
Модель сама хранит всё, что ей нужно, чтобы посчитать признаки для прошлых окон
(например, снимки признаков на сетке окон — см. EEG: `update(..., snapshot_ends)`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ModalityModel:
    name = "base"

    def __init__(self, meta: dict, cfg: dict | None = None):
        self.meta = meta
        self.cfg = dict(cfg or {})

    def load(self, artifacts_dir: str | Path | None) -> None:  # noqa: D401
        """Загрузить предобученные артефакты (по умолчанию — ничего)."""

    @property
    def available(self) -> bool:
        """Есть ли у модели данные (для NIRS: пришёл ли хоть один отсчёт)."""
        return True

    @property
    def fitted(self) -> bool:
        raise NotImplementedError

    def fit(self, ends: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict_proba(self, end: int) -> np.ndarray | None:
        raise NotImplementedError


def subject_of(meta: dict) -> str | None:
    """Идентификатор участника для выбора подучастникового пула: meta["subject"] (судья передаёт,
    например, "N06"), иначе — суффикс `_N<id>` в имени сессии. Нормализация: "n6"/"6" → "N06".
    Нет данных → None (модели откатываются на общий пул)."""
    import re
    raw = meta.get("raw") if isinstance(meta.get("raw"), dict) else meta
    raw = raw if isinstance(raw, dict) else {}
    low = {str(k).lower(): v for k, v in raw.items()}
    cand = [low.get(k) for k in ("subject", "subject_id", "participant", "subj")]
    cand += [low.get(k) for k in ("session_name", "session", "name", "file")]
    for i, v in enumerate(cand):
        if v is None:
            continue
        v = str(np.asarray(v).ravel()[0]) if not isinstance(v, str) else v
        m = re.search(r"(?:^|[_\W])N?(\d{1,3})(?:\.mat)?$", v.strip(), re.I) if i < 4 else \
            re.search(r"_N(\d{1,3})(?:\.mat)?$", v.strip(), re.I)
        if m:
            return f"N{int(m.group(1)):02d}"
    return None
