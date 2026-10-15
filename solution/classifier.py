"""OnlineClassifier — тонкий контракт для замороженного каркаса судьи (ТЗ §3.2).

    clf = OnlineClassifier(meta); clf.load(artifacts_dir)
    clf.push(eeg_chunk, nirs_chunk)      # только новые отсчёты (прошлое)
    clf.predict() -> {"y": 1|2|3, "y_eeg": 1|2|3|None, "y_nirs": 1|2|3|None}
    clf.fit_block(block_idx, labels)     # после завершения блока, метки только этого блока

Точный формат аргументов каркаса судьи не опубликован, поэтому класс «терпим» к вариантам
(подробно — docs/experiments/online_harness.md):

* meta: dict; частоты/число каналов ищутся по ключам eeg_fs|fs_eeg|EEG.Frq|..., при отсутствии —
  ЭЭГ 250 Гц, каналы — по первому чанку, частота NIRS — по числу отсчётов на ≥ 30 с Frame.
* eeg_chunk: (n, C) или (C, n) — ориентация по числу каналов; dict {'Raw'|'data': ...}.
* nirs_chunk: None/пустой; dict {'HbO','HbR'[, 'Frame']}; (HbO, HbR[, Frame]); массив (m, 36)
  [HbO|HbR] (+ опционально столбец Frame). Без Frame отсчёт NIRS считается доступным с
  момента push (Frame := число ЭЭГ-отсчётов, полученных к этому push).
* predict(): окно = последние 1 с полученного ЭЭГ (конец = число полученных отсчётов).
* fit_block(k, labels): официальный формат (ответ кейсодержателя 2026-09-23) —
  {"states": States[mask], "eeg_indices": flatnonzero(mask), "block": k}, mask = (Blocks == k) &
  (States > 0). Также (наследие до ответа): States ЭЭГ-отсчётов блока k (блоки идут подряд с
  отсчёта 0, поэтому блок k = [конец блока k−1, + len(labels))). Также понимаем: кумулятивные
  States с начала сессии; метки окон (по одной на каждый predict с прошлого fit_block);
  пары (end_sample, y); dict/DataFrame с такими полями.

Сигнал хранит сам классификатор (в модальностных моделях); States используются только для
разметки окон при обучении, никогда как признак.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from stream_utils import (as_float, as_int, grid_ends_between, homogeneous_window_ends,  # noqa: E402
                          meta_get, parse_eeg_chunk, parse_labels, parse_nirs_chunk)

DEFAULT_CONFIG = {
    "eeg_model": "eeg_model:EEGTangentPool",
    "nirs_model": "nirs_model:NirsCascadeEns",
    "eeg": {},
    "nirs": {},
    "fusion": "fusion:LogLinearFusion",
    "fusion_params": {},
    "fusion_weights": {"eeg": 1.0, "nirs": 1.0},       # только для fusion:MeanProbaFusion
    "win_s": 1.0,
    "blas_threads": 1,
}


def _load_class(spec: str):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def normalize_meta(meta) -> dict:
    meta = meta if isinstance(meta, dict) else {}
    eeg_fs = as_float(meta_get(meta, "eeg_fs", "fs_eeg", "eeg_frq", "eeg.frq", "eeg.fs", "fs", "frq", "srate"), 250.0)
    n_eeg = as_int(meta_get(meta, "eeg_channels", "n_eeg_channels", "n_eeg", "eeg_n_channels", "eeg.n_channels",
                            "n_eeg_ch", "n_channels"), None)
    if n_eeg is None:
        lbl = meta_get(meta, "eeg_channel_names", "eeg_labels", "eeg.labels", "channels", "ch_names")
        n_eeg = len(lbl) if lbl is not None and np.ndim(lbl) == 1 else None
    nirs_fs = as_float(meta_get(meta, "nirs_fs", "fs_nirs", "nirs_frq", "nirs.frq", "nirs.fs"), None)
    n_nirs = as_int(meta_get(meta, "nirs_channels", "n_nirs_channels", "n_nirs", "n_nirs_ch",
                            "nirs.n_channels"), None)
    has_nirs = meta_get(meta, "has_nirs", "nirs_available", "nirs")
    has_nirs = None if has_nirs is None or isinstance(has_nirs, dict) else bool(has_nirs)
    if n_nirs == 0:
        has_nirs = False
    return dict(eeg_fs=eeg_fs, eeg_channels=n_eeg, nirs_fs=nirs_fs, nirs_channels=n_nirs or 18,
                has_nirs=has_nirs, raw=meta)


class OnlineClassifier:
    def __init__(self, meta: dict):
        self.meta = normalize_meta(meta)
        self.cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        self._built = False
        self.n = 0                    # получено отсчётов ЭЭГ
        self.labels = np.zeros(0, np.int8)   # известные метки отсчётов (−1 — неизвестно)
        self.last_block_end = 0       # абсолютный конец последнего размеченного блока
        self.train_ends = np.zeros(0, np.int64)
        self.train_y = np.zeros(0, np.int64)
        self._last_nirs_fr = 0
        self._eeg_T = False           # чанки ЭЭГ приходят как (C, n) — для квадратных чанков
        # асинхронный режим: решение обновляется на КАЖДОЙ точке сетки 250 мс внутри push
        # (состояние слияния/сглаживания не зависит от того, когда и как часто зовут predict)
        self._tick_end = None
        self._tick_out = None

    # ------------------------------------------------------------------ setup
    def load(self, artifacts_dir=None) -> None:
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else _HERE / "artifacts"
        cfg_path = self.artifacts_dir / "config.json"
        if cfg_path.exists():
            user = json.loads(cfg_path.read_text())
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(self.cfg.get(k), dict):
                    self.cfg[k].update(v)
                else:
                    self.cfg[k] = v
        if self.cfg.get("blas_threads"):
            try:
                from threadpoolctl import threadpool_limits
                threadpool_limits(int(self.cfg["blas_threads"]))
            except Exception:
                pass
        self._build()
        self._warmup()

    def _warmup(self):
        """Прогрев ленивых импортов/инициализаций sklearn/scipy, чтобы первый fit_block не
        платил за них (load не хронометрируется, fit_block — да)."""
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            rng = np.random.default_rng(0)
            X, y = rng.standard_normal((30, 8)), np.repeat([1, 2, 3], 10)
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y).predict_proba(X[:1])
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                       priors=[0.5, 0.5]).fit(X, y > 1).predict_proba(X[:1])
            A = rng.standard_normal((600, 21, 21))                # прогрев батч-eigh (LAPACK)
            A = A @ np.swapaxes(A, -1, -2)
            np.linalg.eigh(A)
            np.linalg.solve(A[0] + 21 * np.eye(21), np.ones((21, 3)))
        except Exception:
            pass

    def _build(self):
        if self._built:
            return
        self.meta["win"] = int(round(self.cfg["win_s"] * self.meta["eeg_fs"]))
        self.win = self.meta["win"]
        self.eeg = _load_class(self.cfg["eeg_model"])(self.meta, self.cfg.get("eeg"))
        self.nirs = None
        if self.meta["has_nirs"] is not False and self.cfg.get("nirs_model"):
            self.nirs = _load_class(self.cfg["nirs_model"])(self.meta, self.cfg.get("nirs"))
        for m in (self.eeg, self.nirs):
            if m is not None:
                m.load(getattr(self, "artifacts_dir", None))
        fcls = _load_class(self.cfg.get("fusion") or "fusion:LogLinearFusion")
        fparams = self.cfg.get("fusion_params")
        if getattr(self.eeg, "spool", None) is not None and self.cfg.get("fusion_params_subject"):
            # подучастниковый пул ЭЭГ загружен (участник знаком) — свои константы слияния
            # (эксперимент subject_pool); иначе — прежние, подобранные для новых участников
            fparams = dict(fparams or {}, **self.cfg["fusion_params_subject"])
        fparams = self._fusion_meta(fparams)
        self.fusion = fcls(fparams if "MeanProba" not in fcls.__name__ else self.cfg.get("fusion_weights"))
        self._built = True

    def _fusion_meta(self, fparams):
        """Веса мета-модели слияния (fusion_stacked:StackedFusion) — отдельный артефакт
        `<artifacts>/<fusion_meta_file>`: {"params": {...}, "models": {...}, "models_subject": {...}}.
        `models_subject` — для прогона с пулом участника (вероятности ЭЭГ/NIRS увереннее),
        `models` — без него. Файла нет (dev-протокол без мета-модели) → params без "models" →
        StackedFusion откатывается на LogLinearFusion с теми же fparams."""
        name = self.cfg.get("fusion_meta_file")
        art = getattr(self, "artifacts_dir", None)
        if not name or art is None or not (Path(art) / name).exists():
            return fparams
        try:
            meta = json.loads((Path(art) / name).read_text())
        except Exception:
            return fparams
        out = dict(fparams or {}, **(meta.get("params") or {}))
        subj = getattr(self.eeg, "spool", None) is not None
        out["models"] = (meta.get("models_subject") if subj else None) or meta.get("models")
        return out

    # ------------------------------------------------------------------ stream
    def push(self, eeg_chunk, nirs_chunk=None) -> None:
        self._build()
        arr = eeg_chunk if isinstance(eeg_chunk, np.ndarray) else None
        if arr is not None and arr.ndim == 2 and arr.shape[0] == arr.shape[1] and self._eeg_T:
            eeg_chunk = arr.T     # квадратный чанк (отсчётов = каналов): ориентация как у прошлых чанков
        x = parse_eeg_chunk(eeg_chunk, self.meta["eeg_channels"])
        if arr is not None and arr.ndim == 2 and arr.shape[0] != arr.shape[1] and x.shape[0]:
            self._eeg_T = x.shape[0] == arr.shape[1]
        if self.meta["eeg_channels"] is None and x.shape[0] > 0:
            self.meta["eeg_channels"] = x.shape[1]
        n0, n1 = self.n, self.n + x.shape[0]
        if x.shape[0]:
            snaps = grid_ends_between(n0, n1, self.meta["eeg_fs"], self.win)
            self.eeg.update(x, snaps)
            self.n = n1
        if self.nirs is not None and nirs_chunk is not None:
            v, fr = parse_nirs_chunk(nirs_chunk, self.meta["nirs_channels"])
            if v is not None and v.shape[0]:
                if fr is None or len(fr) != v.shape[0]:
                    fr = self._nirs_frames(v.shape[0])
                self.nirs.update(v, fr)
        if x.shape[0] and self.cfg.get("async_ticks", True):
            for e in snaps:
                self._tick(int(e))

    def _tick(self, end: int) -> None:
        """Решение для окна, заканчивающегося в точке сетки end (≤ self.n): признаки ЭЭГ — снимок
        на end, NIRS — последний отсчёт с Frame ≤ end; состояние слияния шагает на каждой точке."""
        p_eeg = self.eeg.predict_proba(end)
        p_nirs = self.nirs.predict_proba(end) if (self.nirs is not None and self.nirs.available) else None
        self._tick_out = self.fusion.decide({"eeg": p_eeg, "nirs": p_nirs}, end=end, fs=self.meta["eeg_fs"])
        self._tick_end = end
        self.last_proba = {"eeg": p_eeg, "nirs": p_nirs, "hyb": getattr(self.fusion, "last_smoothed", None)}

    def _nirs_frames(self, m: int) -> np.ndarray:
        """Судья не передаёт NIRS.Frame: отсчёты чанка — это Frame ∈ [начало, конец чанка).
        Восстанавливаем метки времени (1-based индекс ЭЭГ) равномерно с шагом fs_eeg/fs_nirs,
        последний отсчёт — у конца чанка (ошибка ≤ одного периода NIRS); без знания частоты —
        все «сейчас». Метки не превышают self.n, поэтому будущие отсчёты не используются."""
        fs_n = self.meta.get("nirs_fs")
        if not fs_n:
            fr = np.full(m, self.n, np.int64)
        else:
            step = self.meta["eeg_fs"] / float(fs_n)
            fr = np.round(self.n - (m - 1 - np.arange(m)) * step).astype(np.int64)
            fr = np.maximum(fr, self._last_nirs_fr + 1 + np.arange(m))
            fr = np.minimum(fr, self.n)
        self._last_nirs_fr = int(fr[-1])
        return fr

    def predict(self) -> dict:
        self._build()
        if self.cfg.get("async_ticks", True) and self._tick_out is not None:
            # последнее решение на сетке (≤ 250 мс назад); predict ничего не меняет в состоянии
            return dict(self._tick_out)
        end = self.n
        p_eeg = self.eeg.predict_proba(end)
        p_nirs = self.nirs.predict_proba(end) if (self.nirs is not None and self.nirs.available) else None
        out = self.fusion.decide({"eeg": p_eeg, "nirs": p_nirs}, end=end, fs=self.meta["eeg_fs"])
        # справочно для dev-инструментов (паритет с оффлайн-вероятностями); судьёй не используется
        self.last_proba = {"eeg": p_eeg, "nirs": p_nirs, "hyb": getattr(self.fusion, "last_smoothed", None)}
        return out

    # ------------------------------------------------------------------ training
    def _set_labels(self, start: int, lab: np.ndarray):
        end = start + len(lab)
        if end > len(self.labels):
            self.labels = np.r_[self.labels, np.full(end - len(self.labels), -1, np.int8)]
        self.labels[start:end] = lab

    def fit_block(self, block_idx: int, labels) -> None:
        self._build()
        kind, arr = parse_labels(labels)
        new_ends, new_y = np.zeros(0, np.int64), np.zeros(0, np.int64)
        if kind == "samples":
            lab = np.asarray(arr).astype(np.int64)
            # (эвристика «по метке на каждый вызов predict» удалена: метки не сопоставляются
            # с расписанием вызовов; официальный формат — "indexed", ниже)
            since = self.n - self.last_block_end
            if len(lab) > since + self.win and len(lab) >= self.last_block_end + 1 \
                    and len(lab) <= self.n + self.win:
                lab = lab[self.last_block_end:]           # кумулятивные States с начала сессии
            start = self.last_block_end
            self._set_labels(start, lab.astype(np.int8))
            stop = start + len(lab)
            cand = grid_ends_between(start, min(stop, self.n), self.meta["eeg_fs"], self.win)
            new_ends, new_y = homogeneous_window_ends(self.labels, cand, self.win)
            self.last_block_end = stop
        elif kind == "indexed":
            idx, lab = arr[:, 0], arr[:, 1]
            if len(idx):
                end = int(idx.max()) + 1
                if end > len(self.labels):
                    self.labels = np.r_[self.labels, np.full(end - len(self.labels), -1, np.int8)]
                self.labels[idx] = lab.astype(np.int8)
                cand = grid_ends_between(max(int(idx.min()), 0), min(end, self.n), self.meta["eeg_fs"], self.win)
                new_ends, new_y = homogeneous_window_ends(self.labels, cand, self.win)
                self.last_block_end = max(self.last_block_end, end)
        else:  # windows: (end_sample, y)
            new_ends, new_y = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
            if len(new_ends):
                self.last_block_end = max(self.last_block_end, int(new_ends.max()))
        keep = (new_y >= 1) & (new_y <= 3)
        self.train_ends = np.r_[self.train_ends, new_ends[keep]]
        self.train_y = np.r_[self.train_y, new_y[keep]]
        self.eeg.fit(self.train_ends, self.train_y)
        if self.nirs is not None and self.nirs.available:
            self.nirs.fit(self.train_ends, self.train_y)
