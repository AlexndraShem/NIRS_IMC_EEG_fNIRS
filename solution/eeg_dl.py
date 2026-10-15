"""ЭЭГ-нейросеть (FBShallow × N сидов, чистый numpy) и обёртка-ансамбль с основной ЭЭГ-моделью.

Эксперимент: docs/experiments/eeg_deep.md. Подключение — через config.json, без правок classifier.py:

    "eeg_model": "eeg_dl:EEGWithDL",
    "eeg": { ...параметры основной модели...,
             "inner_model": "eeg_model:EEGLRExperts",
             "dl": {"files": ["eeg_dl_0.npz", ...], "weight": 0.3, "a": 0.5} }

EEGWithDL: log P = (1 − weight)·log P_inner + weight·log P_dl (нормировка), если DL есть и обучен;
иначе — P_inner без изменений (нет артефактов / чужое число каналов / до первого fit_block).
Прочие атрибуты (spool и т.п.) проксируются во внутреннюю модель.

EEGDL (всё причинно):
* поток: x − x[0] → sosfilt с состоянием (HP 1 Гц Баттерворт 2 + notch 50 Q30 + LP 40 Баттерворт 6)
  → чётные отсчёты (125 Гц) в буфер;
* нормировка: поканальный масштаб (медиана дисперсий 1-с окон) + Euclidean Alignment (средняя
  ковариация, shrinkage 0.1) по отсчётам размеченных окон завершённых блоков; пересчёт в fit;
* сеть: слитая свёртка K (C·k × n_s) → BN → x² → среднее по n_sub подокнам → log → FC;
  выход пула — среднее лог-softmax сидов, эмбеддинг — конкатенация;
* адаптация: sLDA(0.8) на эмбеддингах размеченных окон (каждое 2-е); P_dl ∝ P_within^(1−a)·P_pool^a.
  Эмбеддинг окна считается с нормировкой, действовавшей в момент окна (кэш predict; окна без
  predict — в ближайшем fit до обновления нормировки; для блока 1 — с первой нормировкой).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import scipy.signal as ss

from base import ModalityModel

DEC = 2
FLOOR = 1e-9


def _dl_sos(fs):
    hp = ss.butter(2, 1.0, "highpass", fs=fs, output="sos")
    b, a = ss.iirnotch(50.0, 30.0, fs=fs)
    lp = ss.butter(6, 40.0, "lowpass", fs=fs, output="sos")
    return np.vstack([hp, ss.tf2sos(b, a), lp])


def _log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


class NpFBShallow:
    def __init__(self, path):
        z = np.load(path)
        self.K, self.k, self.n_sub = z["K"].astype(np.float32), int(z["k"]), int(z["n_sub"])
        self.g, self.b = z["bn_g"].astype(np.float32), z["bn_b"].astype(np.float32)
        self.fc_w, self.fc_b = z["fc_w"].astype(np.float64), z["fc_b"].astype(np.float64)
        self.L = int(z["L"])
        self.C = self.K.shape[0] // self.k

    def embed_many(self, x, starts):
        """x (C, n) нормированный кусок; starts — начала окон длины L внутри куска."""
        x = np.ascontiguousarray(x, np.float32)
        U = np.lib.stride_tricks.sliding_window_view(x, self.k, axis=1)
        y = (U.transpose(1, 0, 2).reshape(U.shape[1], -1) @ self.K) * self.g + self.b
        cs = np.vstack([np.zeros((1, y.shape[1])), np.cumsum((y * y).astype(np.float64), 0)])
        T = self.L - self.k + 1
        st = np.asarray(starts, np.int64)
        out = []
        for i in range(self.n_sub):
            a, b = (i * T) // self.n_sub, -((-(i + 1) * T) // self.n_sub)
            out.append((cs[st + b] - cs[st + a]) / (b - a))
        emb = np.log(np.maximum(np.stack(out, 2), 1e-6)).reshape(len(st), -1)
        return emb @ self.fc_w.T + self.fc_b, emb


class ShrinkLDA:
    def fit(self, X, y, shrink=0.8):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        cls = np.unique(y)
        M = np.stack([Z[y == c].mean(0) for c in cls])
        R = Z - M[np.searchsorted(cls, y)]
        S = R.T @ R / len(Z)
        S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S))
        self.coef = np.linalg.solve(S, M.T).T
        pri = np.array([np.mean(y == c) for c in cls])
        self.icpt = -0.5 * np.sum(self.coef * M, 1) + np.log(pri)
        self.cls = cls
        return self

    def log_proba(self, X):
        z = ((X - self.mu) / self.sd) @ self.coef.T + self.icpt
        full = np.full((len(X), 3), -30.0)
        full[:, self.cls - 1] = z
        return full - np.logaddexp.reduce(full, 1, keepdims=True)


def norm_matrix(x, mask, win, shrink=0.1):
    """x (n, C) 125 Гц, mask (n,) — отсчёты статистики. Масштаб + EA: x_norm = x @ M."""
    idx = np.flatnonzero(mask)
    starts = []
    if len(idx):
        br = np.flatnonzero(np.diff(idx) > 1)
        for s0, s1 in zip(np.r_[0, br + 1], np.r_[br + 1, len(idx)]):
            a, b = idx[s0], idx[s1 - 1] + 1
            starts += list(range(a, b - win + 1, win))
    if not starts:
        return None
    seg = x[np.array(starts)[:, None] + np.arange(win)[None]].astype(np.float64)   # (n_w, win, C)
    seg = seg - seg.mean(1, keepdims=True)
    d = np.sqrt(np.median((seg ** 2).mean(1), 0))
    d = np.maximum(d, 1e-3 * np.median(d) + 1e-30)
    segn = seg / d
    cov = np.einsum("ntc,ntd->cd", segn, segn) / (len(starts) * win)
    C = len(cov)
    c = (1 - shrink) * cov + shrink * np.trace(cov) / C * np.eye(C)
    w, V = np.linalg.eigh(c)
    return np.diag(1.0 / d) @ ((V / np.sqrt(np.maximum(w, 1e-12))) @ V.T)


class EEGDL(ModalityModel):
    name = "eeg_dl"

    def __init__(self, meta: dict, cfg: dict | None = None):
        super().__init__(meta, cfg)
        self.fs = float(meta.get("eeg_fs") or 250.0)
        self.files = list(self.cfg.get("files", []))
        self.a = float(self.cfg.get("a", 0.5))
        self.shrink = float(self.cfg.get("shrinkage", 0.8))
        self.stride = int(self.cfg.get("train_stride", 2))
        self.max_train = int(self.cfg.get("max_train", 0))          # 0 — без ограничения
        self.lab_win = int(meta.get("win") or round(self.fs))
        self.clip = float(self.cfg.get("clip", 20.0))
        self.nets = []
        self.sos = _dl_sos(self.fs)
        self.zi = self.x0 = self.buf = None
        self.n = self.nd = 0
        self.M = None
        self.emb: dict[int, np.ndarray] = {}
        self.lp_pool: dict[int, np.ndarray] = {}
        self.head = None

    def load(self, artifacts_dir) -> None:
        if not artifacts_dir or abs(self.fs - 250.0) > 1e-6:
            return
        nets = []
        for f in self.files:
            p = Path(artifacts_dir) / f
            if p.exists():
                try:
                    nets.append(NpFBShallow(p))
                except Exception:
                    pass
        if nets and len({(m.L, m.C, m.k, m.n_sub, m.K.shape[1]) for m in nets}) == 1:
            self.nets = nets
            self.Kcat = np.ascontiguousarray(np.hstack([m.K for m in nets]))      # общая свёртка всех сетей
            self.gcat = np.concatenate([m.g for m in nets]); self.bcat = np.concatenate([m.b for m in nets])

    @property
    def enabled(self) -> bool:
        return bool(self.nets)

    @property
    def fitted(self) -> bool:
        return self.M is not None

    # ------------------------------------------------------------------ поток
    def update(self, x: np.ndarray, snapshot_ends=()) -> None:
        if not self.nets or x.shape[0] == 0:
            return
        if x.shape[1] != self.nets[0].C:
            self.nets = []
            return
        x = np.asarray(x, np.float64)
        if self.x0 is None:
            self.x0 = x[0].copy()
            self.zi = np.zeros((self.sos.shape[0], 2, x.shape[1]))
            self.buf = np.zeros((int(700 * self.fs / DEC), x.shape[1]), np.float32)
        y, self.zi = ss.sosfilt(self.sos, x - self.x0, axis=0, zi=self.zi)
        keep = y[(-self.n) % DEC::DEC]
        self.n += len(x)
        while self.nd + len(keep) > len(self.buf):
            self.buf = np.concatenate([self.buf, np.zeros_like(self.buf)])
        self.buf[self.nd:self.nd + len(keep)] = keep
        self.nd += len(keep)

    def _embed(self, ends):
        """Лог-вер. пула и эмбеддинги окон ends (одной свёрткой по охватывающему куску)."""
        ends = np.asarray(ends, np.int64)
        L = self.nets[0].L
        e_d = (ends + 1) // DEC
        j0 = int(e_d.min()) - L
        x = np.clip(self.buf[j0:int(e_d.max())].astype(np.float64) @ self.M, -self.clip, self.clip)
        x = np.ascontiguousarray(x.T, np.float32)
        m0 = self.nets[0]
        U = np.lib.stride_tricks.sliding_window_view(x, m0.k, axis=1)                 # (C, T', k)
        y = (U.transpose(1, 0, 2).reshape(U.shape[1], -1) @ self.Kcat) * self.gcat + self.bcat
        cs = np.cumsum((y * y).astype(np.float64), 0)
        cs = np.vstack([np.zeros((1, y.shape[1])), cs])
        T = L - m0.k + 1
        st = e_d - L - j0
        parts = []
        for i in range(m0.n_sub):
            a, b = (i * T) // m0.n_sub, -((-(i + 1) * T) // m0.n_sub)
            parts.append((cs[st + b] - cs[st + a]) / (b - a))                          # (n_win, N·n_s)
        P = np.log(np.maximum(np.stack(parts, 2), 1e-6))                               # (n_win, N·n_s, n_sub)
        ns = m0.K.shape[1]
        lps, embs = [], []
        for j, m in enumerate(self.nets):
            em = P[:, j * ns:(j + 1) * ns].reshape(len(st), -1)
            lps.append(_log_softmax(em @ m.fc_w.T + m.fc_b)); embs.append(em)
        return np.mean(lps, 0), np.concatenate(embs, 1)

    def _embed_missing(self, ends):
        miss = [int(e) for e in ends if int(e) not in self.emb and (int(e) + 1) // DEC >= self.nets[0].L
                and (int(e) + 1) // DEC <= self.nd]
        if not miss:
            return
        miss = np.array(sorted(miss))
        # по непрерывным кускам (разрывы > 10 с режем, чтобы не свёртывать лишнее)
        cut = np.flatnonzero(np.diff(miss) > 10 * self.fs) + 1
        for part in np.split(miss, cut):
            lp, E = self._embed(part)
            for i, e in enumerate(part):
                self.emb[int(e)] = E[i]; self.lp_pool[int(e)] = lp[i]

    def fit(self, ends, y) -> None:
        if not self.nets:
            return
        ends = np.asarray(ends, np.int64); y = np.asarray(y, np.int64)
        if len(ends) == 0:
            return
        stride = max(self.stride, int(np.ceil(len(ends) / self.max_train))) if self.max_train else self.stride
        if self.M is not None:                       # окна прошлой эпохи — со старой нормировкой
            self._embed_missing(ends[::stride])      # (нужны только окна, которые пойдут в голову)
        d = np.zeros(self.nd + 1, np.int64)           # отсчёты размеченных окон (разностный массив)
        lo = np.clip((ends - self.lab_win + 1) // DEC, 0, self.nd)
        hi = np.clip((ends + 1) // DEC, 0, self.nd)
        np.add.at(d, lo, 1); np.add.at(d, hi, -1)
        mask = np.cumsum(d[:-1]) > 0
        M = norm_matrix(self.buf[:self.nd], mask, int(round(self.fs / DEC)))
        if M is None:
            return
        first = self.M is None
        self.M = M
        if first:                                    # блок 1: только окна, нужные голове сейчас
            self._embed_missing(ends[::stride])
        ok = np.array([int(e) in self.emb for e in ends])
        e2, y2 = ends[::stride], y[::stride]
        keep = ok[::stride]
        e2, y2 = e2[keep], y2[keep]
        if len(e2) >= 6 and len(np.unique(y2)) >= 2:
            self.head = ShrinkLDA().fit(np.stack([self.emb[int(e)] for e in e2]), y2, self.shrink)

    def predict_proba(self, end: int):
        if not self.nets or self.M is None:
            return None
        end = int(end)
        if (end + 1) // DEC < self.nets[0].L or (end + 1) // DEC > self.nd:
            return None
        if end not in self.emb:
            lp, E = self._embed([end])
            self.emb[end], self.lp_pool[end] = E[0], lp[0]
        lp = self.lp_pool[end]
        if self.head is not None:
            lp = (1 - self.a) * self.head.log_proba(self.emb[end][None])[0] + self.a * lp
        p = np.exp(lp - lp.max())
        return p / p.sum()


def _load_class(spec):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


class EEGWithDL(ModalityModel):
    """Обёртка: основная ЭЭГ-модель (cfg["inner_model"]) × EEGDL (cfg["dl"])."""

    name = "eeg"

    def __init__(self, meta: dict, cfg: dict | None = None):
        super().__init__(meta, cfg)
        c = dict(self.cfg)
        inner_spec = c.pop("inner_model", "eeg_model:EEGLRExperts")
        dcfg = c.pop("dl", {}) or {}
        self.inner = _load_class(inner_spec)(meta, c)
        self.dl = EEGDL(meta, dcfg)
        self.w = float(dcfg.get("weight", 0.3))
        self.last_dl = None

    def __getattr__(self, k):                        # spool, pool, ... — от основной модели
        if k in ("inner", "dl"):
            raise AttributeError(k)
        return getattr(self.inner, k)

    def load(self, artifacts_dir) -> None:
        self.inner.load(artifacts_dir)
        try:
            self.dl.load(artifacts_dir)
        except Exception:
            self.dl.nets = []

    @property
    def available(self) -> bool:
        return self.inner.available

    @property
    def fitted(self) -> bool:
        return self.inner.fitted

    def update(self, x, snapshot_ends=()) -> None:
        self.inner.update(x, snapshot_ends)
        try:
            self.dl.update(x, snapshot_ends)
        except Exception:
            self.dl.nets = []

    def fit(self, ends, y) -> None:
        self.inner.fit(ends, y)
        try:
            self.dl.fit(ends, y)
        except Exception:
            self.dl.nets = []

    def predict_proba(self, end: int):
        p = self.inner.predict_proba(end)
        self.last_dl = None
        if p is None or self.w <= 0:
            return p
        try:
            q = self.dl.predict_proba(end)
        except Exception:
            self.dl.nets = []
            q = None
        self.last_dl = q
        if q is None or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            return p
        lp = (1 - self.w) * np.log(np.clip(p, FLOOR, 1)) + self.w * np.log(np.clip(q, FLOOR, 1))
        e = np.exp(lp - lp.max())
        return e / e.sum()
