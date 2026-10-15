"""ЭЭГ-модели решения.

* `EEGTangentPool` — **боевая** модель (эксперимент `eeg_ts2s_all`): причинный фильтр-банк →
  ковариация последних 2 с → рецентрированное касательное пространство → смесь
  within-sLDA(shrinkage 0.8) текущей сессии и предобученной кросс-субъектной пул-LR
  (P ∝ P_within^(1−a)·P_pool^a, a = 0.8).
* `EEGLogVarLDA` — baseline stage0 (log-дисперсии 1 с + sLDA), остаётся для отката через config.

Конвейер `EEGTangentPool` (всё причинно, состояние фильтров сохраняется между push):
  x ← x − x[0] (первый отсчёт сессии);
  префильтр HP 1 Гц (Баттерворт 2) + notch 50 Гц (Q=30);
  7 полос Баттерворта 4-го порядка: 4-8, 8-10, 10-13, 13-18, 18-24, 24-30, 8-30 Гц;
  ковариация по окну [end − 2 с, end) для каждой полосы (B, C, C);
  поканальная нормировка (медиана дисперсий референсных окон) + shrinkage 1e-2 к следу;
  касательное пространство относительно лог-евклидова среднего референсных окон →
  вектор 7 × C(C+1)/2 (21 канал → 1617);
  P_within — sLDA с фиксированным shrinkage 0.8 на окнах завершённых блоков сессии;
  P_pool    — логистическая регрессия, обученная на других участниках (артефакт `pool_file`).

`fit` вызывается из `fit_block`: пересчитываются референс (каждое `ref_stride`-е окно) и
sLDA (каждое `train_stride`-е окно) — окна перекрываются на 87.5 %, прореживание почти не
теряет информации, но кратно ускоряет дообучение.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal as ss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from base import ModalityModel, subject_of
from stream_utils import proba_to3

DEFAULT_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 26), (26, 32)]      # baseline stage0
TS_BANDS = [(4, 8), (8, 10), (10, 13), (13, 18), (18, 24), (24, 30), (8, 30)]  # eeg_ts2s_all

# ИСТИННЫЙ порядок столбцов EEG.Raw (официальный io_utils: eeg_channel_names, C3 = 18, C4 = 0).
# Montage.EEG_labels этому порядку НЕ соответствует.
EEG_LABELS = ['c4', 'rpa', 'f8', 'p8', 'f4', 'p4', 'fp2', 'o2', 'cz', 'pz',
              'fz', 'o1', 'fp1', 'p3', 'f3', 'p7', 'f7', 'lpa', 'c3', 'c7', 'c8']
FRONTAL = ["fp1", "fp2", "f7", "f8"]
REF_EAR = ["lpa", "rpa"]
CHSETS = {"all": list(EEG_LABELS),
          "nofront": [c for c in EEG_LABELS if c not in FRONTAL + REF_EAR]}


class CausalFilterBank:
    """Префильтр (HP+notch) и банк полосовых фильтров с сохраняемым состоянием."""

    def __init__(self, fs: float, n_ch: int, bands=DEFAULT_BANDS, order=4, hp=1.0, notch=50.0, notch_q=30.0):
        pre = [ss.butter(2, hp, "highpass", fs=fs, output="sos")]
        if notch and notch < fs / 2:
            b, a = ss.iirnotch(notch, notch_q, fs=fs)
            pre.append(ss.tf2sos(b, a))
        self.pre_sos = np.vstack(pre)
        self.band_sos = [ss.butter(order, [lo, min(hi, 0.99 * fs / 2)], "bandpass", fs=fs, output="sos")
                         for lo, hi in bands]
        self.pre_zi = np.zeros((self.pre_sos.shape[0], 2, n_ch))
        self.band_zi = [np.zeros((s.shape[0], 2, n_ch)) for s in self.band_sos]
        self.x0 = None

    def process(self, x: np.ndarray) -> np.ndarray:
        """x (n, C) сырое → (n, B, C) полосовые сигналы."""
        if self.x0 is None:
            self.x0 = x[:1].copy()
        x = x - self.x0
        x, self.pre_zi = ss.sosfilt(self.pre_sos, x, axis=0, zi=self.pre_zi)
        out = np.empty((x.shape[0], len(self.band_sos), x.shape[1]))
        for i, sos in enumerate(self.band_sos):
            out[:, i], self.band_zi[i] = ss.sosfilt(sos, x, axis=0, zi=self.band_zi[i])
        return out


# --------------------------------------------------------------------- риманова геометрия
_POOL = [None]


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def _eig_fn_mt(C, fn, threads=1):
    """Батч-eigh: numpy отпускает GIL, поэтому на больших батчах помогают потоки."""
    if C.ndim < 3 or len(C) < 512 or threads <= 1:
        return _eig_fn(C, fn)
    if _POOL[0] is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL[0] = ThreadPoolExecutor(threads)
    shp = C.shape
    flat = C.reshape(-1, shp[-2], shp[-1])
    parts = list(_POOL[0].map(lambda a: _eig_fn(a, fn), np.array_split(flat, threads)))
    return np.concatenate(parts).reshape(shp)


def _logm(C, threads=1):
    return _eig_fn_mt(C, lambda w: np.log(np.maximum(w, 1e-30)), threads)


def _invsqrtm(C):
    return _eig_fn(C, lambda w: 1 / np.sqrt(np.maximum(w, 1e-30)))


def _regularize(C, eps):
    C = np.asarray(C, np.float64)
    n = C.shape[-1]
    tr = np.trace(C, axis1=-2, axis2=-1)[..., None, None] / n
    return (1 - eps) * C + eps * tr * np.eye(n)


def _upper(L):
    n = L.shape[-1]
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return L[..., iu[0], iu[1]] * coef


def _diag_scale(covs_ref):
    """Поканальные масштабы по референсным окнам (медиана дисперсий), (B, C)."""
    d = np.sqrt(np.median(np.diagonal(covs_ref, axis1=-2, axis2=-1), axis=0))
    med = np.median(d, axis=-1, keepdims=True)
    return np.maximum(d, 1e-3 * med + 1e-30)            # защита от «мёртвых» каналов


def eeg_channel_labels(meta: dict, n_ch=None):
    """Имена столбцов ЭЭГ: meta["eeg_channel_names"] (официальный каркас) / eeg_labels, иначе —
    истинный порядок EEG_LABELS (если каналов 21); None — неизвестно."""
    raw = meta.get("raw") if isinstance(meta.get("raw"), dict) else meta
    for k in ("eeg_channel_names", "eeg_labels", "ch_names"):
        v = raw.get(k) if isinstance(raw, dict) else None
        if v is not None and np.ndim(v) == 1 and len(v) > 0:
            return [str(x).strip().lower() for x in v]
    return list(EEG_LABELS) if n_ch in (None, len(EEG_LABELS)) else None


class RefState:
    """Референс для рецентрирования: поканальные масштабы + лог-евклидово среднее по полосам."""

    def __init__(self, eps=1e-2, threads=1):
        self.eps, self.threads = eps, threads

    def _norm(self, covs):
        c = covs / self.d[None, :, :, None] / self.d[None, :, None, :]
        return _regularize(c, self.eps)

    def fit(self, covs):
        self.d = _diag_scale(covs)
        c = self._norm(covs)
        self.P = np.stack([_eig_fn(_logm(c[:, b], self.threads).mean(0), np.exp) for b in range(c.shape[1])])
        self.P = np.stack([_invsqrtm(p) for p in self.P])
        return self

    def ts(self, covs):
        c = self._norm(covs)
        W = self.P[None] @ c @ self.P[None]
        return _upper(_logm(W, self.threads)).reshape(len(covs), -1)


class SLDA:
    """LDA с фиксированным shrinkage (эквивалент sklearn lsqr+shrinkage, но быстрее на numpy):
    ковариация каждого класса сжимается к своей диагонали, общая — взвешена приорами."""

    def __init__(self, shrinkage=0.8):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        self.classes = np.unique(y)
        pri = np.array([np.mean(y == k) for k in self.classes])
        means = np.stack([X[y == k].mean(0) for k in self.classes])
        S = np.zeros((X.shape[1], X.shape[1]))
        for k, pk, mk in zip(self.classes, pri, means):
            Xc = X[y == k] - mk
            Ck = Xc.T @ Xc / len(Xc)
            dg = np.diag(Ck).copy()
            Ck *= (1 - self.shrinkage)
            Ck[np.diag_indices_from(Ck)] += self.shrinkage * dg
            S += pk * Ck
        self.coef = np.linalg.solve(S, means.T).T
        self.intercept = -0.5 * np.sum(means * self.coef, 1) + np.log(pri)
        return self

    def predict_log_proba(self, X):
        z = X @ self.coef.T + self.intercept
        out = np.full((len(X), 3), -np.inf)
        out[:, self.classes.astype(int) - 1] = z
        return out - np.logaddexp.reduce(out, axis=1, keepdims=True)


class PoolLR:
    """Кросс-субъектная логистическая регрессия на TS-признаках (обучена оффлайн, только инференс)."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mean"], z["scale"], z["coef"], z["intercept"])

    @property
    def n_features(self) -> int:
        return int(self.coef.shape[1])

    def predict_log_proba(self, X):
        z = ((X - self.mean) / self.scale) @ self.coef.T + self.intercept
        return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


class EEGTangentPool(ModalityModel):
    """Боевая ЭЭГ-модель: TS фильтр-банка × (within-sLDA + пул-LR)."""

    name = "eeg"

    def __init__(self, meta: dict, cfg: dict | None = None):
        super().__init__(meta, cfg)
        self.fs = float(meta["eeg_fs"])
        self.n_ch = meta["eeg_channels"]
        self.L = int(round(float(self.cfg.get("win_s", 2.0)) * self.fs))    # окно признака (2 с)
        self.lab_win = int(meta.get("win") or round(self.fs))               # окно разметки (1 с)
        self.bands = [tuple(b) for b in self.cfg.get("bands", TS_BANDS)]
        self.shrinkage = float(self.cfg.get("shrinkage", 0.8))
        self.alpha = float(self.cfg.get("alpha", 0.8))
        self.ref_stride = int(self.cfg.get("ref_stride", 4))
        self.train_stride = int(self.cfg.get("train_stride", 2))
        self.max_train = int(self.cfg.get("max_train_windows", 0))          # 0 — без ограничения
        self.eps = float(self.cfg.get("eps", 1e-2))
        self.threads = int(self.cfg.get("eig_threads", 1))
        self.chans = self.cfg.get("chans", "all")
        self.pool_file = self.cfg.get("pool_file", "eeg_pool.npz")
        # подучастниковый пул: <artifacts>/<subject_pool_dir>/<участник>.npz (прошлые сессии того же
        # участника); его вес subject_weight берётся из общего веса пулов alpha
        self.subject_pool_dir = self.cfg.get("subject_pool_dir")
        self.subject_weight = float(self.cfg.get("subject_weight", 0.0))
        self.subject = subject_of(meta)
        self.pool = None
        self.spool = None
        self.fb = None
        self.ci = None
        self.buf = None
        self.n = 0
        self.cov: dict[int, np.ndarray] = {}      # конец окна → ковариации полос (B, C, C)
        self.ref = self.within = None

    # ------------------------------------------------------------------ артефакты
    def load(self, artifacts_dir) -> None:
        if not artifacts_dir or not self.pool_file:
            return
        p = Path(artifacts_dir) / self.pool_file
        if p.exists():
            try:
                self.pool = PoolLR.load(p)
            except Exception:
                self.pool = None
        if self.subject_pool_dir and self.subject and self.subject_weight > 0:
            sp = Path(artifacts_dir) / self.subject_pool_dir / f"{self.subject}.npz"
            if sp.exists():
                try:
                    self.spool = PoolLR.load(sp)
                except Exception:
                    self.spool = None

    def _ch_index(self):
        """Индексы используемых каналов. Порядок меток — из meta, иначе стандартный монтаж."""
        if self.chans in (None, "all"):
            return np.arange(self.n_ch)
        labels = eeg_channel_labels(self.meta, self.n_ch)
        if labels is None:
            labels = EEG_LABELS if self.n_ch == len(EEG_LABELS) else None
        want = CHSETS.get(self.chans, CHSETS["all"])
        if labels is None:
            return np.arange(self.n_ch)
        idx = [labels.index(c) for c in want if c in labels]
        return np.array(idx) if idx else np.arange(self.n_ch)

    def _build(self, n_ch_chunk: int):
        if self.n_ch is None:
            self.n_ch = n_ch_chunk
        self.ci = self._ch_index()
        C = len(self.ci)
        self.fb = CausalFilterBank(self.fs, C, self.bands)
        self.buf = np.zeros((len(self.bands), self.L, C))
        # пул обучен на фиксированной размерности TS; при несовпадении — только within
        nf = len(self.bands) * C * (C + 1) // 2
        if self.pool is not None and self.pool.n_features != nf:
            self.pool = None
        if self.spool is not None and self.spool.n_features != nf:
            self.spool = None

    # ------------------------------------------------------------------ поток
    def _append(self, xb):
        """xb (n, B, C) → в кольцевой буфер (B, L, C)."""
        xb = np.ascontiguousarray(np.transpose(xb, (1, 0, 2)))
        k = xb.shape[1]
        if k >= self.L:
            self.buf = xb[:, -self.L:].copy()
        else:
            self.buf = np.concatenate([self.buf[:, k:], xb], axis=1)

    def _current_cov(self):
        return np.einsum("btc,btd->bcd", self.buf, self.buf) / self.L

    def update(self, x: np.ndarray, snapshot_ends=()) -> None:
        if x.shape[0] == 0:
            return
        if self.fb is None:
            self._build(x.shape[1])
        x = np.asarray(x, np.float64)[:, self.ci]
        n0 = self.n
        snaps = [int(e) for e in snapshot_ends if n0 < int(e) <= n0 + x.shape[0]]
        pos = 0
        for e in snaps:                       # режем чанк по точкам сетки: снимок ровно в конце окна
            take = e - n0 - pos
            if take > 0:
                self._append(self.fb.process(x[pos:pos + take]))
                pos += take
            self.n = e
            if self.n >= self.L:
                self.cov[e] = self._current_cov()
        if pos < x.shape[0]:
            self._append(self.fb.process(x[pos:]))
            self.n = n0 + x.shape[0]

    # ------------------------------------------------------------------ модель
    @property
    def fitted(self) -> bool:
        return self.ref is not None

    def _covs_for(self, ends):
        idx, covs = [], []
        for i, e in enumerate(ends):
            c = self.cov.get(int(e))
            if c is not None:
                idx.append(i); covs.append(c)
        return np.array(idx, np.int64), (np.array(covs) if covs else np.zeros((0, len(self.bands), 1, 1)))

    def fit(self, ends, y) -> None:
        idx, covs = self._covs_for(ends)
        if len(covs) < 6:
            return
        yy = np.asarray(y, np.int64)[idx]
        rs, ts_ = self.ref_stride, self.train_stride
        if self.max_train and len(covs) // ts_ > self.max_train:     # страховка по времени
            ts_ = int(np.ceil(len(covs) / self.max_train))
            rs = max(rs, ts_ * 2)
        self.ref = RefState(self.eps, self.threads).fit(covs[::rs])
        self.within = None
        if len(np.unique(yy)) >= 3:
            X = self.ref.ts(covs[::ts_])
            self.within = SLDA(self.shrinkage).fit(X, yy[::ts_])

    def predict_proba(self, end: int):
        if self.ref is None:
            return None
        c = self.cov.get(int(end))
        if c is None:
            if int(end) != self.n or self.buf is None or self.n < self.L:
                return None
            c = self._current_cov()
        f = self.ref.ts(c[None])
        lp = np.zeros((1, 3))
        floor = np.log(1e-9)
        # веса: within 1−alpha, общий пул alpha−ws, пул участника ws; отсутствующая часть отдаёт
        # вес остальным (нет пула участника → всё общему; нет within → пулы нормируются на 1)
        ws = self.subject_weight if self.spool is not None else 0.0
        wg = (self.alpha - ws) if self.pool is not None else 0.0
        if self.pool is None and self.spool is not None:
            ws = self.alpha
        ww = (1 - self.alpha) if self.within is not None else 0.0
        tot = ww + wg + ws
        if tot <= 0:
            ww, tot = 1.0, 1.0
        if self.within is None and tot > 0:
            wg, ws = wg / tot, ws / tot
        elif self.within is not None and wg + ws == 0:
            ww = 1.0
        if wg > 0:
            lp += wg * np.maximum(self.pool.predict_log_proba(f), floor)
        if ws > 0:
            lp += ws * np.maximum(self.spool.predict_log_proba(f), floor)
        if ww > 0 and self.within is not None:
            lp += ww * np.maximum(self.within.predict_log_proba(f), floor)
        p = np.exp(lp[0] - lp[0].max())
        s = p.sum()
        return p / s if s > 0 else np.full(3, 1 / 3)


class EEGLogVarLDA(ModalityModel):
    """Baseline stage0: log-дисперсии полос в окне 1 с + sLDA (оставлен для отката через config)."""

    name = "eeg"

    def __init__(self, meta: dict, cfg: dict | None = None):
        super().__init__(meta, cfg)
        self.fs = meta["eeg_fs"]
        self.n_ch = meta["eeg_channels"]
        self.win = meta["win"]
        bands = [tuple(b) for b in self.cfg.get("bands", DEFAULT_BANDS)]
        self.fb = None if self.n_ch is None else CausalFilterBank(self.fs, self.n_ch, bands)
        self.bands = bands
        self.tail = None            # (≤ win, B, C) последние полосовые отсчёты
        self.n = 0                  # сколько отсчётов обработано
        self.snap: dict[int, np.ndarray] = {}
        self.model = None
        self.shrinkage = self.cfg.get("shrinkage", "auto")

    # ------------------------------------------------------------------ поток
    def _feat(self, seg: np.ndarray) -> np.ndarray:
        # seg (win, B, C) → (B*C,), порядок как в оффлайн-кэше: полоса-major
        return np.log(np.maximum(np.mean(seg * seg, axis=0), 1e-30)).ravel()

    def update(self, x: np.ndarray, snapshot_ends=()) -> None:
        """x (n, C) — новые отсчёты с абсолютного индекса self.n; snapshot_ends — концы окон
        в (self.n, self.n + n], для которых надо запомнить признаки."""
        if x.shape[0] == 0:
            return
        if self.fb is None:
            self.n_ch = x.shape[1]
            self.fb = CausalFilterBank(self.fs, self.n_ch, self.bands)
        xb = self.fb.process(x)
        buf = xb if self.tail is None else np.concatenate([self.tail, xb], axis=0)
        start = self.n + x.shape[0] - buf.shape[0]          # абсолютный индекс buf[0]
        for e in snapshot_ends:
            e = int(e)
            if e - self.win >= start and e - self.win >= 0:
                self.snap[e] = self._feat(buf[e - self.win - start:e - start])
        self.n += x.shape[0]
        self.tail = buf[-self.win:]

    def features_at(self, end: int) -> np.ndarray | None:
        if end in self.snap:
            return self.snap[end]
        if end == self.n and self.tail is not None and self.tail.shape[0] >= self.win:
            f = self._feat(self.tail[-self.win:])
            self.snap[end] = f
            return f
        return None

    # ------------------------------------------------------------------ модель
    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, ends, y) -> None:
        X, Y = [], []
        for e, c in zip(ends, y):
            f = self.snap.get(int(e))
            if f is not None:
                X.append(f); Y.append(int(c))
        if len(Y) < 3 or len(set(Y)) < 2:
            return
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.shrinkage)
        self.model = m.fit(np.asarray(X), np.asarray(Y))

    def predict_proba(self, end: int):
        if self.model is None:
            return None
        f = self.features_at(end)
        if f is None:
            return None
        return proba_to3(self.model, self.model.predict_proba(f[None]))[0]


# ===================================================================== eeg_lr_nbsm4
# Эксперимент docs/experiments/eeg_lr_features.md: base (EEGTangentPool) × эксперт «nb»
# (логдисперсии биполяр+лапласианов в 13 узких полосах, 2 с) × эксперт «sm4» (TS 11 сенсомоторных
# каналов, 4 с). У экспертов within-sLDA обучается с зеркальной аугментацией (полушария L↔R).

NB_BANDS = [(4, 8)] + [(f, f + 2) for f in range(8, 30, 2)] + [(8, 30)]
SM_CHANS = ["c3", "c4", "cz", "f3", "f4", "fz", "p3", "p4", "pz", "c7", "c8"]
MIRROR = {"c3": "c4", "f3": "f4", "p3": "p4", "c7": "c8", "f7": "f8", "p7": "p8", "fp1": "fp2",
          "o1": "o2", "lpa": "rpa", "cz": "cz", "fz": "fz", "pz": "pz"}
MIRROR.update({v: k for k, v in MIRROR.items()})
LAP = {"c3": ["f3", "p3", "cz", "c7"], "c4": ["f4", "p4", "cz", "c8"], "cz": ["fz", "pz", "c3", "c4"],
       "f3": ["fp1", "fz", "c3", "f7"], "f4": ["fp2", "fz", "c4", "f8"], "p3": ["c3", "pz", "o1", "p7"],
       "p4": ["c4", "pz", "o2", "p8"], "c7": ["f7", "p7", "c3"], "c8": ["f8", "p8", "c4"],
       "fz": ["fp1", "fp2", "f3", "f4", "cz"], "pz": ["cz", "p3", "p4", "o1", "o2"]}
BIP = [("f3", "c3"), ("c3", "p3"), ("c7", "c3"), ("c3", "cz"), ("cz", "c4"), ("c4", "c8"),
       ("f4", "c4"), ("c4", "p4"), ("fz", "cz"), ("cz", "pz"), ("f3", "p3"), ("f4", "p4")]
_FLOOR = np.log(1e-9)


def nb_spatial(labels):
    """(23, C): 12 биполяр + 11 малых лапласианов по именам каналов."""
    ix = {c: i for i, c in enumerate(labels)}
    rows = []
    for a, b in BIP:
        w = np.zeros(len(labels)); w[ix[a]] = 1; w[ix[b]] = -1; rows.append(w)
    for c, nb in LAP.items():
        w = np.zeros(len(labels)); w[ix[c]] = 1
        for m in nb:
            w[ix[m]] -= 1 / len(nb)
        rows.append(w)
    return np.array(rows)


def mirror_rows(W, labels):
    """Знаковая перестановка строк W под зеркалом полушарий: (perm, sign)."""
    cm = np.array([labels.index(MIRROR[c]) for c in labels])
    Wm = W[:, cm]
    perm, sign = [], []
    for i in range(len(W)):
        hit = [(j, sg) for j in range(len(W)) for sg in (1, -1) if np.allclose(Wm[i], sg * W[j])]
        perm.append(hit[0][0]); sign.append(hit[0][1])
    return np.array(perm), np.array(sign, float)


def feature_mirror(perm, sign, kind, n_bands):
    """Индексы/знаки зеркалирования вектора признаков (полоса — старший индекс)."""
    if kind == "logvar":
        fi, fs = np.asarray(perm), np.ones(len(perm))
    else:
        k = len(perm)
        iu = np.triu_indices(k)
        pos = {(a, b): t for t, (a, b) in enumerate(zip(*iu))}
        fi = np.array([pos[(min(perm[a], perm[b]), max(perm[a], perm[b]))] for a, b in zip(*iu)])
        fs = np.array([sign[a] * sign[b] for a, b in zip(*iu)])
    m = len(fi)
    return np.concatenate([fi + b * m for b in range(n_bands)]), np.tile(fs, n_bands)


def swap_lr(y):
    y = np.array(y, copy=True)
    l, r = y == 2, y == 3
    y[l], y[r] = 3, 2
    return y


class SLDAFast(SLDA):
    """Та же sLDA (shrinkage к диагонали ковариаций классов), решение через тождество Вудбери:
    S = D + UᵀU (D — диагональ, U — n×d центрированные окна), быстрее при n < d."""

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        n, d = X.shape
        if n >= d:
            return super().fit(X, y)
        self.classes = np.unique(y)
        pri = np.array([np.mean(y == k) for k in self.classes])
        means = np.stack([X[y == k].mean(0) for k in self.classes])
        D = np.zeros(d)
        U = []
        for k, pk, mk in zip(self.classes, pri, means):
            Xc = X[y == k] - mk
            D += pk * self.shrinkage * np.mean(Xc * Xc, 0)
            U.append(Xc * np.sqrt(pk * (1 - self.shrinkage) / len(Xc)))
        U = np.vstack(U)
        D = np.maximum(D, 1e-12)
        M = means.T / D[:, None]                              # D⁻¹ M
        UD = U / D                                            # U D⁻¹
        K = np.eye(len(U)) + UD @ U.T
        self.coef = (M - UD.T @ np.linalg.solve(K, U @ M)).T
        self.intercept = -0.5 * np.sum(means * self.coef, 1) + np.log(pri)
        return self


def _mix_parts(parts):
    """parts: [(вес, лог-вероятности (1,3) | None)] → нормированная смесь; отсутствующие части
    отдают вес остальным."""
    parts = [(w, lp) for w, lp in parts if lp is not None and w > 0]
    if not parts:
        return None
    tot = sum(w for w, _ in parts)
    z = sum(w / tot * np.maximum(lp, _FLOOR) for w, lp in parts)
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def _load_pool(path):
    try:
        return PoolLR.load(path) if path is not None and Path(path).exists() else None
    except Exception:
        return None


class _Expert:
    """within-sLDA с зеркальной аугментацией + общий пул + пул участника."""

    def __init__(self, mirror, shrinkage=0.8):
        self.mirror, self.shrinkage = mirror, shrinkage
        self.within = self.pool = self.spool = None

    def fit(self, X, y):
        self.within = None
        if len(np.unique(y)) < 3:
            return
        Xa = np.vstack([X, X[:, self.mirror[0]] * self.mirror[1]])
        self.within = SLDAFast(self.shrinkage).fit(Xa, np.concatenate([y, swap_lr(y)]))

    def log_proba(self, f, a, rho):
        """a — общий вес пулов, rho — доля пула участника внутри a."""
        ws = a * rho if self.spool is not None else 0.0
        wg = a - ws
        return _mix_parts([(1 - a, self.within.predict_log_proba(f) if self.within is not None else None),
                           (wg, self.pool.predict_log_proba(f) if self.pool is not None else None),
                           (ws, self.spool.predict_log_proba(f) if self.spool is not None else None)])


class EEGLRExperts(EEGTangentPool):
    """Боевая ЭЭГ-модель `eeg_lr_nbsm4`: P ∝ P_base · P_nb · P_sm4.

    * base — EEGTangentPool (TS 21 канал × 7 полос, 2 с): within-sLDA(0.8) + общий пул + пул
      участника; общий вес пулов линейно по блоку b = 2…7: base_alpha = [0.9, 0.7];
      доля пула участника rho = subject_weight / 0.8 (= 0.375 при subject_weight 0.3).
    * nb — логдисперсии 23 пространственных каналов (12 биполяр + 11 лапласианов) в 13 полосах
      (4-8, 2-Гц 8…30, 8-30), окно 2 с, нормировка log(v / медиана v референсных окон);
    * sm4 — TS 11 сенсомоторных каналов × 7 полос, ковариация за 4 с, свой референс.
      Эксперты: within-sLDA(0.8) на окнах + зеркальных копиях (L↔R) + пулы; вес пулов [0.8, 0.5].
    * fit: не больше max_train обучающих окон (train_stride = max(2, ⌈n/max_train⌉),
      ref_stride = 2·train_stride) — ограничение времени fit_block.
    Нет имён 21 канала → эксперты выключены, остаётся base.
    """

    def __init__(self, meta: dict, cfg: dict | None = None):
        super().__init__(meta, cfg)
        self.max_train = int(self.cfg.get("max_train", 400))
        self.base_alpha = [float(v) for v in self.cfg.get("base_alpha", [0.9, 0.7])]
        self.exp_alpha = [float(v) for v in self.cfg.get("expert_alpha", [0.8, 0.5])]
        self.L4 = int(round(float(self.cfg.get("expert_win_s", 4.0)) * self.fs))
        self.files = self.cfg.get("expert_pools", {"nb": "eeg_lrnb_pool.npz", "sm4": "eeg_lrsm4_pool.npz"})
        self.subj_dirs = self.cfg.get("expert_subject_dirs", {"nb": "eeg_lrnb_subj", "sm4": "eeg_lrsm4_subj"})
        self.use_experts = bool(self.cfg.get("experts", True))
        # абляции (по умолчанию всё включено): отключение части ансамбля
        self.use_base = bool(self.cfg.get("use_base", True))
        self.expert_on = {"nb": bool(self.cfg.get("use_nb", True)), "sm4": bool(self.cfg.get("use_sm4", True))}
        self.n_fit = 0
        self.n_train = 0                    # число размеченных окон в последнем fit (объём калибровки)
        self.win_per_block = float(self.cfg.get("win_per_block", 264.0))   # ≈ 72 с размеченного сигнала
        self.snap_x = {}                    # конец окна → (cov4_sm (7,11,11), var_nb (13,23))
        self.nb = self.sm4 = None
        self.ref_sm = self.nb_ref = None
        self._art = None

    def load(self, artifacts_dir) -> None:
        super().load(artifacts_dir)
        self._art = Path(artifacts_dir) if artifacts_dir else None

    @property
    def rho(self):
        return min(1.0, self.subject_weight / max(self.alpha, 1e-9))

    def _build(self, n_ch_chunk: int):
        super()._build(n_ch_chunk)
        labels = eeg_channel_labels(self.meta, self.n_ch)
        need = set(SM_CHANS) | set(MIRROR) | {c for v in LAP.values() for c in v}
        ok = (self.use_experts and labels is not None and len(labels) == self.n_ch and need <= set(labels)
              and len(self.ci) == self.n_ch)
        self.buf = np.zeros((len(self.bands), max(self.L, self.L4), len(self.ci)))
        if not ok:
            self.use_experts = False
            return
        self.sm_ix = np.array([labels.index(c) for c in SM_CHANS])
        self.W = nb_spatial(labels)
        pre = self.fb.pre_sos
        self.nb_pre = pre
        self.nb_sos = [ss.butter(4, b, "bandpass", fs=self.fs, output="sos") for b in NB_BANDS]
        K = self.W.shape[0]
        self.nb_zi = [np.zeros((s.shape[0], 2, K)) for s in self.nb_sos]
        self.nbuf = np.zeros((len(self.nb_sos), self.L, K))
        pm, sg = mirror_rows(self.W, labels)
        self.nb = _Expert(feature_mirror(pm, sg, "logvar", len(self.nb_sos)))
        pos = {c: i for i, c in enumerate(SM_CHANS)}
        pm_sm = np.array([pos[MIRROR[c]] for c in SM_CHANS])
        self.sm4 = _Expert(feature_mirror(pm_sm, np.ones(len(SM_CHANS)), "ts", len(self.bands)))
        if self._art is not None:
            for name, ex in (("nb", self.nb), ("sm4", self.sm4)):
                ex.pool = _load_pool(self._art / self.files[name]) if self.files.get(name) else None
                if self.subject and self.subject_weight > 0 and self.subj_dirs.get(name):
                    ex.spool = _load_pool(self._art / self.subj_dirs[name] / f"{self.subject}.npz")
            nf = {"nb": len(self.nb_sos) * K, "sm4": len(self.bands) * len(SM_CHANS) * (len(SM_CHANS) + 1) // 2}
            for name, ex in (("nb", self.nb), ("sm4", self.sm4)):
                for att in ("pool", "spool"):
                    p = getattr(ex, att)
                    if p is not None and p.n_features != nf[name]:
                        setattr(ex, att, None)

    # ------------------------------------------------------------------ поток
    def _process(self, x):
        """x (n, C) сырые (уже выбранные каналы) → обновить буферы base и nb."""
        xb = self.fb.process(x)                                  # (n, B, C); fb хранит x0 и zi
        self._append(xb)
        if self.use_experts:
            # префильтр fb уже применён внутри process; повторяем его для пространственных каналов
            # с собственным состоянием (линейность: W·prefilt(x) = prefilt(W·x))
            xs = (x - self.fb.x0) @ self.W.T
            if not hasattr(self, "nb_pre_zi"):
                self.nb_pre_zi = np.zeros((self.nb_pre.shape[0], 2, xs.shape[1]))
            xs, self.nb_pre_zi = ss.sosfilt(self.nb_pre, xs, axis=0, zi=self.nb_pre_zi)
            out = np.empty((len(self.nb_sos), xs.shape[0], xs.shape[1]))
            for i, sos in enumerate(self.nb_sos):
                out[i], self.nb_zi[i] = ss.sosfilt(sos, xs, axis=0, zi=self.nb_zi[i])
            k = out.shape[1]
            if k >= self.L:
                self.nbuf = out[:, -self.L:].copy()
            else:
                self.nbuf = np.concatenate([self.nbuf[:, k:], out], axis=1)

    def _append(self, xb):
        xb = np.ascontiguousarray(np.transpose(xb, (1, 0, 2)))
        k = xb.shape[1]
        if k >= self.buf.shape[1]:
            self.buf = xb[:, -self.buf.shape[1]:].copy()
        else:
            self.buf = np.concatenate([self.buf[:, k:], xb], axis=1)

    def _current_cov(self):
        b = self.buf[:, -self.L:]
        return np.einsum("btc,btd->bcd", b, b) / self.L

    def _current_x(self):
        b4 = self.buf[:, -self.L4:][:, :, self.sm_ix]
        c4 = np.einsum("btc,btd->bcd", b4, b4) / self.L4
        v = np.mean(self.nbuf ** 2, axis=1)
        return c4, v

    def update(self, x: np.ndarray, snapshot_ends=()) -> None:
        if x.shape[0] == 0:
            return
        if self.fb is None:
            self._build(x.shape[1])
        x = np.asarray(x, np.float64)[:, self.ci]
        n0 = self.n
        snaps = [int(e) for e in snapshot_ends if n0 < int(e) <= n0 + x.shape[0]]
        pos = 0
        for e in snaps:
            take = e - n0 - pos
            if take > 0:
                self._process(x[pos:pos + take])
                pos += take
            self.n = e
            if self.n >= self.L:
                self.cov[e] = self._current_cov()
                if self.use_experts:
                    self.snap_x[e] = self._current_x()
        if pos < x.shape[0]:
            self._process(x[pos:])
            self.n = n0 + x.shape[0]

    # ------------------------------------------------------------------ модель
    @staticmethod
    def _sched(block, a):
        b = min(max(block, 2), 7)
        return a[0] + (a[1] - a[0]) * (b - 2) / 5

    def fit(self, ends, y) -> None:
        self.n_fit += 1
        idx, covs = self._covs_for(ends)
        if len(covs) < 6:
            return
        yy = np.asarray(y, np.int64)[idx]
        self.n_train = len(covs)
        ts_ = max(2, int(np.ceil(len(covs) / self.max_train))) if self.max_train else self.train_stride
        rs = 2 * ts_ if self.max_train else self.ref_stride
        self.ref = RefState(self.eps, self.threads).fit(covs[::rs])
        self.within = None
        if len(np.unique(yy)) >= 3:
            X = self.ref.ts(covs[::ts_])
            self.within = SLDAFast(self.shrinkage).fit(X, yy[::ts_])
        if not self.use_experts:
            return
        ends_ok = np.asarray(ends, np.int64)[idx]
        xs = [self.snap_x.get(int(e)) for e in ends_ok]
        if any(v is None for v in xs):
            keep = np.array([v is not None for v in xs])
            xs = [v for v in xs if v is not None]
            yy = yy[keep]
        if len(xs) < 6:
            return
        c4 = np.array([v[0] for v in xs]); vv = np.array([v[1] for v in xs])
        self.ref_sm = RefState(self.eps, self.threads).fit(c4[::rs])
        self.sm4.fit(self.ref_sm.ts(c4[::ts_]), yy[::ts_])
        self.nb_ref = np.maximum(np.median(vv[::rs], axis=0), 1e-30)
        self.nb.fit(self._nb_feat(vv[::ts_]), yy[::ts_])

    def _nb_feat(self, v):
        return np.log(np.maximum(v, 1e-30) / self.nb_ref).reshape(len(v), -1)

    def predict_proba(self, end: int):
        if self.ref is None:
            return None
        e = int(end)
        c = self.cov.get(e)
        xe = self.snap_x.get(e) if self.use_experts else None
        if c is None or (self.use_experts and xe is None):
            if e != self.n or self.n < self.L:
                return None
            c = self._current_cov()
            xe = self._current_x() if self.use_experts else None
        # вес пулов — по объёму размеченных in-session данных (а не по числу вызовов fit_block):
        # «эквивалент блока» = 1 + окна / win_per_block → [0.9 … 0.7] при 1 … 6 блоках калибровки
        block = 1.0 + self.n_train / self.win_per_block
        f = self.ref.ts(c[None])
        a = self._sched(block, self.base_alpha)
        ws = a * self.rho if self.spool is not None else 0.0
        z = _mix_parts([(1 - a, self.within.predict_log_proba(f) if self.within is not None else None),
                        (a - ws, self.pool.predict_log_proba(f) if self.pool is not None else None),
                        (ws, self.spool.predict_log_proba(f) if self.spool is not None else None)])
        if z is None:
            return None
        if not self.use_base and self.use_experts and self.ref_sm is not None:
            z = np.zeros_like(z)
        if self.use_experts and self.ref_sm is not None:
            ae = self._sched(block, self.exp_alpha)
            for name, ex, fx in (("nb", self.nb, self._nb_feat(xe[1][None])), ("sm4", self.sm4, self.ref_sm.ts(xe[0][None]))):
                if not self.expert_on[name]:
                    continue
                ze = ex.log_proba(fx, ae, self.rho)
                if ze is not None:
                    z = z + ze
        p = np.exp(z[0] - z[0].max())
        s = p.sum()
        return p / s if s > 0 else np.full(3, 1 / 3)
