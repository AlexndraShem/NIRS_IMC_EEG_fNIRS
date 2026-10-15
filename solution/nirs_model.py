"""NIRS-модели решения.

* `NirsCascadeEns` — **боевая** модель (эксперимент `nirs_cascade_ens`): каскад ансамблей sLDA
  «покой/задача» (8 членов) × «левая/правая» (4 члена) на производных каналах четырёх
  полосовых банков.
* `NirsLagLDA` — baseline stage0 (уровень + разности с лагами → sLDA), остаётся для отката.

Конвейер `NirsCascadeEns` (причинно, состояние `sosfilt` сохраняется между push):
  4 банка полосовых Баттервортов по HbO/HbR: 0.005–0.1 (2), 0.01–0.2 (2), 0.02–0.3 (2),
  0.01–0.5 Гц (3); zi = steady-state по первому отсчёту;
  производные каналы (58): 36 исходных + 9 пар латерализации (левое − правое) для HbO и HbR +
  глобальные средние HbO/HbR + средняя латерализация HbO/HbR;
  масштаб = std каждого производного канала по блоку 1 (с 20-й секунды), далее заморожен;
  признак окна = значения в последнем отсчёте с Frame ≤ end и на лагах назад
  (`diff` = [x(t), x(t) − x(t−lag)…], `level` = [x(t−lag)…]);
  p_t = σ(среднее логитов 8 sLDA «задача/покой»), p_r = σ(среднее логитов 4 sLDA «правая/левая»,
  обучаются только на окнах задачи с позицией ≥ 3 с от начала сегмента);
  P = [1 − p_t, p_t(1 − p_r), p_t·p_r].
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import json
from pathlib import Path

from base import ModalityModel, subject_of
from stream_utils import proba_to3

# имя банка → (lo, hi, order)
BANKS = {
    "bp005_10": (0.005, 0.10, 2),
    "bp01_20": (0.01, 0.20, 2),
    "bp02_30": (0.02, 0.30, 2),
    "bp01_50": (0.01, 0.50, 3),
}
# (банк, набор каналов, лаги [с], режим)
RT_MEMBERS = [
    ("bp01_50", "chan", (0, 4, 8, 12), "diff"),
    ("bp01_50", "all", (0, 4, 8, 12), "diff"),
    ("bp01_20", "chan", (0, 4, 8, 12), "diff"),
    ("bp02_30", "chan", (0, 4, 8, 12), "diff"),
    ("bp005_10", "chan", (0, 4, 8, 12), "diff"),
    ("bp02_30", "all", (0, 4, 8, 12), "diff"),
    ("bp01_20", "chan", (0, 3, 6, 9, 12), "level"),
    ("bp01_50", "chan", (0, 1, 2, 4, 6), "diff"),
]
LR_MEMBERS = [
    ("bp01_20", "all", (0,), "level"),
    ("bp02_30", "all", (0,), "level"),
    ("bp005_10", "all", (0, 3), "diff"),
    ("bp01_20", "chan", (0,), "level"),
]
LR_TMIN = 3.0            # с от начала сегмента — минимальная позиция обучающего окна для L/R
N_CH_HALF = 29           # o0..o17, olat0..8, oglob, olatm
RIGHT = list(range(0, 8)) + [16]
LEFT = list(range(8, 16)) + [17]


def ch_select(kind):
    o = np.arange(N_CH_HALF)
    r = o + N_CH_HALF
    return {"all": np.r_[o, r], "hbo": o, "chan": np.r_[o[:18], r[:18]],
            "latglob": np.r_[o[18:], r[18:]]}[kind]


def bandpass_sos(fs: float, lo: float, hi: float, order: int):
    hi = min(hi, 0.45 * fs)
    if lo and lo > 0:
        return ss.butter(order, [lo, hi], "bandpass", fs=fs, output="sos")
    return ss.butter(order, hi, "lowpass", fs=fs, output="sos")


def derived_channels(hbo: np.ndarray, hbr: np.ndarray) -> np.ndarray:
    """hbo/hbr (m, 18) уже отфильтрованные → (m, 58) производных каналов."""
    cols = []
    for x in (hbo, hbr):
        cols.append(x)
        lat = x[:, LEFT] - x[:, RIGHT]
        cols.append(lat)
        cols.append(x.mean(1, keepdims=True))
        cols.append(lat.mean(1, keepdims=True))
    return np.hstack(cols)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _slda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])


def pos_in_segment(ends, y, fs_eeg=250.0):
    """Время (с) от начала однородного сегмента до конца окна; только для отбора обучающих окон."""
    new = np.r_[True, (np.diff(ends) > 0.28 * fs_eeg) | (np.diff(y) != 0)]
    first = np.maximum.accumulate(np.where(new, np.arange(len(ends)), 0))
    return (ends - ends[first]) / fs_eeg


class _Bank:
    """Один полосовой банк: состояние фильтров + история производных каналов + статистики блока 1."""

    def __init__(self, fs, lo, hi, order, n_half=18):
        self.fs = fs
        self.sos = bandpass_sos(fs, lo, hi, order)
        self.zo = self.zr = None
        self.sig = np.zeros((0, 2 * (n_half + 11)), np.float32)
        self.n = 0
        self.s1 = np.zeros(self.sig.shape[1])
        self.s2 = np.zeros(self.sig.shape[1])
        self.cnt = 0

    def push(self, hbo, hbr, accumulate):
        if self.zo is None:
            z = ss.sosfilt_zi(self.sos)[:, :, None]
            self.zo, self.zr = z * hbo[0][None, None, :], z * hbr[0][None, None, :]
        fo, self.zo = ss.sosfilt(self.sos, hbo, axis=0, zi=self.zo)
        fr, self.zr = ss.sosfilt(self.sos, hbr, axis=0, zi=self.zr)
        d = derived_channels(fo, fr)
        if self.n + len(d) > len(self.sig):
            new = np.zeros((max(2 * len(self.sig), self.n + len(d), 1024), self.sig.shape[1]), np.float32)
            new[:self.n] = self.sig[:self.n]
            self.sig = new
        self.sig[self.n:self.n + len(d)] = d
        start = self.n
        self.n += len(d)
        skip = int(20 * self.fs)
        if accumulate and self.n > skip:
            dd = d[max(0, skip - start):]
            self.s1 += dd.sum(0); self.s2 += (dd ** 2).sum(0); self.cnt += len(dd)

    def scale(self):
        m = self.s1 / max(self.cnt, 1)
        sd = np.sqrt(np.maximum(self.s2 / max(self.cnt, 1) - m ** 2, 0))
        return np.where(sd > 0, sd, 1.0)


class NirsCascadeEns(ModalityModel):
    name = "nirs"

    KNOWN_FS = (3.90625, 7.8125, 10.0, 12.5, 15.625, 25.0)

    def __init__(self, meta: dict, cfg: dict | None = None):
        super().__init__(meta, cfg)
        self.fs = meta.get("nirs_fs")
        self.eeg_fs = float(meta["eeg_fs"])
        self.lab_win = int(meta.get("win") or round(self.eeg_fs))
        rt = [tuple(m) for m in self.cfg.get("rt_members", RT_MEMBERS)]
        lr = [tuple(m) for m in self.cfg.get("lr_members", LR_MEMBERS)]
        self.members = [("RT", (m[0], m[1], tuple(m[2]), m[3])) for m in rt] + \
                       [("LR", (m[0], m[1], tuple(m[2]), m[3])) for m in lr]
        self.lr_tmin = float(self.cfg.get("lr_tmin", LR_TMIN))
        self.train_stride = max(1, int(self.cfg.get("train_stride", 1)))
        self.max_stale_s = float(self.cfg.get("max_stale_s", 10.0))   # NIRS «протух» → ЭЭГ-only
        self.fit_every = max(1, int(self.cfg.get("fit_every", 1)))
        self.banks = None
        self.frames = np.zeros(0, np.int64)
        self.pending: list[tuple[np.ndarray, np.ndarray]] = []
        self.scale = None
        self.feat_cache: dict[int, list[np.ndarray]] = {}
        self.models = None
        self.n_fit = 0
        # подучастниковый пул (прошлые сессии того же участника): <artifacts>/<subject_pool_dir>/<участник>.npz,
        # 12 предобученных бинарных sLDA; логиты смешиваются с within: z = (1−a)·z_within + a·z_pool
        self.subject_pool_dir = self.cfg.get("subject_pool_dir")
        self.a_rt = float(self.cfg.get("pool_weight_rt", 0.0))
        self.a_lr = float(self.cfg.get("pool_weight_lr", 0.0))
        self.subject = subject_of(meta)
        self.spool = None

    def load(self, artifacts_dir) -> None:
        if not artifacts_dir or not self.subject_pool_dir or not self.subject or (self.a_rt <= 0 and self.a_lr <= 0):
            return
        p = Path(artifacts_dir) / self.subject_pool_dir / f"{self.subject}.npz"
        if not p.exists():
            return
        try:
            z = np.load(p)
            spec = [[t, [m[0], m[1], list(m[2]), m[3]]] for t, m in self.members]
            if json.loads(str(z["members"])) != json.loads(json.dumps(spec)):
                return                              # другой состав каскада — пул не подходит
            self.spool = ([np.asarray(z[f"coef_{k}"], np.float64) for k in range(len(self.members))],
                          np.asarray(z["intercept"], np.float64))
        except Exception:
            self.spool = None

    # ------------------------------------------------------------------ поток
    @property
    def available(self) -> bool:
        return len(self.frames) > 0

    def _estimate_fs(self):
        """fs неизвестна из meta: оценка по числу отсчётов на отрезке Frame ≥ 30 с (внутри блока 1);
        NIRS приходит пачками, поэтому медиана шага Frame бесполезна. Притягиваем к типовой."""
        if self.fs is not None or len(self.frames) < 20:
            return
        span = self.frames[-1] - self.frames[0]
        if span < 30 * self.eeg_fs:
            return
        est = self.eeg_fs * (len(self.frames) - 1) / span
        near = min(self.KNOWN_FS, key=lambda f: abs(f - est))
        self.fs = float(near if abs(near - est) / near < 0.05 else est)

    def update(self, x: np.ndarray, frames: np.ndarray) -> None:
        """x (m, 2·n_ch) новые отсчёты [HbO|HbR], frames (m,) — 1-based индексы ЭЭГ."""
        if x is None or x.shape[0] == 0:
            return
        x = np.nan_to_num(np.asarray(x, np.float64))
        h = x.shape[1] // 2
        if h != 18:
            # монтаж каскада (пары латерализации i ↔ i+8, 16 ↔ 17) определён для 18 каналов;
            # при другом числе каналов модальность отключается → гибрид работает как ЭЭГ-only
            return
        self.pending.append((x[:, :h], x[:, h:]))
        self.frames = np.r_[self.frames, np.asarray(frames, np.int64)]
        self._estimate_fs()
        if self.fs is None:
            return
        if self.banks is None:
            self.banks = {k: _Bank(self.fs, *v, n_half=h) for k, v in BANKS.items()}
        hbo = np.concatenate([p[0] for p in self.pending], 0)
        hbr = np.concatenate([p[1] for p in self.pending], 0)
        self.pending = []
        for b in self.banks.values():
            b.push(hbo, hbr, accumulate=self.scale is None)

    # ------------------------------------------------------------------ признаки
    def _features(self, end: int):
        if self.banks is None:
            return None
        ni = int(np.searchsorted(self.frames, end, side="right")) - 1
        nmin = min(b.n for b in self.banks.values())
        ni = min(ni, nmin - 1)
        if ni < 0:
            return None
        if self.max_stale_s and end - int(self.frames[ni]) > self.max_stale_s * self.eeg_fs:
            return None          # поток NIRS оборвался: не подмешиваем устаревшие признаки
        out = []
        for _, (bk, chs, lags, mode) in self.members:
            b = self.banks[bk]
            idx = np.clip(ni - np.round(np.asarray(lags) * self.fs).astype(int), 0, None)
            S = b.sig[idx][:, ch_select(chs)].T          # (C, L)
            if mode == "diff":
                S = np.concatenate([S[:, :1], S[:, :1] - S[:, 1:]], axis=1)
            out.append(S.reshape(-1))
        return out

    def _cached_features(self, end: int):
        f = self.feat_cache.get(end)
        if f is None:
            f = self._features(end)
            if f is not None:
                self.feat_cache[end] = f
        return f

    def _scaled(self, k, X):
        bk, chs, lags, _ = self.members[k][1]
        sc = np.repeat(self.scale[bk][ch_select(chs)], len(lags))
        return X / sc

    # ------------------------------------------------------------------ модель
    @property
    def fitted(self) -> bool:
        return self.models is not None

    def fit(self, ends, y) -> None:
        if self.banks is None:
            return
        if self.scale is None:
            self.scale = {k: b.scale() for k, b in self.banks.items()}
        self.n_fit += 1
        if self.models is not None and self.n_fit % self.fit_every:
            return                                  # обучаем не каждый блок (режим экономии времени)
        keep_e, keep_y, feats = [], [], []
        for e, c in zip(np.asarray(ends, np.int64), np.asarray(y, np.int64)):
            f = self._cached_features(int(e))
            if f is not None:
                keep_e.append(int(e)); keep_y.append(int(c)); feats.append(f)
        if len(keep_y) < 8 or len(set(keep_y)) < 3:
            return
        e_arr = np.array(keep_e); y_arr = np.array(keep_y)
        pos = pos_in_segment(e_arr, y_arr, self.eeg_fs)
        st = self.train_stride
        models = []
        for k, (task, _) in enumerate(self.members):
            X = self._scaled(k, np.stack([f[k] for f in feats]))
            if task == "RT":
                models.append(_slda().fit(X[::st], (y_arr > 1)[::st]))
            else:
                sel = (y_arr > 1) & (pos >= self.lr_tmin)
                Xs, ys = X[sel][::st], (y_arr[sel] == 3)[::st]
                if len(np.unique(ys)) < 2:
                    return
                models.append(_slda().fit(Xs, ys))
        self.models = models

    def predict_proba(self, end: int):
        f = self._cached_features(int(end))
        if f is None or self.scale is None or (self.models is None and self.spool is None):
            return None
        lim = float(_logit(1 - 1e-4))
        zt, zr, pt_, pr_ = [], [], [], []
        for k, (task, _) in enumerate(self.members):
            x = self._scaled(k, f[k])
            if self.models is not None:
                z = _logit(self.models[k].predict_proba(x[None])[0, 1])
                (zt if task == "RT" else zr).append(z)
            if self.spool is not None:
                zp = float(np.clip(x @ self.spool[0][k] + self.spool[1][k], -lim, lim))
                (pt_ if task == "RT" else pr_).append(zp)
        if self.spool is None:
            a_rt = a_lr = 0.0
        elif self.models is None:
            a_rt = a_lr = 1.0
        else:
            a_rt, a_lr = self.a_rt, self.a_lr
        mt = (1 - a_rt) * (np.mean(zt) if zt else 0.0) + a_rt * (np.mean(pt_) if pt_ else 0.0)
        mr = (1 - a_lr) * (np.mean(zr) if zr else 0.0) + a_lr * (np.mean(pr_) if pr_ else 0.0)
        pt, pr = _sigmoid(mt), _sigmoid(mr)
        return np.array([1 - pt, pt * (1 - pr), pt * pr])


class _Growing:
    """Растущий 2D-буфер без квадратичных копирований."""

    def __init__(self, n_cols: int, cap: int = 8192):
        self.a = np.empty((cap, n_cols))
        self.n = 0

    def extend(self, x: np.ndarray):
        need = self.n + x.shape[0]
        if need > self.a.shape[0]:
            b = np.empty((max(need, 2 * self.a.shape[0]), self.a.shape[1]))
            b[:self.n] = self.a[:self.n]
            self.a = b
        self.a[self.n:need] = x
        self.n = need

    @property
    def view(self):
        return self.a[:self.n]


class NirsLagLDA(ModalityModel):
    """Baseline stage0: LP 0.5 Гц / HP 0.01 Гц → уровень + разности с лагами → sLDA."""

    name = "nirs"

    def __init__(self, meta: dict, cfg: dict | None = None):
        super().__init__(meta, cfg)
        self.fs = meta.get("nirs_fs")          # может быть None — оценим по Frame
        self.eeg_fs = meta["eeg_fs"]
        self.lags_s = tuple(self.cfg.get("lags_s", (1.0, 2.0, 4.0, 6.0)))
        self.shrinkage = self.cfg.get("shrinkage", "auto")
        self.pending_x: list[np.ndarray] = []   # сырые отсчёты до того, как известна fs
        self.frames = np.zeros(0, np.int64)
        self.x0 = None
        self.xl = self.xh = None
        self.lp = self.hp = None
        self.model = None

    # ------------------------------------------------------------------ поток
    @property
    def available(self) -> bool:
        return len(self.frames) > 0

    def _design(self):
        fs = self.fs
        self.lp = ss.butter(3, min(0.5, 0.4 * fs), "lowpass", fs=fs, output="sos")
        self.hp = ss.butter(1, 0.01, "highpass", fs=fs, output="sos")
        nc = self.pending_x[0].shape[1]
        self.lp_zi = np.zeros((self.lp.shape[0], 2, nc))
        self.hp_zi = np.zeros((self.hp.shape[0], 2, nc))
        self.xl, self.xh = _Growing(nc), _Growing(nc)

    KNOWN_FS = (3.90625, 7.8125, 10.0, 12.5, 15.625, 25.0)

    def _estimate_fs(self):
        if self.fs is not None or len(self.frames) < 20:
            return
        span = self.frames[-1] - self.frames[0]
        if span < 30 * self.eeg_fs:
            return
        est = self.eeg_fs * (len(self.frames) - 1) / span
        near = min(self.KNOWN_FS, key=lambda f: abs(f - est))
        self.fs = float(near if abs(near - est) / near < 0.05 else est)

    def update(self, x: np.ndarray, frames: np.ndarray) -> None:
        """x (m, 2·n_ch) новые отсчёты [HbO|HbR], frames (m,) — 1-based индексы ЭЭГ."""
        if x is None or x.shape[0] == 0:
            return
        x = np.nan_to_num(np.asarray(x, np.float64))
        if self.x0 is None:
            self.x0 = x[:1].copy()
        self.frames = np.r_[self.frames, np.asarray(frames, np.int64)]
        self.pending_x.append(x - self.x0)
        self._estimate_fs()
        if self.fs is None:
            return
        if self.lp is None:
            self._design()
        xx = np.concatenate(self.pending_x, axis=0)
        self.pending_x = []
        xl, self.lp_zi = ss.sosfilt(self.lp, xx, axis=0, zi=self.lp_zi)
        xh, self.hp_zi = ss.sosfilt(self.hp, xl, axis=0, zi=self.hp_zi)
        self.xl.extend(xl)
        self.xh.extend(xh)

    def features_at(self, end: int) -> np.ndarray | None:
        if self.xl is None or self.xl.n == 0:
            return None
        n = self.xl.n
        i = int(np.searchsorted(self.frames[:n], end, side="right")) - 1
        if i < 0:
            return None
        xl, xh = self.xl.view, self.xh.view
        f = [xh[i]]
        for lag in self.lags_s:
            k = max(1, int(round(lag * self.fs)))
            f.append(xl[i] - xl[max(i - k, 0)])
        return np.concatenate(f)

    # ------------------------------------------------------------------ модель
    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, ends, y) -> None:
        X, Y = [], []
        for e, c in zip(ends, y):
            f = self.features_at(int(e))
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
