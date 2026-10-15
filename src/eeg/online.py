"""Потоковый (онлайн) ЭЭГ-декодер: рецентрированное касательное пространство фильтр-банка,
within-session sLDA (сильный shrinkage) + предобученная на других участниках пул-LR,
геометрическая смесь вероятностей.

Состояние:
  * zi префильтра (HP 1 Гц + notch 50 Гц) и 7 полосовых фильтров (sosfilt, причинно);
  * кольцевой буфер отфильтрованного сигнала последних `win` с по полосам (B, L, C);
  * история ковариаций окон на сетке 250 мс: (end, cov) — для fit_block;
  * RefState: масштабы каналов + LE-среднее по полосам, оценены по всем размеченным окнам
    завершённых блоков (обновляется в fit_block);
  * within-sLDA (coef, intercept) и пул-LR (mean/scale/coef/intercept — загружаются из артефакта).

predict(): ковариации текущего буфера → TS с текущим референсом → P_within, P_pool →
  P ∝ P_within^(1-a) · P_pool^a. До первого fit_block — только пул (с референсом по
  накопленным окнам сессии) либо равномерное распределение.
fit_block(block, states): states — метки отсчётов завершённого блока (как в онлайн-каркасе);
  окна, целиком лежащие в однородном сегменте States∈{1,2,3}, получают метку → пересчёт
  референса и TS всех размеченных окон → обучение sLDA.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from NIRS_IMC_EEG_fNIRS.src.eeg.cache import BANDS, band_sos
from NIRS_IMC_EEG_fNIRS.src.eeg.common import NO_FRONTAL, RefState, ch_idx
from NIRS_IMC_EEG_fNIRS.src.features.extract import eeg_prefilter_sos


class PoolLR:
    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept

    @classmethod
    def from_sklearn(cls, sc, lr):
        return cls(sc.mean_, sc.scale_, lr.coef_, lr.intercept_)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mean"], z["scale"], z["coef"], z["intercept"])

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept)

    def predict_log_proba(self, X):
        z = ((X - self.mean) / self.scale) @ self.coef.T + self.intercept
        return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


class SLDA:
    """LDA с фиксированным shrinkage; эквивалент sklearn LinearDiscriminantAnalysis(solver="lsqr",
    shrinkage=s): ковариация каждого класса сжимается к своей диагонали (shrinkage в пространстве
    корреляций), общая ковариация — взвешенная приорами. Реализовано на numpy (быстрее sklearn)."""

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
        out[:, self.classes - 1] = z
        return out - np.logaddexp.reduce(out, axis=1, keepdims=True)


def fit_within(covs, y, shrinkage=0.8, ref_stride=4, train_stride=2):
    """Общая для онлайна и оценки процедура fit_block: референс по каждому ref_stride-му окну
    (окна перекрываются на 87.5 %, прореживание почти не теряет информации), TS и sLDA по каждому
    train_stride-му. Возвращает (RefState, SLDA|None)."""
    ref = RefState().fit(covs[::ref_stride])
    if len(np.unique(y)) < 3:
        return ref, None
    X = ref.ts(covs[::train_stride])
    return ref, SLDA(shrinkage).fit(X, y[::train_stride])


class OnlineEEG:
    def __init__(self, pool: PoolLR | None = None, fs=250.0, win_s=2.0, chans=NO_FRONTAL,
                 shrinkage=0.8, alpha=0.8, ref_stride=4, train_stride=2):
        self.fs, self.L, self.ci = fs, int(round(win_s * fs)), ch_idx(chans)
        self.pool, self.shrinkage, self.alpha = pool, shrinkage, alpha
        self.ref_stride, self.train_stride = ref_stride, train_stride
        self.pre = eeg_prefilter_sos(fs)
        self.bands = band_sos(fs, BANDS)
        C = len(self.ci)
        self.zi_pre = np.zeros((self.pre.shape[0], 2, C))
        self.zi_b = [np.zeros((s.shape[0], 2, C)) for s in self.bands]
        self.buf = np.zeros((len(self.bands), self.L, C))
        self.n = 0                      # сколько отсчётов принято
        self.x0 = None                  # первый отсчёт (вычитание DC, причинно)
        self.step = fs / 4
        self.next_grid = self.fs        # сетка 250 мс, первая точка — 1 с
        self.hist_end, self.hist_cov = [], []
        self.ref, self.within = None, None
        self.states = np.zeros(0, np.uint8)

    # ---------- поток ----------
    def _filter(self, x):
        x, self.zi_pre = ss.sosfilt(self.pre, x, axis=0, zi=self.zi_pre)
        out = []
        for i, sos in enumerate(self.bands):
            y, self.zi_b[i] = ss.sosfilt(sos, x, axis=0, zi=self.zi_b[i])
            out.append(y)
        return np.stack(out)            # (B, n, C)

    def _append(self, xb):
        k = xb.shape[1]
        if k >= self.L:
            self.buf = xb[:, -self.L:].copy()
        else:
            self.buf = np.concatenate([self.buf[:, k:], xb], axis=1)
        self.n += k

    def current_cov(self):
        return np.einsum("btc,btd->bcd", self.buf, self.buf) / self.L

    def push(self, eeg_chunk):
        """eeg_chunk (k, 21) сырые отсчёты. Внутри режется по точкам сетки 250 мс,
        в каждой точке сохраняется ковариация (для последующего обучения)."""
        x = np.asarray(eeg_chunk, np.float64)[:, self.ci]
        if self.x0 is None:
            self.x0 = x[0].copy()
        x = x - self.x0
        pos = 0
        while pos < len(x):
            g = int(round(self.next_grid))
            take = min(len(x) - pos, g - self.n)
            self._append(self._filter(x[pos:pos + take]))
            pos += take
            if self.n == g:
                if self.n >= self.L:
                    self.hist_end.append(self.n); self.hist_cov.append(self.current_cov())
                self.next_grid += self.step

    # ---------- модель ----------
    def predict_proba(self):
        c = self.current_cov()[None]
        if self.ref is None:
            return np.full(3, 1 / 3)
        f = self.ref.ts(c)
        lp = np.zeros((1, 3))
        a = self.alpha if self.within is not None else 1.0
        floor = np.log(1e-9)            # клиппинг как в оффлайн-оценке (mix): ограничивает «вето» одной модели
        if self.pool is not None:
            lp += a * np.maximum(self.pool.predict_log_proba(f), floor)
        if self.within is not None:
            lp += (1 - a) * np.maximum(self.within.predict_log_proba(f), floor)
        p = np.exp(lp[0] - lp[0].max())
        return p / p.sum()

    def fit_block(self, block_idx, states_block):
        """states_block — States отсчётов только что завершённого блока (в порядке поступления,
        последний отсчёт блока = последний принятый). Метки прошлых блоков накапливаются."""
        self.states = np.concatenate([self.states, np.asarray(states_block, np.uint8)])
        ends = np.array(self.hist_end); n_lab = len(self.states)
        ok = ends <= n_lab
        e = ends[ok]
        st = self.states.astype(int)
        seg = np.r_[0, np.cumsum(np.diff(st) != 0)]
        W = int(round(self.fs))
        good = (seg[e - W] == seg[e - 1]) & (st[e - 1] > 0)
        covs = np.array(self.hist_cov)[ok][good]
        y = st[e[good] - 1]
        self.ref, self.within = fit_within(covs, y, self.shrinkage, self.ref_stride, self.train_stride)
