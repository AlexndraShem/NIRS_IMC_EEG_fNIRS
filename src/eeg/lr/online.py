"""Потоковый ЭЭГ-декодер `eeg_lr_nbsm4`: baseline eeg_ts2s_all (TS 2 с × within-sLDA × пул-LR,
вес пула по блокам 0.9 → 0.7) × два эксперта:

  * «nb»  — логдисперсии 23 пространственных каналов (12 биполяр + 11 лапласианов, истинные имена
    столбцов) в 13 полосах (4-8, 2-Гц 8…30, 8-30), окно 2 с, 299 признаков;
  * «sm4» — касательное пространство 11 сенсомоторных каналов, 7 полос, окно 4 с, 462 признака.

У каждого эксперта within-sLDA(0.8) обучается на окнах прошлых блоков + их зеркальных копиях
(полушария меняются местами, метки L↔R), пул-LR — на других участниках; смесь
P ∝ P_w^(1-a)·P_p^a с a от 0.8 (блок 2) до 0.5 (блок 7). Итог: P ∝ P_base · P_nb · P_sm4.

Обучающих окон не больше max_train=400 (шаг прореживания растёт с числом блоков: train_stride =
max(2, ceil(n/400)), ref_stride = 2·train_stride) — качество то же, fit_block ограничен по времени.

Всё причинно: sosfilt с сохранением zi, окна [end-L, end), статистики только по прошлым блокам.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from NIRS_IMC_EEG_fNIRS.src.eeg.cache import BANDS, band_sos
from NIRS_IMC_EEG_fNIRS.src.eeg.common import RefState
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import (TI, feature_mirror, mirror_signed_perm, spatial_matrix, swap_lr)
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.feats import SM, nb_logvar
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.nbcache import NB_BANDS
from NIRS_IMC_EEG_fNIRS.src.eeg.online import SLDA, PoolLR, fit_within
from NIRS_IMC_EEG_fNIRS.src.features.extract import eeg_prefilter_sos

FLOOR = np.log(1e-9)


def _mix(lw, lp, a):
    """Смесь лог-вероятностей (1,3) с полом; lp=None → только within; lw=None → только пул."""
    z = np.zeros((1, 3)); tot = 0.0
    if lw is not None:
        z += (1 - a) * np.maximum(lw, FLOOR); tot += 1 - a
    if lp is not None:
        z += a * np.maximum(lp, FLOOR); tot += a
    if tot == 0:
        return np.full((1, 3), -np.log(3))
    z = z / tot if (lw is None or lp is None) else z
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def sched(block, a2, a7):
    b = min(max(block, 2), 7)
    return a2 + (a7 - a2) * (b - 2) / 5


class Expert:
    """within-sLDA (с зеркальной аугментацией) + пул-LR на готовых признаках."""

    def __init__(self, pool: PoolLR | None, mirror, a2=0.8, a7=0.5, shrinkage=0.8):
        self.pool, self.mirror, self.a2, self.a7, self.shrinkage = pool, mirror, a2, a7, shrinkage
        self.within = None

    def fit(self, X, y):
        Xa = np.vstack([X, X[:, self.mirror[0]] * self.mirror[1]])
        ya = np.concatenate([y, swap_lr(y)])
        self.within = SLDA(self.shrinkage).fit(Xa, ya) if len(np.unique(y)) >= 3 else None

    def log_proba(self, f, block):
        lw = self.within.predict_log_proba(f) if self.within is not None else None
        lp = self.pool.predict_log_proba(f) if self.pool is not None else None
        return _mix(lw, lp, sched(block, self.a2, self.a7))


class OnlineEEGLR:
    def __init__(self, pool_base: PoolLR | None, pool_nb: PoolLR | None, pool_sm4: PoolLR | None,
                 fs=250.0, max_train=400, base_a=(0.9, 0.7)):
        self.fs = fs
        self.L2, self.L4 = int(round(2 * fs)), int(round(4 * fs))
        self.max_train = max_train          # не больше max_train обучающих окон (время fit_block)
        self.pool_base, self.base_a = pool_base, base_a
        self.pre = eeg_prefilter_sos(fs)
        self.bands = band_sos(fs, BANDS)
        self.nb_sos = [ss.butter(4, b, "bandpass", fs=fs, output="sos") for b in NB_BANDS]
        self.W = np.vstack([spatial_matrix("bip"), spatial_matrix("lap")])     # (23, 21)
        self.sm = np.array([TI[c] for c in SM])
        C, K = 21, self.W.shape[0]
        self.zi_pre = np.zeros((self.pre.shape[0], 2, C))
        self.zi_b = [np.zeros((s.shape[0], 2, C)) for s in self.bands]
        self.zi_nb = [np.zeros((s.shape[0], 2, K)) for s in self.nb_sos]
        self.buf = np.zeros((len(self.bands), self.L4, C))          # 7 полос × 4 с × 21
        self.nbuf = np.zeros((len(self.nb_sos), self.L2, K))        # 13 полос × 2 с × 23
        self.n, self.x0 = 0, None
        self.step, self.next_grid = fs / 4, fs
        self.hist_end, self.h_c2, self.h_c4, self.h_nb = [], [], [], []
        self.states = np.zeros(0, np.uint8)
        self.n_fit = 0
        self.ref = self.within = self.ref_sm = None
        # зеркала признаков экспертов
        pm, sg = mirror_signed_perm(self.W)
        mir_nb = feature_mirror(pm, sg, "logvar", len(self.nb_sos))
        pos = {c: i for i, c in enumerate(SM)}
        from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import MIRROR
        pm_sm = np.array([pos[MIRROR[c]] for c in SM])
        mir_sm = feature_mirror(pm_sm, np.ones(len(SM)), "ts", len(self.bands))
        self.nb = Expert(pool_nb, mir_nb)
        self.sm4 = Expert(pool_sm4, mir_sm)
        self.nb_ref = None

    # ---------------------------------------------------------------- поток
    @staticmethod
    def _roll(buf, xb):
        k = xb.shape[1]
        if k >= buf.shape[1]:
            return xb[:, -buf.shape[1]:].copy()
        return np.concatenate([buf[:, k:], xb], axis=1)

    def _process(self, x):
        x, self.zi_pre = ss.sosfilt(self.pre, x, axis=0, zi=self.zi_pre)
        xb = []
        for i, sos in enumerate(self.bands):
            y, self.zi_b[i] = ss.sosfilt(sos, x, axis=0, zi=self.zi_b[i]); xb.append(y)
        xs = x @ self.W.T
        xn = []
        for i, sos in enumerate(self.nb_sos):
            y, self.zi_nb[i] = ss.sosfilt(sos, xs, axis=0, zi=self.zi_nb[i]); xn.append(y)
        self.buf = self._roll(self.buf, np.stack(xb))
        self.nbuf = self._roll(self.nbuf, np.stack(xn))
        self.n += len(x)

    def _feats_now(self):
        b2 = self.buf[:, -self.L2:]
        c2 = np.einsum("btc,btd->bcd", b2, b2) / self.L2
        b4 = self.buf[:, :, self.sm]
        c4 = np.einsum("btc,btd->bcd", b4, b4) / self.L4
        v = np.mean(self.nbuf ** 2, axis=1)                          # (13, 23)
        return c2, c4, v

    def push(self, eeg_chunk):
        x = np.asarray(eeg_chunk, np.float64)
        if self.x0 is None:
            self.x0 = x[0].copy()
        x = x - self.x0
        pos = 0
        while pos < len(x):
            g = int(round(self.next_grid))
            take = min(len(x) - pos, g - self.n)
            self._process(x[pos:pos + take])
            pos += take
            if self.n == g:
                if self.n >= self.L2:
                    c2, c4, v = self._feats_now()
                    self.hist_end.append(self.n); self.h_c2.append(c2); self.h_c4.append(c4); self.h_nb.append(v)
                self.next_grid += self.step

    # ---------------------------------------------------------------- модель
    def predict_proba(self):
        if self.ref is None:
            return np.full(3, 1 / 3)
        block = self.n_fit + 1
        c2, c4, v = self._feats_now()
        f = self.ref.ts(c2[None])
        lw = self.within.predict_log_proba(f) if self.within is not None else None
        lp = self.pool_base.predict_log_proba(f) if self.pool_base is not None else None
        z = _mix(lw, lp, sched(block, *self.base_a))
        z = z + self.nb.log_proba(nb_logvar(self.nb_ref, v[None]), block)
        z = z + self.sm4.log_proba(self.ref_sm.ts(c4[None]), block)
        p = np.exp(z[0] - z[0].max())
        return p / p.sum()

    def fit_block(self, block_idx, states_block):
        self.states = np.concatenate([self.states, np.asarray(states_block, np.uint8)])
        self.n_fit = int(block_idx)
        ends = np.array(self.hist_end); n_lab = len(self.states)
        ok = ends <= n_lab
        e = ends[ok]
        st = self.states.astype(int)
        seg = np.r_[0, np.cumsum(np.diff(st) != 0)]
        Wl = int(round(self.fs))
        good = (seg[e - Wl] == seg[e - 1]) & (st[e - 1] > 0)
        sel = np.flatnonzero(ok)[good]
        y = st[e[good] - 1]
        ts_ = max(2, int(np.ceil(len(sel) / self.max_train))) if self.max_train else 2
        self.rs, self.ts_ = 2 * ts_, ts_
        c2 = np.array([self.h_c2[i] for i in sel])
        self.ref, self.within = fit_within(c2, y, 0.8, self.rs, self.ts_)
        # эксперт sm4
        c4 = np.array([self.h_c4[i] for i in sel])
        self.ref_sm = RefState().fit(c4[::self.rs])
        self.sm4.fit(self.ref_sm.ts(c4[::self.ts_]), y[::self.ts_])
        # эксперт nb
        v = np.array([self.h_nb[i] for i in sel])
        self.nb_ref = v[::self.rs]
        self.nb.fit(nb_logvar(self.nb_ref, v[::self.ts_]), y[::self.ts_])
