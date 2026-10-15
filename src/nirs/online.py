"""Онлайн-реализация каскадного NIRS-декодера (`src/nirs/cascade.py`).

Состояние:
  * по каждому из 4 банков фильтров — состояние `sosfilt` (zi) для HbO и HbR, история
    производных каналов (58 × m, float32; для лагов нужны только последние 12 с, храним всё —
    ~7.5k × 58 × 4 × 4 Б ≈ 7 МБ на сессию);
  * накопители сумм/квадратов производных каналов с 20-й секунды до первого `fit_block`
    (→ масштаб каждого канала = std по блоку 1, затем замораживается);
  * буфер признаков всех окон, для которых вызывался `predict` (сырые, до масштабирования) +
    их концы в отсчётах ЭЭГ;
  * 8 + 4 обученных sLDA (коэффициенты).

push(hbo, hbr, frames) — новые отсчёты NIRS (frames — 1-based индексы ЭЭГ, как NIRS.Frame).
predict(end)            — признаки окна с концом `end` (exclusive, отсчёты ЭЭГ) по отсчётам
                          NIRS с frame ≤ end; возвращает P(1,2,3) или None, пока нет модели.
fit_block(states, offset) — метки завершённого блока: States на частоте ЭЭГ, `offset` — индекс
                          первого отсчёта блока. Размечает сохранённые окна (однородные, State>0),
                          переобучает 12 sLDA на всех размеченных окнах.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from src.nirs.cascade import (BANKS, LR_MEMBERS, LR_TMIN, RT_MEMBERS, combine, logit,
                              pos_in_segment, sigmoid, slda)
from src.nirs.evaluate import ch_select
from src.nirs.features import bandpass_sos, derived_channels

WIN = 250


class _Bank:
    def __init__(self, fs, lo, hi, order):
        self.fs = fs
        self.sos = bandpass_sos(fs, lo, hi, order)
        self.zo = self.zr = None
        self.sig = np.zeros((0, 58), np.float32)
        self.n = 0
        self.s1 = np.zeros(58)
        self.s2 = np.zeros(58)
        self.cnt = 0

    def push(self, hbo, hbr, accumulate):
        if self.zo is None:
            z = ss.sosfilt_zi(self.sos)[:, :, None]
            self.zo, self.zr = z * hbo[0][None, None, :], z * hbr[0][None, None, :]
        fo, self.zo = ss.sosfilt(self.sos, hbo, axis=0, zi=self.zo)
        fr, self.zr = ss.sosfilt(self.sos, hbr, axis=0, zi=self.zr)
        d, _ = derived_channels(fo, fr)
        if self.n + len(d) > len(self.sig):
            new = np.zeros((max(2 * len(self.sig), self.n + len(d), 1024), 58), np.float32)
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


class OnlineNirsDecoder:
    def __init__(self, nirs_fs: float):
        self.fs = nirs_fs
        self.banks = {k: _Bank(nirs_fs, *v) for k, v in BANKS.items()}
        self.frames = np.zeros(0, np.int64)
        self.scale = None                      # {bank: (58,)} после первого fit_block
        self.members = [("RT", m) for m in RT_MEMBERS] + [("LR", m) for m in LR_MEMBERS]
        self.ends, self.feats = [], []         # сохранённые окна (сырые признаки по членам)
        self.y = np.zeros(0, int)              # метки сохранённых окон (0 — нет)
        self.models = None

    # ---- поток данных ----
    def push(self, hbo, hbr, frames):
        if len(frames) == 0:
            return
        hbo = np.asarray(hbo, np.float64); hbr = np.asarray(hbr, np.float64)
        for b in self.banks.values():
            b.push(hbo, hbr, accumulate=self.scale is None)
        self.frames = np.r_[self.frames, np.asarray(frames, np.int64)]

    def _features(self, end):
        ni = int(np.searchsorted(self.frames, end, side="right")) - 1
        if ni < 0:
            return None
        out = []
        for _, (bk, chs, lags, mode) in self.members:
            b = self.banks[bk]
            idx = np.clip(ni - np.round(np.asarray(lags) * self.fs).astype(int), 0, None)
            S = b.sig[idx][:, ch_select(chs)].T          # (C, L)
            if mode == "diff":
                S = np.concatenate([S[:, :1], S[:, :1] - S[:, 1:]], axis=1)
            out.append(S.reshape(-1))
        return out

    def _scaled(self, k, X):
        bk, chs, lags, _ = self.members[k][1]
        sc = np.repeat(self.scale[bk][ch_select(chs)], len(lags))
        return X / sc

    def predict(self, end: int):
        f = self._features(end)
        if f is None:
            return None
        self.ends.append(end); self.feats.append(f); self.y = np.r_[self.y, 0]
        if self.models is None:
            return None
        zt, zr = [], []
        for k, m in enumerate(self.models):
            z = logit(m.predict_proba(self._scaled(k, f[k])[None])[0, 1])
            (zt if self.members[k][0] == "RT" else zr).append(z)
        return combine(sigmoid(np.mean(zt))[None], sigmoid(np.mean(zr))[None])[0]

    # ---- дообучение ----
    def fit_block(self, states: np.ndarray, offset: int):
        if self.scale is None:
            self.scale = {k: b.scale() for k, b in self.banks.items()}
        ends = np.asarray(self.ends)
        st = np.asarray(states).astype(int)
        rel = ends - offset
        ok = (rel >= WIN) & (rel <= len(st))
        seg = np.r_[0, np.cumsum(np.diff(st) != 0)]
        r = rel[ok]
        good = (seg[r - WIN] == seg[r - 1]) & (st[r - 1] > 0)
        lab = np.zeros(ok.sum(), int)
        lab[good] = st[r - 1][good]
        self.y[np.flatnonzero(ok)] = lab
        L = self.y > 0
        y = self.y[L]
        pos = pos_in_segment(ends[L], y)
        self.models = []
        for k, (task, _) in enumerate(self.members):
            X = self._scaled(k, np.stack([f[k] for f, l in zip(self.feats, L) if l]))
            if task == "RT":
                self.models.append(slda().fit(X, y > 1))
            else:
                sel = (y > 1) & (pos >= LR_TMIN)
                self.models.append(slda().fit(X[sel], y[sel] == 3))
