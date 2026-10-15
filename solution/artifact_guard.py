"""Причинный детектор широкополосных ВЧ-артефактов ЭЭГ (ФЭС / ЭМГ) и «охрана» ЭЭГ-модели.

Аудит (docs/experiments/artifacts_audit.md) показал: во время воображения в части сессий идут
эпизоды широкополосной ВЧ-мощности (+1.3…2 порядка в 30–125 Гц, 2–10 с, отсутствуют в блоке 1 —
совместимо с ФЭС, включавшейся по решению исходного классификатора; плюс ЭМГ), которые
«протекают» в бета-диапазон признаков (18–30 Гц: +0.8…1.5 порядка). Охрана делает так, чтобы
решение на таких окнах не опиралось на загрязнённый сигнал:

* детектор (всё причинно, фильтры sosfilt с сохраняемым zi):
    x − x[0] → полоса 31–46 Гц (Баттерворт 4) + режекторы 37.5 Гц (3-я гармоника кадровой частоты
    NIRS 12.5 Гц) и 50 Гц → лог-энергия по каналам на каждом шаге сетки окон (62 отсчёта);
    базовый уровень канала — `q`-квантиль (0.2) его лог-энергий за последние `hist_s` с (только
    прошлое; до `min_hist_s` с детектор молчит); индекс = медиана по каналам (лог-энергия − база);
    окно «загрязнено», если максимум индекса за последние `lookback_s` с > `thr` (log10).
* predict_proba(end): на загрязнённом окне вероятности ЭЭГ заменяются вероятностями внутренней
  модели для ПОСЛЕДНЕЙ ЧИСТОЙ точки сетки e0 < end (удержание, `mode="hold"`), стягиваемыми к
  равномерным в лог-пространстве с τ = `hold_tau_s` по времени (end − e0); `mode="uniform"` —
  сразу равномерные; `mode="none"` — только исключение из обучения.
  Независимость от расписания вызовов: e0 определяется сигналом (флаги ставятся в update на
  КАЖДОЙ точке сетки, e0 — последняя незагрязнённая), а вероятности для e0 считаются по снимку
  признаков на e0 текущей моделью. Поэтому ответ зависит только от сигнала и меток завершённых
  блоков, а не от того, когда и как часто вызывается predict (совместимо с асинхронным _tick).
* fit_block: загрязнённые окна исключаются из обучения внутри сессии (`exclude_train`).

Метки States детектор не использует. Прочие атрибуты (spool, …) проксируются во внутреннюю модель.
"""
from __future__ import annotations

import importlib
from collections import deque

import numpy as np
import scipy.signal as ss

from NIRS_IMC_EEG_fNIRS.solution.base import ModalityModel


def _load_class(spec):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


class HFBurstDetector:
    def __init__(self, fs, n_ch, band=(31.0, 46.0), notches=(37.5, 50.0), q=0.2, hist_s=300.0,
                 min_hist_s=10.0, step=62, base_every=4):
        self.fs = float(fs)
        sos = [ss.butter(4, [band[0], min(band[1], 0.45 * fs)], "bandpass", fs=fs, output="sos")]
        for f0 in notches:
            if f0 < fs / 2:
                sos.append(ss.tf2sos(*ss.iirnotch(f0, 8.0, fs=fs)))
        self.sos = np.vstack(sos)
        self.zi = np.zeros((self.sos.shape[0], 2, n_ch))
        self.x0 = None
        self.q = float(q)
        self.step = int(step)
        self.hist = deque(maxlen=max(1, int(round(hist_s * fs / step))))
        self.min_hist = int(round(min_hist_s * fs / step))
        self.base_every = int(base_every)
        self.base = None
        self.k = 0
        self.tail = np.zeros((0, n_ch))

    def process(self, x):
        """x (n, C) сырые → отфильтрованные ВЧ (n, C); хвост хранится для энергии шага."""
        if self.x0 is None:
            self.x0 = x[:1].copy()
        h, self.zi = ss.sosfilt(self.sos, x - self.x0, axis=0, zi=self.zi)
        self.tail = np.concatenate([self.tail, h])[-self.step:]
        return h

    def score(self):
        """Индекс для текущего конца потока (вызывать на точках сетки)."""
        if len(self.tail) == 0:
            return 0.0
        e = np.log10(np.mean(self.tail ** 2, axis=0) + 1e-30)
        z = 0.0
        if len(self.hist) >= self.min_hist:
            if self.base is None or self.k % self.base_every == 0:
                self.base = np.quantile(np.asarray(self.hist), self.q, axis=0)
            z = float(np.median(e - self.base))
        self.hist.append(e)
        self.k += 1
        return z


class EEGArtifactGuard(ModalityModel):
    name = "eeg"

    def __init__(self, meta: dict, cfg: dict | None = None):
        super().__init__(meta, cfg)
        c = dict(self.cfg)
        g = dict(c.pop("guard", {}) or {})
        inner_spec = c.pop("guard_inner_model", "eeg_dl:EEGWithDL")
        self.inner = _load_class(inner_spec)(meta, c)
        self.fs = float(meta["eeg_fs"])
        self.thr = float(g.get("thr", 0.7))
        self.mode = str(g.get("mode", "hold"))
        self.hold_tau = float(g.get("hold_tau_s", 4.0))
        self.lookback = max(1, int(round(float(g.get("lookback_s", 1.0)) * self.fs / 62)))
        self.exclude_train = bool(g.get("exclude_train", True))
        self.det_kw = dict(band=tuple(g.get("band", (31.0, 46.0))), notches=tuple(g.get("notches", (37.5, 50.0))),
                           q=float(g.get("q", 0.2)), hist_s=float(g.get("hist_s", 300.0)),
                           min_hist_s=float(g.get("min_hist_s", 10.0)))
        self.enabled = bool(g.get("enabled", True)) and self.fs >= 100
        self.det = None
        self.n = 0
        self.zs = deque(maxlen=self.lookback)
        self.flag: dict[int, bool] = {}
        self.clean_before: dict[int, int | None] = {}   # точка сетки → последняя чистая точка ≤ неё
        self._last_clean_end = None
        self._n_fit = 0
        self._hold_cache: dict[tuple, np.ndarray] = {}
        self.n_flagged = 0

    def __getattr__(self, k):                        # spool, pool, ... — от внутренней модели
        if k in ("inner",):
            raise AttributeError(k)
        return getattr(self.inner, k)

    def load(self, artifacts_dir) -> None:
        self.inner.load(artifacts_dir)

    @property
    def available(self) -> bool:
        return self.inner.available

    @property
    def fitted(self) -> bool:
        return self.inner.fitted

    # ------------------------------------------------------------------ поток
    def _feed(self, x):
        self.det.process(x)
        self.n += x.shape[0]

    def update(self, x, snapshot_ends=()) -> None:
        self.inner.update(x, snapshot_ends)
        if not self.enabled or x.shape[0] == 0:
            return
        x = np.asarray(x, np.float64)
        if self.det is None:
            self.det = HFBurstDetector(self.fs, x.shape[1], **self.det_kw)
        n0 = self.n
        pos = 0
        for e in sorted(int(v) for v in snapshot_ends if n0 < int(v) <= n0 + x.shape[0]):
            take = e - n0 - pos
            if take > 0:
                self._feed(x[pos:pos + take])
                pos += take
            self.zs.append(self.det.score())
            f = max(self.zs) > self.thr
            self.flag[e] = f
            if not f:
                self._last_clean_end = e
            self.clean_before[e] = self._last_clean_end
        if pos < x.shape[0]:
            self._feed(x[pos:])

    def is_flagged(self, end: int) -> bool:
        return bool(self.flag.get(int(end), False))

    # ------------------------------------------------------------------ модель
    def fit(self, ends, y) -> None:
        ends = np.asarray(ends, np.int64)
        y = np.asarray(y, np.int64)
        if self.enabled and self.exclude_train and len(ends):
            keep = np.array([not self.is_flagged(e) for e in ends])
            # защита: если чистых окон какого-то класса почти нет — учим на всех
            if all(np.sum(keep & (y == k)) >= 8 for k in np.unique(y)):
                ends, y = ends[keep], y[keep]
        self.inner.fit(ends, y)
        self._n_fit += 1
        self._hold_cache.clear()

    def predict_proba(self, end: int):
        e = int(end)
        if not self.enabled or self.mode == "none" or not self.is_flagged(e):
            return self.inner.predict_proba(e)
        self.n_flagged += 1
        e0 = self.clean_before.get(e)
        if self.mode == "uniform" or e0 is None:
            return np.full(3, 1 / 3) if self.inner.fitted else self.inner.predict_proba(e)
        key = (e0, self._n_fit)
        p0 = self._hold_cache.get(key)
        if p0 is None:
            p0 = self.inner.predict_proba(e0)
            if p0 is None:
                return np.full(3, 1 / 3) if self.inner.fitted else None
            p0 = np.asarray(p0, np.float64)
            if len(self._hold_cache) > 64:
                self._hold_cache.clear()
            self._hold_cache[key] = p0
        w = np.exp(-(e - e0) / self.fs / self.hold_tau) if self.hold_tau > 0 else 1.0
        lp = w * np.log(np.clip(p0, 1e-9, None))
        q = np.exp(lp - lp.max())
        return q / q.sum()
