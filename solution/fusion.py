"""Объединение модальностей и формирование ответов ТЗ (`y`, `y_eeg`, `y_nirs`).

* `LogLinearFusion` — **боевое** слияние: взвешенное среднее лог-вероятностей + сдвиг лог-приора
  активных классов + причинное экспоненциальное сглаживание (EMA) по прошлым окнам.
* `MeanProbaFusion` — baseline stage0 (простое среднее вероятностей), оставлен для отката.

Интерфейс, который использует classifier.py:

    fusion.decide({"eeg": p_eeg | None, "nirs": p_nirs | None}, end, fs) -> {"y", "y_eeg", "y_nirs"}

`end` — конец окна в отсчётах ЭЭГ, `fs` — частота ЭЭГ: по ним считается реальное время между
соседними решениями (окна идут с шагом 250 мс, но на границах блоков/сегментов бывают разрывы).
"""
from __future__ import annotations

import numpy as np

FLOOR = 1e-9


def _norm_log(p, w=1.0):
    return w * np.log(np.clip(np.asarray(p, np.float64), FLOOR, 1.0))


def _softmax(lp):
    e = np.exp(lp - lp.max())
    s = e.sum()
    return e / s if s > 0 else np.full(len(lp), 1.0 / len(lp))


class _Ema:
    """Причинное EMA с весом по реальному времени: a = exp(−Δt/τ), s ← a·s + (1−a)·p.

    Сброса по сегментам/блокам нет (разрывы гасятся самим весом: при Δt ≫ τ a → 0).
    """

    def __init__(self, tau_s: float):
        self.tau = float(tau_s)
        self.s = None
        self.t = None

    def __call__(self, p, t):
        if p is None:
            return None
        if self.s is None or self.t is None or self.tau <= 0:
            self.s, self.t = np.asarray(p, np.float64), t
            return self.s
        a = float(np.exp(-(t - self.t) / self.tau))
        self.s = a * self.s + (1 - a) * np.asarray(p, np.float64)
        self.t = t
        return self.s


class LogLinearFusion:
    """log P ∝ w·log P_eeg + (1−w)·log P_nirs + prior, затем EMA(τ).

    Параметры (config.json → "fusion_params"):
      eeg_weight  — вес ЭЭГ в лог-смеси (0.75); при отсутствии NIRS берётся чистый log P_eeg;
      prior       — сдвиг лог-приора по классам [покой, левая, правая] для гибрида и y_eeg;
      tau_s       — постоянная времени EMA (с) для гибрида и y_eeg; 0 — без сглаживания;
      nirs_prior  — сдвиг лог-приора для отдельного выхода y_nirs;
      nirs_tau_s  — EMA для y_nirs (по умолчанию 0 — без сглаживания);
      hier_tau_s  — [τ_покой, τ_LR]: иерархическое сглаживание гибрида вместо EMA(tau_s):
                    P(покой) — EMA τ_покой, доля L/(L+R) — EMA τ_LR (docs/experiments/eeg_lr_features.md);
                    None — обычное EMA(tau_s).
    """

    def __init__(self, params: dict | None = None):
        p = dict(params or {})
        self.w = float(p.get("eeg_weight", 0.75))
        self.prior = np.asarray(p.get("prior", [0.0, 2.0, 2.0]), np.float64)
        self.tau = float(p.get("tau_s", 1.5))
        self.nirs_prior = np.asarray(p.get("nirs_prior", [0.0, 1.0, 1.0]), np.float64)
        self.nirs_tau = float(p.get("nirs_tau_s", 0.0))
        self.ema_y = _Ema(self.tau)
        self.ema_eeg = _Ema(self.tau)
        self.ema_nirs = _Ema(self.nirs_tau)
        h = p.get("hier_tau_s")
        self.hier = None if not h else (_Ema(float(h[0])), _Ema(float(h[1])))

    # ---------------------------------------------------------------- вероятности
    def fuse(self, probas: dict):
        """Гибридные вероятности до сглаживания (нужны и для паритета с оффлайн-оценкой)."""
        pe, pn = probas.get("eeg"), probas.get("nirs")
        if pe is not None and not np.all(np.isfinite(pe)):
            pe = None
        if pn is not None and not np.all(np.isfinite(pn)):
            pn = None
        if pe is None and pn is None:
            return None
        if pe is None:
            lp = _norm_log(pn)
        elif pn is None:
            lp = _norm_log(pe)                       # деградация до ЭЭГ-only: без веса w
        else:
            lp = _norm_log(pe, self.w) + _norm_log(pn, 1.0 - self.w)
        return _softmax(lp + self.prior)

    # ---------------------------------------------------------------- ответы
    def decide(self, probas: dict, end: int = 0, fs: float = 250.0) -> dict:
        t = float(end) / float(fs or 250.0)
        pe, pn = probas.get("eeg"), probas.get("nirs")
        self.last_fused = self.fuse(probas)          # до сглаживания — для dev-паритета
        y_hat = self.last_smoothed = self._smooth_y(self.last_fused, t)
        e_hat = self.ema_eeg(_softmax(_norm_log(pe) + self.prior) if pe is not None else None, t)
        n_hat = self.ema_nirs(_softmax(_norm_log(pn) + self.nirs_prior) if pn is not None else None, t)
        pick = lambda v: None if v is None else int(np.argmax(v)) + 1
        return {"y": pick(y_hat) or 1, "y_eeg": pick(e_hat), "y_nirs": pick(n_hat)}


    def _smooth_y(self, p, t):
        if self.hier is None or p is None:
            return self.ema_y(p, t)
        pr, q = float(p[0]), float(p[1] / max(p[1] + p[2], 1e-12))
        pr = float(self.hier[0](np.array([pr, 1 - pr]), t)[0])
        q = float(self.hier[1](np.array([q, 1 - q]), t)[0])
        return np.array([pr, (1 - pr) * q, (1 - pr) * (1 - q)])


class MeanProbaFusion:
    """Baseline stage0: взвешенное среднее вероятностей, ответы — argmax по модальностям."""

    def __init__(self, weights: dict | None = None):
        self.weights = dict(weights or {})

    def fit(self, *args, **kwargs) -> None:  # место для обучаемого стекинга
        pass

    def fuse(self, probas: dict):
        acc, wsum = None, 0.0
        for k, p in probas.items():
            if p is None or not np.all(np.isfinite(p)):
                continue
            w = float(self.weights.get(k, 1.0))
            acc = w * np.asarray(p) if acc is None else acc + w * np.asarray(p)
            wsum += w
        return None if acc is None or wsum <= 0 else acc / wsum

    def decide(self, probas: dict, end: int = 0, fs: float = 250.0) -> dict:
        p = self.last_fused = self.last_smoothed = self.fuse(probas)
        pick = lambda v: None if v is None else int(np.argmax(v)) + 1
        return {"y": pick(p) or 1, "y_eeg": pick(probas.get("eeg")), "y_nirs": pick(probas.get("nirs"))}
