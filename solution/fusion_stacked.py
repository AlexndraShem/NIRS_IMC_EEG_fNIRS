"""Стекинг-слияние `StackedFusion` (копия `src/fusion/stacked.py`, держать синхронно): мета-логрегрессия по многомасштабным причинным признакам
вероятностей ЭЭГ и NIRS (эксперимент `docs/experiments/fusion_decision.md`).

Интерфейс совпадает с `solution/fusion.py::LogLinearFusion`:

    fusion = StackedFusion(params)                     # params — dict из config.json → fusion_params
    fusion.decide({"eeg": p_eeg | None, "nirs": p_nirs | None}, end, fs) -> {"y", "y_eeg", "y_nirs"}

Признаки окна (всё причинно, состояние — только прошлые вызовы `decide`):

* для каждой модальности m и каждого τ из `eeg_taus` / `nirs_taus`: EMA по реальному времени
  (a = exp(−Δt/τ)) от вероятностей P_m, затем лог-отношения [log s₂ − log s₁, log s₃ − log s₁]
  (τ = 0 — без сглаживания);
* «центрированные» признаки для каждого T из `center_T`: лог-отношения EMA с τ = `center_ref_tau`
  минус EMA(T) от мгновенных лог-отношений — причинное вычитание медленного сессионного смещения
  (калибровка по сессии без меток).

Мета-модели — мультиномиальные логрегрессии (веса в params, обучает `src/fusion/tune.py`):

* `models.hyb`  — ЭЭГ+NIRS признаки → ответ `y`, если NIRS есть;
* `models.eeg`  — только ЭЭГ-признаки → `y` при отсутствии NIRS (деградация) и `y_eeg`;
* `y_nirs` — правило по одному P_nirs: argmax(log P_nirs + nirs_prior) (+ опц. EMA `nirs_tau_s`);
  стекинг для NIRS-only не дал выигрыша, поэтому оставлено правило.

Выход мета-модели: log-softmax → сглаживание (EMA `post_tau_s` или иерархическое
`post_hier_tau_s` = [τ_покой, τ_LR], как в LogLinearFusion) → + `bias` → argmax.
Меток модель не видит; `fit_block` не требуется (адаптация по сессии проверена и не помогла).

Нет весов (`params["models"]` пуст/None) → откат на `LogLinearFusion(params)` из `fusion.py`
(те же ключи: eeg_weight, prior, tau_s, hier_tau_s, nirs_prior, nirs_tau_s).
"""
from __future__ import annotations

import numpy as np

FLOOR = 1e-9


def _softmax(lp):
    lp = np.asarray(lp, np.float64)
    e = np.exp(lp - lp.max())
    return e / e.sum()


def _clean(p):
    if p is None:
        return None
    p = np.asarray(p, np.float64).ravel()
    if p.shape[0] != 3 or not np.all(np.isfinite(p)):
        return None
    p = np.clip(p, FLOOR, None)
    return p / p.sum()


def _ratio(p):
    lp = np.log(np.clip(p, FLOOR, None))
    return lp[1:] - lp[0]


class _Ema:
    """Причинное EMA по реальному времени; τ ≤ 0 — пропуск без сглаживания."""

    __slots__ = ("tau", "s", "t")

    def __init__(self, tau):
        self.tau, self.s, self.t = float(tau), None, None

    def __call__(self, x, t):
        x = np.asarray(x, np.float64)
        if self.tau <= 0 or self.s is None:
            self.s, self.t = x.copy(), t
            return self.s
        a = float(np.exp(-max(t - self.t, 0.0) / self.tau))
        self.s = a * self.s + (1.0 - a) * x
        self.t = t
        return self.s


class ModalityFeatures:
    """Многомасштабные признаки одной модальности (состояние между вызовами)."""

    def __init__(self, taus, center_T=(), center_ref_tau=None):
        self.taus = [float(v) for v in taus]
        self.center_T = [float(v) for v in center_T]
        self.ref = self.taus[min(1, len(self.taus) - 1)] if center_ref_tau is None else float(center_ref_tau)
        self.emas = [_Ema(v) for v in self.taus]
        self.ref_ema = _Ema(self.ref) if self.ref not in self.taus else None
        self.cemas = [_Ema(v) for v in self.center_T]

    @property
    def dim(self):
        return 2 * (len(self.taus) + len(self.center_T))

    def step(self, p, t):
        out = [_ratio(e(p, t)) for e in self.emas]
        if self.center_T:
            ref = out[self.taus.index(self.ref)] if self.ref_ema is None else _ratio(self.ref_ema(p, t))
            x0 = _ratio(p)
            out += [ref - c(x0, t) for c in self.cemas]
        return np.concatenate(out)


class _Linear:
    def __init__(self, d):
        self.W = np.asarray(d["W"], np.float64)          # (3, F)
        self.b = np.asarray(d["b"], np.float64)          # (3,)

    def logp(self, x):
        z = self.W @ x + self.b
        z = z - z.max()
        return z - np.log(np.exp(z).sum())


DEFAULTS = {
    "eeg_taus": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "nirs_taus": [0.0, 2.0, 6.0, 12.0, 24.0],
    "center_T": [60.0],
    "center_ref_tau": None,
    "post_tau_s": 1.0,
    "bias": [0.0, 0.0, 0.0],
    "eeg_bias": None,           # None → как bias
    "nirs_prior": [0.0, 1.0, 1.0],
    "nirs_tau_s": 0.0,
    "post_hier_tau_s": None,
}


def _hier(emas, p, t):
    """Иерархическое сглаживание: P(покой) — EMA τ₀, доля L/(L+R) — EMA τ₁."""
    pr, q = float(p[0]), float(p[1] / max(p[1] + p[2], 1e-12))
    pr = float(emas[0](np.array([pr, 1 - pr]), t)[0])
    q = float(emas[1](np.array([q, 1 - q]), t)[0])
    return np.array([pr, (1 - pr) * q, (1 - pr) * (1 - q)])


class StackedFusion:
    def __new__(cls, params: dict | None = None):
        if not (params or {}).get("models"):
            try:                                        # откат: нет мета-модели для этого прогона
                from fusion import LogLinearFusion      # solution/ (каталог решения в sys.path)
            except ImportError:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "solution"))
                from fusion import LogLinearFusion
            return LogLinearFusion(params)
        return super().__new__(cls)

    def __init__(self, params: dict | None = None):
        p = dict(DEFAULTS)
        p.update(params or {})
        self.params = p
        self.m_hyb = _Linear(p["models"]["hyb"]) if p["models"].get("hyb") else None
        self.m_eeg = _Linear(p["models"]["eeg"])
        self.bias = np.asarray(p["bias"], np.float64)
        self.eeg_bias = self.bias if p.get("eeg_bias") is None else np.asarray(p["eeg_bias"], np.float64)
        self.nirs_prior = np.asarray(p["nirs_prior"], np.float64)
        self.fe = ModalityFeatures(p["eeg_taus"], p["center_T"], p["center_ref_tau"])
        self.fn = ModalityFeatures(p["nirs_taus"], p["center_T"], p["center_ref_tau"])
        h = p.get("post_hier_tau_s")
        self.hier_y = (_Ema(h[0]), _Ema(h[1])) if h else None
        self.hier_e = (_Ema(h[0]), _Ema(h[1])) if h else None
        self.post_y = _Ema(p["post_tau_s"])
        self.post_e = _Ema(p["post_tau_s"])
        self.post_n = _Ema(p["nirs_tau_s"])
        self.last_fused = self.last_smoothed = None

    # признаки (для обучения в tune.py — тот же код, что онлайн)
    def features(self, pe, pn, t):
        xe = self.fe.step(pe, t) if pe is not None else None
        xn = self.fn.step(pn, t) if pn is not None else None
        return xe, xn

    def decide(self, probas: dict, end: int = 0, fs: float = 250.0) -> dict:
        t = float(end) / float(fs or 250.0)
        pe, pn = _clean(probas.get("eeg")), _clean(probas.get("nirs"))
        xe, xn = self.features(pe, pn, t)
        pick = lambda v: None if v is None else int(np.argmax(v)) + 1
        y = y_eeg = None
        if xe is not None:
            lpe = self.m_eeg.logp(xe)
            if xn is not None and self.m_hyb is not None:
                lp = self.m_hyb.logp(np.concatenate([xe, xn]))
            else:
                lp = lpe
            self.last_fused = np.exp(lp)
            s = self.last_smoothed = (_hier(self.hier_y, self.last_fused, t) if self.hier_y
                                      else self.post_y(self.last_fused, t))
            y = pick(np.log(np.clip(s, FLOOR, None)) + self.bias)
            se = _hier(self.hier_e, np.exp(lpe), t) if self.hier_e else self.post_e(np.exp(lpe), t)
            y_eeg = pick(np.log(np.clip(se, FLOOR, None)) + self.eeg_bias)
        elif pn is not None:                              # нет ЭЭГ (не должно случаться) — только NIRS
            y = pick(np.log(pn) + self.nirs_prior)
        y_nirs = None
        if pn is not None:
            sn = self.post_n(pn, t)
            y_nirs = pick(np.log(np.clip(sn, FLOOR, None)) + self.nirs_prior)
        return {"y": y or 1, "y_eeg": y_eeg, "y_nirs": y_nirs}
