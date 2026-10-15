"""Комбинации экспертов поверх eeg_ts2s_all: расписание веса пула по блокам, произведение с
дополнительными 3-классовыми экспертами, бинарные L/R-эксперты, иерархическое сглаживание.

Используется как библиотека (см. src/eeg/lr/final.py) и для быстрых сравнений:
python -m src.eeg.lr.comb
"""
from __future__ import annotations

import pickle

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import combine_lr, ema, load_probs
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.feats import SCR

FLOOR = np.log(1e-9)


def _norm(z):
    z = z - np.nanmax(z, 1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def sched_alpha(block, a2=0.9, a7=0.7):
    """Вес пула для блока b: линейно от a2 (блок 2) до a7 (блок 7)."""
    b = np.clip(block, 2, 7)
    return a2 + (a7 - a2) * (b - 2) / 5


def mix_sched(lw, lp, block, a2=0.9, a7=0.7):
    """lw, lp — вероятности или лог-вероятности (n,3); смесь с весом пула по блокам."""
    a = sched_alpha(block, a2, a7)[:, None]
    return _norm((1 - a) * np.maximum(lw, FLOOR) + a * np.maximum(lp, FLOOR))


def as_log(P):
    P = np.asarray(P, np.float64)
    return np.log(np.clip(P, 1e-9, 1)) if np.nanmin(P) >= 0 else P


def load_res(feat):
    return pickle.load(open(SCR / f"res_{feat}.pkl", "rb"))


def hier_smooth(P, ends, tau_rest=0.0, tau_lr=3.0):
    """P(покой) сглаживается с tau_rest, доля L/(L+R) — с tau_lr (причинное EMA)."""
    pr = P[:, 0:1]
    q = P[:, 1:2] / np.maximum(P[:, 1:2] + P[:, 2:3], 1e-12)
    if tau_rest > 0:
        pr = ema(np.hstack([pr, 1 - pr, pr]), ends, tau_rest)[:, 0:1]
    if tau_lr > 0:
        q = ema(np.hstack([q, 1 - q, q]), ends, tau_lr)[:, 0:1]
    return np.hstack([pr, (1 - pr) * q, (1 - pr) * (1 - q)])


def build(data, base_w="ts2s_all_within", base_p="ts2s_all_pool", a2=0.8, a7=0.8,
          experts=(), lr_experts=()):
    """experts: [(feat, a2, a7, g)] — 3-кл. эксперты (P ∝ base · expert^g);
    lr_experts: [(feat, a, beta)] — бинарные лог-шансы (1-a)·rw + a·rp с весом beta."""
    if base_w.startswith("res:"):          # within из скрининга (напр. с зеркальной аугментацией)
        W = {n: r["pw3"] for n, r in load_res(base_w[4:]).items()}
    else:
        W = load_probs(base_w)
    Pp = load_probs(base_p)
    out = {n: np.log(mix_sched(as_log(W[n]), as_log(Pp[n]), data[n]["block"], a2, a7)) for n in W}
    for feat, e2, e7, g in experts:
        R = load_res(feat)
        for n in out:
            pp = R[n]["pp3"] if not np.all(np.isnan(R[n]["pp3"])) else np.zeros_like(R[n]["pw3"])
            out[n] = out[n] + g * np.log(mix_sched(R[n]["pw3"], pp, data[n]["block"], e2, e7))
    for feat, a, beta in lr_experts:
        R = load_res(feat)
        for n in out:
            r = np.clip((1 - a) * R[n]["rw"] + a * R[n]["rp"], -8, 8)
            out[n] = np.log(combine_lr(_norm(out[n]), r, beta))
    return {n: _norm(z) for n, z in out.items()}
