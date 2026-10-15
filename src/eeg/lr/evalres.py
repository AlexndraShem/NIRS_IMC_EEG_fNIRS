"""Оценка результатов скрининга src.eeg.lr.feats против baseline eeg_ts2s_all.

python -m src.eeg.lr.evalres <feat> [<feat> ...]
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_eeg_cache
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import combine_lr, detail, fmt, load_probs
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.feats import SCR


def mix3(lw, lp, a):
    out = {}
    for n in lw:
        z = (1 - a) * np.maximum(lw[n], np.log(1e-9)) + a * np.maximum(lp[n], np.log(1e-9))
        z = z - np.nanmax(z, 1, keepdims=True)
        p = np.exp(z)
        out[n] = p / p.sum(1, keepdims=True)
    return out


def prod(Pa, Pb, g):
    out = {}
    for n in Pa:
        z = np.log(np.clip(Pa[n], 1e-9, 1)) + g * np.log(np.clip(Pb[n], 1e-9, 1))
        z = z - np.nanmax(z, 1, keepdims=True)
        p = np.exp(z)
        out[n] = p / p.sum(1, keepdims=True)
    return out


def lr_mix(res, a, clip=8.0):
    return {n: np.clip((1 - a) * r["rw"] + a * r["rp"], -clip, clip) for n, r in res.items()}


def lr_auc(data, R):
    """Точность знака r на истинных L/R окнах блоков 2–7 (среднее по сессиям)."""
    acc = []
    for n, r in R.items():
        d = data[n]; t = (d["block"] >= 2) & (d["y"] > 1)
        acc.append(np.mean((r[t] > 0) == (d["y"][t] == 2)))
    return float(np.mean(acc))


def evaluate(feat, data, base, quiet=False):
    res = pickle.load(open(SCR / f"res_{feat}.pkl", "rb"))
    lw = {n: r["pw3"] for n, r in res.items()}; lp = {n: r["pp3"] for n, r in res.items()}
    best = []
    for a in (0.0, 0.5, 0.8, 1.0):
        s = detail(data, mix3(lw, lp, a)); best.append((s["pts"], f"3cls a={a}"))
        if not quiet:
            print(fmt(f"{feat} 3cls mix a={a}", s))
    P3 = mix3(lw, lp, 0.8)
    for g in (0.3, 0.5, 1.0):
        s = detail(data, prod(base, P3, g)); best.append((s["pts"], f"base*3cls^{g}"))
        if not quiet:
            print(fmt(f"{feat} base x 3cls(a=.8)^{g}", s))
    for a in (0.0, 0.5, 0.8, 1.0):
        R = lr_mix(res, a)
        acc = lr_auc(data, R)
        for beta in (0.25, 0.5, 1.0):
            s = detail(data, {n: combine_lr(base[n], R[n], beta) for n in base})
            best.append((s["pts"], f"base+LR a={a} b={beta}"))
            if not quiet:
                print(fmt(f"{feat} base+LR a={a} b={beta} (acc {acc:.3f})", s))
    best.sort(reverse=True)
    print(f"== {feat}: top {best[:3]}", flush=True)
    return res


if __name__ == "__main__":
    data = load_eeg_cache(2.0, f64=True)
    base = load_probs("ts2s_all")
    print(fmt("baseline eeg_ts2s_all", detail(data, base), detail(data, base, tau=1.5)))
    for f in sys.argv[1:]:
        evaluate(f, data, base)
