"""Эксперимент 3: FBCSP (one-vs-rest + пара левая/правая) на фильтр-банке, within-session.

python -m src.eeg.exp_csp <win> <chans>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.linalg as sl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.eeg.cache import load_eeg_cache
from src.eeg.common import ALL, NO_FRONTAL, SENSORIMOTOR, diag_scale, normalize, online_eval, report, score, subcov

CHSETS = dict(all=ALL, nofront=NO_FRONTAL, sm=SENSORIMOTOR)
PAIRS = [((1,), (2, 3)), ((2,), (1, 3)), ((3,), (1, 2)), ((2,), (3,))]


def csp_filters(c, y, m=2, shrink=0.1):
    """c (n,C,C). Для каждой пары классов — m фильтров с каждого края, (C, k)."""
    Ws = []
    n = c.shape[-1]
    for A, B in PAIRS:
        Ca = c[np.isin(y, A)].mean(0); Cb = c[np.isin(y, B)].mean(0)
        Ca = (1 - shrink) * Ca + shrink * np.trace(Ca) / n * np.eye(n)
        Cb = (1 - shrink) * Cb + shrink * np.trace(Cb) / n * np.eye(n)
        w, V = sl.eigh(Ca, Ca + Cb)
        Ws.append(np.hstack([V[:, :m], V[:, -m:]]))
    return np.hstack(Ws)


class FBCSP:
    def __init__(self, m=2, shrink=0.1, eps=1e-2):
        self.m, self.shrink, self.eps = m, shrink, eps

    def _f(self, c):
        c = normalize(c, self.d, self.eps)
        v = np.einsum("bck,nbcd,bdk->nbk", self.W, c, self.W)
        return np.log(np.maximum(v, 1e-12)).reshape(len(c), -1)

    def fit(self, c, y):
        self.d = diag_scale(c)
        cn = normalize(c, self.d, self.eps)
        self.W = np.stack([csp_filters(cn[:, b], y, self.m, self.shrink) for b in range(c.shape[1])])
        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(self._f(c), y)
        return self

    def predict_proba(self, c):
        return self.clf.predict_proba(self._f(c))


def make_fp(chans, m=2, shrink=0.1, bands=None):
    def fp(name, d, b):
        c = d["covs"] if bands is None else d["covs"][:, bands]
        c = subcov(c, chans).astype(np.float64)
        tr, te = d["block"] < b, d["block"] == b
        return FBCSP(m, shrink).fit(c[tr], d["y"][tr]).predict_proba(c[te])
    return fp


if __name__ == "__main__":
    win = float(sys.argv[1]); chn = sys.argv[2]
    data = load_eeg_cache(win)
    for m in [1, 2, 3]:
        for sh in [0.05, 0.2]:
            p, t = online_eval(data, make_fp(CHSETS[chn], m, sh))
            report(f"w{win:g} fbcsp {chn} m={m} sh={sh}", score(data, p), t)
