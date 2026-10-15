"""Эксперимент 1: within-session модели на фильтр-банке (длина окна, каналы, полосы, TS/logvar).

python -m src.eeg.exp_within <win> [feat ...]
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.eeg.cache import load_eeg_cache
from src.eeg.common import ALL, NO_FRONTAL, SENSORIMOTOR, RefState, diag_scale, online_eval, report, score, subcov

CHSETS = dict(all=ALL, nofront=NO_FRONTAL, sm=SENSORIMOTOR)


def slda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def make_fp(feat="ts", chans=ALL, bands=None, clf=slda, eps=1e-2):
    def fp(name, d, b):
        c = d["covs"]
        if bands is not None:
            c = c[:, bands]
        c = subcov(c, chans).astype(np.float64)
        tr, te = d["block"] < b, d["block"] == b
        ref = RefState(eps)
        if feat == "ts":
            ref.fit(c[tr])
        else:
            ref.d = diag_scale(c[tr])
        f = getattr(ref, feat)
        m = clf().fit(f(c[tr]), d["y"][tr])
        return m.predict_proba(f(c[te]))
    return fp


if __name__ == "__main__":
    win = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    feats = sys.argv[2:] or ["logvar", "ts"]
    data = load_eeg_cache(win)
    for feat in feats:
        for cn in ["all", "nofront", "sm"]:
            for bn, bands in [("fb6", [0, 1, 2, 3, 4, 5]), ("fb7", None), ("wide", [6])]:
                if feat == "logvar" and bn == "wide":
                    continue
                p, t = online_eval(data, make_fp(feat, CHSETS[cn], bands))
                report(f"w{win:g} {feat} {cn} {bn}", score(data, p), t)
