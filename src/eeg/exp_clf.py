"""Эксперимент 2c: регуляризация within-классификатора на рецентрированном TS (калибровка
уверенности) и смесь с пул-LR. Работает на кэше признаков exp_cross2 (tsF_*.pkl) и его пул-модели.

python -m src.eeg.exp_clf <win> <chans> <cache_dir>
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eeg.common import report, score
from src.eeg.exp_cross2 import get_feats, mix

_G = {}


def within_one(n):
    data, F, spec = _G["data"], _G["F"], _G["spec"]
    d = data[n]
    P = np.full((len(d["y"]), 3), np.nan)
    for b in range(2, 8):
        Xb = F[n][b]; blk = d["block"][d["block"] <= b]; yb = d["y"][d["block"] <= b]
        tr, te = blk < b, blk == b
        kind, par = spec
        if kind == "lda":
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=par).fit(Xb[tr], yb[tr])
            p = m.predict_proba(Xb[te])
        else:
            sc = StandardScaler().fit(Xb[tr])
            m = LogisticRegression(C=par, max_iter=200, class_weight="balanced").fit(sc.transform(Xb[tr]), yb[tr])
            p = m.predict_proba(sc.transform(Xb[te]))
        P[d["block"] == b] = p
    return n, P


if __name__ == "__main__":
    win, chn, cdir = float(sys.argv[1]), sys.argv[2], sys.argv[3]
    data, F = get_feats(win, chn, cdir)
    names = list(data)
    cand = sorted(Path(cdir).glob(f"cross2_res_w{win:g}_{chn}_lr*.pkl"))
    res = pickle.load(open(cand[0], "rb")) if cand else None
    Pp = res["lrC0.01"][1] if res else None
    out = {}
    for spec in [("lda", "auto"), ("lda", 0.5), ("lda", 0.8), ("lda", 0.95), ("lr", 0.001), ("lr", 0.01), ("lr", 0.1)]:
        _G.update(data=data, F=F, spec=spec)
        with ProcessPoolExecutor(10, mp_context=mp.get_context("fork")) as ex:
            Pw = dict(ex.map(within_one, names))
        out[spec] = Pw
        report(f"w{win:g} {chn} within {spec}", score(data, Pw))
        if Pp is not None:
            for a in [0.3, 0.5, 0.7, 0.9]:
                report(f"   + pool a={a}", score(data, mix(Pw, Pp, a)))
    pickle.dump(out, open(Path(cdir) / f"clf_res_w{win:g}_{chn}.pkl", "wb"))
