"""Быстрая оффлайн-оценка по кэшу признаков с онлайн-порядком обучения.

Для каждой сессии и блока b = 2..7: модель видит только
  * внешний пул (другие участники, leave-subject-out), и/или
  * блоки < b текущей сессии,
и предсказывает окна блока b. Метрики — по ТЗ §5.1, агрегат — среднее по сессиям.
"""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset.io import PROJECT_DIR
from src.dataset.windows import confusion, metrics

FEAT_DIR = PROJECT_DIR / "data" / "features"


def load_features(tag="v1_win1"):
    out = {}
    for p in sorted((FEAT_DIR / tag).glob("*.npz")):
        z = np.load(p)
        out[p.stem] = {k: z[k] for k in z.files}
        out[p.stem]["subject"] = str(z["subject"])
    return out


def slda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def logreg(C=0.01):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300))


def run_within(feats, key, make_model=slda):
    """Только данные текущей сессии, дообучение после каждого блока."""
    res = {}
    for name, d in feats.items():
        X = d.get(key)
        if X is None:
            continue
        yt, yp = [], []
        for b in range(2, 8):
            tr, te = d["block"] < b, d["block"] == b
            m = make_model().fit(X[tr], d["y"][tr])
            yp.append(m.predict(X[te])); yt.append(d["y"][te])
        res[name] = metrics(confusion(np.concatenate(yt), np.concatenate(yp)))
    return res


def run_proba(feats, key, fit_predict, names=None):
    """Общий цикл: fit_predict(name, b, tr_mask, te_mask) → proba (n_te, 3)."""
    probs = {}
    for name in names or feats:
        d = feats[name]
        if d.get(key) is None:
            continue
        P = np.full((len(d["y"]), 3), np.nan)
        for b in range(2, 8):
            te = d["block"] == b
            P[te] = fit_predict(name, b, d["block"] < b, te)
        probs[name] = P
    return probs


def score_probs(feats, probs):
    res = {}
    for name, P in probs.items():
        d = feats[name]
        te = d["block"] >= 2
        res[name] = metrics(confusion(d["y"][te], P[te].argmax(1) + 1))
    return res


def summary(res, label=""):
    f = np.mean([r["min_f1"] for r in res.values()])
    r = np.mean([r["recall"] for r in res.values()])
    print(f"{label:40s} n={len(res):2d} minF1={f:.3f} recall={r:.3f} pts(60F+30P)={60*f+30*r:.1f}", flush=True)
    return f, r
