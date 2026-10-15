"""Переигрывание слияния по dev-прогонам официальной сетки (`solution/run.py --dump-probs` →
`<runs>/<сессия>/probs.csv`: end_sample, eeg1..3, nirs1..3 на КАЖДЫЙ вызов predict).

Загрузка (`load_protocol`) + признаки `StackedFusion` по всем вызовам predict (как онлайн) +
векторный прогон решающего правила и метрики судьи (окна блоков ≥ 2, однородные 1 с).
Флаг `subj` — в прогоне был пул участника (в overlay-каталоге протокола есть `eeg_subj/*.npz`),
т.е. судья использовал `fusion_params_subject` / мета-модель «с пулом».
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from NIRS_IMC_EEG_fNIRS.src.dataset.io import RAW_DIR, load_session
from NIRS_IMC_EEG_fNIRS.src.dataset.windows import confusion, metrics
from NIRS_IMC_EEG_fNIRS.src.fusion.stacked import FLOOR, ModalityFeatures, _clean


def load_protocol(runs, overlay=None):
    out = {}
    for d in sorted(Path(runs).iterdir()):
        f = d / "probs.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        s = load_session(RAW_DIR / "Data" / f"{d.name}.mat")
        e = p.end_sample.values.astype(int)
        st = s.states.astype(int)
        seg = np.r_[0, np.cumsum(np.diff(st) != 0)]
        ok = (e >= 250) & (seg[np.maximum(e - 250, 0)] == seg[e - 1]) & (st[e - 1] > 0) & (s.blocks[e - 1] >= 2)
        pn = p[["nirs1", "nirs2", "nirs3"]].values
        out[d.name] = dict(e=e, t=e / 250.0, y=st[e - 1], ok=ok, pe=p[["eeg1", "eeg2", "eeg3"]].values,
                           pn=pn, has_nirs=bool(np.isfinite(pn).all(1).any()),
                           subj=bool(overlay) and any((Path(overlay) / d.name / "eeg_subj").glob("*.npz")),
                           subject=d.name.split("_")[-1])
    return out


def add_features(R, feat):
    """Признаки StackedFusion по всем вызовам predict сессии (то же состояние, что онлайн)."""
    for r in R.values():
        fe = ModalityFeatures(feat["eeg_taus"], feat["center_T"], feat.get("center_ref_tau"))
        fn = ModalityFeatures(feat["nirs_taus"], feat["center_T"], feat.get("center_ref_tau"))
        n = len(r["e"])
        Xe = np.full((n, fe.dim), np.nan)
        Xn = np.full((n, fn.dim), np.nan)
        for i in range(n):
            a = _clean(r["pe"][i])
            if a is not None:
                Xe[i] = fe.step(a, r["t"][i])
            b = _clean(r["pn"][i])
            if b is not None:
                Xn[i] = fn.step(b, r["t"][i])
        r["Xe"], r["Xn"] = Xe, Xn
    return R


def ema_seq(P, t, tau):
    """Как онлайн `_Ema`: строки с NaN пропускаются (состояние не меняется)."""
    out = np.array(P, np.float64)
    s = tp = None
    for i in range(len(P)):
        if not np.all(np.isfinite(P[i])):
            continue
        if s is None or tau <= 0:
            s = np.array(P[i], np.float64)
        else:
            a = np.exp(-(t[i] - tp) / tau)
            s = a * s + (1 - a) * P[i]
        out[i] = s
        tp = t[i]
    return out


def hier_seq(P, t, tr, tl):
    pr = P[:, :1]
    q = P[:, 1:2] / np.maximum(P[:, 1:2] + P[:, 2:3], 1e-12)
    pr = ema_seq(np.hstack([pr, 1 - pr]), t, tr)[:, :1] if tr > 0 else pr
    q = ema_seq(np.hstack([q, 1 - q]), t, tl)[:, :1] if tl > 0 else q
    return np.hstack([pr, (1 - pr) * q, (1 - pr) * (1 - q)])


def sess_fp(r, P):
    ok = r["ok"]
    m = metrics(confusion(r["y"][ok], P[ok].argmax(1) + 1))
    return m["min_f1"], m["recall"]


def summarize(per):
    a = np.array(list(per.values()))
    F, P = float(a[:, 0].mean()), float(a[:, 1].mean())
    return {"min_f1": round(F, 4), "recall": round(P, 4), "pts": round(60 * F + 30 * P, 2), "n": len(a)}


# ---------------------------------------------------------------- LogLinearFusion (сдача)
def loglinear(r, w, prior, tau=1.5, hier=None):
    lpe = np.log(np.clip(r["pe"], FLOOR, 1))
    lpn = np.log(np.clip(r["pn"], FLOOR, 1))
    has_n = np.all(np.isfinite(r["pn"]), 1)
    lp = np.where(has_n[:, None], w * lpe + (1 - w) * lpn, lpe) + np.asarray(prior, float)
    lp -= lp.max(1, keepdims=True)
    P = np.exp(lp)
    P /= P.sum(1, keepdims=True)
    return ema_seq(P, r["t"], tau) if not hier else hier_seq(P, r["t"], *hier)


def loglinear_cfg(r, fp, fps):
    p = dict(fp, **fps) if (r["subj"] and fps) else dict(fp)
    return loglinear(r, p.get("eeg_weight", 0.75), p.get("prior", [0, 2, 2]), p.get("tau_s", 1.5), p.get("hier_tau_s"))


# ---------------------------------------------------------------- стекинг
def _lsm(Z):
    Z = Z - Z.max(1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(1, keepdims=True))


def meta_logp(r, models):
    """Лог-вероятности мета-модели по окнам (гибрид при наличии NIRS, иначе ЭЭГ-модель)."""
    We, be = np.asarray(models["eeg"]["W"]), np.asarray(models["eeg"]["b"])
    okE = np.isfinite(r["Xe"]).all(1)
    L = np.full((len(r["e"]), 3), np.nan)
    L[okE] = _lsm(r["Xe"][okE] @ We.T + be)
    okH = okE & np.isfinite(r["Xn"]).all(1)
    if models.get("hyb") and okH.any():
        Wh, bh = np.asarray(models["hyb"]["W"]), np.asarray(models["hyb"]["b"])
        L[okH] = _lsm(np.hstack([r["Xe"], r["Xn"]])[okH] @ Wh.T + bh)
    return L


def stacked_out(r, models, post_tau=1.0, bias=0.0, hier=None):
    L = meta_logp(r, models)
    P = np.exp(L)
    S = ema_seq(P, r["t"], post_tau) if not hier else hier_seq(P, r["t"], *hier)
    out = np.log(np.clip(S, FLOOR, None)) + np.r_[0, bias, bias]
    out[~np.isfinite(out).all(1)] = [0, -1, -1]            # нет ЭЭГ-вероятностей → покой
    return out


def fit_meta(R, names, C=1.0, max_rows=None, seed=0):
    """Мета-LR `eeg` и `hyb` по размеченным окнам (ok) сессий names."""
    from sklearn.linear_model import LogisticRegression
    Xh, yh, Xe, ye = [], [], [], []
    for k in names:
        r = R[k]
        okE = r["ok"] & np.isfinite(r["Xe"]).all(1)
        okH = okE & np.isfinite(r["Xn"]).all(1)
        Xe.append(r["Xe"][okE]); ye.append(r["y"][okE])
        if okH.any():
            Xh.append(np.hstack([r["Xe"], r["Xn"]])[okH]); yh.append(r["y"][okH])

    def lr(X, y):
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)
        return {"W": m.coef_.tolist(), "b": m.intercept_.tolist()}
    return {"eeg": lr(np.vstack(Xe), np.concatenate(ye)),
            "hyb": lr(np.vstack(Xh), np.concatenate(yh)) if Xh else None}
