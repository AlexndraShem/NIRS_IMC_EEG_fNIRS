"""Переигрывание слияния по dev-прогонам (solution/run.py --dump-probs → <runs>/<сессия>/probs.csv):
LogLinearFusion с разными параметрами и иерархическое сглаживание гибрида
(P(покой) — EMA τ_rest, доля L/(L+R) — EMA τ_lr). Метрика — как у судьи: окна прогноза, метка по
States, однородные окна 1 с блоков ≥ 2, min-F1 и macro-recall гибрида по сессиям.

python -m src.eeg.lr.fusion_eval <runs_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from NIRS_IMC_EEG_fNIRS.src.dataset.io import RAW_DIR, load_session
from NIRS_IMC_EEG_fNIRS.src.dataset.windows import confusion, metrics

FLOOR = 1e-9


def load_runs(runs, overlay=None):
    """overlay — каталог артефактов протокола (<S>/eeg_subj/…): есть пул участника → в судейском
    прогоне действовали fusion_params_subject."""
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
        out[d.name] = dict(e=e, y=st[e - 1], ok=ok, pe=p[["eeg1", "eeg2", "eeg3"]].values,
                           pn=p[["nirs1", "nirs2", "nirs3"]].values,
                           subj=bool(overlay) and any((Path(overlay) / d.name / "eeg_subj").glob("*.npz")))
    return out


def ema_seq(P, t, tau):
    out = np.array(P, np.float64)
    s = tp = None
    for i in range(len(P)):
        if not np.all(np.isfinite(P[i])):
            continue
        if s is None or tau <= 0:
            s = P[i].copy()
        else:
            a = np.exp(-(t[i] - tp) / tau)
            s = a * s + (1 - a) * P[i]
        out[i] = s; tp = t[i]
    return out


def fuse(r, w=0.75, prior=(0, 2, 2), tau=1.5, hier=None):
    pe, pn = r["pe"], r["pn"]
    lpe = np.log(np.clip(pe, FLOOR, 1))
    lpn = np.log(np.clip(pn, FLOOR, 1))
    has_n = np.all(np.isfinite(pn), 1)
    lp = np.where(has_n[:, None], w * lpe + (1 - w) * lpn, lpe) + np.asarray(prior, float)
    lp -= lp.max(1, keepdims=True)
    P = np.exp(lp); P /= P.sum(1, keepdims=True)
    t = r["e"] / 250.0
    if hier is None:
        return ema_seq(P, t, tau)
    tr, tl = hier
    pr = P[:, :1]
    q = P[:, 1:2] / np.maximum(P[:, 1:2] + P[:, 2:3], 1e-12)
    pr = ema_seq(np.hstack([pr, 1 - pr]), t, tr)[:, :1] if tr > 0 else pr
    q = ema_seq(np.hstack([q, 1 - q]), t, tl)[:, :1] if tl > 0 else q
    return np.hstack([pr, (1 - pr) * q, (1 - pr) * (1 - q)])


def score(R, subj_kw=None, **kw):
    """subj_kw — параметры для сессий с пулом участника (иначе те же kw)."""
    mf, rc = [], []
    for r in R.values():
        P = fuse(r, **(dict(kw, **subj_kw) if subj_kw and r["subj"] else kw))
        ok = r["ok"]
        m = metrics(confusion(r["y"][ok], P[ok].argmax(1) + 1))
        mf.append(m["min_f1"]); rc.append(m["recall"])
    f, q = float(np.mean(mf)), float(np.mean(rc))
    return f, q, 60 * f + 30 * q


if __name__ == "__main__":
    R = load_runs(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"сессий: {len(R)}, с пулом участника: {sum(r['subj'] for r in R.values())}")
    cur = dict(w=0.75, prior=(0, 2, 2)); cur_s = dict(w=0.7, prior=(0, 2.5, 2.5))
    rows = [("ТЕКУЩЕЕ (.75/2; участник .7/2.5)", score(R, subj_kw=cur_s, **cur))]
    for w in (0.55, 0.6, 0.65, 0.7, 0.75):
        for pr in (1.5, 2.0, 2.5, 3.0):
            rows.append((f"все w={w} prior={pr}", score(R, w=w, prior=(0, pr, pr))))
    for tau in (0.75, 1.0, 2.0):
        rows.append((f"текущее, ema {tau}", score(R, subj_kw=cur_s, tau=tau, **cur)))
    for tr in (0.0, 0.75, 1.5):
        for tl in (1.5, 2.0, 3.0, 4.0):
            rows.append((f"текущее, hier {tr}/{tl}", score(R, subj_kw=cur_s, hier=(tr, tl), **cur)))
    for lab, (f, q, pts) in rows:
        print(f"{lab:36s} minF1={f:.4f} rec={q:.4f} 60F+30P={pts:.2f}", flush=True)
