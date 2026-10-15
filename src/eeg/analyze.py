"""Разбор сохранённых вероятностей: метрики, по-классовые F1, L/R-точность, по блокам, сдвиг приоров.

python -m src.eeg.analyze eeg_ts2s eeg_ts2s_within ...
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

from src.dataset.windows import confusion, metrics
from src.eeg.common import PROBS_DIR
from src.features.build_cache import FEAT_DIR


def load_meta(names):
    out = {}
    for n in names:
        z = np.load(FEAT_DIR / "v1_win1" / f"{n}.npz")
        out[n] = dict(y=z["y"], block=z["block"])
    return out


def agg(meta, probs, mask_fn=lambda d: d["block"] >= 2, bias=None):
    mf, rc, f1s, lr = [], [], [], []
    for n, P in probs.items():
        d = meta[n]; m = mask_fn(d)
        lp = np.log(np.clip(P[m], 1e-9, 1)) + (0 if bias is None else bias)
        r = metrics(confusion(d["y"][m], lp.argmax(1) + 1))
        mf.append(r["min_f1"]); rc.append(r["recall"]); f1s.append(r["f1"])
        t = m & (d["y"] >= 2)
        lr.append(np.mean((P[t, 2] > P[t, 1]) + 2 == d["y"][t]))
    return np.mean(mf), np.mean(rc), np.mean(f1s, 0), np.mean(lr)


if __name__ == "__main__":
    for key in sys.argv[1:]:
        probs = pickle.load(open(PROBS_DIR / f"{key}.pkl", "rb"))
        meta = load_meta(probs)
        f, r, f1, lr = agg(meta, probs)
        print(f"{key:24s} minF1={f:.3f} recall={r:.3f} pts={60*f+30*r:.1f} F1(rest,L,R)={np.round(f1,3)} L/R acc={lr:.3f}")
        blk = [agg(meta, probs, lambda d, b=b: d["block"] == b)[1] for b in range(2, 8)]
        print(f"{'':24s} recall по блокам 2..7: {np.round(blk, 3)}")
        for bb in [-0.2, 0.2, 0.4]:
            f, r, *_ = agg(meta, probs, bias=np.array([0, bb, bb]))
            print(f"{'':24s} сдвиг log-приора задач {bb:+.1f}: minF1={f:.3f} recall={r:.3f} pts={60*f+30*r:.1f}")
