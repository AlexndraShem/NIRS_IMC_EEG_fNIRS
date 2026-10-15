"""Проверка вклада лобных (глазных) каналов: классификатор только на fp1/fp2/f7/f8 и
на остальных, по-классовые F1 (особенно левая vs правая), где «пик» по времени внутри сегмента.

python -m src.eeg.exp_frontal <win>
"""
from __future__ import annotations

import sys

import numpy as np

from src.eeg.cache import load_eeg_cache
from src.eeg.common import ALL, FRONTAL, NO_FRONTAL, online_eval, score
from src.eeg.exp_within import make_fp


def lr_acc(data, probs):
    """Точность различения левая/правая среди окон задач (argmax по P2 vs P3)."""
    acc = []
    for n, P in probs.items():
        d = data[n]; m = (d["block"] >= 2) & (d["y"] >= 2)
        acc.append(np.mean((P[m, 1] < P[m, 2]) + 2 == d["y"][m]))
    return np.mean(acc)


if __name__ == "__main__":
    win = float(sys.argv[1])
    data = load_eeg_cache(win)
    for cn, ch in [("frontal4", FRONTAL), ("nofront15", NO_FRONTAL), ("all21", ALL)]:
        for feat in ["logvar", "ts"]:
            p, _ = online_eval(data, make_fp(feat, ch, [0, 1, 2, 3, 4, 5]))
            sc = score(data, p)
            f1 = np.mean([r["f1"] for r in sc["per"].values()], 0)
            print(f"w{win:g} {cn:10s} {feat:6s} minF1={sc['min_f1']:.3f} rec={sc['recall']:.3f} pts={sc['pts']:.1f} "
                  f"F1(rest,L,R)={np.round(f1, 3)} L/R acc={lr_acc(data, p):.3f}", flush=True)
