"""Диагностика: детализация baseline (eeg_ts2s_all) и ERD-карта лево/право по истинным каналам.

python -m src.eeg.lr.analyze
"""
from __future__ import annotations

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_eeg_cache
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import TRUE_NAMES, apply_spatial, detail, fmt, load_probs, spatial_matrix


def erd_map(data, band_ix, W=None, names=TRUE_NAMES):
    """Для каждой сессии: mean logvar(L) - mean logvar(R) по каналам (d-prime-подобно). Медиана по сессиям."""
    diffs, dpr = [], []
    for n, d in data.items():
        c = d["covs"][:, band_ix].astype(np.float64).mean(1)      # (n,21,21)
        if W is not None:
            c = apply_spatial(c, W)
        lv = np.log(np.maximum(np.diagonal(c, axis1=-2, axis2=-1), 1e-30))
        L, R, Q = lv[d["y"] == 2], lv[d["y"] == 3], lv[d["y"] == 1]
        diffs.append(np.stack([L.mean(0) - R.mean(0), L.mean(0) - Q.mean(0), R.mean(0) - Q.mean(0)]))
        dpr.append((L.mean(0) - R.mean(0)) / np.sqrt(0.5 * (L.var(0) + R.var(0)) + 1e-12))
    diffs = np.array(diffs); dpr = np.array(dpr)
    med = np.median(diffs, 0)
    order = np.argsort(med[0])
    print(f"{'ch':6s} {'L-R':>7s} {'L-rest':>7s} {'R-rest':>7s} {'dprime':>7s} {'sign+%':>6s}")
    for i in order:
        print(f"{names[i]:6s} {med[0, i]:7.3f} {med[1, i]:7.3f} {med[2, i]:7.3f} "
              f"{np.median(dpr[:, i]):7.3f} {np.mean(diffs[:, 0, i] > 0) * 100:6.0f}")


if __name__ == "__main__":
    data = load_eeg_cache(2.0, f64=True)
    P = load_probs("ts2s_all")
    print(fmt("eeg_ts2s_all", detail(data, P)))
    print(fmt("eeg_ts2s_all_within", detail(data, load_probs("ts2s_all_within"))))
    print(fmt("eeg_ts2s_all_pool", detail(data, load_probs("ts2s_all_pool"))))
    # confusion суммарная
    G = np.zeros((3, 3), int)
    for n, p in P.items():
        d = data[n]; te = d["block"] >= 2
        for t, q in zip(d["y"][te], p[te].argmax(1) + 1):
            G[q - 1, t - 1] += 1
    print("confusion (строки — ответ, столбцы — истина):\n", G)
    print("\n== мю 8-13 Гц, отдельные каналы ==")
    erd_map(data, [1, 2])
    print("\n== бета 13-24 Гц ==")
    erd_map(data, [3, 4])
    lap = spatial_matrix("lap")
    from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import LAP
    print("\n== мю, лапласианы ==")
    erd_map(data, [1, 2], lap, [f"L{c}" for c in LAP])
    print("\n== бета, лапласианы ==")
    erd_map(data, [3, 4], lap, [f"L{c}" for c in LAP])
