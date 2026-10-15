"""Рисунки аудита артефактов → results/artifacts_audit/*.png (оффлайн-анализ).

  fig_bursts_example.png   — ВЧ-всплески в сессии N02 (2025.12.08): индекс по времени + спектры;
  fig_data_auc.png         — AUC «левая vs правая» и «покой vs воображение» по каналам × полосам
                             (медиана по 56 сессиям, данные, без модели);
  fig_model_patterns.png   — «паттерны» решения (Haufe et al. 2014): Спирмен между выходом ЭЭГ-части
                             (dev-прогон с --dump-probs) и лог-мощностью канала×полосы, медиана по сессиям;
  fig_topo.png             — топокарты мю/бета: данные (AUC L/R) и модель (паттерн L/R).

    python -m src.visualization.artifacts_maps --runs <dev_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from src.artifacts.audit import FR, HF, L, auc, burst_index  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BANDS = [("1–3", 1, 3), ("4–7", 4, 7), ("8–13 мю", 8, 13), ("14–20 β1", 14, 20), ("21–30 β2", 21, 30),
         ("31–45 γ/ЭМГ", 31, 45)]
ORDER = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 'c7', 'c3', 'cz', 'c4', 'c8',
         'p7', 'p3', 'pz', 'p4', 'p8', 'o1', 'o2', 'lpa', 'rpa']
# приблизительная 2D-раскладка 10–20 (нос вверх); c7/c8 — латеральные позиции T7/T8
POS = {'fp1': (-.3, .95), 'fp2': (.3, .95), 'f7': (-.8, .58), 'f3': (-.4, .5), 'fz': (0, .5), 'f4': (.4, .5),
       'f8': (.8, .58), 'c7': (-1, 0), 'c3': (-.5, 0), 'cz': (0, 0), 'c4': (.5, 0), 'c8': (1, 0),
       'p7': (-.8, -.58), 'p3': (-.4, -.5), 'pz': (0, -.5), 'p4': (.4, -.5), 'p8': (.8, -.58),
       'o1': (-.3, -.95), 'o2': (.3, -.95), 'lpa': (-1.12, -.2), 'rpa': (1.12, -.2)}
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#e66767", "#a8281f"])
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def bandpow(lp):
    return np.stack([lp[:, :, (FR >= lo) & (FR <= hi)].mean(2) for _, lo, hi in BANDS], 2)  # (n, C, B)


def heat(ax, M, title, vmax, fmt="{:.2f}", center=0.5):
    ci = [L.index(c) for c in ORDER]
    M = M[ci]
    im = ax.imshow(M.T, cmap=DIV, vmin=center - vmax, vmax=center + vmax, aspect="auto")
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=90)
    ax.set_yticks(range(len(BANDS)), [b[0] for b in BANDS])
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j] - center) >= 0.6 * vmax:
                ax.text(i, j, fmt.format(M[i, j]).lstrip("0").replace("-0.", "−."), ha="center", va="center",
                        fontsize=6.5, color="white")
    return im


def topo(ax, vals, title, vmax, center=0.5):
    xy = np.array([POS[c] for c in L])
    gx, gy = np.mgrid[-1.2:1.2:120j, -1.1:1.1:120j]
    gz = griddata(xy, vals, (gx, gy), method="cubic")
    gz[gx ** 2 + gy ** 2 > 1.21] = np.nan
    ax.contourf(gx, gy, gz, levels=np.linspace(center - vmax, center + vmax, 15), cmap=DIV, extend="both")
    ax.add_patch(plt.Circle((0, 0), 1.1, fill=False, color=INK2, lw=1))
    ax.plot([-.1, 0, .1], [1.08, 1.2, 1.08], color=INK2, lw=1)
    for c, (x, y) in POS.items():
        ax.plot(x, y, "o", ms=2.5, color=INK)
        if c in ("c3", "c4", "cz", "fp1", "fp2", "f7", "f8", "o1", "o2"):
            ax.text(x, y + .07, c.upper(), ha="center", fontsize=6.5, color=INK)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.2, 1.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, color=INK, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(ROOT / "data/features/artifacts_audit"))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/artifacts_audit"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A_lr, A_rm, P_lr, P_rm = [], [], [], []
    for fz in sorted(Path(a.feat).glob("*_N??.npz")):
        z = np.load(fz)
        y = z["y"]
        bp = bandpow(z["lp"].astype(np.float32))
        n, C, B = bp.shape
        A_lr.append(np.array([[auc(bp[y == 2, c, b], bp[y == 3, c, b]) for b in range(B)] for c in range(C)]))
        A_rm.append(np.array([[auc(bp[y > 1, c, b], bp[y == 1, c, b]) for b in range(B)] for c in range(C)]))
        pp = Path(a.runs) / fz.stem / "probs.csv"
        if pp.exists():
            p = pd.read_csv(pp).set_index("end_sample")
            pos = pd.Series(np.arange(n), index=z["ends"])
            p = p[p.index.isin(pos.index)]
            ii = pos.loc[p.index].values
            lr = np.log(p.eeg2 + 1e-12) - np.log(p.eeg3 + 1e-12)
            rm = np.log(p.eeg2 + p.eeg3 + 1e-12) - np.log(p.eeg1 + 1e-12)
            mi = y[ii] > 1

            def corr(v, X):
                rv = rankdata(v); rv = (rv - rv.mean()) / rv.std()
                R = np.apply_along_axis(rankdata, 0, X.reshape(len(X), -1))
                R = (R - R.mean(0)) / (R.std(0) + 1e-12)
                return (rv @ R / len(rv)).reshape(C, B)
            P_lr.append(corr(lr.values[mi], bp[ii][mi]))
            P_rm.append(corr(rm.values, bp[ii]))
    A_lr, A_rm = np.nanmedian(A_lr, 0), np.nanmedian(A_rm, 0)
    P_lr, P_rm = np.nanmedian(P_lr, 0), np.nanmedian(P_rm, 0)
    np.savez(out / "maps.npz", A_lr=A_lr, A_rm=A_rm, P_lr=P_lr, P_rm=P_rm, labels=L,
             bands=[b[0] for b in BANDS])

    fig, axs = plt.subplots(2, 1, figsize=(8.5, 5.6), constrained_layout=True)
    im = heat(axs[0], A_lr, "Данные: AUC «левая > правая» (медиана 56 сессий); синий у C4 — ERD при левой", 0.08)
    fig.colorbar(im, ax=axs[0], shrink=.8)
    im = heat(axs[1], A_rm, "Данные: AUC «воображение > покой»; синий — мощность ниже при воображении", 0.15)
    fig.colorbar(im, ax=axs[1], shrink=.8)
    fig.savefig(out / "fig_data_auc.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(8.5, 5.6), constrained_layout=True)
    im = heat(axs[0], P_lr, "Модель: ρ(logit L/R ЭЭГ-части, лог-мощность) на окнах воображения", 0.15, center=0)
    fig.colorbar(im, ax=axs[0], shrink=.8)
    im = heat(axs[1], P_rm, "Модель: ρ(logit воображение/покой, лог-мощность), все окна", 0.3, center=0)
    fig.colorbar(im, ax=axs[1], shrink=.8)
    fig.savefig(out / "fig_model_patterns.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(10, 3), constrained_layout=True)
    topo(axs[0], A_lr[:, 2], "Данные L>R, мю 8–13", 0.06)
    topo(axs[1], A_lr[:, 4], "Данные L>R, β2 21–30", 0.06)
    topo(axs[2], P_lr[:, 2], "Модель L/R, мю 8–13", 0.12, center=0)
    topo(axs[3], P_lr[:, 4], "Модель L/R, β2 21–30", 0.12, center=0)
    fig.suptitle("Левая рука против правой: синий над C4 — контралатеральная ERD, красный над C3/C7", color=INK, fontsize=9)
    fig.savefig(out / "fig_topo.png", dpi=150); plt.close(fig)

    # пример всплесков
    s = "2025.12.08.14.19.05_N02"
    z = np.load(Path(a.feat) / f"{s}.npz")
    lp = z["lp"].astype(np.float32); y, b = z["y"], z["block"]
    bi = burst_index(lp, y, b)
    t = z["ends"] / 250.0
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True, gridspec_kw={"width_ratios": [2.2, 1]})
    cols = {1: "#c3c2b7", 2: "#2a78d6", 3: "#eb6834"}
    for k, nm in ((1, "покой"), (2, "левая"), (3, "правая")):
        m = y == k
        axs[0].scatter(t[m], bi[m], s=2, color=cols[k], label=nm)
    axs[0].set_xlabel("время, с"); axs[0].set_ylabel("индекс ВЧ-всплеска, z")
    axs[0].set_title(f"{s}: ВЧ (32–85 Гц без линий) относительно покоя блока 1", loc="left", fontsize=9)
    axs[0].legend(frameon=False, markerscale=4, fontsize=8)
    for k in range(2, 8):
        axs[0].axvline(t[b == k][0], color="#e0dfdb", lw=0.8, zorder=0)
    T = z["trial_psd"]; ty, tb = z["trial_y"], z["trial_block"]
    f = np.arange(T.shape[2]) * 0.25
    c = L.index("cz")
    for sel, nm, col in (((ty > 1) & (tb >= 6), "воображение, блоки 6–7", "#e34948"),
                         ((ty == 1) & (tb >= 6), "покой, блоки 6–7", "#52514e"),
                         ((ty > 1) & (tb == 1), "воображение, блок 1", "#2a78d6")):
        axs[1].plot(f, T[sel][:, c].mean(0), lw=1.2, color=col, label=nm)
    axs[1].set_xlabel("частота, Гц"); axs[1].set_ylabel("log10 мощности (Cz)")
    axs[1].legend(frameon=False, fontsize=7)
    fig.savefig(out / "fig_bursts_example.png", dpi=150); plt.close(fig)
    print("ok", out)


if __name__ == "__main__":
    main()
