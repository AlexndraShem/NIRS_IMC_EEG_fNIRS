"""Опирается ли решение на артефакты? Внутриклассовые корреляции (Спирмен) выходов ЭЭГ-части
решения с индексами артефактов и с нейрофизиологическими индексами (оффлайн-анализ).

Вход — dev-прогоны официальной сетки с `--dump-probs` (probs.csv: eeg1..3) и признаки
`src.artifacts.extract` / `src.artifacts.eog`.

  logit_rm = log((P2+P3)/P1)  — «воображение против покоя», корреляции считаются ВНУТРИ класса
  logit_lr = log(P2/P3)       — «левая против правой», внутри окон воображения каждого класса.

    python -m src.artifacts.reliance --runs <dev_dir> --out results/artifacts_audit/reliance_<proto>.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.artifacts.audit import FRONTAL, HF, LEFT, RIGHT, band, burst_index

ROOT = Path(__file__).resolve().parents[2]


def partial_rho(q, a, b, ctrl):
    """Спирмен a~b после линейного вычитания рангов контрольных переменных."""
    R = np.column_stack([rankdata(q[c]) for c in ctrl] + [np.ones(len(q))])
    ra, rb = rankdata(q[a]), rankdata(q[b])
    ea = ra - R @ np.linalg.lstsq(R, ra, rcond=None)[0]
    eb = rb - R @ np.linalg.lstsq(R, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--feat", default=str(ROOT / "data/features/artifacts_audit"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--col", default="eeg", help="eeg | hyb")
    a = ap.parse_args()
    rows = []
    for d in sorted(Path(a.runs).iterdir()):
        pp = d / "probs.csv"
        fz = Path(a.feat) / f"{d.name}.npz"
        if not pp.exists() or not fz.exists():
            continue
        z = np.load(fz); e = np.load(Path(a.feat) / f"{d.name}_eog.npz")
        lp = z["lp"].astype(np.float32); y = z["y"]; b = z["block"]
        idx = pd.DataFrame({
            "y": y,
            "burst": burst_index(lp, y, b),
            "front_low": band(lp, FRONTAL, 1, 3),
            "veog_ptp": np.log(e["ptp"][:, 1] + 1e-9),
            "heog_mean": e["mean"][:, 0],
            "hf_lat": band(lp, RIGHT, 32, 45) - band(lp, LEFT, 32, 45),
            "mu_lat": band(lp, ['c4'], 8, 13) - band(lp, ['c3'], 8, 13),
            "beta_lat": band(lp, ['c4'], 14, 30) - band(lp, ['c3'], 14, 30),
            "sm_mu": band(lp, ['c3', 'c4'], 8, 13),
            "sm_beta": band(lp, ['c3', 'c4'], 14, 30),
        }, index=z["ends"])
        p = pd.read_csv(pp).set_index("end_sample")
        c = a.col
        p = p.join(idx, how="inner")
        eps = 1e-12
        p["logit_rm"] = np.log(p[f"{c}2"] + p[f"{c}3"] + eps) - np.log(p[f"{c}1"] + eps)
        p["logit_lr"] = np.log(p[f"{c}2"] + eps) - np.log(p[f"{c}3"] + eps)
        r = dict(session=d.name)
        rest = p[p.y == 1]
        for k in ("burst", "front_low", "veog_ptp", "sm_mu", "sm_beta"):
            r[f"rest:rm~{k}"] = spearmanr(rest.logit_rm, rest[k]).correlation
        for k in ("burst", "front_low", "veog_ptp"):
            r[f"rest:rm~{k}|sm"] = partial_rho(rest, "logit_rm", k, ["sm_mu", "sm_beta"])
        for cls, nm in ((2, "L"), (3, "R")):
            q = p[p.y == cls]
            for k in ("heog_mean", "hf_lat", "mu_lat", "beta_lat"):
                r[f"{nm}:lr~{k}"] = spearmanr(q.logit_lr, q[k]).correlation
            for k in ("burst", "front_low", "veog_ptp", "sm_mu", "sm_beta"):
                r[f"{nm}:rm~{k}"] = spearmanr(q.logit_rm, q[k]).correlation
            for k in ("burst", "front_low", "veog_ptp"):
                r[f"{nm}:rm~{k}|sm"] = partial_rho(q, "logit_rm", k, ["sm_mu", "sm_beta"])
            r[f"{nm}:lr~heog_mean|lat"] = partial_rho(q, "logit_lr", "heog_mean", ["mu_lat", "beta_lat"])
            r[f"{nm}:lr~mu_lat|heog"] = partial_rho(q, "logit_lr", "mu_lat", ["heog_mean"])
        rows.append(r)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    s = df.drop(columns="session")
    summ = pd.DataFrame({"median_rho": s.median(), "mean_rho": s.mean(),
                         "frac_|rho|>0.2": (s.abs() > 0.2).mean()})
    summ.to_csv(Path(a.out).with_name(Path(a.out).stem + "_summary.csv"))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
