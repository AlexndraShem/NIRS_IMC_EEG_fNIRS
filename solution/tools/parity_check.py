"""Паритет онлайн-прогона `solution/` с оффлайн-эталонами (dev-инструмент, судьёй не используется).

Сравниваются вероятности по окнам блоков 2–7:

| выход | эталон |
|---|---|
| ЭЭГ   | `data/probs/eeg_ts2s_all.pkl`    (TS 2 с, 21 канал, within-sLDA(0.8) × пул-LR, a = 0.8) |
| NIRS  | `data/probs/nirs_cascade_ens.pkl` (каскад 8 × 4 sLDA) |
| гибрид| `data/probs/fused_stage1.pkl`     (0.75·log P_eeg + 0.25·log P_nirs + приор [0,2,2] + EMA τ=1.5 с) |

Онлайн-вероятности берутся из `probs.csv` (нужен прогон `run.py --dump-probs`): столбцы
`eeg*`/`nirs*` — вероятности модальностей (сравниваются с сырыми эталонами), `hyb*` — итог
гибрида после EMA (сравнивается с `fused_stage1`). Ответы (`y`, `y_eeg`, `y_nirs`) сравниваются
с эталоном, к которому применены те же приор и EMA, что и в онлайне (см. config.json).

Эталонные пулы обучены leave-one-subject-out, поэтому прогон для паритета делается с
`--loso-artifacts data/features/eeg_models` (иначе ЭЭГ-часть будет отличаться законно).

    python solution/tools/dev_run.py --input data/raw/Data --output-dir <runs> --jobs 6 --dump-probs \
                                     --loso-artifacts data/features/eeg_models
    python solution/tools/parity_check.py --runs-dir <runs> [--out results/metrics/parity_offline.csv]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = {"eeg": "eeg_ts2s_all", "nirs": "nirs_cascade_ens", "hyb": "fused_stage1"}


def load_ref(name):
    p = ROOT / "data" / "probs" / f"{name}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else {}


def read_probs(path):
    rows = list(csv.DictReader(open(path, newline="")))
    out = {}
    for r in rows:
        d = {}
        for m in ("eeg", "nirs", "hyb"):
            v = [r[f"{m}{k}"] for k in (1, 2, 3)]
            d[m] = None if any(x == "" for x in v) else np.array([float(x) for x in v])
        out[int(r["end_sample"])] = d
    return out


def ema(P, ends, tau, fs=250.0):
    """Причинное EMA эталонных вероятностей (как в онлайне): a = exp(−Δt/τ)."""
    Q = np.full_like(P, np.nan)
    prev = None
    for i in range(len(P)):
        if not np.isfinite(P[i]).all():
            continue
        if prev is None:
            Q[i] = P[i]
        else:
            a = np.exp(-(ends[i] - ends[prev]) / fs / tau)
            Q[i] = a * Q[prev] + (1 - a) * P[i]
        prev = i
    return Q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--features", default=str(ROOT / "data/features/v1_win1"))
    ap.add_argument("--config", default=str(ROOT / "solution/artifacts/config.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fus = json.loads(Path(a.config).read_text()).get("fusion_params", {})
    tau = float(fus.get("tau_s", 1.5))
    prior = np.asarray(fus.get("prior", [0.0, 2.0, 2.0]), float)
    nprior = np.asarray(fus.get("nirs_prior", [0.0, 1.0, 1.0]), float)
    refs = {k: load_ref(v) for k, v in REF.items()}
    rows = []
    for d in sorted(p for p in Path(a.runs_dir).iterdir() if (p / "predictions.csv").exists()):
        fpath = Path(a.features) / f"{d.name}.npz"
        if not fpath.exists():
            continue
        z = np.load(fpath)
        te = z["block"] >= 2
        ends = z["ends"][te]
        pred = {int(r["end_sample"]): r for r in csv.DictReader(open(d / "predictions.csv", newline=""))}
        assert set(pred) == set(ends.tolist()), f"{d.name}: сетки окон не совпали"
        pr = read_probs(d / "probs.csv") if (d / "probs.csv").exists() else None
        r = dict(session=d.name, n=int(te.sum()))
        for key, col, onl in (("eeg", "y_eeg", "eeg"), ("nirs", "y_nirs", "nirs"), ("hyb", "y", "hyb")):
            R = refs[key].get(d.name)
            if R is None:
                continue
            Roff = np.asarray(R[te], np.float64)
            ok = np.isfinite(Roff).all(1)
            ref_p = Roff                 # с чем сравнивать вероятности из probs.csv
            if key == "eeg":             # ответ онлайна — argmax после приора и EMA
                Z = np.where(ok[:, None], np.log(np.clip(Roff, 1e-9, 1)) + prior, np.nan)
                Q = np.exp(Z - np.nanmax(Z, 1, keepdims=True)); Q /= np.nansum(Q, 1, keepdims=True)
                dec = ema(Q, ends, tau) if tau else Q
            elif key == "nirs":          # только приор, без сглаживания
                dec = np.where(ok[:, None], np.log(np.clip(Roff, 1e-9, 1)) + nprior, np.nan)
            else:                        # гибрид: эталон уже с приором и EMA
                dec = Roff
            y_off_dec = np.where(ok, np.nan_to_num(dec, nan=-np.inf).argmax(1) + 1, 0)
            y_on = np.array([int(pred[int(e)][col] or 0) for e in ends])
            m = ok & (y_on > 0)
            r[f"{key}_agree_y"] = float(np.mean(y_off_dec[m] == y_on[m])) if m.any() else float("nan")
            if pr is not None:
                P_on = np.array([pr[int(e)][onl] if pr[int(e)][onl] is not None else [np.nan] * 3 for e in ends])
                mm = m & np.isfinite(P_on).all(1) & np.isfinite(ref_p).all(1)
                r[f"{key}_maxdP"] = float(np.max(np.abs(P_on[mm] - ref_p[mm]))) if mm.any() else float("nan")
                r[f"{key}_agree_p"] = float(np.mean(P_on[mm].argmax(1) == ref_p[mm].argmax(1))) if mm.any() else float("nan")
        rows.append(r)
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    keys = sorted({k for r in rows for k in r if k not in ("session", "n")})
    print("\nСВОДКА по сессиям:")
    for k in keys:
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        if not v:
            continue
        agg = np.max(v) if k.endswith("maxdP") else np.mean(v)
        print(f"  {k:16s} {'max' if k.endswith('maxdP') else 'mean'}={agg:.6f}  (n={len(v)}, "
              f"min={np.min(v):.6f}, худшая сессия: {rows[int(np.argmin([r.get(k, np.inf) for r in rows]))]['session']})")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, ["session", "n"] + keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
