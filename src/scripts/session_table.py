"""Таблица показателей по сессиям (ТЗ §5.2) из прогона официального каркаса.

    python -m src.scripts.session_table [--metrics-dir results/metrics] [--md]

Читает <metrics-dir>/per_session/*.json (официальный score.py) и <metrics-dir>/runs/<S>/predictions.csv
(ЭЭГ-only — по колонке y_eeg). Пишет <metrics-dir>/summary.csv (строка MEAN — среднее по сессиям,
как в §5.2); с --md печатает markdown-таблицу для отчёта.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "solution"))
from score import confusion_from_preds, metrics_block  # noqa: E402


def build(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((metrics_dir / "per_session").glob("*.json")):
        m = json.loads(f.read_text())
        pr = pd.read_csv(metrics_dir / "runs" / f.stem / "predictions.csv")
        e = metrics_block(confusion_from_preds(pr.true_y.values, pr.y_eeg.values.astype(int)))
        n = m.get("nirs_only") or {}
        rows.append(dict(
            session=f.stem, subject=f.stem.split("_")[-1], n_windows=m["hybrid"]["n_windows"],
            hyb_min_f1=m["hybrid"]["min_f1"], hyb_recall=m["hybrid"]["macro_recall"],
            eeg_min_f1=e["min_f1"], eeg_recall=e["macro_recall"],
            nirs_min_f1=n.get("min_f1", np.nan), nirs_recall=n.get("macro_recall", np.nan),
            score_total=m["score"]["total"], predict_max_s=m["timings"]["predict_max_s"],
            fit_max_s=m["timings"]["fit_max_s"], penalties=m["timings"]["penalty_total"],
            G_hybrid=json.dumps(m["hybrid"]["confusion"]),
            G_nirs=json.dumps(n["confusion"]) if n else ""))
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    mean = {c: df[c].mean() for c in num}
    mean.update(session="MEAN", subject="", G_hybrid="", G_nirs="")
    mean["predict_max_s"], mean["fit_max_s"] = df.predict_max_s.max(), df.fit_max_s.max()
    return pd.concat([df, pd.DataFrame([mean])], ignore_index=True)


def to_md(df: pd.DataFrame) -> str:
    f3 = lambda v: "—" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
    out = ["| # | сессия | участник | гибрид minF1 | гибрид recall | ЭЭГ-only minF1 | ЭЭГ-only recall "
           "| NIRS-only minF1 | NIRS-only recall | балл сессии |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in df.iterrows():
        last = r.session == "MEAN"
        name = "**среднее по 56 сессиям**" if last else r.session
        b = (lambda s: f"**{s}**") if last else (lambda s: s)
        out.append(f"| {'' if last else i + 1} | {name} | {r.subject} | " + " | ".join(
            b(f3(r[c])) for c in ("hyb_min_f1", "hyb_recall", "eeg_min_f1", "eeg_recall", "nirs_min_f1",
                                  "nirs_recall")) + f" | {b(f'{r.score_total:.2f}')} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default=str(ROOT / "results" / "metrics"))
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    d = Path(a.metrics_dir)
    df = build(d)
    df.to_csv(d / "summary.csv", index=False, float_format="%.6g")
    print("written", d / "summary.csv", len(df) - 1, "sessions")
    if a.md:
        print(to_md(df))


if __name__ == "__main__":
    main()
