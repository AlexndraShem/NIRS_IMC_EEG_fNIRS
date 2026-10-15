"""Замороженный подсчёт метрик по predictions.csv и timings.csv.

    python score.py --session session.mat --predictions predictions.csv \\
                    --timings timings.csv --output metrics.json

НЕ ИЗМЕНЯЙТЕ этот файл. Судья подменяет его эталонной копией.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import io_utils as iu  # noqa: E402

CLASSES = (1, 2, 3)
PREDICT_BUDGET_S = 0.20
FIT_BUDGET_S = 1.5
PREDICT_PENALTY_STEP_S = 0.05  # −1 балл за каждые 0.05 с сверх бюджета
FIT_PENALTY_STEP_S = 0.5      # −1 балл за каждые 0.5 с сверх бюджета
MAX_PREDICT_PENALTY = 20
MAX_FIT_PENALTY = 10


def confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """G[i, j] = число окон истинного класса (j+1), предсказанных как (i+1).
    Строки — ответы, столбцы — истинные классы.
    """
    g = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if t in CLASSES and p in CLASSES:
            g[p - 1, t - 1] += 1
    return g


def f1_per_class(g: np.ndarray) -> np.ndarray:
    f1 = np.zeros(3, dtype=float)
    for k in range(3):
        tp = float(g[k, k])
        row = float(g[k, :].sum())
        col = float(g[:, k].sum())
        denom = row + col
        f1[k] = (2.0 * tp / denom) if denom > 0 else 0.0
    return f1


def macro_recall(g: np.ndarray) -> float:
    recalls = []
    for k in range(3):
        col = float(g[:, k].sum())
        recalls.append((float(g[k, k]) / col) if col > 0 else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def metrics_block(g: np.ndarray) -> dict:
    f1 = f1_per_class(g)
    return {
        "confusion": g.tolist(),
        "f1": {str(k): float(f1[k - 1]) for k in CLASSES},
        "min_f1": float(f1.min()),
        "macro_recall": macro_recall(g),
        "n_windows": int(g.sum()),
    }


def time_penalties(timings: pd.DataFrame) -> dict:
    pred = timings.loc[timings["event"] == "predict", "seconds"].astype(float)
    fit = timings.loc[timings["event"] == "fit_block", "seconds"].astype(float)

    pred_over = pred[pred > PREDICT_BUDGET_S] - PREDICT_BUDGET_S
    fit_over = fit[fit > FIT_BUDGET_S] - FIT_BUDGET_S

    # суммарное превышение, квантованное
    pred_pen = int(math.floor(float(pred_over.sum()) / PREDICT_PENALTY_STEP_S)) if len(pred_over) else 0
    fit_pen = int(math.floor(float(fit_over.sum()) / FIT_PENALTY_STEP_S)) if len(fit_over) else 0
    pred_pen = min(pred_pen, MAX_PREDICT_PENALTY)
    fit_pen = min(fit_pen, MAX_FIT_PENALTY)

    return {
        "predict_mean_s": float(pred.mean()) if len(pred) else 0.0,
        "predict_max_s": float(pred.max()) if len(pred) else 0.0,
        "fit_mean_s": float(fit.mean()) if len(fit) else 0.0,
        "fit_max_s": float(fit.max()) if len(fit) else 0.0,
        "n_predict": int(len(pred)),
        "n_fit": int(len(fit)),
        "penalty_predict": pred_pen,
        "penalty_fit": fit_pen,
        "penalty_total": pred_pen + fit_pen,
    }


def nirs_bonus(nirs_metrics: dict | None) -> dict:
    """До 20 баллов: 5 за валидную матрицу + 15 × min-F1 NIRS (F_global^NIRS)."""
    if nirs_metrics is None or nirs_metrics.get("n_windows", 0) <= 0:
        return {"present": False, "points": 0.0, "detail": "нет y_nirs"}
    if "confusion" in nirs_metrics:
        g = np.asarray(nirs_metrics["confusion"])
        if g.shape != (3, 3) or int(g.sum()) == 0:
            return {"present": False, "points": 0.0, "detail": "пустая матрица"}
    elif "min_f1" not in nirs_metrics:
        return {"present": False, "points": 0.0, "detail": "нет матрицы/min_f1"}
    min_f1 = float(nirs_metrics["min_f1"])
    points = 5.0 + 15.0 * max(0.0, min(1.0, min_f1))
    return {
        "present": True,
        "points": round(points, 4),
        "min_f1": min_f1,
        "macro_recall": float(nirs_metrics.get("macro_recall", 0.0)),
        "detail": "валидная NIRS-only матрица",
    }


def aggregate_session_metrics(session_results: list[dict]) -> dict:
    """Невзвешенное среднее метрик по сессиям. Не суммирует G.

    Каждый элемент — результат score_files / metrics.json одной сессии.
    """
    if not session_results:
        raise ValueError("пустой список сессий")

    def _mean(key_path: tuple[str, ...]) -> float | None:
        vals: list[float] = []
        for r in session_results:
            node = r
            for k in key_path:
                if node is None or k not in node:
                    node = None
                    break
                node = node[k]
            if node is None:
                continue
            vals.append(float(node))
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    hybrid_min_f1 = _mean(("hybrid", "min_f1"))
    hybrid_recall = _mean(("hybrid", "macro_recall"))
    nirs_min_f1 = _mean(("nirs_only", "min_f1"))
    nirs_recall = _mean(("nirs_only", "macro_recall"))

    total_windows = sum(int(r["hybrid"]["n_windows"]) for r in session_results)
    pred_pen = min(
        sum(int(r.get("timings", {}).get("penalty_predict", 0)) for r in session_results),
        MAX_PREDICT_PENALTY,
    )
    fit_pen = min(
        sum(int(r.get("timings", {}).get("penalty_fit", 0)) for r in session_results),
        MAX_FIT_PENALTY,
    )
    pens = {
        "penalty_predict": pred_pen,
        "penalty_fit": fit_pen,
        "penalty_total": pred_pen + fit_pen,
    }

    hybrid = {
        "min_f1": hybrid_min_f1 if hybrid_min_f1 is not None else 0.0,
        "macro_recall": hybrid_recall if hybrid_recall is not None else 0.0,
        "n_windows": total_windows,
        "aggregation": "unweighted_mean_over_sessions",
        "n_sessions": len(session_results),
    }
    nirs_block = None
    if nirs_min_f1 is not None:
        nirs_block = {
            "min_f1": nirs_min_f1,
            "macro_recall": nirs_recall,
            "n_windows": sum(
                int(r["nirs_only"]["n_windows"])
                for r in session_results
                if r.get("nirs_only")
            ),
        }

    sc = final_score(hybrid, nirs_block, pens, engineering=0.0)
    return {
        "status": "ok",
        "aggregation": "unweighted_mean_over_sessions",
        "n_sessions": len(session_results),
        "hybrid": hybrid,
        "nirs_only": nirs_block,
        "timings": pens,
        "score": sc,
        "note": "Матрицы G не суммируются; ЭЭГ-only не входит в балл; см. per-session metrics.json",
    }


def final_score(
    hybrid: dict,
    nirs: dict | None,
    penalties: dict,
    engineering: float = 0.0,
) -> dict:
    """База 130: 60*F_global + 30*P_global + nirs_bonus(≤20) + engineering(≤20) − penalties."""
    base_quality = 60.0 * hybrid["min_f1"] + 30.0 * hybrid["macro_recall"]
    nb = nirs_bonus(nirs)
    eng = max(0.0, min(20.0, float(engineering)))
    pen = int(penalties.get("penalty_total", 0))
    total = base_quality + nb["points"] + eng - pen
    return {
        "base_quality": round(base_quality, 4),
        "nirs_bonus": nb,
        "engineering": eng,
        "penalties": pen,
        "total": round(total, 4),
        "max_possible": 130.0,
    }


def score_files(
    predictions_csv: Path,
    timings_csv: Path,
    engineering: float = 0.0,
    session_path: Path | None = None,
) -> dict:
    preds = pd.read_csv(predictions_csv)
    timings = pd.read_csv(timings_csv)

    required = {"true_y", "y"}
    if not required.issubset(preds.columns):
        raise ValueError(f"в predictions.csv нет колонок {required}")

    y_true = preds["true_y"].to_numpy()
    y_pred = preds["y"].to_numpy()
    hybrid = metrics_block(confusion_from_preds(y_true, y_pred))

    nirs_m = None
    if "y_nirs" in preds.columns and preds["y_nirs"].notna().any():
        nirs_m = metrics_block(
            confusion_from_preds(y_true, preds["y_nirs"].fillna(-1).to_numpy())
        )

    pens = time_penalties(timings)
    sc = final_score(hybrid, nirs_m, pens, engineering=engineering)

    out = {
        "status": "ok",
        "session": str(session_path) if session_path else None,
        "hybrid": hybrid,
        "nirs_only": nirs_m,
        "timings": pens,
        "score": sc,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Подсчёт метрик хакатона FES EEG+NIRS")
    ap.add_argument("--session", default=None, help="путь к .mat (для метаданных)")
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--timings", default=None)
    ap.add_argument("--output", default="metrics.json")
    ap.add_argument("--engineering", type=float, default=0.0, help="баллы инженерии 0..20")
    ap.add_argument(
        "--aggregate",
        nargs="+",
        default=None,
        help="пути к per-session metrics.json -> невзвешенное среднее (без суммирования G)",
    )
    args = ap.parse_args()

    if args.aggregate:
        sessions = [
            json.loads(Path(p).read_text(encoding="utf-8")) for p in args.aggregate
        ]
        result = aggregate_session_metrics(sessions)
        if args.engineering:
            result["score"] = final_score(
                result["hybrid"],
                result.get("nirs_only"),
                result["timings"],
                engineering=args.engineering,
            )
    else:
        if not args.predictions or not args.timings:
            ap.error("нужны --predictions и --timings (или --aggregate)")
        result = score_files(
            Path(args.predictions),
            Path(args.timings),
            engineering=args.engineering,
            session_path=Path(args.session) if args.session else None,
        )
        if args.session:
            try:
                s = iu.load_session(args.session)
                n_exp = sum(
                    1
                    for _ in iu.iter_scored_windows(
                        s["states"], s["blocks"], s["fs_eeg"]
                    )
                )
                result["expected_windows"] = n_exp
                result["windows_match"] = n_exp == result["hybrid"]["n_windows"]
            except Exception as exc:
                result["session_check_error"] = str(exc)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.get("score", result), ensure_ascii=False, indent=2))
    print(f"Записано: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
