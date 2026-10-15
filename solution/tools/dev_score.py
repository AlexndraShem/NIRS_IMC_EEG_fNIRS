"""Подсчёт метрик по ТЗ §5 (реконструкция; официальный замороженный score.py не выдан).

Одна сессия:
    python solution/score.py --session <session.mat> --predictions <predictions.csv>
                             --timings <timings.csv> --output <metrics.json>

Агрегат по многим сессиям (каталог с результатами run.py в режиме каталога):
    python solution/score.py --runs-dir <out> [--sessions-dir <каталог .mat>] --output-dir results/metrics

Матрица ошибок G: строки — ответы классификатора, столбцы — истинные классы.
F1_k = 2 g_kk / (Σ_j g_kj + Σ_j g_jk);  P = 1/3 Σ_k g_kk / Σ_i g_ik;  F = min_k F1_k.
Считается для гибрида (y), отдельно для NIRS (y_nirs) и справочно для ЭЭГ (y_eeg).
Агрегат (§5.2) — среднее по сессиям (NIRS — по сессиям, где есть y_nirs); матрицы не суммируются.
Балл (§5.4, без дизайна/отчёта): 60·F + 30·P + [5 + 15·F_nirs] − штрафы времени.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PREDICT_LIMIT, PREDICT_STEP, PREDICT_CAP = 0.20, 0.05, 20
FIT_LIMIT, FIT_STEP, FIT_CAP = 1.5, 0.5, 10


def confusion(y_true, y_pred, k=3):
    G = np.zeros((k, k), int)
    for t, p in zip(y_true, y_pred):
        G[int(p) - 1, int(t) - 1] += 1
    return G


def metrics_from_G(G):
    G = np.asarray(G)
    d = np.diag(G).astype(float)
    f1 = 2 * d / np.maximum(G.sum(1) + G.sum(0), 1)
    rec = d / np.maximum(G.sum(0), 1)
    return dict(G=G.tolist(), f1=f1.round(6).tolist(), recall_k=rec.round(6).tolist(),
                min_f1=float(f1.min()), recall=float(rec.mean()))


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def timing_stats(rows):
    out = {}
    for m in sorted({r["method"] for r in rows}):
        t = np.array([float(r["seconds"]) for r in rows if r["method"] == m])
        out[m] = dict(n=int(len(t)), mean=float(t.mean()), p50=float(np.percentile(t, 50)),
                      p95=float(np.percentile(t, 95)), max=float(t.max()))
    return out


def penalty(t, limit, step, cap):
    """Кейсодержатель (2026-09-23): штрафуются только полные превышения — −1 за каждые полные
    `step` сверх `limit` (первый балл при limit + step), по СРЕДНЕМУ времени вызова."""
    return float(min(cap, math.floor(max(0.0, t - limit) / step + 1e-12))) if t > limit else 0.0


def score_session(pred_path, timings_path=None, session_path=None) -> dict:
    rows = _read_csv(pred_path)
    ends = np.array([int(r["end_sample"]) for r in rows])
    yt = np.array([int(r["true_y"]) for r in rows])
    res = dict(n_windows=int(len(rows)))
    checks = {}
    if session_path:
        from session_io import load_session, window_ends
        s = load_session(session_path)
        res["session"] = s["name"]
        e_ref, y_ref, b_ref = window_ends(s["states"], s["blocks"], s["eeg_fs"])
        e_ref, y_ref = e_ref[b_ref >= 2], y_ref[b_ref >= 2]
        ref = dict(zip(e_ref.tolist(), y_ref.tolist()))
        # истинные метки берём из сессии, а не из CSV
        yt_ref = np.array([ref.get(int(e), -1) for e in ends])
        checks = dict(expected_windows=int(len(e_ref)), missing_windows=int(len(set(ref) - set(ends.tolist()))),
                      extra_windows=int(np.sum(yt_ref < 0)), true_y_mismatch=int(np.sum((yt_ref != yt) & (yt_ref > 0))))
        ok = yt_ref > 0
        rows = [r for r, o in zip(rows, ok) if o]
        yt = yt_ref[ok]
    else:
        res["session"] = Path(pred_path).parent.name
    res["checks"] = checks
    for key, col in (("hybrid", "y"), ("eeg", "y_eeg"), ("nirs", "y_nirs")):
        yp = [_to_int(r.get(col)) for r in rows]
        valid = [p is not None and 1 <= p <= 3 for p in yp]
        if not rows or not all(valid):
            res[key] = None
            continue
        res[key] = metrics_from_G(confusion(yt, yp))
    if timings_path and Path(timings_path).exists():
        res["timings"] = timing_stats(_read_csv(timings_path))
    h, nr = res["hybrid"], res["nirs"]
    res["points_60F_30P"] = 60 * h["min_f1"] + 30 * h["recall"] if h else None
    res["points_nirs"] = 5 + 15 * nr["min_f1"] if nr else 0.0
    return res


def aggregate(per_session: list[dict]) -> dict:
    def mean_of(key, field):
        v = [r[key][field] for r in per_session if r.get(key)]
        return (float(np.mean(v)) if v else None), len(v)

    F, n = mean_of("hybrid", "min_f1")
    P, _ = mean_of("hybrid", "recall")
    Fe, ne = mean_of("eeg", "min_f1")
    Pe, _ = mean_of("eeg", "recall")
    Fn, nn = mean_of("nirs", "min_f1")
    Pn, _ = mean_of("nirs", "recall")
    tim = {}
    for m in ("predict", "fit_block", "push"):
        st = [r["timings"][m] for r in per_session if r.get("timings", {}).get(m)]
        if st:
            tim[m] = dict(max=float(max(s["max"] for s in st)), mean=float(np.mean([s["mean"] for s in st])),
                          mean_max_over_sessions=float(max(s["mean"] for s in st)),
                          p95_max_over_sessions=float(max(s["p95"] for s in st)),
                          p50_median_over_sessions=float(np.median([s["p50"] for s in st])))
    # по среднему; консервативно — худшее по сессиям среднее (как именно судья агрегирует — не уточняли)
    pen_pred = penalty(tim["predict"]["mean_max_over_sessions"], PREDICT_LIMIT, PREDICT_STEP, PREDICT_CAP) \
        if "predict" in tim else 0.0
    pen_fit = penalty(tim["fit_block"]["mean_max_over_sessions"], FIT_LIMIT, FIT_STEP, FIT_CAP) \
        if "fit_block" in tim else 0.0
    pen_pred_max = penalty(tim["predict"]["max"], PREDICT_LIMIT, PREDICT_STEP, PREDICT_CAP) if "predict" in tim else 0.0
    pen_fit_max = penalty(tim["fit_block"]["max"], FIT_LIMIT, FIT_STEP, FIT_CAP) if "fit_block" in tim else 0.0
    base = 60 * F + 30 * P if F is not None else None
    nirs_pts = 5 + 15 * Fn if Fn is not None else 0.0
    return dict(
        n_sessions=n, F_global=F, P_global=P,
        F_global_nirs=Fn, P_global_nirs=Pn, n_sessions_nirs=nn,
        F_global_eeg=Fe, P_global_eeg=Pe, n_sessions_eeg=ne,
        points=dict(hybrid_60F_30P=base, nirs=nirs_pts,
                    penalty_predict=pen_pred, penalty_fit_block=pen_fit,
                    penalty_predict_if_by_max=pen_pred_max, penalty_fit_block_if_by_max=pen_fit_max,
                    total_without_design=(base + nirs_pts - pen_pred - pen_fit) if base is not None else None),
        timings=tim,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session")
    ap.add_argument("--predictions")
    ap.add_argument("--timings")
    ap.add_argument("--output")
    ap.add_argument("--runs-dir", help="каталог с <сессия>/predictions.csv (режим агрегата)")
    ap.add_argument("--sessions-dir", help="каталог с .mat (для сверки сетки окон и меток)")
    ap.add_argument("--output-dir", help="куда писать metrics.json, per_session/*.json, summary.csv")
    ap.add_argument("--note", default=None, help="пометка в metrics.json (условия прогона, артефакты)")
    a = ap.parse_args(argv)

    if a.runs_dir:
        out = Path(a.output_dir or a.runs_dir)
        (out / "per_session").mkdir(parents=True, exist_ok=True)
        per = []
        for d in sorted(p for p in Path(a.runs_dir).iterdir() if (p / "predictions.csv").exists()):
            sp = Path(a.sessions_dir) / f"{d.name}.mat" if a.sessions_dir else None
            r = score_session(d / "predictions.csv", d / "timings.csv", sp if sp and sp.exists() else None)
            r["session"] = d.name
            (out / "per_session" / f"{d.name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
            per.append(r)
        agg = aggregate(per)
        if a.note:
            agg = {"note": a.note, **agg}
        (out / "metrics.json").write_text(json.dumps(agg, indent=1, ensure_ascii=False))
        write_summary(per, out / "summary.csv")
        print(json.dumps({k: agg[k] for k in ("n_sessions", "F_global", "P_global", "F_global_nirs",
                                             "P_global_nirs", "F_global_eeg", "P_global_eeg", "points")},
                         indent=1, ensure_ascii=False))
        return 0

    if not a.predictions:
        ap.error("нужен --predictions (или --runs-dir)")
    r = score_session(a.predictions, a.timings, a.session)
    txt = json.dumps(r, indent=1, ensure_ascii=False)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(txt)
    print(txt)
    return 0


def write_summary(per, path):
    cols = ["session", "n_windows", "hyb_min_f1", "hyb_recall", "eeg_min_f1", "eeg_recall", "nirs_min_f1",
            "nirs_recall", "predict_p50", "predict_p95", "predict_max", "fit_block_max",
            "G_hybrid", "G_nirs"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in per:
            def g(k, fld):
                return "" if not r.get(k) else round(r[k][fld], 4)
            t = r.get("timings", {})
            w.writerow([r["session"], r["n_windows"], g("hybrid", "min_f1"), g("hybrid", "recall"),
                        g("eeg", "min_f1"), g("eeg", "recall"), g("nirs", "min_f1"), g("nirs", "recall"),
                        round(t.get("predict", {}).get("p50", float("nan")), 5),
                        round(t.get("predict", {}).get("p95", float("nan")), 5),
                        round(t.get("predict", {}).get("max", float("nan")), 5),
                        round(t.get("fit_block", {}).get("max", float("nan")), 5),
                        json.dumps(r["hybrid"]["G"]) if r.get("hybrid") else "",
                        json.dumps(r["nirs"]["G"]) if r.get("nirs") else ""])
        # строка среднего (§5.2)
        agg = aggregate(per)
        w.writerow(["MEAN", "", round(agg["F_global"], 4), round(agg["P_global"], 4),
                    round(agg["F_global_eeg"], 4) if agg["F_global_eeg"] is not None else "",
                    round(agg["P_global_eeg"], 4) if agg["P_global_eeg"] is not None else "",
                    round(agg["F_global_nirs"], 4) if agg["F_global_nirs"] is not None else "",
                    round(agg["P_global_nirs"], 4) if agg["P_global_nirs"] is not None else "",
                    "", "", "", "", "", ""])


if __name__ == "__main__":
    sys.exit(main())
