"""Онлайн-симуляция на каталоге сессий + метрики (ТЗ §5.3): run.py для каждой сессии → score.py.

    python solution/run_all.py --input <каталог .mat> [--runs-dir results/online_runs]
                               [--metrics-dir results/metrics] [--jobs 6] [--artifacts solution/artifacts]

Результат:
  <runs-dir>/<сессия>/{predictions.csv, timings.csv}
  <metrics-dir>/metrics.json            — агрегат §5.2 (среднее по сессиям) + баллы + времена
  <metrics-dir>/per_session/<сессия>.json — метрики и матрицы ошибок G (гибрид, ЭЭГ, NIRS)
  <metrics-dir>/summary.csv             — сводная таблица по сессиям + строка MEAN
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]   # каталог solution/
sys.path.insert(0, str(HERE))

import dev_run as run_mod  # noqa: E402
import dev_score as score_mod  # noqa: E402
from session_io import list_sessions  # noqa: E402


def main(argv=None):
    root = HERE.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="каталог с .mat (или один .mat)")
    ap.add_argument("--runs-dir", default=str(root / "results" / "online_runs"))
    ap.add_argument("--metrics-dir", default=str(root / "results" / "metrics"))
    ap.add_argument("--artifacts", default=str(HERE / "artifacts"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--loso-artifacts", default=None,
                    help="dev: каталог пулов leave-one-subject-out (data/features/eeg_models) — "
                         "честная оценка на train (см. run.py::loso_artifacts)")
    ap.add_argument("--loseso-artifacts", default=None,
                    help="dev: каталог leave-one-session-out артефактов по сессиям (см. run.py::loseso_artifacts)")
    ap.add_argument("--note", default=None, help="пометка в metrics.json (условия прогона)")
    a = ap.parse_args(argv)
    files = list_sessions(a.input)
    sess_dir = files[0].parent if files else Path(a.input)
    t0 = time.time()
    rc = run_mod.main(["--input", str(sess_dir if Path(a.input).is_dir() else a.input),
                       "--output-dir", str(Path(a.runs_dir) if Path(a.input).is_dir() else Path(a.runs_dir) / files[0].stem),
                       "--artifacts", a.artifacts, "--jobs", str(a.jobs)]
                      + (["--loso-artifacts", a.loso_artifacts] if a.loso_artifacts else [])
                      + (["--loseso-artifacts", a.loseso_artifacts] if a.loseso_artifacts else []))
    print(f"run: {time.time() - t0:.1f} s, rc={rc}", flush=True)
    score_mod.main(["--runs-dir", a.runs_dir, "--sessions-dir", str(sess_dir), "--output-dir", a.metrics_dir]
                   + (["--note", a.note] if a.note else []))
    return rc


if __name__ == "__main__":
    sys.exit(main())
