"""Прогон решения ОФИЦИАЛЬНЫМ каркасом судьи (run.py / io_utils.py / score.py из репозитория
команды, без изменений) на всех train-сессиях, с честной подменой ЭЭГ-пула (LOSubjO).

    python -m src.scripts.judge_eval --official ../hormigas/solution --out <dir> [--jobs 8] [--no-loso]
                                     [--loseso-dir <каталог LOSessO-артефактов по сессиям>]
                                     [--data <каталог .mat, по умолчанию data/raw/Data>]

В репозитории сдачи официальный каркас лежит в самом solution/ → `--official solution`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True,
                    help="каталог с официальными run.py / io_utils.py / score.py (в репозитории сдачи — solution)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/raw/Data"))
    ap.add_argument("--loso-dir", default=str(ROOT / "data/features/eeg_models"))
    ap.add_argument("--no-loso", action="store_true")
    ap.add_argument("--loseso-dir", default=None,
                    help="leave-one-session-out: <dir>/<сессия>/ накладывается поверх артефактов "
                         "(см. loseso_artifacts в solution/tools/dev_run.py или solution/run.py) вместо LOSubjO-подмены пула")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--config", default=None, help="заменить artifacts/config.json в копии решения (сравнение вариантов)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sol = out / "solution"
    if sol.exists():
        shutil.rmtree(sol)
    shutil.copytree(ROOT / "solution", sol, ignore=shutil.ignore_patterns("__pycache__", "tools"))
    for f in ("run.py", "io_utils.py", "score.py"):
        shutil.copy2(Path(a.official) / f, sol / f)
    if a.config:
        shutil.copy2(a.config, sol / "artifacts" / "config.json")
    sys.path.insert(0, str(ROOT / "solution"))
    # наши dev-хелперы подмены пула: в репозитории сдачи solution/run.py — официальный, хелперы —
    # в solution/tools/dev_run.py; в исследовательской копии — в реконструкции solution/run.py
    if (ROOT / "solution" / "tools" / "dev_run.py").exists():
        sys.path.insert(0, str(ROOT / "solution" / "tools"))
        from dev_run import loseso_artifacts, loso_artifacts
    else:
        from run import loseso_artifacts, loso_artifacts

    sessions = sorted(Path(a.data).glob("*.mat"))
    if not sessions:
        sys.exit(f"нет *.mat в {a.data}: распакуйте data.zip так, чтобы сессии лежали в data/raw/Data/, "
                 "или укажите --data <каталог .mat>")

    def job(p: Path):
        d = out / "runs" / p.stem
        d.mkdir(parents=True, exist_ok=True)
        if a.loseso_dir:
            art = loseso_artifacts(p, a.loseso_dir, sol / "artifacts")
        else:
            art = str(sol / "artifacts") if a.no_loso else loso_artifacts(p, a.loso_dir, sol / "artifacts")
        env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "PATH": "/usr/bin:/bin"}
        r = subprocess.run([PY, str(sol / "run.py"), "--input", str(p), "--output-dir", str(d),
                            "--artifacts", art], capture_output=True, text=True, env=env)
        if a.loseso_dir:
            shutil.rmtree(Path(art).parent, ignore_errors=True)
        elif not a.no_loso:
            shutil.rmtree(art, ignore_errors=True)
        if r.returncode:
            return p.stem, f"run rc={r.returncode}: {r.stderr[-2000:]}"
        r = subprocess.run([PY, str(sol / "score.py"), "--session", str(p), "--predictions",
                            str(d / "predictions.csv"), "--timings", str(d / "timings.csv"),
                            "--output", str(d / "metrics.json")], capture_output=True, text=True)
        return p.stem, None if r.returncode == 0 else f"score rc={r.returncode}: {r.stderr[-2000:]}"

    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(job, sessions))
    errs = [(s, e) for s, e in res if e]
    for s, e in errs:
        print("FAIL", s, e)
    ms = [str(out / "runs" / s / "metrics.json") for s, e in res if not e]
    subprocess.run([PY, str(sol / "score.py"), "--aggregate", *ms, "--output", str(out / "metrics.json")], check=True)
    agg = json.loads((out / "metrics.json").read_text())
    per = [json.loads(Path(m).read_text()) for m in ms]
    agg["windows_match_all"] = all(r.get("windows_match", False) for r in per)
    agg["fit_max_s"] = max(r["timings"]["fit_max_s"] for r in per)
    agg["fit_mean_s"] = sum(r["timings"]["fit_mean_s"] for r in per) / len(per)
    agg["predict_max_s"] = max(r["timings"]["predict_max_s"] for r in per)
    agg["predict_mean_s"] = sum(r["timings"]["predict_mean_s"] for r in per) / len(per)
    (out / "metrics.json").write_text(json.dumps(agg, ensure_ascii=False, indent=1))
    print(json.dumps(agg, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
