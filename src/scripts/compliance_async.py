"""Тест независимости решения от расписания вызовов (аудит compliance, docs/experiments/compliance_audit.md).

Драйвер повторяет логику официального run.py (push → predict/fit_block в хронологическом порядке,
те же чанки NIRS через io_utils.nirs_slice_for_eeg_range, те же метки fit_block), но позволяет менять
расписание:

  --predict judge|all   predict только на окнах судьи или на ВСЕХ точках сетки (end = 250 + 62·j,
                        включая блок 1, состояние 0 и межсегментные паузы);
  --chunk K             push мелкими чанками по K отсчётов (0 — как судья: push только до события);
  --fit-shift-s S       fit_block(b) на S секунд позже конца блока (метки те же);
  --no-fit              fit_block не вызывается вообще.

Пишет <out>/<сессия>.npz: end, y, y_eeg, y_nirs, p_eeg, p_inner, p_dl, p_nirs, hyb (NaN — нет).

    python -m src.scripts.compliance_async --sol <копия solution | solution> --out <dir> --predict all --chunk 20 --jobs 16
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _vec(p):
    return np.full(3, np.nan) if p is None else np.asarray(p, np.float64).ravel()[:3]


def run_one(args):
    path, a = args
    sol = Path(a["sol"]).resolve()
    sys.path.insert(0, str(sol))
    if (sol / "tools" / "dev_run.py").exists():     # репозиторий сдачи: --sol solution
        sys.path.insert(0, str(sol / "tools"))
    import io_utils as iu  # официальный
    from classifier import OnlineClassifier
    from dev_run import loseso_artifacts, loso_artifacts

    path = Path(path)
    if a.get("loso_dir"):
        art = loso_artifacts(path, a["loso_dir"], sol / "artifacts")
    else:
        art = loseso_artifacts(path, a["overlay"], sol / "artifacts") if a["overlay"] else str(sol / "artifacts")
    s = iu.load_session(path)
    meta = iu.session_meta(s)
    clf = OnlineClassifier(meta)
    clf.load(art)
    eeg, states, blocks, fs = s["eeg"], s["states"], s["blocks"], s["fs_eeg"]
    n = eeg.shape[0]
    win, step = int(round(fs)), int(round(fs / 4))
    judge = [e for _, e, _ in iu.iter_scored_windows(states, blocks, fs)]
    pred_pts = judge if a["predict"] == "judge" else list(range(win, n + 1, step))
    block_end = {}
    for b in sorted({int(b) for b in np.unique(blocks) if int(b) >= 1}):
        idx = np.flatnonzero(blocks == b)
        block_end[b] = int(idx[-1]) + 1
    ev = [(e, 0, None) for e in pred_pts]
    if not a["no_fit"]:
        sh = int(round(a["fit_shift_s"] * fs))
        ev += [(min(e + sh, n), 1, b) for b, e in block_end.items()]
    ev.sort(key=lambda t: (t[0], t[1]))

    cursor = 0

    def push_to(e):
        nonlocal cursor
        if e <= cursor:
            return
        cuts = [e] if not a["chunk"] else list(range(cursor + a["chunk"], e, a["chunk"])) + [e]
        for c in cuts:
            if s["has_nirs"]:
                nirs = iu.nirs_slice_for_eeg_range(s, cursor, c)
            else:
                nirs = (np.zeros((0, 0)), np.zeros((0, 0)))
            clf.push(eeg[cursor:c], nirs)
            cursor = c

    rec = {k: [] for k in ("end", "y", "y_eeg", "y_nirs", "p_eeg", "p_inner", "p_dl", "p_nirs", "hyb", "t")}
    for e, kind, b in ev:
        push_to(e)
        if kind == 1:
            mask = (blocks == b) & (states > 0)
            clf.fit_block(b, {"states": states[mask].copy(), "eeg_indices": np.flatnonzero(mask), "block": b})
            continue
        t0 = time.perf_counter()
        out = clf.predict()
        dt = time.perf_counter() - t0
        lp = getattr(clf, "last_proba", {}) or {}
        inner = getattr(clf.eeg, "inner", None)
        p_inner = inner.predict_proba(e) if inner is not None else None
        rec["end"].append(e); rec["t"].append(dt)
        for k in ("y", "y_eeg", "y_nirs"):
            v = out.get(k)
            rec[k].append(-1 if v is None else int(v))
        rec["p_eeg"].append(_vec(lp.get("eeg"))); rec["p_nirs"].append(_vec(lp.get("nirs")))
        rec["hyb"].append(_vec(lp.get("hyb"))); rec["p_inner"].append(_vec(p_inner))
        rec["p_dl"].append(_vec(getattr(clf.eeg, "last_dl", None)))
    push_to(n)
    if a.get("loso_dir"):
        shutil.rmtree(art, ignore_errors=True)
    elif a["overlay"]:
        shutil.rmtree(Path(art).parent, ignore_errors=True)
    out = Path(a["out"]) / f"{path.stem}.npz"
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in rec.items()})
    return path.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol", required=True, help="каталог решения (копия solution/ + официальный io_utils.py + dev_run.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/raw/Data"))
    ap.add_argument("--overlay", default=str(ROOT / "data/features/chrono"),
                    help="хронологические артефакты (как judge_eval --loseso-dir); '' — артефакты сдачи")
    ap.add_argument("--loso-dir", default=None, help="LOSubjO: подмена пулов как judge_eval без --loseso-dir")
    ap.add_argument("--predict", choices=["judge", "all"], default="judge")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--fit-shift-s", type=float, default=0.0)
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--sessions", default="", help="подстрока-фильтр имён сессий через запятую")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.data).glob("*.mat"))
    if a.sessions:
        keys = a.sessions.split(",")
        files = [f for f in files if any(k in f.name for k in keys)]
    cfg = dict(sol=a.sol, out=a.out, overlay=a.overlay, predict=a.predict, chunk=a.chunk,
               fit_shift_s=a.fit_shift_s, no_fit=a.no_fit, loso_dir=a.loso_dir)
    with ProcessPoolExecutor(a.jobs) as ex:
        for name in ex.map(run_one, [(str(f), cfg) for f in files]):
            print("done", name, flush=True)


if __name__ == "__main__":
    main()
