"""Замороженный онлайн-каркас.

Владеет часами и метками. Классификатор видит только прошлое.

    python run.py --input /data/session.mat --output-dir /out [--artifacts solution/artifacts]

НЕ ИЗМЕНЯЙТЕ этот файл. Судья подменяет его эталонной копией.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import classifier as clf_mod  # noqa: E402
import io_utils as iu  # noqa: E402

EXIT_OK, EXIT_DATA, EXIT_MODEL = 0, 2, 3


def fail(code: int, message: str) -> None:
    print(f"ОШИБКА: {message}", file=sys.stderr)
    sys.exit(code)


def _empty_nirs(session: dict) -> tuple[np.ndarray, np.ndarray]:
    nch = int(session.get("n_nirs_ch") or 0)
    return (
        np.zeros((0, nch), dtype=np.float64),
        np.zeros((0, nch), dtype=np.float64),
    )


def run_session(
    session_path: Path,
    output_dir: Path,
    artifacts_dir: Path | None,
) -> dict:
    try:
        session = iu.load_session(session_path)
    except Exception as exc:
        fail(EXIT_DATA, f"не удалось прочитать {session_path}: {exc}")

    meta = iu.session_meta(session)
    try:
        classifier = clf_mod.OnlineClassifier(meta)
        if artifacts_dir is not None and Path(artifacts_dir).exists():
            classifier.load(artifacts_dir)
    except Exception:
        fail(EXIT_MODEL, f"инициализация классификатора:\n{traceback.format_exc()}")

    eeg = session["eeg"]
    states = session["states"]
    blocks = session["blocks"]
    fs = session["fs_eeg"]
    n = eeg.shape[0]

    windows = list(iu.iter_scored_windows(states, blocks, fs))
    # fit_block вызывается для ВСЕХ блоков (включая 1-й — первичное обучение);
    # окна блока 1 при этом не оцениваются (см. iter_scored_windows).
    block_ids = sorted({int(b) for b in np.unique(blocks) if int(b) >= 1})
    block_end: dict[int, int] = {}
    for b in block_ids:
        idxs = np.flatnonzero(blocks == b)
        if idxs.size:
            block_end[b] = int(idxs[-1]) + 1  # exclusive end sample

    pred_rows: list[dict] = []
    timing_rows: list[dict] = []
    fitted: set[int] = set()
    cursor = 0  # сколько отсчётов ЭЭГ уже отдано через push

    def push_upto(end_sample: int) -> None:
        nonlocal cursor
        if end_sample <= cursor:
            return
        eeg_chunk = eeg[cursor:end_sample]
        if session["has_nirs"]:
            hbo, hbr = iu.nirs_slice_for_eeg_range(session, cursor, end_sample)
            nirs_chunk = (hbo, hbr)
        else:
            nirs_chunk = _empty_nirs(session)
        try:
            classifier.push(eeg_chunk, nirs_chunk)
        except Exception:
            fail(EXIT_MODEL, f"push() упал на [{cursor},{end_sample}):\n{traceback.format_exc()}")
        cursor = end_sample

    # хронологический проход: окна и fit_block в порядке времени
    events: list[tuple[int, str, object]] = []
    for start, end, lab in windows:
        events.append((end, "predict", (start, end, lab)))
    for b, end_s in block_end.items():
        events.append((end_s, "fit", b))
    events.sort(key=lambda t: (t[0], 0 if t[1] == "predict" else 1))

    for end_sample, kind, payload in events:
        push_upto(end_sample)

        if kind == "fit":
            b = int(payload)
            if b in fitted:
                continue
            # метки завершённых блоков: только отсчёты с States>0 внутри блока
            mask = (blocks == b) & (states > 0)
            labels = {
                "states": states[mask].copy(),
                "eeg_indices": np.flatnonzero(mask),
                "block": b,
            }
            t0 = time.perf_counter()
            try:
                classifier.fit_block(b, labels)
            except Exception:
                fail(EXIT_MODEL, f"fit_block({b}) упал:\n{traceback.format_exc()}")
            dt = time.perf_counter() - t0
            timing_rows.append({"event": "fit_block", "block": b, "seconds": round(dt, 6)})
            fitted.add(b)
            continue

        start, end, lab = payload  # type: ignore[misc]
        t0 = time.perf_counter()
        try:
            out = classifier.predict()
        except Exception:
            fail(EXIT_MODEL, f"predict() упал на окне [{start},{end}):\n{traceback.format_exc()}")
        dt = time.perf_counter() - t0
        timing_rows.append(
            {"event": "predict", "end_sample": end, "seconds": round(dt, 6)}
        )

        if not isinstance(out, dict) or "y" not in out:
            fail(EXIT_MODEL, f"predict() должен вернуть dict с ключом 'y', получено: {type(out)}")
        y = int(out["y"])
        if y not in (1, 2, 3):
            fail(EXIT_MODEL, f"predict()['y'] должен быть 1|2|3, получено: {y}")

        row = {
            "end_sample": end,
            "start_sample": start,
            "true_y": lab,
            "y": y,
        }
        if "y_eeg" in out and out["y_eeg"] is not None:
            row["y_eeg"] = int(out["y_eeg"])
        if "y_nirs" in out and out["y_nirs"] is not None:
            row["y_nirs"] = int(out["y_nirs"])
        pred_rows.append(row)

    # дочитываем хвост сессии (на случай, если fit последнего блока после последнего окна)
    push_upto(n)
    for b in block_ids:
        if b not in fitted and b in block_end:
            push_upto(block_end[b])
            mask = (blocks == b) & (states > 0)
            labels = {
                "states": states[mask].copy(),
                "eeg_indices": np.flatnonzero(mask),
                "block": b,
            }
            t0 = time.perf_counter()
            try:
                classifier.fit_block(b, labels)
            except Exception:
                fail(EXIT_MODEL, f"fit_block({b}) упал:\n{traceback.format_exc()}")
            dt = time.perf_counter() - t0
            timing_rows.append({"event": "fit_block", "block": b, "seconds": round(dt, 6)})
            fitted.add(b)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.csv"
    time_path = output_dir / "timings.csv"

    pred_fields = ["end_sample", "start_sample", "true_y", "y"]
    if any("y_eeg" in r for r in pred_rows):
        pred_fields.append("y_eeg")
    if any("y_nirs" in r for r in pred_rows):
        pred_fields.append("y_nirs")
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(pred_rows)

    time_fields = ["event", "seconds", "end_sample", "block"]
    with time_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=time_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(timing_rows)

    print(f"Сессия: {session_path.name}")
    print(f"Окон:   {len(pred_rows)}")
    print(f"fit:    {len(fitted)} блоков")
    print(f"Записано: {pred_path}, {time_path}")
    return {"n_windows": len(pred_rows), "n_fits": len(fitted)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Онлайн-прогон классификатора на одной сессии")
    ap.add_argument("--input", required=True, help="путь к .mat сессии")
    ap.add_argument("--output-dir", default=".", help="куда писать predictions.csv и timings.csv")
    ap.add_argument("--artifacts", default=None, help="каталог с параметрами классификатора (опционально)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        fail(EXIT_DATA, f"файл не найден: {inp}")
    artifacts = Path(args.artifacts) if args.artifacts else (HERE / "artifacts")
    if not artifacts.exists():
        artifacts = None

    run_session(inp, Path(args.output_dir), artifacts)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
