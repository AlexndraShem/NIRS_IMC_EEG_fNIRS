"""Тесты деградации (dev): классификатор не должен падать при неполных/оборванных данных.

Сценарии на одной сессии (по умолчанию — первая в каталоге):
  1. `no_fit`        — `fit_block` не вызывается вообще (судья не даёт метки);
  2. `one_class`     — в метках блока 1 остался только покой (нет классов 2/3);
  3. `nirs_cut`      — поток NIRS обрывается в середине сессии (после блока 4);
  4. `no_nirs`       — meta обещает NIRS, но ни одного отсчёта не приходит;
  5. `nirs_late`     — NIRS начинается только с блока 3;
  6. `empty_pushes`  — чередование пустых чанков ЭЭГ/NIRS.

Проверяется: отсутствие исключений, корректность формата ответа (y ∈ {1,2,3}, y_nirs — int или
None) и осмысленная деградация (например, при обрыве NIRS `y_nirs` становится пустым).

    python solution/tools/degrade_test.py [--input <session.mat>]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from classifier import OnlineClassifier  # noqa: E402
from dev_run import make_meta  # noqa: E402
from session_io import list_sessions, load_session, window_ends  # noqa: E402


def simulate(s, scenario):
    n = len(s["states"])
    states, blocks = s["states"], s["blocks"]
    ends, ys, wb = window_ends(states, blocks, s["eeg_fs"])
    wins = set(ends[wb >= 2].tolist())
    ch = np.flatnonzero(np.diff(blocks)) + 1
    b_start, b_end = np.r_[0, ch], np.r_[ch, n]
    cut_at = int(b_end[3]) if len(b_end) > 3 else n // 2          # конец блока 4
    start_at = int(b_start[2]) if len(b_start) > 2 else 0         # начало блока 3
    bounds = np.unique(np.r_[np.round(np.arange(62.5, n + 1e-9, 62.5)).astype(int), b_end, n])

    clf = OnlineClassifier(make_meta(s))
    clf.load(HERE / "artifacts")
    fr = s["nirs_frame"]
    ptr, prev = 0, 0
    out = Counter()
    y_nirs_after_cut = Counter()
    for b in bounds:
        b = int(b)
        nc = None
        if fr is not None and scenario != "no_nirs":
            j = int(np.searchsorted(fr, b, side="right"))
            take = (scenario != "nirs_cut" or b <= cut_at) and (scenario != "nirs_late" or b >= start_at)
            if j > ptr and take:
                nc = {"HbO": s["hbo"][ptr:j], "HbR": s["hbr"][ptr:j], "Frame": fr[ptr:j]}
            if scenario != "nirs_late" or b >= start_at:
                ptr = j
        x = s["eeg"][prev:b]
        if scenario == "empty_pushes":
            clf.push(np.zeros((0, x.shape[1])), None)
        clf.push(x, nc)
        prev = b
        if b in wins:
            r = clf.predict()
            assert r["y"] in (1, 2, 3), f"{scenario}: y={r['y']}"
            assert r.get("y_nirs") in (None, 1, 2, 3)
            out[("y", r["y"])] += 1
            out[("y_nirs", r.get("y_nirs"))] += 1
            if scenario == "nirs_cut" and b > cut_at + 15 * s["eeg_fs"]:
                y_nirs_after_cut[r.get("y_nirs")] += 1
        k = np.flatnonzero(b_end == b)
        if len(k) and scenario != "no_fit":
            k = int(k[0])
            lab = states[b_start[k]:b].copy()
            if scenario == "one_class" and k == 0:
                lab = np.where(lab > 0, 1, lab)      # в блоке 1 «только покой»
            clf.fit_block(k + 1, lab)
    return out, y_nirs_after_cut


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None, help=".mat сессии (по умолчанию — первая из data/raw/Data)")
    a = ap.parse_args()
    if a.input is None:
        raw = Path(__file__).resolve().parents[2] / "data/raw/Data"
        if not raw.is_dir() or not list_sessions(raw):
            ap.error(f"нет сессий в {raw} — укажите --input <session.mat> "
                     "(или распакуйте data.zip так, чтобы сессии лежали в data/raw/Data/)")
    path = a.input or str(list_sessions(raw)[0])
    s = load_session(path)
    print(f"сессия: {Path(path).stem}")
    ok = True
    for sc in ("no_fit", "one_class", "nirs_cut", "no_nirs", "nirs_late", "empty_pushes"):
        try:
            out, after = simulate(s, sc)
            ys = {k[1]: v for k, v in out.items() if k[0] == "y"}
            yn = {k[1]: v for k, v in out.items() if k[0] == "y_nirs"}
            note = ""
            if sc == "nirs_cut":
                note = f" | после обрыва (+15 с) y_nirs: {dict(after)}"
            print(f"{sc:14s} OK  y={ys}  y_nirs={yn}{note}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"{sc:14s} ПАДЕНИЕ: {type(e).__name__}: {e}")
    print("OK" if ok else "ЕСТЬ ПАДЕНИЯ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
