"""Самопроверка «терпимости» OnlineClassifier к вариантам контракта (dev-инструмент).

Прогоняет одну сессию в нескольких вариантах подачи данных и сверяет ответы с каноническим
вариантом (как в solution/run.py). Ожидается 100 % совпадение.

    python solution/tools/contract_selftest.py --input data/raw/Data/<session>.mat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from classifier import OnlineClassifier  # noqa: E402
from dev_run import make_meta  # noqa: E402
from session_io import load_session, window_ends  # noqa: E402


def simulate(s, variant, rng):
    n = len(s["states"])
    states, blocks = s["states"], s["blocks"]
    ends, ys, wb = window_ends(states, blocks, s["eeg_fs"])
    wins = set(ends[wb >= 2].tolist())
    ch = np.flatnonzero(np.diff(blocks)) + 1
    b_end = np.r_[ch, n]
    b_start = np.r_[0, ch]
    if variant == "random_chunks":
        cuts = np.cumsum(rng.integers(1, 200, size=n // 50))
        bounds = np.unique(np.r_[cuts[cuts < n], list(wins), b_end, n])
    else:   # как у судьи: push только перед событиями (концы окон, концы блоков, хвост)
        bounds = np.unique(np.r_[list(wins), b_end, n])
    meta = {} if variant == "empty_meta_T_cumlabels" else make_meta(s)
    if variant == "canonical_nosubj":      # эталон для вариантов без subject (откат без пула участника)
        meta = {k: v for k, v in meta.items() if k not in ("subject", "session_name", "session")}
    if variant == "nested_meta":
        meta = {"EEG": {"Frq": s["eeg_fs"]}, "NIRS": {"Frq": s["nirs_fs"]}}
    clf = OnlineClassifier(meta)
    clf.load(HERE / "artifacts")
    fr = s["nirs_frame"]
    ptr, prev, out, pend = 0, 0, {}, []
    for b in bounds:
        b = int(b)
        nc = None
        if fr is not None:
            j = int(np.searchsorted(fr, b, side="right"))
            if j > ptr:
                if variant == "official":                                         # формат судьи
                    nc = (s["hbo"][ptr:j], s["hbr"][ptr:j])
                elif variant == "empty_meta_T_cumlabels":
                    nc = np.c_[fr[ptr:j], s["hbo"][ptr:j], s["hbr"][ptr:j]]      # Frame первым столбцом
                elif variant == "array_noframe":
                    nc = np.hstack([s["hbo"][ptr:j], s["hbr"][ptr:j]])           # без Frame
                elif variant == "nested_meta":
                    nc = (s["hbo"][ptr:j], s["hbr"][ptr:j], fr[ptr:j])
                else:
                    nc = {"hbo": s["hbo"][ptr:j], "hbr": s["hbr"][ptr:j], "frame": fr[ptr:j]}
            ptr = j
        x = s["eeg"][prev:b]
        clf.push(x.T if variant == "empty_meta_T_cumlabels" else x, nc)
        prev = b
        if b in wins:
            r = clf.predict()
            out[b] = (r["y"], r["y_eeg"], r["y_nirs"])
            pend.append(b)
        k = np.flatnonzero(b_end == b)
        if len(k):
            k = int(k[0])
            if variant == "official":
                mask = (blocks == k + 1) & (states > 0)
                lab = {"states": states[mask].copy(), "eeg_indices": np.flatnonzero(mask), "block": k + 1}
            elif variant == "empty_meta_T_cumlabels":
                lab = states[:b]
            else:
                lab = states[b_start[k]:b]
            clf.fit_block(k + 1, lab)
            pend = []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    a = ap.parse_args()
    s = load_session(a.input)
    rng = np.random.default_rng(0)
    ref = simulate(s, "canonical", rng)
    ref_ns = simulate(s, "canonical_nosubj", rng)
    # (вариант, эталон, допуски (y, y_eeg, y_nirs)): без Frame (формат судьи) время отсчётов NIRS
    # восстанавливается по границам push — y_nirs на части окон законно отличается от эталона с
    # истинным Frame (≈ 97–99.6 % совпадений), гибрид y — через слияние слабее (≥ 99 %), ЭЭГ — точно;
    # без subject решение законно откатывается на модель без пула участника
    noframe = (0.99, 1.0, 0.95)
    checks = [("official", ref, noframe), ("random_chunks", ref, (1.0, 1.0, 1.0)),
              ("empty_meta_T_cumlabels", ref_ns, (1.0, 1.0, 1.0)), ("array_noframe", ref, noframe),
              ("nested_meta", ref_ns, (1.0, 1.0, 1.0))]
    ok = True
    for v, r0, tol in checks:
        o = simulate(s, v, rng)
        agree = [float(np.mean([o[e][i] == r0[e][i] for e in r0])) for i in range(3)]
        good = all(a >= t for a, t in zip(agree, tol))
        print(f"{v:28s} совпадение y / y_eeg / y_nirs{' (без subject)' if r0 is ref_ns else ''}: "
              f"{agree[0]:.4f} / {agree[1]:.4f} / {agree[2]:.4f} (допуск {tol}) {'ok' if good else 'FAIL'}")
        ok &= good
    print("OK" if ok else "РАСХОЖДЕНИЯ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
