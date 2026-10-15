"""Сравнение прогонов compliance_async с эталонным (расписание судьи): совпадение предсказаний на
окнах судьи по компонентам, метрики ТЗ и распределение ответов на неоцениваемых окнах.

    python -m src.scripts.compliance_compare --ref <judge dir> --runs <dir1> <dir2> ... [--out json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path.home() / "projects/hackatons/hormigas/solution"))
import io_utils as iu  # noqa: E402


def mets(yt, yp):
    g = np.zeros((3, 3))
    np.add.at(g, (yp - 1, yt - 1), 1)
    tp = np.diag(g)
    den = g.sum(0) + g.sum(1)
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0)
    rec = tp / np.maximum(g.sum(0), 1)
    return float(f1.min()), float(rec.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/raw/Data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--after-fit-shift-s", type=float, default=0.0,
                    help="исключить из сравнения окна в [конец блока, +S с) (fit ещё не случился)")
    a = ap.parse_args()
    ref = Path(a.ref)
    stems = sorted(p.stem for p in ref.glob("*.npz"))
    cache = {}
    report = {}
    for rd in [a.ref] + a.runs:
        rd = Path(rd)
        acc = {k: [] for k in ("y", "y_eeg", "y_nirs")}
        dev = {k: [] for k in ("p_inner", "p_dl", "p_eeg", "p_nirs", "hyb")}
        q = {"hyb": [], "eeg": [], "nirs": []}
        s0 = np.zeros(3); trans = np.zeros(3); b1 = np.zeros(3)
        nwin = 0
        for st in stems:
            f = rd / f"{st}.npz"
            if not f.exists():
                continue
            if st not in cache:
                s = iu.load_session(Path(a.data) / f"{st}.mat")
                block_end = {int(b): int(np.flatnonzero(s["blocks"] == b)[-1]) + 1 for b in np.unique(s["blocks"])}
                cache[st] = (s["states"], s["blocks"], block_end)
            states, blocks, block_end = cache[st]
            r = np.load(ref / f"{st}.npz"); d = np.load(f)
            m = np.isin(d["end"], r["end"])
            mr = np.isin(r["end"], d["end"])
            if a.after_fit_shift_s:
                sh = int(a.after_fit_shift_s * 250)
                bad = np.zeros(len(r["end"]), bool)
                for be in block_end.values():
                    bad |= (r["end"] > be) & (r["end"] <= be + sh)
                keep_r = mr & ~bad
                keep_d = np.isin(d["end"], r["end"][keep_r])
            else:
                keep_r, keep_d = mr, m
            for k in acc:
                acc[k].append(np.mean(d[k][keep_d] == r[k][keep_r]))
            for k in dev:
                x = np.abs(d[k][keep_d] - r[k][keep_r])
                dev[k].append(np.nanmax(x) if np.isfinite(x).any() else np.nan)
            yt = states[d["end"][m] - 1]
            q["hyb"].append(mets(yt, d["y"][m]))
            q["eeg"].append(mets(yt, np.where(d["y_eeg"][m] > 0, d["y_eeg"][m], 1)))
            if (d["y_nirs"][m] > 0).all():
                q["nirs"].append(mets(yt, d["y_nirs"][m]))
            nwin += m.sum()
            # неоцениваемые окна (только в прогонах с --predict all)
            e = d["end"][~m]
            if len(e):
                seg_s = np.stack([states[x - 250:x] for x in e])
                seg_b = np.stack([blocks[x - 250:x] for x in e])
                is0 = (seg_s == 0).all(1)
                is_b1 = (seg_b == 1).all(1)
                y = d["y"][~m]
                for arr, sel in ((s0, is0 & ~is_b1), (trans, ~is0 & ~is_b1), (b1, is_b1)):
                    arr += np.bincount(y[sel] - 1, minlength=3)[:3]
        rep = {"n_sessions": len(q["hyb"]), "n_judge_windows": int(nwin),
               "agree": {k: float(np.mean(v)) for k, v in acc.items()},
               "agree_min_session": {k: float(np.min(v)) for k, v in acc.items()},
               "max_abs_dev": {k: (float(np.nanmax(v)) if np.isfinite(v).any() else None) for k, v in dev.items()}}
        for k, v in q.items():
            if v:
                v = np.array(v)
                rep[f"{k}_minF1"] = float(v[:, 0].mean()); rep[f"{k}_recall"] = float(v[:, 1].mean())
        rep["base60_30"] = 60 * rep["hyb_minF1"] + 30 * rep["hyb_recall"]
        if "nirs_minF1" in rep:
            rep["nirs_bonus"] = 5 + 15 * rep["nirs_minF1"]
            rep["total"] = rep["base60_30"] + rep["nirs_bonus"]
        if s0.sum():
            rep["state0_answers_share[rest,L,R]"] = (s0 / s0.sum()).round(4).tolist()
            rep["transition_answers_share"] = (trans / trans.sum()).round(4).tolist()
            rep["block1_answers_share"] = (b1 / max(b1.sum(), 1)).round(4).tolist()
        report[rd.name] = rep
    txt = json.dumps(report, ensure_ascii=False, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
