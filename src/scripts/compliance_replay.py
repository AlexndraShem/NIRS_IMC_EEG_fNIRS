"""Эксплуатирует ли слияние (StackedFusion: EMA-признаки + мета-LR + сглаживание) порядок проб?

Берём вероятности ЭЭГ/NIRS окон судьи из прогона compliance_async (--predict judge), переставляем
сегменты (однородные пробы) внутри каждого блока и заново прогоняем ТОЛЬКО слияние (свежий объект
fusion из тех же хронологических артефактов). Вероятности базовых моделей фиксированы — меряется
вклад слияния/сглаживания.

Порядки:
  orig      — исходный (проверка: должен совпасть с прогоном);
  alt       — покой переставлен между слотами покоя, задачи — между слотами задач (чередование
              покой↔задача сохранено, порядок L/R случаен);
  rand      — случайная перестановка всех сегментов блока (чередование нарушено случайно);
  grouped   — все покои блока подряд, затем все задачи (чередования нет);
  gap20     — исходный порядок, но между сегментами +20 с (память EMA τ ≤ 4 с между пробами обнулена).

    python -m src.scripts.compliance_replay --sol <копия solution | solution> --runs <dir прогона judge> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def seg_bounds(end, y, step=62):
    new = np.r_[True, (np.diff(end) != step) | (np.diff(y) != 0)]
    st = np.flatnonzero(new)
    return list(zip(st, np.r_[st[1:], len(end)]))


def order_for(kind, segs, labs, rng):
    idx = np.arange(len(segs))
    if kind in ("orig", "gap20"):
        return idx
    if kind == "alt":
        out = idx.copy()
        for grp in (labs == 1, labs > 1):
            g = idx[grp]
            out[g] = rng.permutation(g)
        return out
    if kind == "rand":
        return rng.permutation(idx)
    if kind == "grouped":
        return np.r_[idx[labs == 1], idx[labs > 1]]
    raise ValueError(kind)


def metrics(yt, yp):
    g = np.zeros((3, 3))
    for t, p in zip(yt, yp):
        g[p - 1, t - 1] += 1           # строки — предсказание, столбцы — истина (как score.py)
    tp = np.diag(g)
    prec = tp / np.maximum(g.sum(1), 1)
    rec = tp / np.maximum(g.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return float(f1.min()), float(rec.mean())


def run_one(args):
    stem, a = args
    sol = Path(a["sol"]).resolve()
    sys.path.insert(0, str(sol))
    if (sol / "tools" / "dev_run.py").exists():     # репозиторий сдачи: --sol solution
        sys.path.insert(0, str(sol / "tools"))
    import io_utils as iu
    from classifier import OnlineClassifier
    from dev_run import loseso_artifacts
    import shutil
    path = Path(a["data"]) / f"{stem}.mat"
    s = iu.load_session(path)
    art = loseso_artifacts(path, a["overlay"], sol / "artifacts") if a["overlay"] else str(sol / "artifacts")
    d = np.load(Path(a["runs"]) / f"{stem}.npz")
    end = d["end"]
    lab = s["states"][end - 1]
    blk = s["blocks"][end - 1]
    res = {}
    rng = np.random.default_rng(int(a["seed"]))
    for kind in a["kinds"]:
        reps = a["reps"] if kind in ("alt", "rand") else 1
        acc = []
        for _ in range(reps):
            clf = OnlineClassifier(iu.session_meta(s))
            clf.load(art)
            fus = clf.fusion
            yt, yp, ye = [], [], []
            same = []
            for b in np.unique(blk):
                ib = np.flatnonzero(blk == b)
                segs = seg_bounds(end[ib], lab[ib])
                slabs = np.array([lab[ib][s0] for s0, _ in segs])
                o = order_for(kind, segs, slabs, rng)
                shift = 0
                for k, j in enumerate(o):
                    s0, s1 = segs[j]
                    t0_slot = end[ib][segs[k][0]]
                    rel = end[ib][s0:s1] - end[ib][s0]
                    if kind == "gap20":
                        shift += int(20 * 250)
                    for r, w in zip(rel, ib[s0:s1]):
                        pe = None if np.isnan(d["p_eeg"][w]).any() else d["p_eeg"][w]
                        pn = None if np.isnan(d["p_nirs"][w]).any() else d["p_nirs"][w]
                        # сдвиг блока: блоки идут в исходном времени, внутри — слоты
                        out = fus.decide({"eeg": pe, "nirs": pn}, end=int(t0_slot + r + shift), fs=250.0)
                        yt.append(int(lab[w])); yp.append(int(out["y"]))
                        ye.append(int(out["y_eeg"]) if out.get("y_eeg") else 1)
                        if kind == "orig":
                            same.append(int(out["y"]) == int(d["y"][w]))
            f, p = metrics(yt, yp)
            fe, pe_ = metrics(yt, ye)
            acc.append([f, p, fe, pe_] + ([float(np.mean(same))] if kind == "orig" else []))
        res[kind] = np.mean(acc, 0).tolist()
    if a["overlay"]:
        shutil.rmtree(Path(art).parent, ignore_errors=True)
    return stem, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/raw/Data"))
    ap.add_argument("--overlay", default=str(ROOT / "data/features/chrono"))
    ap.add_argument("--kinds", default="orig,alt,rand,grouped,gap20")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=28)
    a = ap.parse_args()
    stems = sorted(p.stem for p in Path(a.runs).glob("*.npz"))
    cfg = dict(sol=a.sol, runs=a.runs, data=a.data, overlay=a.overlay, kinds=a.kinds.split(","),
               reps=a.reps, seed=a.seed)
    per = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for stem, r in ex.map(run_one, [(s, cfg) for s in stems]):
            per[stem] = r
    agg = {}
    for k in cfg["kinds"]:
        m = np.mean([per[s][k] for s in per], 0)
        agg[k] = {"hyb_minF1": m[0], "hyb_recall": m[1], "hyb_base60_30": 60 * m[0] + 30 * m[1],
                  "eeg_minF1": m[2], "eeg_recall": m[3], "eeg_base60_30": 60 * m[2] + 30 * m[3]}
        if k == "orig":
            agg[k]["agree_with_run"] = m[4]
    Path(a.out).write_text(json.dumps({"agg": agg, "per_session": per}, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
