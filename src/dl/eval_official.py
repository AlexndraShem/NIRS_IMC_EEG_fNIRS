"""Добавочный выигрыш DL поверх текущей ЭЭГ сдачи на официальной сетке (dev-прогоны с --dump-probs).

log P_eeg = (1−w)·log P_new + w·log P_dl, затем текущее слияние (`src/eeg/lr/fusion_eval.fuse`
с параметрами solution/artifacts/config.json: w_eeg 0.55, приор [0,3.5,3.5], hier (0.75, 2.0);
для сессий с пулом участника — fusion_params_subject). w выбирается вложенно по участникам.

    python -m src.dl.eval_official <runs_dir> <dl.pkl> [<overlay_dir>]
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np

from src.dataset.io import PROJECT_DIR, subject_of
from src.dataset.windows import confusion, metrics
from src.eeg.lr.fusion_eval import fuse, load_runs

FLOOR = 1e-9
CFG = json.load(open(PROJECT_DIR / "solution/artifacts/config.json"))
FP = CFG["fusion_params"]
KW = dict(w=FP["eeg_weight"], prior=tuple(FP["prior"]), hier=tuple(FP["hier_tau_s"]))
FPS = CFG.get("fusion_params_subject", {})
KW_S = dict(KW, w=FPS.get("eeg_weight", KW["w"]), prior=tuple(FPS.get("prior", KW["prior"])))


def mix_eeg(pe, pd, w):
    lp = (1 - w) * np.log(np.clip(pe, FLOOR, 1)) + w * np.log(np.clip(pd, FLOOR, 1))
    bad = ~np.all(np.isfinite(lp), 1)
    lp[bad] = np.log(np.clip(pe[bad], FLOOR, 1))
    lp -= lp.max(1, keepdims=True)
    P = np.exp(lp)
    return P / P.sum(1, keepdims=True)


def per_session(R, D, w, hybrid=True):
    out = {}
    for n, r in R.items():
        rr = dict(r)
        rr["pe"] = mix_eeg(r["pe"], D[n]["P"], w) if w > 0 else r["pe"]
        if not hybrid:
            rr["pn"] = np.full_like(r["pn"], np.nan)
        P = fuse(rr, **(KW_S if r["subj"] else KW))
        ok = r["ok"]
        out[n] = metrics(confusion(r["y"][ok], P[ok].argmax(1) + 1))
    return out


def agg(per):
    f = float(np.mean([m["min_f1"] for m in per.values()]))
    q = float(np.mean([m["recall"] for m in per.values()]))
    return f, q, 60 * f + 30 * q


def nested(R, D, grid, hybrid=True):
    per = {w: per_session(R, D, w, hybrid) for w in grid}
    subs = sorted({subject_of(n) for n in R})
    out, ch = {}, {}
    for s in subs:
        best = max(grid, key=lambda w: agg({n: m for n, m in per[w].items() if subject_of(n) != s})[2])
        ch[s] = best
        out.update({n: m for n, m in per[best].items() if subject_of(n) == s})
    return agg(out), ch, per


def fmt(t):
    return f"{t[0]:.3f} / {t[1]:.3f} ({t[2]:.2f})"


def main():
    runs, dl = sys.argv[1], sys.argv[2]
    R = load_runs(runs, sys.argv[3] if len(sys.argv) > 3 else None)
    D = pickle.load(open(dl, "rb"))
    assert all(np.array_equal(D[n]["e"], R[n]["e"]) for n in R)
    grid = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    for hyb in (False, True):
        lab = "гибрид" if hyb else "ЭЭГ-only"
        (sc, ch, per) = nested(R, D, grid, hyb)
        print(f"[{lab}] новая модель: {fmt(agg(per[0.0]))} | + DL вложенно: {fmt(sc)} w={sorted(set(ch.values()))}")
        print("   по w: " + " ".join(f"{w}:{agg(per[w])[2]:.2f}" for w in grid))
        d = [agg({n: per[max(set(ch.values()), key=list(ch.values()).count)][n]})[2] - agg({n: per[0.0][n]})[2]
             for n in R]
    subs = sorted({subject_of(n) for n in R})
    wbest = max(set(ch.values()), key=list(ch.values()).count)
    print(f"по участникам (гибрид, w={wbest}): " + " ".join(
        f"{s}:{agg({n: per[wbest][n] for n in R if subject_of(n) == s})[2] - agg({n: per[0.0][n] for n in R if subject_of(n) == s})[2]:+.1f}"
        for s in subs))
    # DL одна: после того же слияния
    Dm = {n: dict(r, pe=np.where(np.isfinite(D[n]["P"]), D[n]["P"], r["pe"])) for n, r in R.items()}
    print(f"DL одна (вместо ЭЭГ) гибрид: {fmt(agg(per_session(Dm, D, 0.0)))} ЭЭГ-only: {fmt(agg(per_session(Dm, D, 0.0, False)))}")


def joint(R, D, wdl=(0.0, 0.2, 0.3, 0.4, 0.5), ew=(0.55, 0.6, 0.65, 0.7), pr=(3.0, 3.5, 4.0)):
    """Вложенный совместный подбор (w_dl, eeg_weight, prior) — и отдельно для базы (w_dl = 0)."""
    from concurrent.futures import ProcessPoolExecutor
    combos = [(a, b, c) for a in wdl for b in ew for c in pr]
    with ProcessPoolExecutor(4) as ex:
        res = dict(zip(combos, ex.map(_combo, [(R, D, cb) for cb in combos])))
    subs = sorted({subject_of(n) for n in R})
    for lab, cands in (("база (w_dl=0), слияние вложенно", [c for c in combos if c[0] == 0]),
                       ("+ DL, всё вложенно", combos)):
        out, ch = {}, []
        for s in subs:
            best = max(cands, key=lambda c: agg({n: m for n, m in res[c].items() if subject_of(n) != s})[2])
            ch.append(best)
            out.update({n: m for n, m in res[best].items() if subject_of(n) == s})
        print(f"{lab}: {fmt(agg(out))}  выборы: {sorted(set(ch))}")
    top = sorted(combos, key=lambda c: -agg(res[c])[2])[:6]
    print("лучшие (не вложенно): " + "; ".join(f"{c}: {agg(res[c])[2]:.2f}" for c in top))
    print("текущие ew/prior (0.55, 3.5) по w_dl: " + " ".join(f"{a}:{agg(res[(a, 0.55, 3.5)])[2]:.2f}" for a in wdl))


def _combo(args):
    R, D, (a, b, c) = args
    kw = dict(KW, w=b, prior=(0, c, c))
    out = {}
    for n, r in R.items():
        rr = dict(r, pe=mix_eeg(r["pe"], D[n]["P"], a) if a > 0 else r["pe"])
        P = fuse(rr, **kw)
        ok = r["ok"]
        out[n] = metrics(confusion(r["y"][ok], P[ok].argmax(1) + 1))
    return out


if __name__ == "__main__":
    if "--joint" in sys.argv:
        sys.argv.remove("--joint")
        R = load_runs(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None)
        joint(R, pickle.load(open(sys.argv[2], "rb")), ew=(0.55, 0.6, 0.65, 0.7, 0.75, 0.8), pr=(2.5, 3.0, 3.5))
    else:
        main()
