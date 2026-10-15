"""Анализ эксперимента subject_pool: смеси within × общий пул × пул участника (ЭЭГ), NIRS-пул
участника, слияние модальностей как в `solution/fusion.py` (w=0.75, приор [0,2,2], EMA 1.5 с).

    python -m src.eeg.exp_subjpool_eval <eeg_pools.pkl> <nirs_pools.pkl>

Метрики: среднее по сессиям min-F1 и macro-recall (блоки 2–7), балл 60F+30P гибрида,
NIRS — 15·F_nirs (приор [0,1,1], без EMA). Сессии без NIRS — ЭЭГ-only (в NIRS-метрику не входят).
"""
from __future__ import annotations

import itertools
import pickle
import sys

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.dataset.io import PROJECT_DIR
from NIRS_IMC_EEG_fNIRS.src.dataset.windows import confusion, metrics

R = PROJECT_DIR
FLOOR = 1e-9
W = {p.stem: dict(np.load(p)) for p in sorted((R / "data/features/v1_win1").glob("*.npz"))}
NAMES = sorted(W)
SUBJ = {n: n.split("_")[-1] for n in NAMES}


def _lp(P):
    return np.log(np.clip(P, FLOOR, 1))


def _norm(lp):
    lp = lp - lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def ema(P, ends, tau=1.5):
    s = np.full_like(P, np.nan)
    prev = None
    for i in range(len(P)):
        if not np.isfinite(P[i]).all():
            continue
        if prev is None:
            s[i] = P[i]
        else:
            a = np.exp(-(ends[i] - ends[prev]) / 250.0 / tau)
            s[i] = a * s[prev] + (1 - a) * P[i]
        prev = i
    return s


def sess_metrics(n, P):
    z = W[n]
    te = z["block"] >= 2
    return metrics(confusion(z["y"][te], np.argmax(np.nan_to_num(P[te], nan=-1), 1) + 1))


def agg(res):
    f = float(np.mean([r["min_f1"] for r in res.values()]))
    r = float(np.mean([r["recall"] for r in res.values()]))
    return f, r, 60 * f + 30 * r


def hybrid(Pe, Pn, w=0.75, prior=2.0, tau=1.5):
    out = {}
    for n in NAMES:
        lp = _lp(Pe[n])
        pn = Pn.get(n)
        if pn is not None:
            lp = w * lp + (1 - w) * _lp(pn)
        p = _norm(lp + np.array([0, prior, prior]))
        out[n] = sess_metrics(n, ema(p, W[n]["ends"], tau))
    return agg(out)


def eeg_only(Pe, prior=2.0, tau=1.5):
    out = {n: sess_metrics(n, ema(_norm(_lp(Pe[n]) + np.array([0, prior, prior])), W[n]["ends"], tau))
           for n in NAMES}
    return agg(out)


def nirs_only(Pn, prior=1.0):
    out = {n: sess_metrics(n, _norm(_lp(P) + np.array([0, prior, prior]))) for n, P in Pn.items()}
    return agg(out)


def eeg_mix(Pw, Pg, Ps, ww, wg, ws):
    """log P = ww·log Pw + wg·log Pg + ws·log Ps; нет Ps (нет прошлых сессий) → его вес уходит в Pg;
    нет Pg → только within."""
    out = {}
    for n in NAMES:
        g = Pg.get(n) if Pg is not None else None
        s = Ps.get(n) if Ps is not None else None
        if s is None:
            s = g
        if g is None:
            g = s
        lp = ww * _lp(Pw[n])
        if g is not None:
            lp = lp + wg * _lp(g) + ws * _lp(s)
        else:
            lp = _lp(Pw[n])
        out[n] = _norm(lp)
    return out


def sig(z):
    return 1 / (1 + np.exp(-z))


def nirs_probs(NR, kind=None, a_rt=0.0, a_lr=0.0):
    """Каскад: z = (1−a)·z_within + a·z_pool по задачам RT (8 членов) и LR (4 члена)."""
    out = {}
    for n, d in NR.items():
        Zw = d["within"]
        Zp = d.get(kind) if kind else None
        if Zp is None:
            Zp = Zw
        zt = (1 - a_rt) * Zw[:8].mean(0) + a_rt * Zp[:8].mean(0)
        zr = (1 - a_lr) * Zw[8:].mean(0) + a_lr * Zp[8:].mean(0)
        pt, pr = sig(zt), sig(zr)
        out[n] = np.c_[1 - pt, pt * (1 - pr), pt * pr]
    return out


def main(eeg_pk, nirs_pk):
    EP = pickle.load(open(eeg_pk, "rb"))
    L = lambda f: pickle.load(open(R / "data/probs" / f, "rb"))
    Pw = L("eeg_ts2s_all_within.pkl")
    Pn0 = L("nirs_cascade_ens.pkl")
    EP["G0_old"] = L("eeg_ts2s_all_pool.pkl")
    print("ключи ЭЭГ-пулов:", sorted(EP))
    fmt = lambda t: f"minF1={t[0]:.3f} rec={t[1]:.3f} pts={t[2]:.2f}"

    # ---------- ЭЭГ ----------
    print("\n=== ЭЭГ (EEG-only с приором/EMA; гибрид с базовым NIRS) ===")
    rows = []

    def ev(label, Pe):
        e, h = eeg_only(Pe), hybrid(Pe, Pn0)
        rows.append((label, e, h))
        print(f"{label:52s} EEG {fmt(e)} | HYB {fmt(h)}", flush=True)
        return h[2]

    ev("база: within^.2 · G0_old^.8", eeg_mix(Pw, EP["G0_old"], None, 0.2, 0.8, 0))
    for prot, G, S, S1 in (("A", "GA_C0.01", "SA_C0.01", "SA_C0.1"), ("B", "GB_C0.01", "SB_C0.01", "SB_C0.1")):
        print(f"--- протокол {prot}")
        ev(f"{prot}: within^.2 · G^.8 ({G})", eeg_mix(Pw, EP[G], None, 0.2, 0.8, 0))
        ev(f"{prot}: within^.2 · S^.8 ({S})", eeg_mix(Pw, EP[S], None, 0.2, 0.8, 0))
        ev(f"{prot}: within^.2 · S^.8 ({S1})", eeg_mix(Pw, EP[S1], None, 0.2, 0.8, 0))
        ev(f"{prot}: S only ({S})", eeg_mix(Pw, EP[S], None, 0.0, 1.0, 0))
        ev(f"{prot}: G only ({G})", eeg_mix(Pw, EP[G], None, 0.0, 1.0, 0))
        g0b = "G0_old" if prot == "A" else "G0B_C0.01"
        best = {}
        for gname in (G, g0b):
            for sname in (S, S1):
                for ww, ws in itertools.product((0.1, 0.2, 0.3), (0.2, 0.3, 0.4, 0.5, 0.6)):
                    wg = round(1 - ww - ws, 2)
                    if wg < 0:
                        continue
                    Pe = eeg_mix(Pw, EP[gname], EP[sname], ww, wg, ws)
                    h = hybrid(Pe, Pn0)
                    best[(gname, sname, ww, wg, ws)] = h
        top = sorted(best.items(), key=lambda kv: -kv[1][2])[:8]
        for k, h in top:
            print(f"   {prot} смесь {k}: HYB {fmt(h)}")
        pickle.dump(best, open(eeg_pk.replace(".pkl", f"_grid{prot}.pkl"), "wb"))

    # ---------- NIRS ----------
    if nirs_pk:
        NR = pickle.load(open(nirs_pk, "rb"))
        print("\n=== NIRS (15·F) ===")
        print("база nirs_cascade_ens:", fmt(nirs_only(Pn0)), f"nirs_pts={15 * nirs_only(Pn0)[0]:.2f}")
        print("within (пересчёт):   ", fmt(nirs_only(nirs_probs(NR))))
        for kind in ("SA", "SB", "GA", "SA_aug", "SB_aug"):
            if kind not in next(iter(NR.values())):
                continue
            for a_rt, a_lr in itertools.product((0, 0.25, 0.5, 0.75, 1.0), (0, 0.25, 0.5, 0.75, 1.0)):
                if a_rt == 0 and a_lr == 0:
                    continue
                Pn = nirs_probs(NR, kind, a_rt, a_lr)
                t = nirs_only(Pn)
                print(f"  {kind:7s} a_rt={a_rt:.2f} a_lr={a_lr:.2f} {fmt(t)} nirs_pts={15 * t[0]:.2f}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 1 else None)
