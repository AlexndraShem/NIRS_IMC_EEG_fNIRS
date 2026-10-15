"""Оценка DL-выходов LOSubjO: одиночная сеть, адаптация (within-голова на эмбеддингах по прошлым
блокам), ансамбль с римановой `eeg_ts2s_all`, ЭЭГ-only с приором/EMA и гибрид с NIRS
(как `solution/fusion.py`: w_eeg 0.75, приор [0,2,2], EMA 1.5 с).

    python -m src.dl.evaluate <tag>[+<tag2>...] [<tag>...] [--full] [--save <имя> --a <a> --w <w>]

`tag1+tag2` — ансамбль сетей/сидов (среднее лог-вероятностей, эмбеддинги конкатенируются).
--save: сохраняет data/probs/eeg_dl_<имя>.pkl (DL с адаптацией, смесь a) и
eeg_dl_<имя>_ens.pkl (Riem^(1−w)·DL^w) в формате eeg_ts2s_all.pkl (NaN на блоке 1).
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.dl.train import OUT_DIR
from NIRS_IMC_EEG_fNIRS.src.eeg.exp_subjpool_eval import NAMES, SUBJ, W, R, _norm, agg, ema, eeg_only, hybrid, sess_metrics

FLOOR = 1e-9


def lp_(P):
    return np.log(np.clip(P, FLOOR, 1))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_tags(tags):
    L, E = {}, {}
    for t in tags:
        for f in sorted((OUT_DIR / t).glob("*.npz")):
            z = np.load(f)
            for k in z.files:
                n, kind = k.split("__")
                if kind == "logits":
                    L.setdefault(n, []).append(np.log(softmax(z[k].astype(np.float64))))
                else:
                    E.setdefault(n, []).append(z[k])
    Pdl = {n: softmax(np.mean(v, 0)) for n, v in L.items()}
    Emb = {n: np.concatenate(v, 1) for n, v in E.items()}
    for n in Pdl:
        Pdl[n][W[n]["block"] < 2] = np.nan
    return Pdl, Emb


class ShrinkLDA:
    """LDA с фиксированным shrinkage на стандартизованных признаках (≈ sklearn lsqr)."""

    def fit(self, X, y, shrink=0.8):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        cls = np.unique(y)
        M = np.stack([Z[y == c].mean(0) for c in cls])
        R_ = Z - M[np.searchsorted(cls, y)]
        S = R_.T @ R_ / len(Z)
        S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S))
        self.coef = np.linalg.solve(S, M.T).T
        pri = np.array([np.mean(y == c) for c in cls])
        self.icpt = -0.5 * np.sum(self.coef * M, 1) + np.log(pri)
        self.cls = cls
        return self

    def log_proba(self, X):
        z = ((X - self.mu) / self.sd) @ self.coef.T + self.icpt
        full = np.full((len(X), 3), -30.0)
        full[:, self.cls - 1] = z
        return full - np.logaddexp.reduce(full, 1, keepdims=True)


def within_head(emb, n, shrink=0.8, stride=2):
    """Эмуляция fit_block: для блока b — sLDA на эмбеддингах окон блоков < b (каждое 2-е)."""
    z = W[n]
    P = np.full((len(z["y"]), 3), np.nan)
    for b in range(2, 8):
        tr = np.flatnonzero(z["block"] < b)[::stride]
        te = z["block"] == b
        m = ShrinkLDA().fit(emb[tr].astype(np.float64), z["y"][tr], shrink)
        P[te] = np.exp(m.log_proba(emb[te].astype(np.float64)))
    return P


def mix(Pa, Pb, a):
    """P ∝ Pa^(1-a)·Pb^a."""
    return {n: softmax((1 - a) * lp_(Pa[n]) + a * lp_(Pb[n])) for n in Pa}


def raw(P):
    return agg({n: sess_metrics(n, P[n]) for n in NAMES})


def fmt(t):
    return f"{t[0]:.3f}/{t[1]:.3f} ({t[2]:.2f})"


def nested_weight(Pa, Pb, grid, fn=None):
    """Вес смеси выбирается по остальным участникам (LOSubjO), применяется к исключённому.
    fn(P) -> {сессия: metrics}; по умолчанию raw argmax."""
    fn = fn or (lambda P: {n: sess_metrics(n, P[n]) for n in NAMES})
    subs = sorted(set(SUBJ.values()))
    per = {a: fn(mix(Pa, Pb, a)) for a in grid}
    out, chosen = {}, {}
    for s in subs:
        best = max(grid, key=lambda a: agg({n: r for n, r in per[a].items() if SUBJ[n] != s})[2])
        chosen[s] = best
        out.update({n: per[best][n] for n in NAMES if SUBJ[n] == s})
    return agg(out), chosen


def hybrid_per(Pe, Pn, w=0.75, prior=2.0, tau=1.5):
    """Как exp_subjpool_eval.hybrid, но по сессиям (для вложенного выбора веса)."""
    out = {}
    for n in NAMES:
        lp = lp_(Pe[n])
        if Pn is not None and Pn.get(n) is not None:
            lp = w * lp + (1 - w) * lp_(Pn[n])
        out[n] = sess_metrics(n, ema(_norm(lp + np.array([0, prior, prior])), W[n]["ends"], tau))
    return out


def final_table(Pr, Pn, Pv, w_grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    """Сводка: одиночная DL / ансамбль с римановой / гибрид; вес — вложенно по каждой метрике."""
    rows = [("риманова eeg_ts2s_all", Pr, None), ("DL", Pv, None)]
    print(f"{'вариант':34s} {'raw argmax':>22s} {'ЭЭГ-only pr+EMA':>22s} {'гибрид с NIRS':>22s}")
    for lab, P, _ in rows:
        print(f"{lab:34s} {fmt(raw(P)):>22s} {fmt(eeg_only(P)):>22s} {fmt(hybrid(P, Pn)):>22s}")
    fns = [lambda P: {n: sess_metrics(n, P[n]) for n in NAMES}, lambda P: hybrid_per(P, None),
           lambda P: hybrid_per(P, Pn)]
    nested = [nested_weight(Pr, Pv, list(w_grid), fn) for fn in fns]
    print(f"{'ансамбль, w вложенно (LOSubjO)':34s} " + " ".join(f"{fmt(s):>22s}" for s, _ in nested))
    print(f"{'   выбранные w':34s} " + " ".join(f"{str(sorted(set(c.values()))):>22s}" for _, c in nested))
    for w in w_grid[1:]:
        Pm = mix(Pr, Pv, w)
        print(f"{'ансамбль, w=' + str(w):34s} {fmt(raw(Pm)):>22s} {fmt(eeg_only(Pm)):>22s} {fmt(hybrid(Pm, Pn)):>22s}")


def baselines():
    L = lambda f: pickle.load(open(R / "data/probs" / f, "rb"))
    return L("eeg_ts2s_all.pkl"), L("nirs_cascade_ens.pkl")


def summarize(spec, Pr, Pn, full=False, a_grid=(0.5, 0.7), w_grid=(0.2, 0.3, 0.4, 0.5)):
    Pdl, Emb = load_tags(spec.split("+"))
    Pw = {n: within_head(Emb[n], n) for n in NAMES}
    print(f"== {spec}")
    print(f"  DL пул: raw {fmt(raw(Pdl))} | EEG pr+EMA {fmt(eeg_only(Pdl))} | within-emb raw {fmt(raw(Pw))}")
    variants = {"pool": Pdl}
    for a in a_grid:
        variants[f"ad{a}"] = mix(Pw, Pdl, a)
    for vn, Pv in variants.items():
        sc, ch = nested_weight(Pr, Pv, [0.0] + list(w_grid))
        line = f"  [{vn}] alone {fmt(raw(Pv))} | ens nested {fmt(sc)} w∈{sorted(set(ch.values()))}"
        for w in w_grid if full else (0.3, 0.4):
            Pm = mix(Pr, Pv, w)
            line += f" | w={w}: raw {raw(Pm)[2]:.2f} EEG {eeg_only(Pm)[2]:.2f} hyb {fmt(hybrid(Pm, Pn))}"
        print(line, flush=True)
    return Pdl, Pw


def main():
    argv = sys.argv[1:]
    opt = {}
    for k in ("--save", "--a", "--w"):
        if k in argv:
            i = argv.index(k); opt[k] = argv[i + 1]; del argv[i:i + 2]
    full = "--full" in argv
    specs = [x for x in argv if not x.startswith("--")]
    Pr, Pn = baselines()
    print(f"риманова eeg_ts2s_all: raw {fmt(raw(Pr))} | EEG pr+EMA {fmt(eeg_only(Pr))} | hyb {fmt(hybrid(Pr, Pn))}")
    for spec in specs:
        Pdl, Pw = summarize(spec, Pr, Pn, full)
        if "--table" in argv:
            final_table(Pr, Pn, mix(Pw, Pdl, float(opt.get("--a", 0.5))))
    if "--save" in opt:
        from NIRS_IMC_EEG_fNIRS.src.eeg.common import save_probs
        a, w = float(opt["--a"]), float(opt["--w"])
        Pv = mix(Pw, Pdl, a) if a < 1 else Pdl
        Pm = mix(Pr, Pv, w)
        for P in (Pv, Pm):
            for n in P:
                P[n][W[n]["block"] < 2] = np.nan
        save_probs(f"dl_{opt['--save']}", Pv)
        save_probs(f"dl_{opt['--save']}_ens", Pm)
        print(f"saved eeg_dl_{opt['--save']}.pkl: raw {fmt(raw(Pv))}; _ens (w={w}): raw {fmt(raw(Pm))} "
              f"EEG {fmt(eeg_only(Pm))} hyb {fmt(hybrid(Pm, Pn))}")


if __name__ == "__main__":
    main()
