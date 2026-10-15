"""Подбор слияния `StackedFusion` одной командой + честная вложенная оценка.

    python -m src.fusion.tune --eeg eeg_ts2s_all --nirs nirs_cascade_ens --out stack [--nested] [--jobs 4]

Вход: `data/probs/<eeg>.pkl`, `data/probs/<nirs>.pkl` — {сессия: (n_окон, 3)} на сетке
`data/features/v1_win1/<сессия>.npz` (ends, y, block); блок 1 — NaN (predict не вызывается).

Что делает:
1. Для каждой сессии прогоняет признаки `StackedFusion` (тот же код, что онлайн) по окнам блоков 2–7.
2. OOF по участникам (LOSubjO): мета-LR `hyb` (ЭЭГ+NIRS) и `eeg` (только ЭЭГ) обучаются на других
   участниках → логиты сессий участника.
3. По OOF-логитам всех сессий выбирает `post_tau_s`, сдвиг приора `bias` (гибрид, 60F+30P) и
   `eeg_bias` (y_eeg), а также правило `y_nirs` (nirs_prior, nirs_tau_s; критерий — min-F1 NIRS).
4. Финальные мета-LR обучаются на всех сессиях → `data/fusion/<out>_params.json` (готовый блок
   `fusion_params` для config.json) и `data/probs/fused_<out>.pkl` (OOF-выход гибрида после EMA и
   сдвига: argmax = y) + `data/probs/fused_<out>_metrics.json`.
5. `--nested`: честная оценка — для каждого участника s весь шаг 2–3 повторяется на остальных 13
   (внутренний LOSubjO), выбранные гиперпараметры применяются к логитам s из внешней модели.
   Рядом считается текущее `LogLinearFusion` (w 0.75, приор [0,2,2], τ 1.5) на тех же сессиях.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from NIRS_IMC_EEG_fNIRS.src.dataset.io import PROJECT_DIR
from NIRS_IMC_EEG_fNIRS.src.dataset.windows import confusion, metrics
from NIRS_IMC_EEG_fNIRS.src.fusion.stacked import DEFAULTS, FLOOR, ModalityFeatures, _clean

PROBS = PROJECT_DIR / "data" / "probs"
GRID = PROJECT_DIR / "data" / "features" / "v1_win1"
OUT_DIR = PROJECT_DIR / "data" / "fusion"

POST_TAUS = [0.0, 0.5, 1.0, 1.5, 2.0]
BIASES = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
NIRS_PRIORS = [0.0, 0.5, 1.0, 1.5, 2.0]
NIRS_TAUS = [0.0, 1.0]


# ------------------------------------------------------------------ данные
def subject(name):
    return name.split("_")[-1]


def load_sessions(eeg, nirs, feat):
    E = pickle.load(open(PROBS / f"{eeg}.pkl", "rb"))
    N = pickle.load(open(PROBS / f"{nirs}.pkl", "rb")) if nirs else {}
    S = {}
    for k in sorted(E):
        z = np.load(GRID / f"{k}.npz", allow_pickle=True)
        assert len(z["ends"]) == len(E[k]), f"{k}: сетка окон не совпадает"
        m = z["block"] >= 2
        pe, pn = E[k][m], (N[k][m] if k in N else None)
        t = z["ends"][m] / 250.0
        fe = ModalityFeatures(feat["eeg_taus"], feat["center_T"], feat["center_ref_tau"])
        fn = ModalityFeatures(feat["nirs_taus"], feat["center_T"], feat["center_ref_tau"])
        n = int(m.sum())
        Xe = np.full((n, fe.dim), np.nan)
        Xn = np.full((n, fn.dim), np.nan)
        for i in range(n):
            a = _clean(pe[i])
            if a is not None:
                Xe[i] = fe.step(a, t[i])
            b = _clean(pn[i]) if pn is not None else None
            if b is not None:
                Xn[i] = fn.step(b, t[i])
        ln = None if pn is None else np.log(np.clip(pn, FLOOR, 1))
        S[k] = dict(y=z["y"][m].astype(int), t=t, block=z["block"][m].astype(int), Xe=Xe, Xn=Xn,
                    le=np.log(np.clip(pe, FLOOR, 1)), ln=ln, subj=subject(k))
    return S


# ------------------------------------------------------------------ мета-модели
def _fit_lr(X, y, C):
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X, y)
    return {"W": m.coef_.tolist(), "b": m.intercept_.tolist()}


def fit_models(S, names, C):
    Xh, yh, Xe, ye = [], [], [], []
    for k in names:
        d = S[k]
        okE = np.isfinite(d["Xe"]).all(1)
        okH = okE & np.isfinite(d["Xn"]).all(1)
        Xe.append(d["Xe"][okE]); ye.append(d["y"][okE])
        if okH.any():
            Xh.append(np.hstack([d["Xe"], d["Xn"]])[okH]); yh.append(d["y"][okH])
    out = {"eeg": _fit_lr(np.vstack(Xe), np.concatenate(ye), C)}
    out["hyb"] = _fit_lr(np.vstack(Xh), np.concatenate(yh), C) if Xh else None
    return out


def _lsm(Z):
    Z = Z - Z.max(1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(1, keepdims=True))


def logits(S, k, models):
    """(лог-вероятности гибрида/деградации, лог-вероятности ЭЭГ-only) для окон сессии."""
    d = S[k]
    We, be = np.asarray(models["eeg"]["W"]), np.asarray(models["eeg"]["b"])
    Xe = np.nan_to_num(d["Xe"])
    le = _lsm(Xe @ We.T + be)
    ly = le.copy()
    okH = np.isfinite(d["Xn"]).all(1) & np.isfinite(d["Xe"]).all(1)
    if models.get("hyb") and okH.any():
        Wh, bh = np.asarray(models["hyb"]["W"]), np.asarray(models["hyb"]["b"])
        ly[okH] = _lsm(np.hstack([Xe, np.nan_to_num(d["Xn"])])[okH] @ Wh.T + bh)
    return ly, le


# ------------------------------------------------------------------ решающее правило
def ema_seq(P, t, tau):
    if tau <= 0:
        return P
    a = np.exp(-np.maximum(np.diff(t, prepend=t[0]), 0) / tau)
    out = np.empty_like(P)
    s = P[0]
    for i in range(len(P)):
        s = a[i] * s + (1 - a[i]) * P[i]
        out[i] = s
    return out


def sess_fp(y, yhat):
    m = metrics(confusion(y, yhat))
    return m["min_f1"], m["recall"]


def decide_seq(L, t, tau, bias):
    S = np.log(np.clip(ema_seq(np.exp(L), t, tau), FLOOR, None)) + np.r_[0, bias, bias]
    return S


def select_rule(S, names, L, kind="hyb"):
    """Выбор (post_tau, bias) по среднему по сессиям 60F+30P (kind='hyb'/'eeg') или
    (nirs_tau, nirs_prior) по min-F1 NIRS (kind='nirs'). L: {сессия: лог-вероятности}."""
    best, table = None, {}
    taus, biases = (NIRS_TAUS, NIRS_PRIORS) if kind == "nirs" else (POST_TAUS, BIASES)
    for tau in taus:
        sm = {k: np.log(np.clip(ema_seq(np.exp(L[k]), S[k]["t"], tau), FLOOR, None)) for k in names if L.get(k) is not None}
        for b in biases:
            fp = np.array([sess_fp(S[k]["y"], (sm[k] + [0, b, b]).argmax(1) + 1) for k in sm])
            val = fp[:, 0].mean() if kind == "nirs" else 60 * fp[:, 0].mean() + 30 * fp[:, 1].mean()
            table[(tau, b)] = val
            if best is None or val > best[0] + 1e-12:
                best = (val, tau, b)
    return best[1], best[2]


# ------------------------------------------------------------------ LOSubjO
_G = {}


def _fit_excl(excl):
    S, C = _G["S"], _G["C"]
    names = [k for k in S if S[k]["subj"] not in excl]
    return tuple(sorted(excl)), fit_models(S, names, C)


def oof_logits(S, names, models_by_subj):
    Ly, Le = {}, {}
    for k in names:
        Ly[k], Le[k] = logits(S, k, models_by_subj[S[k]["subj"]])
    return Ly, Le


def run(args):
    t0 = time.time()
    feat = {k: DEFAULTS[k] for k in ("eeg_taus", "nirs_taus", "center_T", "center_ref_tau")}
    if args.eeg_taus: feat["eeg_taus"] = [float(v) for v in args.eeg_taus.split(",")]
    if args.nirs_taus: feat["nirs_taus"] = [float(v) for v in args.nirs_taus.split(",")]
    if args.center_T is not None:
        feat["center_T"] = [float(v) for v in args.center_T.split(",") if v]
    S = load_sessions(args.eeg, args.nirs, feat)
    names = sorted(S)
    subs = sorted({S[k]["subj"] for k in names})
    print(f"сессий {len(names)}, участников {len(subs)}, признаки {feat}, {time.time()-t0:.0f} с", flush=True)
    _G["S"], _G["C"] = S, args.C

    # все нужные фиты: LOSubjO (исключить 1 участника) и, для --nested, исключить пару
    excls = [(s,) for s in subs]
    if args.nested:
        excls += [tuple(sorted((s, u))) for i, s in enumerate(subs) for u in subs[i + 1:]]
    with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("fork")) as ex:
        fits = dict(ex.map(_fit_excl, excls, chunksize=1))
    fits[()] = fit_models(S, names, args.C)
    print(f"мета-LR обучены ({len(fits)} шт.), {time.time()-t0:.0f} с", flush=True)

    # --- OOF-логиты (внешний LOSubjO) и выбор правила на всех сессиях → финальные params
    Ly, Le = oof_logits(S, names, {s: fits[(s,)] for s in subs})
    Ln = {k: S[k]["ln"] for k in names}
    tau, bias = select_rule(S, names, Ly, "hyb")
    etau, ebias = select_rule(S, names, Le, "eeg")
    if args.nirs_prior == "auto":
        ntau, nprior = select_rule(S, names, Ln, "nirs")
    else:                      # по умолчанию фиксировано: выбор по данным не лучше шума (см. --nested)
        ntau, nprior = 0.0, float(args.nirs_prior)
    params = dict(feat, post_tau_s=tau, bias=[0.0, bias, bias], eeg_bias=[0.0, ebias, ebias],
                  nirs_prior=[0.0, nprior, nprior], nirs_tau_s=ntau, models=fits[()],
                  source={"eeg": args.eeg, "nirs": args.nirs, "C": args.C})
    if etau != tau:
        print(f"  (для y_eeg лучший τ {etau}, используется общий post_tau_s {tau})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(OUT_DIR / f"{args.out}_params.json", "w"), indent=1)

    # --- OOF-выход (оптимистично только в выборе 3 чисел правила) → fused_<out>.pkl
    res = {"rule": {"post_tau_s": tau, "bias": bias, "eeg_bias": ebias, "nirs_prior": nprior, "nirs_tau_s": ntau}}
    fused, per = {}, {}
    for k in names:
        z = np.load(GRID / f"{k}.npz")
        full = np.full((len(z["ends"]), 3), np.nan)
        Sm = decide_seq(Ly[k], S[k]["t"], tau, bias)
        full[z["block"] >= 2] = np.exp(_lsm(Sm))
        fused[k] = full
        per[k] = sess_fp(S[k]["y"], Sm.argmax(1) + 1)
    pickle.dump(fused, open(PROBS / f"fused_{args.out}.pkl", "wb"))
    res["oof"] = _summ(per)

    if args.nested:
        out_h, out_e, out_n, out_b, chosen = {}, {}, {}, {}, {}
        for s in subs:
            inner = [k for k in names if S[k]["subj"] != s]
            test = [k for k in names if S[k]["subj"] == s]
            iLy, iLe = oof_logits(S, inner, {u: fits[tuple(sorted((s, u)))] for u in subs if u != s})
            it, ib = select_rule(S, inner, iLy, "hyb")
            et, eb = select_rule(S, inner, iLe, "eeg")
            nt, nb = select_rule(S, inner, {k: Ln[k] for k in inner}, "nirs")
            chosen[s] = (it, ib, eb, nt, nb)
            for k in test:
                d = S[k]
                out_h[k] = sess_fp(d["y"], decide_seq(Ly[k], d["t"], it, ib).argmax(1) + 1)
                out_e[k] = sess_fp(d["y"], decide_seq(Le[k], d["t"], it, eb).argmax(1) + 1)
                if Ln[k] is not None:
                    out_n[k] = sess_fp(d["y"], decide_seq(Ln[k], d["t"], nt, nb).argmax(1) + 1)
                L0 = d["le"] if d["ln"] is None else 0.75 * d["le"] + 0.25 * d["ln"]
                P0 = np.exp(_lsm(L0 + [0, 2, 2]))
                out_b[k] = sess_fp(d["y"], ema_seq(P0, d["t"], 1.5).argmax(1) + 1)
        res["nested"] = {"hyb": _summ(out_h), "eeg": _summ(out_e), "nirs": _summ(out_n),
                         "baseline_loglinear": _summ(out_b), "chosen": chosen}
    json.dump(res, open(PROBS / f"fused_{args.out}_metrics.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "per"}, indent=1, default=str))
    print(f"готово за {time.time()-t0:.0f} с → {OUT_DIR / (args.out + '_params.json')}")


def _summ(per):
    a = np.array(list(per.values()))
    F, P = float(a[:, 0].mean()), float(a[:, 1].mean())
    return {"min_f1": round(F, 4), "recall": round(P, 4), "pts": round(60 * F + 30 * P, 2), "n": len(a)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg", default="eeg_ts2s_all")
    ap.add_argument("--nirs", default="nirs_cascade_ens")
    ap.add_argument("--out", default="stack")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--eeg-taus", default=None, help="через запятую, напр. 0,0.5,1,2,4,8,16")
    ap.add_argument("--nirs-taus", default=None)
    ap.add_argument("--center-T", default=None, help="через запятую; пустая строка — без центрирования")
    ap.add_argument("--nirs-prior", default="1.0", help="сдвиг приора задачи для y_nirs или 'auto'")
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
