"""Эксперимент 2b: варианты кросс-субъектного переноса на рецентрированном TS.

TS-признаки (см. exp_cross.feats_session) кэшируются в scratch-npz для повторных прогонов.
Варианты: вес смеси, C пул-LR, MLP-пул на GPU, рекалибровка пул-логитов по прошлым блокам.

python -m src.eeg.exp_cross2 <win> <chans> <cache_dir>
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_eeg_cache
from NIRS_IMC_EEG_fNIRS.src.eeg.common import report, save_probs, score, subcov
from NIRS_IMC_EEG_fNIRS.src.eeg.exp_cross import CHSETS, feats_session

_P = {}


def get_feats(win, chn, cache_dir):
    f = Path(cache_dir) / f"tsF_w{win:g}_{chn}.pkl"
    data = load_eeg_cache(win)
    names = list(data)
    if f.exists():
        return data, pickle.load(open(f, "rb"))
    with ProcessPoolExecutor(10) as ex:
        F = dict(zip(names, ex.map(feats_session, [(subcov(data[n]["covs"], CHSETS[chn]).astype(np.float64),
                                                   data[n]["block"], "ts") for n in names])))
    F = {n: {k: v.astype(np.float32) for k, v in d.items()} for n, d in F.items()}
    pickle.dump(F, open(f, "wb"), protocol=4)
    return data, F


def fit_pool_lr(args):
    s, C = args
    F, data, names = _P["F"], _P["data"], _P["names"]
    tr = [n for n in names if data[n]["subject"] != s]
    X = np.vstack([F[n]["pool"] for n in tr]); y = np.concatenate([data[n]["y"] for n in tr])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(sc.transform(X), y)
    return sc, m


def pool_mlp(data, F, names, subjects, epochs=30, hidden=256, drop=0.5, wd=1e-3, seed=0):
    """MLP на TS пула (GPU), leave-subject-out. Возвращает {subject: predict_fn}."""
    import torch
    import torch.nn as nn
    dev = "cuda"
    out = {}
    for s in subjects:
        torch.manual_seed(seed)
        tr = [n for n in names if data[n]["subject"] != s]
        X = np.vstack([F[n]["pool"] for n in tr]); y = np.concatenate([data[n]["y"] for n in tr]) - 1
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - mu) / sd, device=dev); yt = torch.tensor(y, device=dev)
        w = torch.tensor(len(y) / (3 * np.bincount(y, minlength=3)), dtype=torch.float32, device=dev)
        net = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Dropout(drop),
                            nn.Linear(hidden, 3)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
        lossf = nn.CrossEntropyLoss(weight=w)
        for ep in range(epochs):
            perm = torch.randperm(len(yt), device=dev)
            net.train()
            for i in range(0, len(perm), 512):
                j = perm[i:i + 512]
                opt.zero_grad(); lossf(net(Xt[j]), yt[j]).backward(); opt.step()
        net.eval()

        def pred(Xn, net=net, mu=mu, sd=sd):
            with torch.no_grad():
                return torch.softmax(net(torch.tensor((Xn - mu) / sd, device=dev, dtype=torch.float32)), 1).cpu().numpy()
        out[s] = pred
    return out


def mix(Pa, Pb, a):
    out = {}
    for n in Pa:
        p = np.exp((1 - a) * np.log(np.clip(Pa[n], 1e-9, 1)) + a * np.log(np.clip(Pb[n], 1e-9, 1)))
        out[n] = p / p.sum(1, keepdims=True)
    return out


def run_variant(data, F, names, predict_pool, recal=False):
    """Возвращает (within, pool, pool_recal) вероятности."""
    Pw, Pp, Pr = {}, {}, {}
    for n in names:
        d = data[n]
        Pw[n] = np.full((len(d["y"]), 3), np.nan); Pp[n] = Pw[n].copy(); Pr[n] = Pw[n].copy()
        for b in range(2, 8):
            Xb = F[n][b]; blk = d["block"][d["block"] <= b]; yb = d["y"][d["block"] <= b]
            tr, te = blk < b, blk == b
            if Pw is not None:
                m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xb[tr], yb[tr])
                Pw[n][d["block"] == b] = m.predict_proba(Xb[te])
            pp = predict_pool(n, Xb)
            Pp[n][d["block"] == b] = pp[te]
            if recal:   # LR на лог-вероятностях пула по прошлым блокам текущей сессии
                L = np.log(np.clip(pp, 1e-9, 1))
                r = LogisticRegression(C=1.0, max_iter=200).fit(L[tr], yb[tr])
                Pr[n][d["block"] == b] = r.predict_proba(L[te])
    return Pw, Pp, Pr


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    win, chn, cdir = float(sys.argv[1]), sys.argv[2], sys.argv[3]
    what = sys.argv[4:] or ["lr", "mlp"]
    data, F = get_feats(win, chn, cdir)
    names = list(data); subjects = sorted({data[n]["subject"] for n in names})
    tag = f"w{win:g} {chn}"
    res = {}
    if "lr" in what or "lr1" in what:
        _P.update(F=F, data=data, names=names)
        for C in ([0.01, 0.1] if "lr" in what else [0.01]):
            with ProcessPoolExecutor(10, mp_context=mp.get_context("fork")) as ex:
                pools = dict(zip(subjects, ex.map(fit_pool_lr, [(s, C) for s in subjects])))
            pp = lambda n, X: pools[data[n]["subject"]][1].predict_proba(pools[data[n]["subject"]][0].transform(X))
            Pw, Pp, Pr = run_variant(data, F, names, pp, recal=True)
            res[f"lrC{C}"] = (Pw, Pp, Pr)
    if "mlp" in what:
        for hid, ep in [(256, 20), (512, 40)]:
            mlps = pool_mlp(data, F, names, subjects, epochs=ep, hidden=hid)
            pp = lambda n, X: mlps[data[n]["subject"]](X)
            res[f"mlp{hid}e{ep}"] = run_variant(data, F, names, pp, recal=True)
    for k, (Pw, Pp, Pr) in res.items():
        report(f"{tag} within", score(data, Pw))
        report(f"{tag} pool {k}", score(data, Pp))
        report(f"{tag} pool {k} recal", score(data, Pr))
        for a in [0.5, 0.65, 0.8, 0.9]:
            report(f"{tag} mix within+{k} a={a}", score(data, mix(Pw, Pp, a)))
            report(f"{tag} mix within+{k}recal a={a}", score(data, mix(Pw, Pr, a)))
    pickle.dump(res, open(Path(cdir) / f"cross2_res_w{win:g}_{chn}_{'_'.join(what)}.pkl", "wb"))
