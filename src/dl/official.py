"""DL-вероятности (FBShallow, stats=past, адаптация sLDA по эмбеддингам) на ОФИЦИАЛЬНОЙ сетке судьи
(концы окон из dev-прогонов `<runs>/<сессия>/probs.csv`, 250 + 62·j) — точная эмуляция онлайна:

* окно блока k нормируется статистиками (масштаб + EA) по State>0 блоков < k (последний fit_block);
* эмбеддинги окон блока 1 — на сетке 250 + 62·j в fit_block(1) (нормировка по блоку 1);
* голова within для окна блока k — sLDA(0.8) на размеченных окнах блоков < k (блок 1 + окна,
  на которых вызывался predict), каждое 2-е; P_dl ∝ P_within^(1−a)·P_pool^a.

    python -m src.dl.official --runs <dev_losubjo> --tags fb_w2_eapast+..._s3 --fold losubjo --out <pkl> [--gpu 0]

--fold: losubjo — модель <tag>_<участник>.pt; all — <tag>_all.pt; dir:<каталог> — <каталог>/<сессия>/<tag>.pt.
Выход: pkl {сессия: dict(e=концы, pool=(n,3) лог-вер. пула, emb_ends/…, P=(n,3) итог a=0.5)}.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from NIRS_IMC_EEG_fNIRS.src.dataset.io import subject_of
from NIRS_IMC_EEG_fNIRS.src.dl.evaluate import ShrinkLDA
from NIRS_IMC_EEG_fNIRS.src.dl.models import build
from NIRS_IMC_EEG_fNIRS.src.dl.prep import DEC, load_prep
from NIRS_IMC_EEG_fNIRS.src.dl.train import MODEL_DIR, apply_norm, band_stats, split_bands

GRID0, GRID_STEP, WIN = 250, 62, 250


def load_model(path, gpu):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = build(cfg["model"], C=21, T=cfg["L"], **cfg["kw"])
    m.load_state_dict(ck["state"]); m.eval().cuda(gpu)
    return m, cfg


def model_path(tag, fold, name):
    if fold == "losubjo":
        return MODEL_DIR / f"{tag}_{subject_of(name)}.pt"
    if fold == "all":
        return MODEL_DIR / f"{tag}_all.pt"
    if fold.startswith("dir:"):
        return Path(fold[4:]) / name / f"{tag}.pt"
    raise ValueError(fold)


def homogeneous(st, e):
    seg = np.r_[0, np.cumsum(np.diff(st) != 0)]
    return (e >= WIN) & (seg[np.maximum(e - WIN, 0)] == seg[e - 1]) & (st[e - 1] > 0)


@torch.no_grad()
def session_dl(name, d, ends_pred, st250, bl250, models, a=0.5, shrink=0.8, stride=2):
    """models: список (torch-модель, L). Возвращает лог-вер. пула и итоговые P на ends_pred."""
    bl_end = {b: int(np.flatnonzero(bl250 == b).max()) + 1 for b in range(1, 8)}
    g = np.arange(GRID0, bl_end[1] + 1, GRID_STEP)
    ends_b1 = g[homogeneous(st250, g) & (bl250[g - 1] == 1)]
    allends = np.r_[ends_b1, ends_pred]
    comp = np.array([sum(bl_end[b] <= e for b in bl_end) for e in allends])     # завершённых блоков
    comp[: len(ends_b1)] = 1                                                    # блок 1 — в fit_block(1)
    xb = split_bands(d["x"], [])
    lp = np.zeros((len(allends), 3)); emb = None
    y_all = st250[allends - 1]
    lab_all = homogeneous(st250, allends)
    blk_all = bl250[allends - 1]
    for c in np.unique(comp):
        sel = comp == c
        # статистика — по отсчётам размеченных окон завершённых блоков (как solution/eeg_dl.py)
        mask = np.zeros(len(d["x"]), bool)
        for e in allends[lab_all & (blk_all <= c)]:
            mask[max((e - WIN + 1) // DEC, 0):(e + 1) // DEC] = True
        M = apply_norm(np.eye(21)[None], *band_stats(xb, mask), "ea", clip=np.inf)
        x = np.clip(d["x"].astype(np.float64) @ M, -20, 20).astype(np.float32)
        X = torch.from_numpy(x).cuda()
        ed = torch.from_numpy((allends[sel] + 1) // DEC).cuda()
        ls, es = [], []
        for m, L in models:
            idx = ed[:, None] + torch.arange(-L, 0, device="cuda")[None]
            lo, em = m(X[idx].transpose(1, 2))
            ls.append(torch.log_softmax(lo.double(), 1).cpu().numpy()); es.append(em.double().cpu().numpy())
        lp[sel] = np.mean(ls, 0)
        E = np.concatenate(es, 1)
        if emb is None:
            emb = np.zeros((len(allends), E.shape[1]))
        emb[sel] = E
    y = st250[allends - 1]
    lab = homogeneous(st250, allends)
    blk = bl250[allends - 1]
    npred = len(ends_pred)
    out_lp = lp[len(ends_b1):].copy()
    P = np.full((npred, 3), np.nan)
    for k in range(2, 8):
        te = np.flatnonzero(blk[len(ends_b1):] == k)
        if not len(te):
            continue
        tr = np.flatnonzero(lab & (blk < k))[::stride]
        h = ShrinkLDA().fit(emb[tr], y[tr], shrink)
        lw = h.log_proba(emb[len(ends_b1) + te])
        z = (1 - a) * lw + a * out_lp[te]
        z -= z.max(1, keepdims=True)
        P[te] = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return dict(e=ends_pred, pool=out_lp, P=P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--fold", default="losubjo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--names", default="")
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    from NIRS_IMC_EEG_fNIRS.src.dataset.io import RAW_DIR, load_session
    names = sorted(p.name for p in Path(a.runs).iterdir() if (p / "probs.csv").exists())
    if a.names:
        names = [n for n in names if n in a.names.split(",")]
    data = load_prep(set(names))
    out = {}
    for n in names:
        s = load_session(RAW_DIR / "Data" / f"{n}.mat")
        e = pd.read_csv(Path(a.runs) / n / "probs.csv").end_sample.values.astype(int)
        paths = [model_path(t, a.fold, n) for t in a.tags.split("+")]
        if not all(p.exists() for p in paths):      # нет пула (ранние сессии хронологии) → без DL
            out[n] = dict(e=e, pool=np.full((len(e), 3), np.nan), P=np.full((len(e), 3), np.nan))
            print(n, "нет DL-пула", flush=True)
            continue
        models = []
        for p in paths:
            m, cfg = load_model(p, a.gpu)
            models.append((m, cfg["L"]))
        out[n] = session_dl(n, data[n], e, s.states.astype(int), s.blocks.astype(int), models)
        print(n, flush=True)
    pickle.dump(out, open(a.out, "wb"))


if __name__ == "__main__":
    main()
