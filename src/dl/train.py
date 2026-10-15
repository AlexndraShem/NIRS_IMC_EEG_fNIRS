"""Предобучение ЭЭГ-сети на пуле сессий (leave-one-subject-out) и выгрузка логитов/эмбеддингов
на оценочной сетке окон исключённого участника.

    python -m src.dl.train --model fbshallow --win 2 --norm ea --stats past --fb 4-8,8-13,13-20,20-30 \
        --fold N01 --gpu 0 --tag fbs_past
    (--fold all — обучение на всех 56 сессиях, для сдачи; --fold sess:<имя> — leave-one-session-out)

Вход сети: сигнал 125 Гц после причинного префильтра (src/dl/prep.py), опционально разложенный
причинными полосовыми фильтрами (`--fb`, каналы = полосы × 21). Нормировка сессии (по полосам):
  z  — деление на поканальный масштаб (медиана дисперсий 1-с окон);
  ea — плюс Euclidean Alignment: отбеливание средней ковариацией (shrinkage 0.1).
Откуда статистики (`--stats`):
  b1   — блок 1 сессии (все отсчёты; блок 1 не оценивается и завершён до первого окна оценки);
  past — как референс римановой модели: для окон блока k ≥ 2 — отсчёты State>0 блоков < k
         (пересчёт в fit_block), для окон блока 1 — блок 1; обучающие сессии пула — по всей сессии.
Выход:
  data/models/dl/<tag>_<fold>.pt   — веса + конфиг;
  <scratch>/dl/out/<tag>/<fold>.npz — {сессия__logits, сессия__emb} на окнах data/features/v1_win1.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.signal as ss
import torch
import torch.nn.functional as F

from src.dataset.io import PROJECT_DIR
from src.dataset.windows import confusion, metrics
from src.dl.models import build
from src.dl.prep import FS_DL, SCRATCH, load_prep

MODEL_DIR = PROJECT_DIR / "data" / "models" / "dl"
OUT_DIR = SCRATCH / "out"


def parse_fb(s):
    return [tuple(float(v) for v in b.split("-")) for b in s.split(",")] if s else []


def fb_sos(bands, fs=FS_DL, order=4):
    return [ss.butter(order, b, "bandpass", fs=fs, output="sos") for b in bands]


def split_bands(x, bands):
    """(n, C) → (B, n, C): причинные полосовые фильтры по всей сессии; без полос — B=1."""
    if not bands:
        return x[None].astype(np.float64)
    return np.stack([ss.sosfilt(s, x.astype(np.float64), axis=0) for s in fb_sos(bands)])


def band_stats(xb, mask, win=int(FS_DL)):
    """xb (B, n, C) → масштабы (B, C) и ковариации (B, C, C) по отсчётам mask (1-с окна)."""
    idx = np.flatnonzero(mask)
    # разбиваем на непрерывные куски, в каждом — целые 1-с окна
    starts = []
    if len(idx):
        br = np.flatnonzero(np.diff(idx) > 1)
        for s0, s1 in zip(np.r_[0, br + 1], np.r_[br + 1, len(idx)]):
            a, b = idx[s0], idx[s1 - 1] + 1
            starts += list(range(a, b - win + 1, win))
    starts = np.array(starts, int)
    seg = xb[:, starts[:, None] + np.arange(win)[None]]          # (B, n_w, win, C)
    seg = seg - seg.mean(2, keepdims=True)
    var = (seg ** 2).mean(2)                                        # (B, n_w, C)
    d = np.sqrt(np.median(var, 1))
    d = np.maximum(d, 1e-3 * np.median(d, 1, keepdims=True) + 1e-30)
    segn = seg / d[:, None, None, :]
    cov = np.einsum("bntc,bntd->bcd", segn, segn) / (len(starts) * win)
    return d, cov


def ea_matrix(cov, shrink=0.1):
    C = len(cov)
    c = (1 - shrink) * cov + shrink * np.trace(cov) / C * np.eye(C)
    w, V = np.linalg.eigh(c)
    return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def apply_norm(xb, d, cov, mode, clip=20.0, shrink=0.1):
    """xb (B, n, C) → (n, B·C) float32."""
    out = []
    for b in range(len(xb)):
        M = np.diag(1.0 / d[b])
        if mode == "ea":
            M = M @ ea_matrix(cov[b], shrink)
        out.append(xb[b] @ M)
    return np.clip(np.concatenate(out, 1), -clip, clip).astype(np.float32)


def train_ends(d, L, hom=int(FS_DL)):
    """Концы (в отсчётах 125 Гц, искл.) всех окон, чьи последние 1 с однородны и State>0."""
    st = d["st"].astype(int)
    seg = np.r_[0, np.cumsum(np.diff(st) != 0)]
    e = np.arange(max(L, hom), len(st) + 1)
    ok = (seg[e - hom] == seg[e - 1]) & (st[e - 1] > 0)
    return e[ok], st[e[ok] - 1] - 1


def grid_ends(d):
    """Сетка оценки (250 Гц, искл.) → конец в отсчётах 125 Гц: берём j с 2j < e."""
    return (d["ends"] + 1) // 2


def gather(X, ends, L):
    idx = ends[:, None] + torch.arange(-L, 0, device=X.device)[None]
    return X[idx].transpose(1, 2)              # (B, C, L)


@torch.no_grad()
def infer(model, x, ends, L, bs=2048):
    model.eval()
    X = torch.from_numpy(x).cuda()
    e = torch.from_numpy(np.maximum(ends, L)).cuda()
    lo, em = [], []
    for i in range(0, len(e), bs):
        l, m = model(gather(X, e[i:i + bs], L))
        lo.append(l.float().cpu()); em.append(m.float().cpu())
    return torch.cat(lo).numpy(), torch.cat(em).numpy()


def train_input(d, bands, mode, stats):
    xb = split_bands(d["x"], bands)
    mask = (d["bl"] == 1) if stats == "b1" else (d["st"] > 0)
    return apply_norm(xb, *band_stats(xb, mask), mode)


def test_infer(model, d, bands, mode, stats, L):
    """Логиты/эмбеддинги на сетке оценки с нормировкой, доступной онлайн в момент окна."""
    xb = split_bands(d["x"], bands)
    ends = grid_ends(d)
    blk = d["block"]
    if stats == "b1":
        return infer(model, apply_norm(xb, *band_stats(xb, d["bl"] == 1), mode), ends, L)
    lo = np.zeros((len(ends), 3), np.float32); em = None
    for k in range(2, 8):
        sel = (blk == k) | ((blk == 1) if k == 2 else False)
        if not sel.any():
            continue
        x = apply_norm(xb, *band_stats(xb, (d["bl"] < k) & (d["st"] > 0)), mode)
        l, m = infer(model, x, ends[sel], L)
        if em is None:
            em = np.zeros((len(ends), m.shape[1]), np.float32)
        lo[sel], em[sel] = l, m
    return lo, em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fbshallow")
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--norm", default="ea")
    ap.add_argument("--stats", default="b1")
    ap.add_argument("--fb", default="")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="", help="несколько сидов подряд в одном процессе (данные грузятся один раз)")
    ap.add_argument("--save-dir", default="", help="сохранять веса как <save-dir>/<tag>_s<seed>.pt, без выгрузки выходов")
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--eval-every", type=int, default=0)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(1)
    L = int(round(a.win * FS_DL))
    bands = parse_fb(a.fb)

    data = load_prep()
    if a.fold == "all":
        te_names = []
    elif a.fold.startswith("sess:"):
        te_names = [a.fold[5:]]
    elif a.fold.startswith("chrono:"):          # только сессии раньше этой (имя начинается с даты)
        te_names = [n for n in data if n >= a.fold[7:]]
    else:
        te_names = [n for n, d in data.items() if d["subject"] == a.fold]
    tr_names = [n for n in data if n not in te_names]

    xs, E, Y, off = [], [], [], 0
    for n in tr_names:
        x = train_input(data[n], bands, a.norm, a.stats)
        e, y = train_ends(data[n], L)
        xs.append(x); E.append(e + off); Y.append(y); off += len(x)
    X = torch.from_numpy(np.concatenate(xs)).cuda()
    del xs
    E = np.concatenate(E); Y = np.concatenate(Y)
    by_cls = [torch.from_numpy(E[Y == k]).cuda() for k in range(3)]
    print(f"[{a.tag} {a.fold}] train sess={len(tr_names)} windows={len(E)} C={X.shape[1]}", flush=True)

    for seed in ([int(v) for v in a.seeds.split(",")] if a.seeds else [a.seed]):
        a.seed = seed
        torch.manual_seed(seed); np.random.seed(seed)
        train_one(a, X, by_cls, data, te_names, tr_names, bands, L)


def train_one(a, X, by_cls, data, te_names, tr_names, bands, L):
    kw = json.loads(a.kw)
    model = build(a.model, C=X.shape[1], T=L, **kw).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    C = X.shape[1]

    def evaluate(out=None):
        res = {}
        for n in te_names:
            lo, em = test_infer(model, data[n], bands, a.norm, a.stats, L)
            if out is not None:
                out[n + "__logits"] = lo; out[n + "__emb"] = em
            d = data[n]; m = d["block"] >= 2
            res[n] = metrics(confusion(d["y"][m], lo[m].argmax(1) + 1))
        f = np.mean([r["min_f1"] for r in res.values()]); r = np.mean([r["recall"] for r in res.values()])
        return f, r

    t0 = time.time()
    for step in range(a.steps):
        model.train()
        k = torch.randint(0, 3, (a.bs,), device="cuda")
        idx = torch.empty(a.bs, dtype=torch.long, device="cuda")
        for c in range(3):
            m = k == c
            idx[m] = by_cls[c][torch.randint(0, len(by_cls[c]), (int(m.sum()),), device="cuda")]
        xb = gather(X, idx, L)
        if a.aug > 0:
            xb = xb * torch.exp(0.15 * a.aug * torch.randn(a.bs, C, 1, device="cuda"))
            xb = xb + 0.1 * a.aug * torch.randn_like(xb)
            xb = xb * (torch.rand(a.bs, C, 1, device="cuda") > 0.1 * a.aug)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, k, label_smoothing=a.ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if a.eval_every and te_names and (step + 1) % a.eval_every == 0:
            f, r = evaluate()
            print(f"  step {step + 1} loss={loss.item():.3f} test minF1={f:.3f} recall={r:.3f} "
                  f"pts={60 * f + 30 * r:.1f} t={time.time() - t0:.0f}s", flush=True)

    if a.save_dir:
        from pathlib import Path
        Path(a.save_dir).mkdir(parents=True, exist_ok=True)
        cfg = dict(model=a.model, win=a.win, L=L, norm=a.norm, stats=a.stats, bands=bands, kw=kw, fs=FS_DL,
                   fold=a.fold, steps=a.steps, seed=a.seed, train_sessions=tr_names)
        torch.save({"state": model.state_dict(), "cfg": cfg}, Path(a.save_dir) / f"{a.tag}_s{a.seed}.pt")
        print(f"[{a.tag} {a.fold} s{a.seed}] saved, time={time.time() - t0:.0f}s", flush=True)
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(model=a.model, win=a.win, L=L, norm=a.norm, stats=a.stats, bands=bands, kw=kw, fs=FS_DL,
               fold=a.fold, steps=a.steps, seed=a.seed, train_sessions=tr_names)
    torch.save({"state": model.state_dict(), "cfg": cfg}, MODEL_DIR / f"{a.tag}_{a.fold.replace(':', '_')}.pt")
    if te_names:
        out = {}
        f, r = evaluate(out)
        (OUT_DIR / a.tag).mkdir(parents=True, exist_ok=True)
        np.savez(OUT_DIR / a.tag / f"{a.fold.replace(':', '_')}.npz", **out)
        print(f"[{a.tag} {a.fold}] FINAL test minF1={f:.3f} recall={r:.3f} pts={60 * f + 30 * r:.1f} "
              f"time={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
