"""Потоковая DL-модель ЭЭГ (numpy, без torch): прототип для внедрения в solution/.

    OnlineDL(model_npz_paths, a=0.5)
      push(eeg_chunk)            — (n, 21) сырое ЭЭГ 250 Гц; причинная фильтрация с сохранением zi
      predict_proba(end) -> (3,) | None — P ∝ P_within^(1−a)·P_pool^a (до fit_block(1) — None)
      fit_block(b, labels)       — labels = {"states", "eeg_indices", "block"} как у судьи

Нормировка (`stats=past`): поканальный масштаб + EA по отсчётам State>0 всех завершённых блоков,
пересчёт в каждом fit_block. Эмбеддинги окон запоминаются в момент predict (т.е. с нормировкой,
действовавшей тогда); для окон блока 1 (predict на них не оценивается) — считаются в fit_block(1).
Голова within — sLDA(0.8) на эмбеддингах размеченных окон прошлых блоков (каждое 2-е окно).

    python -m src.dl.online <tag> <сессия> [n_seeds_tags через +]   # паритет с оффлайн и тайминги
"""
from __future__ import annotations

import sys
import time

import numpy as np
import scipy.signal as ss

from src.dl.evaluate import ShrinkLDA
from src.dl.np_infer import NpFBShallow
from src.dl.prep import DEC, dl_sos
from src.dl.train import apply_norm, band_stats, split_bands

FLOOR = 1e-9


def _log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


class OnlineDL:
    def __init__(self, model_paths, a=0.5, shrink=0.8, stride=2, fs=250.0, cap_s=700):
        self.models = [NpFBShallow(p) for p in model_paths]
        self.L = self.models[0].L
        self.a, self.shrink, self.stride = a, shrink, stride
        self.fs = fs
        self.sos = dl_sos(fs)
        self.zi = None
        self.x0 = None
        self.n = 0                                          # отсчётов 250 Гц принято
        cap = int(cap_s * fs / DEC)
        self.buf = None                                     # (cap, C) отфильтрованный 125 Гц
        self.cap = cap
        self.nd = 0                                         # отсчётов 125 Гц в буфере
        self.lab = np.zeros(cap, np.int8)                   # State (125 Гц) завершённых блоков, 0 — нет
        self.lab250 = np.zeros(cap * DEC, np.int8)          # то же на 250 Гц — разметка окон
        self.M = None                                       # матрица нормировки
        self.emb = {}                                       # end(250 Гц) -> эмбеддинг
        self.head = None

    # ------------------------------------------------------------------ поток
    def push(self, x):
        x = np.asarray(x, np.float64)
        if len(x) == 0:
            return
        if self.x0 is None:
            self.x0 = x[0].copy()
            self.zi = np.zeros((self.sos.shape[0], 2, x.shape[1]))
            self.buf = np.zeros((self.cap, x.shape[1]), np.float32)
        y, self.zi = ss.sosfilt(self.sos, x - self.x0, axis=0, zi=self.zi)
        keep = y[(-self.n) % DEC::DEC]                      # глобальные чётные отсчёты
        self.n += len(x)
        if self.nd + len(keep) > self.cap:                  # растим буфер при длинной сессии
            self.buf = np.concatenate([self.buf, np.zeros_like(self.buf)])
            self.lab = np.concatenate([self.lab, np.zeros_like(self.lab)])
            self.lab250 = np.concatenate([self.lab250, np.zeros_like(self.lab250)])
            self.cap *= 2
        self.buf[self.nd:self.nd + len(keep)] = keep
        self.nd += len(keep)

    def _forward(self, end):
        e = (end + 1) // DEC
        x = self.buf[e - self.L:e].astype(np.float64) @ self.M
        x = np.clip(x, -20, 20).T
        lps, embs = [], []
        for m in self.models:
            lo, em = m.forward(x)
            lps.append(_log_softmax(lo)); embs.append(em)
        return np.mean(lps, 0), np.concatenate(embs)

    def predict_proba(self, end=None):
        end = self.n if end is None else end
        if self.M is None or (end + 1) // DEC < self.L:
            return None
        lp, em = self._forward(end)
        self.emb[end] = em
        if self.head is not None:
            lw = self.head.log_proba(em[None])[0]
            lp = (1 - self.a) * lw + self.a * lp
        p = np.exp(lp - lp.max())
        return p / p.sum()

    # ------------------------------------------------------------------ дообучение
    def fit_block(self, b, labels):
        idx = np.asarray(labels["eeg_indices"]); st = np.asarray(labels["states"])
        d = idx % DEC == 0
        self.lab[idx[d] // DEC] = st[d]
        self.lab250[idx] = st
        mask = self.lab[:self.nd] > 0
        xb = split_bands(self.buf[:self.nd], [])
        d_, cov = band_stats(xb, mask)
        I = np.eye(xb.shape[2])
        self.M = apply_norm(I[None], d_, cov, "ea", clip=np.inf)          # (C, C): x @ M
        if b == 1:                                          # эмбеддинги окон блока 1 — сейчас,
            ends = np.array(self._grid(idx.min(), idx.max() + 1))   # одной свёрткой по всему блоку
            if len(ends):
                e_d = (ends + 1) // DEC
                j0 = e_d.min() - self.L
                x = np.clip(self.buf[j0:e_d.max()].astype(np.float64) @ self.M, -20, 20).T
                E = np.concatenate([m.embed_many(x, e_d - self.L - j0)[1] for m in self.models], 1)
                self.emb.update({int(e): E[i] for i, e in enumerate(ends)})
        ends, ys = self._labeled_windows()
        if len(ends) >= 6 and len(np.unique(ys)) >= 2:
            sel = slice(None, None, self.stride)
            X = np.stack([self.emb[e] for e in ends[sel]])
            self.head = ShrinkLDA().fit(X, ys[sel], self.shrink)

    def _grid(self, lo, hi, win=250, step=62.5):
        g = np.round(np.arange(win, self.n + 1e-9, step)).astype(int)
        return [int(e) for e in g if lo + win <= e <= hi]

    def _labeled_windows(self):
        """Окна с запомненным эмбеддингом, у которых последние 1 с однородны по State>0."""
        ends, ys = [], []
        for e in sorted(self.emb):
            s = self.lab250[max(e - 250, 0):e]
            if len(s) and s[0] > 0 and (s == s[0]).all():
                ends.append(e); ys.append(int(s[0]))
        return np.array(ends), np.array(ys)


def simulate(tags, name):
    """Поток по сессии как у судьи: push до конца окна → predict; в конце блока — fit_block.
    Сверка с оффлайн-выходом evaluate (адаптация a=0.5) и тайминги."""
    import pickle
    from src.dataset.io import PROJECT_DIR, RAW_DIR, load_session
    from src.dl.train import MODEL_DIR
    from src.eeg.cache import load_raw64
    s = load_session(RAW_DIR / "Data" / f"{name}.mat")
    x = load_raw64(RAW_DIR / "Data" / f"{name}.mat")
    z = np.load(PROJECT_DIR / "data/features/v1_win1" / f"{name}.npz")
    paths = [MODEL_DIR / f"{t}_{s.subject}.npz" for t in tags]
    m = OnlineDL(paths)
    blocks, states = s.blocks.astype(int), s.states.astype(int)
    bl_end = {b: int(np.flatnonzero(blocks == b).max()) + 1 for b in range(1, 8)}
    events = sorted([(int(e), 0, i) for i, e in enumerate(z["ends"])] + [(bl_end[b], 1, b) for b in bl_end])
    P = np.full((len(z["ends"]), 3), np.nan)
    tp, tf, pos = [], [], 0
    for e, kind, i in events:
        m.push(x[pos:e]); pos = e
        if kind == 0:
            t0 = time.perf_counter(); p = m.predict_proba(e); tp.append(time.perf_counter() - t0)
            if p is not None and z["block"][i] >= 2:
                P[i] = p
        else:
            msk = (blocks == i) & (states > 0)
            t0 = time.perf_counter()
            m.fit_block(i, {"states": states[msk], "eeg_indices": np.flatnonzero(msk), "block": i})
            tf.append(time.perf_counter() - t0)
    tp, tf = np.array(tp) * 1e3, np.array(tf)
    print(f"{name}: predict mean {tp.mean():.2f} ms p99 {np.percentile(tp, 99):.2f} max {tp.max():.2f}; "
          f"fit_block " + " ".join(f"{t:.2f}" for t in tf) + " s")
    return P


def offline_ref(tags, name):
    """Оффлайн-выход evaluate (DL с адаптацией a=0.5) для одной сессии."""
    from src.dl.evaluate import load_tags, mix, within_head
    Pdl, Emb = load_tags(tags)
    Pw = within_head(Emb[name], name)
    return mix({name: Pw}, {name: Pdl[name]}, 0.5)[name]


if __name__ == "__main__":
    from src.dataset.io import subject_of
    from src.dl.np_infer import export
    tags = sys.argv[1].split("+")
    for name in sys.argv[2:]:
        for t in tags:
            export(f"{t}_{subject_of(name)}")
        P = simulate(tags, name)
        ref = offline_ref(tags, name)
        ok = np.isfinite(P).all(1) & np.isfinite(ref).all(1)
        print(f"  паритет с оффлайн: окон {ok.sum()}, max|ΔP|={np.abs(P[ok] - ref[ok]).max():.2e}, "
              f"argmax совп. {np.mean(P[ok].argmax(1) == ref[ok].argmax(1)):.4f}", flush=True)
