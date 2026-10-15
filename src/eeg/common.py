"""Общие хелперы ЭЭГ-экспериментов: каналы, TS-признаки, онлайн-цикл оценки, сохранение вероятностей."""
from __future__ import annotations

import pickle
import time

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.dataset.io import EEG_LABELS, PROJECT_DIR
from NIRS_IMC_EEG_fNIRS.src.dataset.windows import confusion, metrics
from NIRS_IMC_EEG_fNIRS.src.utils.riemann import _eig_fn, invsqrtm, regularize

CH = {c: i for i, c in enumerate(EEG_LABELS)}
FRONTAL = ["fp1", "fp2", "f7", "f8"]
REF_EAR = ["lpa", "rpa"]
SENSORIMOTOR = ["f3", "f4", "fz", "c3", "c4", "cz", "c7", "c8", "p3", "p4", "pz"]
NO_FRONTAL = [c for c in EEG_LABELS if c not in FRONTAL + REF_EAR]      # 15 каналов
ALL = list(EEG_LABELS)
PROBS_DIR = PROJECT_DIR / "data" / "probs"


def ch_idx(names):
    return np.array([CH[c] for c in names])


def subcov(covs, names):
    ix = ch_idx(names)
    return covs[..., ix[:, None], ix[None, :]]


# ---------- риманова часть ----------
EIG_THREADS = 4          # батч-eigh numpy отпускает GIL → потоки дают ~3x на fit_block
_POOL = None


def _eig_fn_mt(C, fn):
    """_eig_fn для больших батчей, разбитых по потокам."""
    global _POOL
    if C.ndim < 3 or len(C) < 1024 or EIG_THREADS <= 1:
        return _eig_fn(C, fn)
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(EIG_THREADS)
    shp = C.shape
    flat = C.reshape(-1, shp[-2], shp[-1])
    parts = list(_POOL.map(lambda a: _eig_fn(a, fn), np.array_split(flat, EIG_THREADS)))
    return np.concatenate(parts).reshape(shp)


def logm(C):
    return _eig_fn_mt(C, lambda w: np.log(np.maximum(w, 1e-30)))


def sqrtm(C):
    return _eig_fn(C, lambda w: np.sqrt(np.maximum(w, 0)))


def upper(L):
    n = L.shape[-1]
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return L[..., iu[0], iu[1]] * coef


def diag_scale(covs_ref):
    """Поканальные масштабы по референсным окнам (медиана дисперсий), (B, C)."""
    d = np.sqrt(np.median(np.diagonal(covs_ref, axis1=-2, axis2=-1), axis=0))
    med = np.median(d, axis=-1, keepdims=True)
    return np.maximum(d, 1e-3 * med + 1e-30)          # защита от «мёртвых» каналов


def normalize(covs, d, eps):
    """covs (n,B,C,C), d (B,C) -> поканально нормированные + shrinkage."""
    c = covs / d[None, :, :, None] / d[None, :, None, :]
    return regularize(c, eps)


def le_mean(c):
    return _eig_fn(logm(c).mean(0), np.exp)


class RefState:
    """Референс для TS/рецентрирования: поканальные масштабы + лог-евклидово среднее по полосам,
    оценённые на прошлых окнах (обновляется в fit_block по всем прошлым окнам сессии)."""

    def __init__(self, eps=1e-2):
        self.eps = eps

    def fit(self, covs):
        self.d = diag_scale(covs)
        c = normalize(covs, self.d, self.eps)
        self.P = np.stack([invsqrtm(le_mean(c[:, b])) for b in range(c.shape[1])])   # (B,C,C)
        return self

    def ts(self, covs):
        c = normalize(covs, self.d, self.eps)
        W = self.P[None] @ c @ self.P[None]
        return upper(logm(W)).reshape(len(covs), -1)

    def logvar(self, covs):
        c = normalize(covs, self.d, self.eps)
        return np.log(np.diagonal(c, axis1=-2, axis2=-1)).reshape(len(covs), -1)


# ---------- оценка ----------
_G = {}


def _eval_one(name):
    d = _G["data"][name]
    P = np.full((len(d["y"]), 3), np.nan)
    times = []
    for b in range(2, 8):
        te = d["block"] == b
        t0 = time.perf_counter()
        P[te] = _G["fp"](name, d, b)
        times.append(time.perf_counter() - t0)
    return name, P, times


def online_eval(data, fit_predict, names=None, n_jobs=10):
    """fit_predict(name, d, b) -> proba (n_te,3) для окон блока b (модель видит блоки < b).
    Параллелизм по сессиям (fork). Возвращает probs {name: (n,3)} (блок 1 — NaN) и времена
    fit+predict всего блока (в однопоточном процессе, эмуляция — не онлайн-замер)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    _G["data"], _G["fp"] = data, fit_predict
    names = list(names or data)
    probs, times = {}, []
    if n_jobs == 1:
        res = list(map(_eval_one, names))
    else:
        with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(_eval_one, names))
    for name, P, t in res:
        probs[name] = P
        times += t
    return probs, np.array(times)


def score(data, probs, bias=None):
    res = {}
    for name, P in probs.items():
        d = data[name]
        te = d["block"] >= 2
        lp = np.log(np.clip(P[te], 1e-9, 1))
        if bias is not None:
            lp = lp + bias
        res[name] = metrics(confusion(d["y"][te], lp.argmax(1) + 1))
    f = float(np.mean([r["min_f1"] for r in res.values()]))
    r = float(np.mean([r["recall"] for r in res.values()]))
    return dict(min_f1=f, recall=r, pts=60 * f + 30 * r, per=res)


def report(label, sc, times=None):
    t = "" if times is None else f" fit+pred/block mean={times.mean():.3f}s max={times.max():.3f}s"
    print(f"{label:48s} minF1={sc['min_f1']:.3f} recall={sc['recall']:.3f} pts={sc['pts']:.1f}{t}", flush=True)


def save_probs(name, probs):
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROBS_DIR / f"eeg_{name}.pkl", "wb") as f:
        pickle.dump(probs, f)
