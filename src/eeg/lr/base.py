"""Общие хелперы исследования «лево/право» для ЭЭГ (src/eeg/lr).

Правильный порядок столбцов EEG.Raw (из официального io_utils судьи): C3 = 18, C4 = 0.
Все признаки считаются из кэша ковариаций data/features/eeg_fb64 (линейный пространственный
фильтр W переводит ковариацию C в W C Wᵀ, поэтому лапласианы/биполяры/CSP не требуют
пересчёта сигналов).
"""
from __future__ import annotations

import pickle

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.dataset.windows import confusion, metrics
from NIRS_IMC_EEG_fNIRS.src.eeg.common import PROBS_DIR

# истинные имена столбцов EEG.Raw
TRUE_NAMES = ["c4", "rpa", "f8", "p8", "f4", "p4", "fp2", "o2", "cz", "pz",
              "fz", "o1", "fp1", "p3", "f3", "p7", "f7", "lpa", "c3", "c7", "c8"]
TI = {c: i for i, c in enumerate(TRUE_NAMES)}
EAR = ["lpa", "rpa"]
FRONTAL = ["fp1", "fp2", "f7", "f8"]

# соседи для малого лапласиана (c7/c8 — латеральные центрально-височные, лево/право)
LAP = {
    "c3": ["f3", "p3", "cz", "c7"],
    "c4": ["f4", "p4", "cz", "c8"],
    "cz": ["fz", "pz", "c3", "c4"],
    "f3": ["fp1", "fz", "c3", "f7"],
    "f4": ["fp2", "fz", "c4", "f8"],
    "p3": ["c3", "pz", "o1", "p7"],
    "p4": ["c4", "pz", "o2", "p8"],
    "c7": ["f7", "p7", "c3"],
    "c8": ["f8", "p8", "c4"],
    "fz": ["fp1", "fp2", "f3", "f4", "cz"],
    "pz": ["cz", "p3", "p4", "o1", "o2"],
}


def idx(names):
    return np.array([TI[c] for c in names])


def spatial_matrix(kind):
    """Строки — пространственные фильтры (k, 21)."""
    rows = []
    if kind == "lap":
        for c, nb in LAP.items():
            w = np.zeros(21); w[TI[c]] = 1; w[idx(nb)] -= 1 / len(nb); rows.append(w)
    elif kind == "lap_core":            # только моторные: c3, c4, cz
        for c in ["c3", "c4", "cz"]:
            w = np.zeros(21); nb = LAP[c]; w[TI[c]] = 1; w[idx(nb)] -= 1 / len(nb); rows.append(w)
    elif kind == "bip":
        pairs = [("f3", "c3"), ("c3", "p3"), ("c7", "c3"), ("c3", "cz"), ("cz", "c4"), ("c4", "c8"),
                 ("f4", "c4"), ("c4", "p4"), ("fz", "cz"), ("cz", "pz"), ("f3", "p3"), ("f4", "p4")]
        for a, b in pairs:
            w = np.zeros(21); w[TI[a]] = 1; w[TI[b]] = -1; rows.append(w)
    else:
        raise ValueError(kind)
    return np.array(rows)


MIRROR = {"c3": "c4", "f3": "f4", "p3": "p4", "c7": "c8", "f7": "f8", "p7": "p8", "fp1": "fp2",
          "o1": "o2", "lpa": "rpa", "cz": "cz", "fz": "fz", "pz": "pz"}
MIRROR.update({v: k for k, v in MIRROR.items()})


def channel_mirror():
    """Перестановка 21 столбца: зеркало левое↔правое полушарие."""
    return idx([MIRROR[c] for c in TRUE_NAMES])


def mirror_signed_perm(W):
    """Для пространственных фильтров W (k,21): найти знаковую перестановку (perm, sign) такую, что
    зеркальный фильтр строки i = sign[i] * W[perm[i]]."""
    Wm = W[:, channel_mirror()]
    perm, sign = [], []
    for i in range(len(W)):
        for j in range(len(W)):
            for sg in (1, -1):
                if np.allclose(Wm[i], sg * W[j]):
                    perm.append(j); sign.append(sg); break
            else:
                continue
            break
        else:
            raise ValueError(f"нет зеркальной строки для {i}")
    return np.array(perm), np.array(sign, float)


def feature_mirror(perm, sign, kind, n_bands):
    """Индексы и знаки для зеркалирования вектора признаков (полосы — старший индекс)."""
    k = len(perm)
    if kind == "logvar":
        fi, fs = perm, np.ones(k)
    else:
        iu = np.triu_indices(k)
        pos = {(a, b): t for t, (a, b) in enumerate(zip(*iu))}
        fi, fs = [], []
        for a, b in zip(*iu):
            pa, pb = perm[a], perm[b]
            fi.append(pos[(min(pa, pb), max(pa, pb))]); fs.append(sign[a] * sign[b])
        fi, fs = np.array(fi), np.array(fs)
    m = len(fi)
    return (np.concatenate([fi + bi * m for bi in range(n_bands)]), np.tile(fs, n_bands))


def swap_lr(y):
    y = y.copy(); l, r = y == 2, y == 3
    y[l], y[r] = 3, 2
    return y


def apply_spatial(covs, W):
    """covs (..., 21, 21) -> (..., k, k)."""
    return np.einsum("kc,...cd,ld->...kl", W, covs, W, optimize=True)


def load_probs(name):
    with open(PROBS_DIR / f"eeg_{name}.pkl", "rb") as f:
        return pickle.load(f)


def ema(P, ends, tau=1.5, fs=250.0):
    """Причинное EMA вероятностей по реальному времени (как solution/fusion._Ema, tau=1.5 с).
    Окна кэша — только однородные, поэтому это приближение онлайн-сглаживания."""
    out = np.array(P, np.float64)
    t = ends / fs
    s = None
    for i in range(len(P)):
        if np.isnan(P[i, 0]):
            continue
        if s is None:
            s = out[i]
        else:
            a = np.exp(-(t[i] - tp) / tau)
            s = a * s + (1 - a) * P[i]
        out[i] = s; tp = t[i]
    return out


def detail(data, probs, bias=None, tau=0.0):
    """Средние по сессиям: minF1, recall, F1 по классам, точность L/R среди истинных L/R окон.
    tau > 0 — предварительное EMA-сглаживание вероятностей."""
    mf, rc, f1s, lr = [], [], [], []
    for n, P in probs.items():
        d = data[n]
        te = d["block"] >= 2
        if tau > 0:
            P = ema(P, d["ends"], tau)
        lp = np.log(np.clip(P[te], 1e-9, 1))
        if bias is not None:
            lp = lp + bias
        yp = lp.argmax(1) + 1
        m = metrics(confusion(d["y"][te], yp))
        mf.append(m["min_f1"]); rc.append(m["recall"]); f1s.append(m["f1"])
        t = np.isin(d["y"][te], [2, 3])
        lr.append(np.mean((lp[t, 1] > lp[t, 2]) == (d["y"][te][t] == 2)))
    f, r = float(np.mean(mf)), float(np.mean(rc))
    return dict(min_f1=f, recall=r, pts=60 * f + 30 * r, f1=np.mean(f1s, 0), lr_acc=float(np.mean(lr)))


def fmt(label, s, s2=None):
    f1 = s["f1"]
    out = (f"{label:44s} minF1={s['min_f1']:.3f} rec={s['recall']:.3f} pts={s['pts']:.2f} "
           f"F1={f1[0]:.3f}/{f1[1]:.3f}/{f1[2]:.3f} LRacc={s['lr_acc']:.3f}")
    if s2 is not None:
        out += f" | EMA1.5: minF1={s2['min_f1']:.3f} rec={s2['recall']:.3f} pts={s2['pts']:.2f}"
    return out


def combine_lr(P, r, beta):
    """Добавить к 3-классовым вероятностям бинарные лог-шансы r = log P(L)/P(R) с весом beta."""
    lp = np.log(np.clip(P, 1e-9, 1)).copy()
    lp[:, 1] += beta * r / 2
    lp[:, 2] -= beta * r / 2
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)
