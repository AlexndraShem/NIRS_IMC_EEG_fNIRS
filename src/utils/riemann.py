"""Минимальная риманова геометрия SPD-матриц (без pyriemann)."""
from __future__ import annotations

import numpy as np


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def regularize(C, eps=1e-3):
    """Shrinkage к масштабированной единичной матрице — защита от вырожденных каналов."""
    C = np.asarray(C, np.float64)
    n = C.shape[-1]
    tr = np.trace(C, axis1=-2, axis2=-1)[..., None, None] / n
    return (1 - eps) * C + eps * tr * np.eye(n)


def log_euclid_mean(covs):
    return _eig_fn(_eig_fn(covs, np.log).mean(0), np.exp)


def invsqrtm(C):
    return _eig_fn(C, lambda w: 1 / np.sqrt(np.maximum(w, 1e-30)))


def tangent_space(covs, ref):
    """covs (..., C, C), ref (C, C) → векторы (..., C(C+1)/2)."""
    P = invsqrtm(ref)
    W = P @ covs @ P
    L = _eig_fn(W, lambda w: np.log(np.maximum(w, 1e-30)))
    n = covs.shape[-1]
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return L[..., iu[0], iu[1]] * coef
