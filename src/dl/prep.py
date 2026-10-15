"""Причинная предобработка ЭЭГ для нейросетей: HP 1 Гц + notch 50 (как в онлайне) + LP 40 Гц,
децимация 250 → 125 Гц. Всё — `sosfilt` по всей сессии от первого отсчёта (= потоковая
фильтрация с сохранением состояния, т.е. причинно). Плюс статистики блока 1 для нормировки.

    python -m src.dl.prep            # -> <scratch>/dl/prep/<сессия>.npz

Поля npz: x (n/2, 21) float32 — отфильтрованный сигнал 125 Гц (отсчёт j = исходный 2j);
st, bl — States/Blocks на той же сетке; ends/y/block — сетка оценочных окон (250 Гц, из
data/features/v1_win1); scale (21,) — поканальный масштаб блока 1; cov1 (21,21) — средняя
ковариация 1-с окон блока 1 после поканальной нормировки (для Euclidean Alignment).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.signal as ss

from src.dataset.io import PROJECT_DIR, list_sessions, load_session
from src.eeg.cache import load_raw64
from src.features.extract import eeg_prefilter_sos

SCRATCH = Path(os.environ.get("DL_SCRATCH", "/place/vartmp/claude-740548/-home-vkborisov16-projects-hackatons-medic/"
                              "62fd6f60-50fc-45a1-9add-e0b5b7dac57f/scratchpad/dl"))
PREP_DIR = SCRATCH / "prep"
FS = 250.0
DEC = 2
FS_DL = FS / DEC


def dl_sos(fs=FS):
    """Префильтр онлайна (HP 1 Гц, notch 50) + антиалиасный LP 40 Гц (Баттерворт 6)."""
    return np.vstack([eeg_prefilter_sos(fs), ss.butter(6, 40.0, "lowpass", fs=fs, output="sos")])


def block_stats(x, mask, win=int(FS_DL)):
    """Поканальный масштаб (медиана дисперсий 1-с окон) и средняя ковариация нормированного
    сигнала по отсчётам mask (блок 1). Пол масштаба 1e-3·медианы — «мёртвые» каналы."""
    xs = x[mask].astype(np.float64)
    n = len(xs) // win
    seg = xs[: n * win].reshape(n, win, -1)
    seg = seg - seg.mean(1, keepdims=True)
    var = (seg ** 2).mean(1)
    d = np.sqrt(np.median(var, 0))
    d = np.maximum(d, 1e-3 * np.median(d) + 1e-30)
    segn = seg / d
    cov = np.einsum("ntc,ntd->cd", segn, segn) / (n * win)
    return d, cov


def prep_one(path):
    s = load_session(path)
    x = load_raw64(path)
    x = ss.sosfilt(dl_sos(), x, axis=0)[::DEC]
    st, bl = s.states[::DEC], s.blocks[::DEC]
    z = np.load(PROJECT_DIR / "data/features/v1_win1" / f"{s.name}.npz")
    d, cov = block_stats(x, bl == 1)
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PREP_DIR / f"{s.name}.npz", x=x.astype(np.float32), st=st, bl=bl, ends=z["ends"], y=z["y"],
             block=z["block"], subject=s.subject, scale=d, cov1=cov)
    return s.name


def load_prep(names=None):
    out = {}
    for p in sorted(PREP_DIR.glob("*.npz")):
        if names is not None and p.stem not in names:
            continue
        z = np.load(p)
        out[p.stem] = {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(4) as ex:
        for n in ex.map(prep_one, list_sessions()):
            print(n, flush=True)
