"""Синтетические сессии с переставленными пробами (тест на эксплуатацию чередования покой↔задача).

Сессия режется на «юниты» = пауза (состояние 0) перед сегментом + сам сегмент (9 с покоя/задачи);
юниты переставляются ВНУТРИ каждого блока и склеиваются. На стыках сырой ЭЭГ и NIRS снимается
скачок постоянной составляющей (поканально: первый отсчёт юнита приравнивается к последнему
отсчёту предыдущего), Frame NIRS переносится вместе с юнитом. States/Blocks переставляются вместе
с сигналом, так что официальный каркас видит обычную сессию того же формата.

  alt      — покой ↔ покой, задача ↔ задача (чередование сохранено, порядок L/R/покоев случаен);
  grouped  — в каждом блоке сначала все покои, затем все задачи (чередования нет);
  rand     — случайная перестановка юнитов блока.

    python -m src.scripts.compliance_splice --mode grouped --out <dir> [--seed 0]
"""
from __future__ import annotations

import argparse
import zlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[2]


def units_of_block(states, bs, be):
    st = states[bs:be]
    ch = np.flatnonzero(np.diff(st) != 0) + 1
    starts = np.r_[0, ch]
    ends = np.r_[ch, len(st)]
    segs = [(bs + a, bs + b, int(st[a])) for a, b in zip(starts, ends) if st[a] > 0]
    units, prev = [], bs
    for i, (a, b, lab) in enumerate(segs):
        stop = b if i < len(segs) - 1 else be       # хвост блока (если есть) — к последнему юниту
        units.append((prev, stop, lab))
        prev = stop
    return units


def make_order(mode, labs, rng):
    idx = np.arange(len(labs))
    if mode == "alt":
        out = idx.copy()
        for grp in (labs == 1, labs > 1):
            g = idx[grp]
            out[g] = rng.permutation(g)
        return out
    if mode == "grouped":
        return np.r_[idx[labs == 1], idx[labs > 1]]
    if mode == "rand":
        return rng.permutation(idx)
    if mode == "orig":
        return idx
    raise ValueError(mode)


def build(args):
    path, out_dir, mode, seed = args
    rng = np.random.default_rng(seed + zlib.crc32(Path(path).stem.encode()) % 1000)
    m = loadmat(str(path), simplify_cells=True)
    E = m["EEG"]
    raw = np.asarray(E["Raw"], np.float64)
    states = np.asarray(E["States"]).astype(np.int32).ravel()
    blocks = np.asarray(E["Blocks"]).astype(np.int32).ravel()
    has_nirs = "NIRS" in m and isinstance(m["NIRS"], dict) and "HbO" in m["NIRS"]
    if has_nirs:
        N = m["NIRS"]
        hbo, hbr = np.asarray(N["HbO"], np.float64), np.asarray(N["HbR"], np.float64)
        frame = np.asarray(N["Frame"]).astype(np.int64).ravel()
        nst = np.asarray(N["States"]).ravel(); nbl = np.asarray(N["Blocks"]).ravel()
    order_units = []
    b_ids = np.unique(blocks)
    cur = 0
    for b in b_ids:
        idx = np.flatnonzero(blocks == b)
        bs, be = int(idx[0]), int(idx[-1]) + 1
        if bs > cur:                                       # что-то между блоками (не бывает) — как есть
            order_units.append((cur, bs, 0))
        us = units_of_block(states, bs, be)
        if not us:
            order_units.append((bs, be, 0)); cur = be; continue
        if us[0][0] > bs:
            order_units.append((bs, us[0][0], 0))
        labs = np.array([u[2] for u in us])
        o = make_order(mode, labs, rng)
        order_units += [us[j] for j in o]
        cur = be
    if cur < len(states):
        order_units.append((cur, len(states), 0))
    pieces, st_p, bl_p = [], [], []
    n_pieces, n_fr, n_st, n_bl = [], [], [], []
    last_e = None; last_o = last_r = None
    new0 = 0
    prev_u1 = None
    K = 25                                                  # стык: средние 25 отсчётов (0.1 с) по краям
    for (u0, u1, _) in order_units:
        x = raw[u0:u1].copy()
        joint = prev_u1 is not None and u0 != prev_u1       # разрыв исходной записи
        if joint:
            x += last_e - x[:K].mean(0)
        pieces.append(x); last_e = x[-K:].mean(0)
        prev_u1 = u1
        st_p.append(states[u0:u1]); bl_p.append(blocks[u0:u1])
        if has_nirs:
            sel = np.flatnonzero((frame - 1 >= u0) & (frame - 1 < u1))
            if len(sel):
                o_, r_ = hbo[sel].copy(), hbr[sel].copy()
                if last_o is not None and joint:
                    o_ += last_o - o_[0]; r_ += last_r - r_[0]
                last_o, last_r = o_[-1], r_[-1]
                n_pieces.append((o_, r_)); n_fr.append(frame[sel] - u0 + new0)
                n_st.append(nst[sel]); n_bl.append(nbl[sel])
        new0 += u1 - u0
    E2 = {"Raw": np.vstack(pieces), "States": np.concatenate(st_p)[None], "Blocks": np.concatenate(bl_p)[None],
          "Frq": E["Frq"]}
    out = {"EEG": E2}
    if has_nirs:
        out["NIRS"] = {"HbO": np.vstack([p[0] for p in n_pieces]), "HbR": np.vstack([p[1] for p in n_pieces]),
                       "Frame": np.concatenate(n_fr)[None], "States": np.concatenate(n_st)[None],
                       "Blocks": np.concatenate(n_bl)[None], "Frq": N["Frq"]}
    savemat(str(Path(out_dir) / Path(path).name), out, do_compression=False)
    return Path(path).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["alt", "grouped", "rand", "orig"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/raw/Data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.data).glob("*.mat"))
    with ProcessPoolExecutor(a.jobs) as ex:
        for s in ex.map(build, [(str(f), a.out, a.mode, a.seed) for f in files]):
            print("built", s, flush=True)


if __name__ == "__main__":
    main()
