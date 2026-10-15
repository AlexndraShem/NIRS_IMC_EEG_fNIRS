"""Итоговые варианты ЭЭГ «лево/право»: сохранение вероятностей и пул-артефактов экспертов.

Вероятности собираются из результатов скрининга (src.eeg.lr.feats, та же процедура, что в онлайне:
референс по каждому 4-му окну прошлых блоков, sLDA по каждому 2-му) и baseline-компонент
eeg_ts2s_all_{within,pool}. Пулы экспертов (LR на признаках других участников) переобучаются здесь
и сохраняются: data/features/eeg_models/pool_lr<эксперт>_<участник>.npz (LOSubjO) и _all.npz.

python -m src.eeg.lr.final [--pools]
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from NIRS_IMC_EEG_fNIRS.src.eeg.cache import load_eeg_cache
from NIRS_IMC_EEG_fNIRS.src.eeg.common import save_probs
from NIRS_IMC_EEG_fNIRS.src.eeg.lr import feats as FT
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import detail, fmt
from NIRS_IMC_EEG_fNIRS.src.eeg.lr.comb import build, hier_smooth
from NIRS_IMC_EEG_fNIRS.src.eeg.online import PoolLR
from NIRS_IMC_EEG_fNIRS.src.eeg.final import MODEL_DIR

# имя -> параметры build (base = eeg_ts2s_all within/pool с расписанием веса пула)
VARIANTS = {
    "lr_sched": dict(a2=.9, a7=.7),
    "lr_nb": dict(a2=.9, a7=.7, experts=[("nbboth+mw", .8, .5, 1.0)]),
    # основной: как в онлайне (OnlineEEGLR) — не больше 400 обучающих окон во всех within-моделях
    "lr_nbsm4": dict(base_w="res:tsall+np+cap400", a2=.9, a7=.7,
                     experts=[("nbboth+mw+cap400", .8, .5, 1.0), ("tssm+mw+cap400_w4", .8, .5, 1.0)]),
    "lr_ens4": dict(a2=.9, a7=.7, experts=[("nbboth+mw", .8, .5, .5), ("nbboth+mw_w4", .8, .5, .5),
                                           ("tssm+mw", .8, .5, .5), ("tssm+mw_w4", .8, .5, .5)]),
}
# эксперт -> (признак feats.py, окно)
EXPERT_POOLS = {"nb": ("nbboth", 2.0), "sm4": ("tssm", 4.0)}


def _pool(args):
    s, = args
    F, data = FT._G["F"], FT._G["data"]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr = [n for n in data if data[n]["subject"] != s]
    X = np.vstack([F[n]["pool"] for n in tr]).astype(np.float64)
    y = np.concatenate([data[n]["y"] for n in tr])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.01, max_iter=300, class_weight="balanced").fit(sc.transform(X), y)
    return s, PoolLR.from_sklearn(sc, m)


def fit_pools():
    ctx = mp.get_context("fork")
    for name, (feat, win) in EXPERT_POOLS.items():
        data = load_eeg_cache(win, f64=True)
        FT._G.clear()
        FT._G.update(data=data, feat=feat, opts=[], mir=FT.mirror_fn(feat))
        if FT.FEATS[feat][1] == "nblogvar":
            from NIRS_IMC_EEG_fNIRS.src.eeg.lr.nbcache import OUT as NB_OUT
            FT._G["nb"] = {n: np.load(NB_OUT / f"{n}.npz")[f"var_w{win:g}"] for n in data}
        with ProcessPoolExecutor(FT.NJ, mp_context=ctx) as ex:
            FT._G["F"] = dict(ex.map(FT.feats_session, list(data)))
        subjects = sorted({d["subject"] for d in data.values()}) + ["all"]
        with ProcessPoolExecutor(FT.NJ, mp_context=ctx) as ex:
            for s, p in ex.map(_pool, [(s if s != "all" else "__none__",) for s in subjects]):
                p.save(MODEL_DIR / f"pool_lr{name}_{'all' if s == '__none__' else s}.npz")
        print(f"пулы {name}: {len(subjects)} файлов, dim={p.coef.shape[1]}", flush=True)
        del data


def main():
    data = load_eeg_cache(2.0, f64=True)
    from NIRS_IMC_EEG_fNIRS.src.eeg.lr.base import load_probs
    base = load_probs("ts2s_all")
    print(fmt("eeg_ts2s_all", detail(data, base), detail(data, base, tau=1.5)))
    for name, kw in VARIANTS.items():
        P = build(data, **kw)
        save_probs(name, P)
        H = {n: hier_smooth(p, data[n]["ends"], 0.0, 3.0) for n, p in P.items()}
        print(fmt(f"eeg_{name}", detail(data, P), detail(data, P, tau=1.5)),
              f"| hier(0/3) {detail(data, H)['pts']:.2f}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if "--pools" in sys.argv:
        fit_pools()
    main()
