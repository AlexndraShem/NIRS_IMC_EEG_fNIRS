"""Переобучение мета-LR StackedFusion (хронологический протокол) на вероятностях, снятых на ВСЕХ
точках сетки (асинхронный режим, прогон compliance_async --predict all), и переигрывание слияния.

То же, что src/fusion/build_meta.py для протокола chrono (PARAMS/FEAT оттуда же), но вход — npz
драйвера compliance_async. Пишет overlay-каталог (копия data/features/chrono с новыми
fusion_meta.json) для прогона официальным каркасом / драйвером.

    python -m src.scripts.compliance_meta --runs <npz dir> [--write-overlay <dir>] [--min-prev 5]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from src.dataset.io import RAW_DIR, load_session  # noqa: E402
from src.fusion.build_meta import FEAT, PARAMS  # noqa: E402
from src.fusion.replay import add_features, fit_meta, loglinear_cfg, sess_fp, stacked_out, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CHRONO = ROOT / "data" / "features" / "chrono"
_R = {}


def load_runs(d):
    out = {}
    for f in sorted(Path(d).glob("*.npz")):
        z = np.load(f)
        s = load_session(RAW_DIR / "Data" / f"{f.stem}.mat")
        e = z["end"].astype(int)
        st = s.states.astype(int)
        seg = np.r_[0, np.cumsum(np.diff(st) != 0)]
        ok = (e >= 250) & (seg[np.maximum(e - 250, 0)] == seg[e - 1]) & (st[e - 1] > 0) & (s.blocks[e - 1] >= 2)
        pn = z["p_nirs"]
        out[f.stem] = dict(e=e, t=e / 250.0, y=st[e - 1], ok=ok, pe=z["p_eeg"], pn=pn,
                           has_nirs=bool(np.isfinite(pn).all(1).any()),
                           subj=any((CHRONO / f.stem / "eeg_subj").glob("*.npz")), subject=f.stem.split("_")[-1])
    return out


def _fit(job):
    n, names = job
    return n, fit_meta(_R["r"], list(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--min-prev", type=int, default=5)
    ap.add_argument("--write-overlay", default=None)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--feat", default="F0_current", help="набор признаков из compliance_order.FEATS")
    a = ap.parse_args()
    from src.scripts.compliance_order import FEATS
    feat = dict(FEAT, **FEATS[a.feat])
    params = dict(PARAMS, **FEATS[a.feat])
    R = add_features(load_runs(a.runs), feat)
    _R["r"] = R
    names = sorted(R)
    cfg = json.load(open(ROOT / "solution" / "artifacts" / "config.json"))
    jobs = [(n, tuple(names[:i])) for i, n in enumerate(names) if i >= a.min_prev]
    with ProcessPoolExecutor(a.jobs) as ex:
        M = dict(ex.map(_fit, jobs))
    # 1) мета-модели сдачи (обучены на расписании судьи) — как есть: из data/features/chrono
    per_old, per_new, per_ll = {}, {}, {}
    for n in names:
        r = R[n]
        ll = sess_fp(r, loglinear_cfg(r, cfg["fusion_params"], cfg.get("fusion_params_subject")))
        per_ll[n] = ll
        fm = CHRONO / n / "fusion_meta.json"
        old = json.load(open(fm))["models"] if (fm.exists() and a.feat == "F0_current") else None
        per_old[n] = ll if old is None else sess_fp(r, stacked_out(r, old, bias=0.0, hier=tuple(PARAMS["post_hier_tau_s"])))
        m = M.get(n)
        per_new[n] = ll if m is None else sess_fp(r, stacked_out(r, m, bias=0.0, hier=tuple(PARAMS["post_hier_tau_s"])))
    res = {"loglinear": summarize(per_ll), "stacked_old_meta": summarize(per_old),
           "stacked_retrained_meta": summarize(per_new)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.write_overlay:
        ov = Path(a.write_overlay)
        if ov.exists():
            shutil.rmtree(ov)
        shutil.copytree(CHRONO, ov)
        for n in names:
            f = ov / n / "fusion_meta.json"
            if n in M:
                json.dump({"params": params, "models": M[n], "models_subject": M[n],
                           "info": {"protocol": "chrono-async", "n_prev_sessions": names.index(n)}}, open(f, "w"))
            else:
                f.unlink(missing_ok=True)
        print("overlay:", ov)


if __name__ == "__main__":
    main()
