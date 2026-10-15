"""Мета-модели слияния `StackedFusion` («stack_short») для сдачи и dev-протоколов — одной командой.

    python -m src.fusion.build_meta --runs-dir <scratch>/eeg_lr [--jobs 4] [--min-prev 5] [--eval-only]

Вход — dev-прогоны с `--dump-probs` на официальной сетке (`<runs-dir>/dev_{chrono,loseso,losubjo}/
<сессия>/probs.csv`, вероятности ЭЭГ/NIRS на каждый predict). Мета-модель — тоже модель,
обученная на train-сессиях, поэтому для каждого протокола она обучается честно:

* **LOSubjO** — на LOSubjO-вероятностях сессий других участников →
  `data/features/eeg_models/fusion_meta_<участник>.json` (подмена через
  `config.loso_extra_pools["fusion_meta.json"]`);
* **LOSessO** — `models_subject` на LOSessO-вероятностях всех сессий, кроме этой (в прогоне всегда
  есть пул участника), `models` (откат без пула) — LOSubjO-мета без участника →
  `data/features/subjpool_loseso/<S>/fusion_meta.json`;
* **хронологический** — одна мета-модель на хронологических вероятностях всех сессий раньше S
  (и с пулом участника, и без: так данных больше, и это лучше раздельных моделей); если раньше S
  меньше `--min-prev` сессий, файла нет → `LogLinearFusion` сдачи → `data/features/chrono/<S>/`;
* **сдача** — `models_subject` на LOSessO-вероятностях всех 56 сессий (ближе всего к тесту: те же
  участники, пул участника есть), `models` (незнакомый участник / нет `subject`) — на
  LOSubjO-вероятностях всех 56 → `solution/artifacts/fusion_meta.json`.

Признаки и правило решения фиксированы (`PARAMS`, вариант s4c из `docs/experiments/fusion_decision.md`):
τ ЭЭГ ≤ 4 с, τ NIRS ≤ 4 с, центрирование 60 с, иерархическое сглаживание [0.75, 2.0] как у
LogLinearFusion сдачи, сдвиг 0. Скрипт же печатает переигранные метрики трёх протоколов
(LogLinear сдачи против стекинга) — совпадают с судьёй до ~0.01 балла.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from NIRS_IMC_EEG_fNIRS.src.dataset.io import PROJECT_DIR
from NIRS_IMC_EEG_fNIRS.src.fusion.replay import (add_features, fit_meta, load_protocol, loglinear_cfg, sess_fp,
                               stacked_out, summarize)

# compliance_audit (P3): τ ≤ 1 с — без памяти о предыдущей пробе
FEAT = dict(eeg_taus=[0.0, 0.5, 1.0], nirs_taus=[0.0, 1.0], center_T=[60.0],
            center_ref_tau=None)
PARAMS = dict(FEAT, post_tau_s=1.0, post_hier_tau_s=[0.75, 2.0], bias=[0.0, 0.0, 0.0],
              eeg_bias=[0.0, 0.0, 0.0], nirs_prior=[0.0, 1.0, 1.0], nirs_tau_s=0.0)
FEAT_DIR = PROJECT_DIR / "data" / "features"
OVERLAY = {"chrono": FEAT_DIR / "chrono", "loseso": FEAT_DIR / "subjpool_loseso", "losubjo": None}
LOSO_DIR = FEAT_DIR / "eeg_models"
SHIP = PROJECT_DIR / "solution" / "artifacts" / "fusion_meta.json"

_R = {}


def _fit(job):
    key, proto, names = job
    return key, fit_meta(_R[proto], list(names))


def _dump(path, models, models_subject, info):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"params": PARAMS, "models": models, "models_subject": models_subject, "info": info},
              open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="каталог с dev_{chrono,loseso,losubjo}/<S>/probs.csv")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--min-prev", type=int, default=5, help="хронологический: минимум более ранних сессий")
    ap.add_argument("--eval-only", action="store_true", help="не писать артефакты")
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.load(open(PROJECT_DIR / "solution" / "artifacts" / "config.json"))
    for p in OVERLAY:
        _R[p] = add_features(load_protocol(Path(a.runs_dir) / f"dev_{p}", OVERLAY[p]), FEAT)
    names = sorted(_R["losubjo"])                       # имя начинается с даты → хронология
    subs = sorted({n.split("_")[-1] for n in names})
    print(f"загружено {len(names)} сессий × 3 протокола, {time.time() - t0:.0f} с", flush=True)

    jobs = [(("losubjo", s), "losubjo", tuple(n for n in names if not n.endswith("_" + s))) for s in subs]
    jobs += [(("loseso", n), "loseso", tuple(m for m in names if m != n)) for n in names]
    jobs += [(("chrono", n), "chrono", tuple(names[:i])) for i, n in enumerate(names) if i >= a.min_prev]
    jobs += [(("ship", "models"), "losubjo", tuple(names)), (("ship", "models_subject"), "loseso", tuple(names))]
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
        M = dict(ex.map(_fit, jobs, chunksize=1))
    print(f"мета-LR: {len(M)} фитов, {time.time() - t0:.0f} с", flush=True)

    def pick(proto, n):
        s = n.split("_")[-1]
        if proto == "losubjo":
            return M[("losubjo", s)]
        if proto == "loseso":
            return M[("loseso", n)] if _R[proto][n]["subj"] else M[("losubjo", s)]
        return M.get(("chrono", n))

    res = {}
    for proto in OVERLAY:
        R = _R[proto]
        base = {n: sess_fp(R[n], loglinear_cfg(R[n], cfg["fusion_params"], cfg.get("fusion_params_subject"))) for n in names}
        st = {}
        for n in names:
            m = pick(proto, n)
            st[n] = base[n] if m is None else sess_fp(R[n], stacked_out(R[n], m, bias=0.0, hier=tuple(PARAMS["post_hier_tau_s"])))
        res[proto] = {"loglinear": summarize(base), "stacked": summarize(st)}
        print(f"{proto:8s} LogLinear {res[proto]['loglinear']}  →  стекинг {res[proto]['stacked']}", flush=True)
    if a.eval_only:
        return
    for s in subs:
        _dump(LOSO_DIR / f"fusion_meta_{s}.json", M[("losubjo", s)], None, {"protocol": "losubjo", "excluded_subject": s})
    for n in names:
        _dump(OVERLAY["loseso"] / n / "fusion_meta.json", M[("losubjo", n.split("_")[-1])], M[("loseso", n)],
              {"protocol": "loseso", "excluded_session": n})
        cm = M.get(("chrono", n))
        f = OVERLAY["chrono"] / n / "fusion_meta.json"
        if cm is None:
            f.unlink(missing_ok=True)                   # мало ранних сессий → LogLinearFusion
        else:
            _dump(f, cm, cm, {"protocol": "chrono", "n_prev_sessions": names.index(n)})
    _dump(SHIP, M[("ship", "models")], M[("ship", "models_subject")],
          {"models": "LOSubjO-вероятности 56 сессий", "models_subject": "LOSessO-вероятности 56 сессий"})
    (PROJECT_DIR / "data" / "fusion").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(PROJECT_DIR / "data" / "fusion" / "build_meta_replay.json", "w"), indent=1)
    print(f"артефакты записаны ({SHIP}), {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
