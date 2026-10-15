"""Исследовательский каркас: эмуляция судьи (до получения официального solution/run.py) с оверлеем
артефактов протоколов (--loso-artifacts / --loseso-artifacts) и --dump-probs. --legacy — прежняя
реконструкция. Итоговые метрики считаются официальным run.py (src/scripts/judge_eval.py).

    python solution/tools/dev_run.py --input <session.mat | каталог> --output-dir <dir> [--artifacts solution/artifacts]
                           [--jobs N]

Подаёт сессию хронологически: чанками по 250 мс ЭЭГ (границы — глобальная сетка шагом fs/4 и
границы блоков) + отсчёты NIRS с Frame ≤ текущего конца. После каждого чанка:
  * predict — если текущий конец совпадает с концом окна 1 с, целиком лежащего в однородном
    сегменте States∈{1,2,3} блоков 2..7;
  * fit_block(k, States отсчётов блока k) — если чанк закончил блок k (включая блок 1);
    predict для последнего окна блока выполняется до fit_block этого блока.
Каждый вызов predict / fit_block (и push — справочно) замеряется time.perf_counter.

Выход (для одной сессии — прямо в --output-dir; для каталога — в <output-dir>/<сессия>/):
  predictions.csv: end_sample, block, true_y, y, y_eeg, y_nirs
  timings.csv:     method, block, end_sample, seconds
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]   # каталог solution/
sys.path.insert(0, str(HERE))

from session_io import list_sessions, load_session, window_ends  # noqa: E402

# истинный порядок столбцов EEG.Raw (как eeg_channel_names официального io_utils)
EEG_LABELS = ['c4', 'rpa', 'f8', 'p8', 'f4', 'p4', 'fp2', 'o2', 'cz', 'pz',
              'fz', 'o1', 'fp1', 'p3', 'f3', 'p7', 'f7', 'lpa', 'c3', 'c7', 'c8']


def make_meta(s: dict, legacy: bool = False) -> dict:
    has_nirs = s["hbo"] is not None
    if legacy:
        return dict(session=s["name"], eeg_fs=s["eeg_fs"], eeg_channels=int(s["eeg"].shape[1]),
                    eeg_labels=EEG_LABELS if s["eeg"].shape[1] == 21 else None,
                    has_nirs=has_nirs, nirs_fs=s["nirs_fs"] if has_nirs else None,
                    nirs_channels=int(s["hbo"].shape[1]) if has_nirs else 0)
    # ключи session_meta судьи (ответ кейсодержателя 2026-09-23); имена каналов — порядок
    # столбцов Raw не совпадает с Montage.EEG_labels (столбец 0 = C4, 18 = C3), точный список —
    # в официальном run.py, здесь не используется
    return dict(fs_eeg=float(s["eeg_fs"]), fs_nirs=float(s["nirs_fs"]) if has_nirs else None,
                n_eeg_ch=int(s["eeg"].shape[1]), n_nirs_ch=int(s["hbo"].shape[1]) if has_nirs else 0,
                has_nirs=has_nirs, subject=s["name"].split("_")[-1], session_name=s["name"] + ".mat",
                eeg_channel_names=EEG_LABELS if s["eeg"].shape[1] == 21 else None,
                eeg_c3_idx=18, eeg_c4_idx=0, n_samples_eeg=int(len(s["states"])))


def loso_artifacts(path, loso_dir, base_artifacts) -> str:
    """Dev-режим честной оценки на train: подменить пул-артефакт ЭЭГ на leave-one-subject-out.

    В `solution/artifacts/` лежит пул, обученный на всех 56 сессиях (`pool_<имя>_all.npz`) — он
    корректен для закрытого теста с новыми участниками, но на train «подсматривает» в сессии
    того же участника. Для честных отчётных метрик берём `pool_<имя>_<участник>.npz` (участник —
    из имени файла сессии `<дата>_N<id>.mat`). Возвращает путь к временному каталогу артефактов.
    """
    import json
    import shutil
    import tempfile
    base = Path(base_artifacts)
    cfg = json.loads((base / "config.json").read_text())
    pool_name = (cfg.get("eeg") or {}).get("pool_file", "eeg_pool.npz")
    pattern = cfg.get("loso_pool_pattern", "pool_ts2s_all_{subject}.npz")
    subject = Path(path).stem.split("_")[-1]
    src = Path(loso_dir) / pattern.format(subject=subject)
    if not src.exists():
        raise FileNotFoundError(f"нет LOSubjO-артефакта {src}")
    # дополнительные пулы (эксперты eeg_lr_nbsm4): {файл в артефактах: шаблон LOSubjO-файла}
    extra = {k: Path(loso_dir) / v.format(subject=subject) for k, v in (cfg.get("loso_extra_pools") or {}).items()}
    for k, v in extra.items():
        if not v.exists():
            raise FileNotFoundError(f"нет LOSubjO-артефакта {v}")
    tmp = Path(tempfile.mkdtemp(prefix="loso_art_"))
    for f in base.iterdir():          # подкаталоги (пулы участников) не копируются — участник «новый»
        if f.is_file() and f.name != pool_name and f.name not in extra:
            shutil.copy2(f, tmp / f.name)
    shutil.copy2(src, tmp / pool_name)
    for k, v in extra.items():
        shutil.copy2(v, tmp / k)
    return str(tmp)


def loseso_artifacts(path, overlay_dir, base_artifacts) -> str:
    """Dev-режим leave-one-session-out (прокси закрытого теста: те же участники, новая сессия):
    копия `base_artifacts` (со всеми подкаталогами), поверх которой накладывается
    `<overlay_dir>/<имя сессии>/` — пулы, обученные на всех сессиях, КРОМЕ этой (общий
    `eeg_pool.npz`, подучастниковые `eeg_subj/<участник>.npz`, `nirs_subj/<участник>.npz`).
    Тот же формат — хронологический протокол (`data/features/chrono/<сессия>/`, только сессии
    раньше этой). Возвращает путь к временному каталогу артефактов."""
    import shutil
    import tempfile
    src = Path(overlay_dir) / Path(path).stem
    if not src.is_dir():
        raise FileNotFoundError(f"нет LOSessO-артефактов {src}")
    tmp = Path(tempfile.mkdtemp(prefix="loseso_art_")) / "art"
    shutil.copytree(base_artifacts, tmp)
    # строго: все пулы сдачи и мета-модель слияния (видевшие эту сессию) убираются; остаются
    # только файлы из overlay —
    # если какого-то там нет (напр., у первой по времени сессии в хронологическом протоколе),
    # модель откатывается так же, как в сдаче без пула
    for f in ("eeg_pool.npz", "eeg_lrnb_pool.npz", "eeg_lrsm4_pool.npz", "fusion_meta.json"):
        (tmp / f).unlink(missing_ok=True)
    for f in tmp.glob("eeg_dl_*.npz"):   # DL-сети сдачи видели все сессии → только из overlay
        f.unlink()
    for d in ("eeg_subj", "nirs_subj", "eeg_lrnb_subj", "eeg_lrsm4_subj"):
        shutil.rmtree(tmp / d, ignore_errors=True)
    shutil.copytree(src, tmp, dirs_exist_ok=True)
    return str(tmp)


def run_session(path, out_dir, artifacts=None, loso_dir=None, dump_probs=False, legacy=False,
                loseso_dir=None) -> dict:
    from classifier import OnlineClassifier
    tmp_art = None
    if loso_dir:
        tmp_art = loso_artifacts(path, loso_dir, artifacts or HERE / "artifacts")
        artifacts = tmp_art
    elif loseso_dir:
        tmp_art = loseso_artifacts(path, loseso_dir, artifacts or HERE / "artifacts")
        artifacts = tmp_art
    s = load_session(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = s["eeg_fs"]
    n = len(s["states"])
    states, blocks = s["states"], s["blocks"]

    ends, ys, wb = window_ends(states, blocks, fs)
    sel = wb >= 2
    win_info = {int(e): (int(y), int(b)) for e, y, b in zip(ends[sel], ys[sel], wb[sel])}

    # границы блоков: блок k занимает [start_k, end_k)
    ch = np.flatnonzero(np.diff(blocks)) + 1
    b_start = np.r_[0, ch]
    b_end = np.r_[ch, n]
    block_end_at = {int(e): (int(blocks[st]), int(st)) for st, e in zip(b_start, b_end)}

    if legacy:   # старая реконструкция: push каждые 250 мс по глобальной сетке
        step = fs / 4
        grid = np.round(np.arange(step, n + 1e-9, step)).astype(int)
        bounds = np.unique(np.r_[grid, b_end, n])
    else:        # как у судьи: push только перед событиями (конец окна, конец блока, хвост сессии)
        bounds = np.unique(np.r_[list(win_info), b_end, n])
    bounds = bounds[(bounds > 0) & (bounds <= n)]

    clf = OnlineClassifier(make_meta(s, legacy))
    clf.load(artifacts or HERE / "artifacts")

    frame = s["nirs_frame"]
    n_nirs = 0 if frame is None else s["hbo"].shape[1]
    nirs_ptr = 0
    preds, timings, probs = [], [], []
    prev = 0
    pc = time.perf_counter
    for b in bounds:
        b = int(b)
        if legacy:
            nirs_chunk = None
            if frame is not None:
                j = int(np.searchsorted(frame, b, side="right"))     # Frame ≤ b
                if j > nirs_ptr:
                    nirs_chunk = {"HbO": s["hbo"][nirs_ptr:j], "HbR": s["hbr"][nirs_ptr:j],
                                  "Frame": frame[nirs_ptr:j]}
                nirs_ptr = j
        else:
            # судья: кортеж (HbO, HbR) отсчётов с NIRS.Frame ∈ [eeg_start, eeg_end), без Frame;
            # нет отсчётов / нет NIRS — пустые массивы (0, n_ch)
            if frame is not None:
                j = int(np.searchsorted(frame, b, side="left"))
                nirs_chunk = (s["hbo"][nirs_ptr:j], s["hbr"][nirs_ptr:j])
                nirs_ptr = j
            else:
                nirs_chunk = (np.zeros((0, 0)), np.zeros((0, 0)))
        t0 = pc()
        clf.push(s["eeg"][prev:b], nirs_chunk)
        timings.append(("push", int(blocks[b - 1]), b, pc() - t0))
        prev = b
        if b in win_info:
            y_true, blk = win_info[b]
            t0 = pc()
            r = clf.predict()
            timings.append(("predict", blk, b, pc() - t0))
            preds.append((b, blk, y_true, int(r["y"]), r.get("y_eeg"), r.get("y_nirs")))
            if dump_probs:
                lp = getattr(clf, "last_proba", {})
                probs.append((b, lp.get("eeg"), lp.get("nirs"), lp.get("hyb")))
        if b in block_end_at:
            k, st = block_end_at[b]
            if legacy:
                lab = states[st:b].copy()
            else:
                mask = (blocks == k) & (states > 0)
                lab = {"states": states[mask].copy(), "eeg_indices": np.flatnonzero(mask), "block": k}
            t0 = pc()
            clf.fit_block(k, lab)
            timings.append(("fit_block", k, b, pc() - t0))

    def _fmt(v):
        return "" if v is None else int(v)

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["end_sample", "block", "true_y", "y", "y_eeg", "y_nirs"])
        for e, blk, t, y, ye, yn in preds:
            w.writerow([e, blk, t, y, _fmt(ye), _fmt(yn)])
    with open(out_dir / "timings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "block", "end_sample", "seconds"])
        for m, blk, e, sec in timings:
            w.writerow([m, blk, e, f"{sec:.6f}"])
    if dump_probs:
        with open(out_dir / "probs.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["end_sample"] + [f"{m}{k}" for m in ("eeg", "nirs", "hyb") for k in (1, 2, 3)])
            for e, pe, pn, pf in probs:
                row = [e]
                for v in (pe, pn, pf):
                    row += ["", "", ""] if v is None else [f"{x:.9g}" for x in v]
                w.writerow(row)
    if tmp_art:
        import shutil
        shutil.rmtree(Path(tmp_art).parent if loseso_dir and not loso_dir else tmp_art, ignore_errors=True)
    return dict(session=s["name"], n_pred=len(preds), out_dir=str(out_dir))


def _job(args):
    path, out, art, loso, dump, legacy, loseso = args
    try:
        return run_session(path, out, art, loso, dump, legacy, loseso)
    except Exception:
        return dict(session=Path(path).stem, error=traceback.format_exc())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help=".mat одной сессии или каталог с .mat")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--artifacts", default=str(HERE / "artifacts"))
    ap.add_argument("--jobs", type=int, default=1, help="процессов для каталога (≤ 6 на общей машине)")
    ap.add_argument("--loso-artifacts", default=None,
                    help="dev: каталог с пулами leave-one-subject-out (data/features/eeg_models) — "
                         "для честной оценки на train пул подменяется по участнику из имени сессии")
    ap.add_argument("--loseso-artifacts", default=None,
                    help="dev: каталог <сессия>/{eeg_pool.npz, eeg_subj/, nirs_subj/} leave-one-session-out — "
                         "накладывается поверх --artifacts для каждой сессии (прокси теста на тех же участниках)")
    ap.add_argument("--dump-probs", action="store_true",
                    help="dev: писать probs.csv (вероятности ЭЭГ, NIRS и гибрида после EMA) для паритета")
    ap.add_argument("--legacy", action="store_true",
                    help="dev: старая реконструкция каркаса (push каждые 250 мс, dict NIRS с Frame, States блока)")
    a = ap.parse_args(argv)
    files = list_sessions(a.input)
    single = Path(a.input).is_file()
    jobs = [(p, Path(a.output_dir) if single else Path(a.output_dir) / p.stem, a.artifacts, a.loso_artifacts,
             a.dump_probs, a.legacy, a.loseso_artifacts) for p in files]
    if a.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    bad = 0
    for r in results:
        if "error" in r:
            bad += 1
            print(f"[FAIL] {r['session']}\n{r['error']}", file=sys.stderr, flush=True)
        else:
            print(f"[ok] {r['session']}: {r['n_pred']} predictions → {r['out_dir']}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
