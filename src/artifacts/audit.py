"""Аудит артефактов по признакам `src.artifacts.extract` / `src.artifacts.eog` (оффлайн-анализ).

Для каждого признака и сессии считаем AUC:
  * rest-vs-MI (положительный класс — воображение 2/3),
  * L-vs-R (положительный класс — левая рука, 2),
на уровне окон судьи и на уровне проб (среднее по окнам пробы; Манна–Уитни p).
Плюс: доля «ВЧ-всплесков» по классам/блокам и точность текущего решения на окнах со всплеском и без.

    python -m src.artifacts.audit --out results/artifacts_audit [--runs results/stack_dl/chrono/runs]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

ROOT = Path(__file__).resolve().parents[2]
L = ['c4', 'rpa', 'f8', 'p8', 'f4', 'p4', 'fp2', 'o2', 'cz', 'pz',
     'fz', 'o1', 'fp1', 'p3', 'f3', 'p7', 'f7', 'lpa', 'c3', 'c7', 'c8']
IX = {c: i for i, c in enumerate(L)}
FR = np.arange(1, 125)
# ВЧ без линий сети (50/100) и гармоник кадровой частоты NIRS 12.5 Гц
HF = ((FR >= 32) & (FR <= 45)) | ((FR >= 52) & (FR <= 60)) | ((FR >= 65) & (FR <= 72)) | ((FR >= 78) & (FR <= 85))
RIGHT = ['c4', 'f4', 'p4', 'f8', 'p8', 'fp2', 'o2', 'c8', 'rpa']
LEFT = ['c3', 'f3', 'p3', 'f7', 'p7', 'fp1', 'o1', 'c7', 'lpa']
TEMPORAL = ['f7', 'f8', 'c7', 'c8', 'lpa', 'rpa', 'p7', 'p8']
FRONTAL = ['fp1', 'fp2', 'f7', 'f8']


def band(lp, chans, lo, hi):
    m = (FR >= lo) & (FR <= hi)
    return lp[:, [IX[c] for c in chans]][:, :, m].mean((1, 2))


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def burst_index(lp, y, block):
    """Медиана по каналам робастного z ВЧ-мощности относительно покоя блока 1 (оффлайн-описание)."""
    v = lp[:, :, HF].mean(2)
    r = (y == 1) & (block == 1)
    med = np.median(v[r], 0)
    mad = np.median(np.abs(v[r] - med), 0) + 1e-6
    return np.median((v - med) / mad, 1)


def session_features(z, e):
    lp = z["lp"].astype(np.float32)
    y, b = z["y"], z["block"]
    f = {
        "HF all (32–85 Гц, медиана каналов)": np.median(lp[:, :, HF].mean(2), 1),
        "HF височные/периферия": band(lp, TEMPORAL, 32, 45),
        "HF латерализация (прав − лев)": band(lp, RIGHT, 32, 45) - band(lp, LEFT, 32, 45),
        "ВЧ-всплеск (z)": burst_index(lp, y, b),
        "лобные 1–3 Гц (fp1,fp2,f7,f8)": band(lp, FRONTAL, 1, 3),
        "hEOG f8−f7 среднее": e["mean"][:, 0],
        "hEOG f8−f7 размах": np.log(e["ptp"][:, 0] + 1e-9),
        "vEOG размах": np.log(e["ptp"][:, 1] + 1e-9),
        "c3 мю 8–13": band(lp, ['c3'], 8, 13),
        "c4 мю 8–13": band(lp, ['c4'], 8, 13),
        "c3 бета 14–30": band(lp, ['c3'], 14, 30),
        "c4 бета 14–30": band(lp, ['c4'], 14, 30),
        "латерализация мю c4 − c3": band(lp, ['c4'], 8, 13) - band(lp, ['c3'], 8, 13),
        "латерализация бета c4 − c3": band(lp, ['c4'], 14, 30) - band(lp, ['c3'], 14, 30),
        "лобная латерализация мю f8+fp2 − f7+fp1": band(lp, ['f8', 'fp2'], 8, 13) - band(lp, ['f7', 'fp1'], 8, 13),
    }
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(ROOT / "data/features/artifacts_audit"))
    ap.add_argument("--out", default=str(ROOT / "results/artifacts_audit"))
    ap.add_argument("--runs", default=str(ROOT / "results/stack_dl/chrono/runs"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, burst_rows, acc_rows = [], [], []
    for fz in sorted(Path(a.feat).glob("*_N??.npz")):
        s = fz.stem
        subj = re.search(r"_(N\d+)", s).group(1)
        z = np.load(fz); e = np.load(fz.with_name(s + "_eog.npz"))
        y, b, tr = z["y"], z["block"], z["trial"]
        feats = session_features(z, e)
        for name, v in feats.items():
            mi, re_ = v[y > 1], v[y == 1]
            lft, rgt = v[y == 2], v[y == 3]
            # уровень проб
            tv = pd.DataFrame({"v": v, "y": y, "t": tr}).groupby("t").agg(v=("v", "mean"), y=("y", "first"))
            tl, trr, tm, trs = (tv.v[tv.y == 2].values, tv.v[tv.y == 3].values,
                                tv.v[tv.y > 1].values, tv.v[tv.y == 1].values)
            rows.append(dict(session=s, subject=subj, feature=name,
                             auc_rm=auc(mi, re_), auc_lr=auc(lft, rgt),
                             auc_lr_late=auc(v[(y == 2) & (b > 1)], v[(y == 3) & (b > 1)]),
                             p_lr=mannwhitneyu(tl, trr).pvalue, p_rm=mannwhitneyu(tm, trs).pvalue,
                             auc_lr_trial=auc(tl, trr)))
        bi = feats["ВЧ-всплеск (z)"]
        thr = max(3.0, np.quantile(bi[(y == 1) & (b == 1)], 0.99))
        ex = bi > thr
        br = dict(session=s, subject=subj, has_nirs=bool(z["has_nirs"]))
        for c, nm in ((1, "rest"), (2, "left"), (3, "right")):
            br[f"burst_{nm}_b1"] = ex[(y == c) & (b == 1)].mean()
            br[f"burst_{nm}_b2_7"] = ex[(y == c) & (b > 1)].mean()
        burst_rows.append(br)
        np.save(out / f"_burst_{s}.npy", np.c_[z["ends"], ex])
        # точность решения на окнах со всплеском / без
        pp = Path(a.runs) / s / "predictions.csv"
        if pp.exists():
            p = pd.read_csv(pp)
            m = pd.Series(ex, index=z["ends"])
            p = p[p.end_sample.isin(m.index)]
            p["burst"] = m.loc[p.end_sample].values
            for bflag in (False, True):
                q = p[p.burst == bflag]
                if len(q):
                    qm = q[q.true_y > 1]
                    acc_rows.append(dict(session=s, burst=bflag, n=len(q), acc=(q.y == q.true_y).mean(),
                                         acc_eeg=(q.y_eeg == q.true_y).mean(),
                                         pred_mi=(q.y > 1).mean(), true_mi=(q.true_y > 1).mean(),
                                         lr_acc_mi=(qm.y == qm.true_y).mean() if len(qm) else np.nan))
    df = pd.DataFrame(rows); df.to_csv(out / "auc_per_session.csv", index=False)
    bd = pd.DataFrame(burst_rows); bd.to_csv(out / "bursts_per_session.csv", index=False)
    ad = pd.DataFrame(acc_rows); ad.to_csv(out / "accuracy_burst_vs_clean.csv", index=False)

    def summ(g):
        d = g.auc_lr - 0.5
        return pd.Series(dict(
            auc_rm_median=g.auc_rm.median(), rm_sessions_absdev_gt_0_1=(np.abs(g.auc_rm - .5) > .1).mean(),
            rm_sessions_p05=(g.p_rm < .05).mean(),
            auc_lr_median=g.auc_lr.median(), auc_lr_mean_abs_dev=np.abs(d).mean(),
            lr_sessions_absdev_gt_0_1=(np.abs(d) > .1).mean(), lr_sessions_trial_p05=(g.p_lr < .05).mean(),
            lr_sign_consistency=max((d > 0).mean(), (d < 0).mean())))
    sm = df.groupby("feature", sort=False).apply(summ)
    sm.to_csv(out / "auc_summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(sm.round(3))
    print("\nВсплески (доля окон, среднее по сессиям):")
    print(bd.drop(columns=["session", "subject", "has_nirs"]).mean().round(3))
    affected = bd[(bd.burst_left_b2_7 + bd.burst_right_b2_7) / 2 > bd.burst_rest_b2_7 + 0.05]
    print("сессий с MI-всплесками (MI − покой > 5 п.п., блоки 2–7):", len(affected), "участники:",
          sorted(affected.subject.unique()))
    (out / "affected_sessions.json").write_text(json.dumps(affected.session.tolist(), indent=1))
    if len(ad):
        g = ad.groupby("burst").apply(lambda q: pd.Series(dict(n=q.n.sum(), acc=np.average(q.acc, weights=q.n),
                                                               acc_eeg=np.average(q.acc_eeg, weights=q.n),
                                                               pred_mi=np.average(q.pred_mi, weights=q.n),
                                                               true_mi=np.average(q.true_mi, weights=q.n))))
        print("\nТочность решения (хронология) на окнах со всплеском и без:\n", g.round(3))


if __name__ == "__main__":
    main()
