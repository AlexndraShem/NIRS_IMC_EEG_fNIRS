"""Сводка по всем сессиям: длины, частоты, классы, NIRS, плохие каналы."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset.io import EEG_LABELS, list_sessions, load_session  # noqa: E402


def summarize(path):
    s = load_session(path)
    x = s.eeg.astype(np.float64)
    dstd = np.diff(x, axis=0).std(0)
    med = np.median(dstd)
    bad = [EEG_LABELS[i] for i in np.flatnonzero((dstd > 20 * med) | (dstd < med / 20))]
    st = s.states.astype(int)
    ch = np.flatnonzero(np.diff(st)) + 1
    seg_start = np.r_[0, ch]
    seg_len = np.diff(np.r_[seg_start, len(st)]) / s.eeg_fs
    task = st[seg_start] > 0
    row = dict(name=s.name, subject=s.subject, dur_s=len(st) / s.eeg_fs, eeg_fs=s.eeg_fs,
               n_blocks=int(s.blocks.max()),
               **{f"cnt{k}": int((st == k).sum()) for k in range(4)},
               n_trials=int(task.sum()), trial_len_min=seg_len[task].min(), trial_len_max=seg_len[task].max(),
               has_nirs=s.hbo is not None, bad_eeg=",".join(bad), eeg_scale=float(med))
    if s.hbo is not None:
        row.update(nirs_fs=s.nirs_fs, nirs_n=len(s.hbo), nirs_nan=int(np.isnan(s.hbo).sum()),
                   nirs_ch=s.hbo.shape[1], frame_last=int(s.nirs_frame[-1]),
                   nirs_state_agree=float((s.nirs_states == s.states[np.clip(s.nirs_frame - 1, 0, len(st) - 1)]).mean()))
    return row


if __name__ == "__main__":
    paths = list_sessions()
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(summarize, paths))
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[2] / "data" / "sessions_summary.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 100)
    print(df.drop(columns=["eeg_fs"]).to_string())
