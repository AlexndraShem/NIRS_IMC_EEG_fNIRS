"""Минимальный пример чтения .mat сессии (не обязателен для сдачи).

Использование:
  python example.py /path/to/session.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

from scipy.io import loadmat


def main(path: str | Path) -> None:
    path = Path(path)
    mat = loadmat(path, simplify_cells=True)
    eeg = mat["EEG"]
    print(f"file: {path.name}")
    print(f"EEG.Raw: {eeg['Raw'].shape}  fs={eeg['Frq']} Hz")
    print(f"EEG.States unique: {sorted(set(eeg['States'].ravel().tolist()))}")
    print(f"EEG.Blocks unique: {sorted(set(eeg['Blocks'].ravel().tolist()))}")
    nirs = mat.get("NIRS")
    if isinstance(nirs, dict):
        print(f"NIRS.HbO: {nirs['HbO'].shape}  HbR: {nirs['HbR'].shape}  fs={nirs['Frq']} Hz")
        print(f"NIRS.Frame: {nirs['Frame'].shape}  (1-based EEG sample index)")
    else:
        print("NIRS: отсутствует")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "session.mat")
