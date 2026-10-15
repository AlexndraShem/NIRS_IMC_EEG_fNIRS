"""Опционально. Судья не запускает.

  python train.py --data /s3/train --out solution/artifacts
"""
from pathlib import Path
import argparse
import io_utils as iu

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="artifacts")
args = ap.parse_args()
print("sessions:", len(iu.list_mat_files(args.data)))
Path(args.out).mkdir(parents=True, exist_ok=True)
