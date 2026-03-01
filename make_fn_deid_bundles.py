#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_fn_deid_bundles.py  (Python 3.6.8 friendly)

Find the FN patients (Gold Stage2=1, Pred Stage2=0) from the validation MERGED file
and run your existing batch de-id bundler script to export QA bundles for them.

Assumes you run from: /home/apokol/Breast_Restore

Default inputs/outputs:
  IN  : ./_outputs/validation_merged_STAGE2_ANCHOR_FIXED.csv
  OUT : ./_outputs/FN_patient_ids.csv
  RUN : ./batch_export_deid_bundles.py  (the script you pasted earlier)

Usage:
  python make_fn_deid_bundles.py
  python make_fn_deid_bundles.py /path/to/validation_merged.csv
"""

from __future__ import print_function
import os
import sys
import subprocess
import pandas as pd


# --------- HARD-CODED (edit if needed) ----------
BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"
DEFAULT_MERGED = os.path.join(BREAST_RESTORE_DIR, "_outputs", "validation_merged_STAGE2_ANCHOR_FIXED.csv")
OUT_CSV = os.path.join(BREAST_RESTORE_DIR, "_outputs", "FN_patient_ids.csv")

# This is the batch exporter script you already have (paste/save it to this path):
BATCH_EXPORT_SCRIPT = os.path.join(BREAST_RESTORE_DIR, "batch_export_deid_note_bundles.py")
# -----------------------------------------------


def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))


def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


def to01(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def main():
    merged_path = DEFAULT_MERGED
    if len(sys.argv) == 2:
        merged_path = sys.argv[1]
    elif len(sys.argv) > 2:
        print("Usage: {} [optional:/path/to/validation_merged.csv]".format(sys.argv[0]), file=sys.stderr)
        sys.exit(2)

    if not os.path.isfile(merged_path):
        raise IOError("Merged validation file not found: {}".format(merged_path))

    if not os.path.isfile(BATCH_EXPORT_SCRIPT):
        raise IOError("Batch export script not found: {}".format(BATCH_EXPORT_SCRIPT))

    print("Using merged:", merged_path)
    df = normalize_cols(read_csv_robust(merged_path, dtype=str, low_memory=False))

    # Required columns from validation merge
    gold_col = pick_first_existing(df, ["GOLD_HAS_STAGE2", "gold_has_stage2"])
    pred_col = pick_first_existing(df, ["PRED_HAS_STAGE2", "pred_has_stage2"])
    if not gold_col or not pred_col:
        raise ValueError("Merged file missing GOLD_HAS_STAGE2 and/or PRED_HAS_STAGE2. Found: {}".format(list(df.columns)))

    # Patient ID column to feed your exporter/bundle dirs
    # Prefer actual patient_id if present; otherwise fall back to ENCRYPTED_PAT_ID (most common in your pipeline).
    pid_col = pick_first_existing(df, [
        "patient_id", "PATIENT_ID",
        "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID",
        "PatientID", "PATIENTID"
    ])
    if not pid_col:
        raise ValueError(
            "Could not find a patient id column. Need one of: patient_id / ENCRYPTED_PAT_ID / PatientID. "
            "Found: {}".format(list(df.columns))
        )

    # Compute FN: gold=1, pred=0
    gold01 = df[gold_col].map(to01).astype(int)
    pred01 = df[pred_col].map(to01).astype(int)

    fn = df[(gold01 == 1) & (pred01 == 0)].copy()

    # Build patient_id list
    pids = fn[pid_col].fillna("").astype(str).str.strip().tolist()
    pids = [p for p in pids if p]

    # unique while preserving order
    seen = set()
    uniq = []
    for p in pids:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    # Write CSV for batch exporter
    out_df = pd.DataFrame({"patient_id": uniq})
    out_dir = os.path.dirname(OUT_CSV)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_df.to_csv(OUT_CSV, index=False)
    print("FN patients:", len(uniq))
    print("Wrote:", OUT_CSV)

    # Run your existing batch exporter on this list
    cmd = [sys.executable, BATCH_EXPORT_SCRIPT, OUT_CSV]
    print("\nRunning:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError("Batch exporter returned non-zero exit code: {}".format(rc))

    print("\nDone.")


if __name__ == "__main__":
    main()
