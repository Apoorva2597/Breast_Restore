#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_fn_deid_bundles.py (Python 3.6.8 compatible)

1) Read validation merged file:
   ./_outputs/validation_merged_STAGE2_ANCHOR_FIXED.csv

2) Extract FN patients:
   GOLD_HAS_STAGE2 == 1 AND PRED_HAS_STAGE2 == 0

3) Write:
   ./_outputs/FN_patient_ids.csv   (single column: patient_id)

4) Run your existing batch exporter:
   ./batch_export_deid_note_bundles.py ./_outputs/FN_patient_ids.csv

IMPORTANT:
- This fixes the common failure where you accidentally pass MRN/excel_row instead of patient_id.
- We deliberately prefer columns that are likely to be true patient identifiers used by the exporter.
"""

from __future__ import print_function

import os
import sys
import subprocess
import pandas as pd


ROOT = os.path.abspath(".")
MERGED_PATH = os.path.join(ROOT, "_outputs", "validation_merged_STAGE2_ANCHOR_FIXED.csv")
OUT_IDS = os.path.join(ROOT, "_outputs", "FN_patient_ids.csv")

BATCH_EXPORTER = os.path.join(ROOT, "batch_export_deid_note_bundles.py")


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


def clean_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def main():
    if not os.path.isfile(MERGED_PATH):
        raise IOError("Missing merged validation file: {}".format(MERGED_PATH))
    if not os.path.isfile(BATCH_EXPORTER):
        raise IOError("Missing batch exporter script: {}".format(BATCH_EXPORTER))

    df = normalize_cols(read_csv_robust(MERGED_PATH, dtype=str, low_memory=False))

    # Required cols for FN logic
    if "GOLD_HAS_STAGE2" not in df.columns or "PRED_HAS_STAGE2" not in df.columns:
        raise ValueError(
            "Merged file missing GOLD_HAS_STAGE2 or PRED_HAS_STAGE2. Found: {}".format(list(df.columns))
        )

    # Pick the *right* patient identifier column for the exporter
    pid_col = pick_first_existing(df, ["patient_id", "PatientID", "PATIENT_ID", "ENCRYPTED_PAT_ID"])
    if not pid_col:
        raise ValueError(
            "Could not find a patient identifier column. Need one of: "
            "patient_id, PatientID, PATIENT_ID, ENCRYPTED_PAT_ID. Found: {}".format(list(df.columns))
        )

    # Compute FN mask
    gold = df["GOLD_HAS_STAGE2"].map(to01)
    pred = df["PRED_HAS_STAGE2"].map(to01)
    fn_mask = (gold == 1) & (pred == 0)

    fn = df.loc[fn_mask, [pid_col]].copy()
    fn[pid_col] = fn[pid_col].map(clean_id)
    fn = fn[fn[pid_col] != ""].drop_duplicates()

    # Write output CSV in the exact schema the batch exporter expects
    out_df = pd.DataFrame({"patient_id": fn[pid_col].tolist()})
    out_df.to_csv(OUT_IDS, index=False)

    print("Using merged: {}".format(MERGED_PATH))
    print("Patient id col: {}".format(pid_col))
    print("FN patients: {}".format(len(out_df)))
    print("Wrote: {}".format(OUT_IDS))
    print("")

    # Run the batch exporter
    cmd = [sys.executable, BATCH_EXPORTER, OUT_IDS]
    print("Running: {}".format(" ".join(cmd)))
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
