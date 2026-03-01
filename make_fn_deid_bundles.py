#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_fn_deid_bundles.py  (Python 3.6.8 friendly)

Fixes:
1) Avoids hanging/traceback on Ctrl+C (KeyboardInterrupt) while running batch exporter.
2) Lets you force WHICH ID column to use (your run tried exporting 4,5,6... which is almost certainly the wrong id).
3) Defaults to using ENCRYPTED_PAT_ID (or PatientID) before patient_id/MRN.

Run from: /home/apokol/Breast_Restore

Default:
  IN  : ./_outputs/validation_merged_STAGE2_ANCHOR_FIXED.csv
  OUT : ./_outputs/FN_patient_ids.csv
  RUN : ./batch_export_deid_note_bundles.py  (your existing batch script)

Usage:
  python make_fn_deid_bundles.py
  python make_fn_deid_bundles.py --pid_col ENCRYPTED_PAT_ID
  python make_fn_deid_bundles.py --no_run
  python make_fn_deid_bundles.py --merged /path/to/merged.csv --pid_col PatientID
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

# Your batch exporter script filename (adjust to match your actual file):
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


def unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def pick_default_pid_col(df_cols):
    """
    Choose the most likely ID column that matches your PATIENT_BUNDLES / de-id exporter.
    Prefer ENCRYPTED_PAT_ID or PatientID (seen in your screenshots) over MRN/patient_id.
    """
    priority = [
        "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID",
        "PatientID", "PATIENTID",
        "patient_id", "PATIENT_ID",
        "MRN", "mrn",
    ]
    for c in priority:
        if c in df_cols:
            return c
    return None


def parse_args(argv):
    args = {
        "merged": DEFAULT_MERGED,
        "pid_col": None,
        "no_run": False
    }

    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ["--merged", "-m"]:
            i += 1
            if i >= len(argv):
                raise ValueError("Missing value for --merged")
            args["merged"] = argv[i]
        elif a in ["--pid_col", "-c"]:
            i += 1
            if i >= len(argv):
                raise ValueError("Missing value for --pid_col")
            args["pid_col"] = argv[i]
        elif a == "--no_run":
            args["no_run"] = True
        else:
            raise ValueError("Unknown argument: {}".format(a))
        i += 1

    return args


def main():
    args = parse_args(sys.argv)

    merged_path = args["merged"]
    if not os.path.isfile(merged_path):
        raise IOError("Merged validation file not found: {}".format(merged_path))

    print("Using merged:", merged_path)
    df = normalize_cols(read_csv_robust(merged_path, dtype=str, low_memory=False))

    # Required columns from validation merge
    if "GOLD_HAS_STAGE2" not in df.columns or "PRED_HAS_STAGE2" not in df.columns:
        raise ValueError(
            "Merged file must contain GOLD_HAS_STAGE2 and PRED_HAS_STAGE2. Found: {}".format(list(df.columns))
        )

    # Pick patient id column
    pid_col = args["pid_col"]
    if pid_col is None:
        pid_col = pick_default_pid_col(list(df.columns))

    if not pid_col or pid_col not in df.columns:
        raise ValueError(
            "Could not find pid_col='{}'. Available columns: {}".format(pid_col, list(df.columns))
        )

    print("Using pid_col:", pid_col)

    # Compute FN: gold=1, pred=0
    gold01 = df["GOLD_HAS_STAGE2"].map(to01).astype(int)
    pred01 = df["PRED_HAS_STAGE2"].map(to01).astype(int)
    fn = df[(gold01 == 1) & (pred01 == 0)].copy()

    pids = fn[pid_col].fillna("").astype(str).str.strip().tolist()
    pids = [p for p in pids if p]
    pids = unique_preserve_order(pids)

    out_df = pd.DataFrame({"patient_id": pids})
    out_dir = os.path.dirname(OUT_CSV)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_df.to_csv(OUT_CSV, index=False)
    print("FN patients:", len(pids))
    print("Wrote:", OUT_CSV)

    if args["no_run"]:
        print("Not running batch exporter (--no_run).")
        return

    if not os.path.isfile(BATCH_EXPORT_SCRIPT):
        raise IOError("Batch export script not found: {}".format(BATCH_EXPORT_SCRIPT))

    cmd = [sys.executable, BATCH_EXPORT_SCRIPT, OUT_CSV]
    print("\nRunning:", " ".join(cmd))
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
        print("You can resume anytime by running:")
        print("  {} {}".format(sys.executable, " ".join(cmd[1:])))
        return

    if rc != 0:
        print("\nBatch exporter exited with code:", rc)
        print("Check logs in: {}/QA_DEID_BUNDLES/logs/".format(BREAST_RESTORE_DIR))
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
