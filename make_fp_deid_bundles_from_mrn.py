#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_fp_deid_bundles_from_mrn.py

FP definition (for your frozen summary file):
    GOLD  = HAS_STAGE2
    PRED  = HIT_COUNT > 0

We export ENCRYPTED_PAT_IDs for FP cases only.

This version auto-selects the latest:
  _outputs/stage2_rules_*_patient_stage_summary.csv
  _outputs/stage2_rules_*__patient_stage_summary.csv
"""

from __future__ import print_function

import os
import sys
import glob
import subprocess
import pandas as pd


ROOT = os.path.abspath(".")

OUTPUTS_DIR = os.path.join(ROOT, "_outputs")

# Auto-find the latest frozen summary (handles _patient_... and __patient_...)
SUMMARY_GLOBS = [
    os.path.join(OUTPUTS_DIR, "stage2_rules_*_patient_stage_summary.csv"),
    os.path.join(OUTPUTS_DIR, "stage2_rules_*__patient_stage_summary.csv"),
]

OP_PATH = os.path.join(ROOT, "_staging_inputs", "HPI11526 Operation Notes.csv")

OUT_IDS = os.path.join(ROOT, "_outputs", "FP_deid_patient_ids.csv")
BATCH_EXPORTER = os.path.join(ROOT, "batch_export_deid_note_bundles.py")


def find_latest_summary():
    candidates = []
    for g in SUMMARY_GLOBS:
        candidates.extend(glob.glob(g))
    candidates = sorted(set(candidates))
    if not candidates:
        return None
    # pick most recently modified
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


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


def normalize_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit():
            return head
    return s


def normalize_mrn(x):
    return normalize_id(x)


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


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


def main():

    merged_path = find_latest_summary()
    if not merged_path:
        raise IOError("No frozen stage2 summary found in _outputs. Looked for: {}".format(SUMMARY_GLOBS))

    if not os.path.isfile(OP_PATH):
        raise IOError("Missing op-notes bridge file: {}".format(OP_PATH))
    if not os.path.isfile(BATCH_EXPORTER):
        raise IOError("Missing batch exporter script: {}".format(BATCH_EXPORTER))

    merged = normalize_cols(read_csv_robust(merged_path, dtype=str, low_memory=False))
    op = normalize_cols(read_csv_robust(OP_PATH, dtype=str, low_memory=False))

    # Required columns
    required = ["MRN", "HAS_STAGE2", "HIT_COUNT"]
    for r in required:
        if r not in merged.columns:
            raise ValueError("Summary file missing required column {}. Found: {}".format(r, list(merged.columns)))

    merged["MRN"] = merged["MRN"].map(normalize_mrn)

    # GOLD = HAS_STAGE2
    gold = merged["HAS_STAGE2"].map(to01)

    # PRED = HIT_COUNT > 0
    pred = merged["HIT_COUNT"].apply(lambda x: 1 if to01(x) > 0 else 0)

    # FP mask
    fp_mask = (gold == 0) & (pred == 1)

    fp_mrns = merged.loc[fp_mask, ["MRN"]].copy()
    fp_mrns = fp_mrns[fp_mrns["MRN"] != ""].drop_duplicates()

    # Build MRN <-> ENCRYPTED_PAT_ID map from op notes
    op_mrn_col = pick_first_existing(op, ["MRN", "mrn"])
    op_enc_col = pick_first_existing(op, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not op_mrn_col or not op_enc_col:
        raise ValueError("Op notes must contain MRN and ENCRYPTED_PAT_ID. Found: {}".format(list(op.columns)))

    op["MRN"] = op[op_mrn_col].map(normalize_mrn)
    op["ENCRYPTED_PAT_ID"] = op[op_enc_col].map(normalize_id)

    id_map = op[["MRN", "ENCRYPTED_PAT_ID"]].drop_duplicates()
    id_map = id_map[(id_map["MRN"] != "") & (id_map["ENCRYPTED_PAT_ID"] != "")]

    # Map FP MRNs -> ENCRYPTED_PAT_ID
    fp_ids = fp_mrns.merge(id_map, on="MRN", how="left")
    fp_ids["ENCRYPTED_PAT_ID"] = fp_ids["ENCRYPTED_PAT_ID"].fillna("").map(normalize_id)
    fp_ids = fp_ids[fp_ids["ENCRYPTED_PAT_ID"] != ""].drop_duplicates(subset=["ENCRYPTED_PAT_ID"])

    # Write export list WITHOUT MRN column
    out_df = pd.DataFrame({"patient_id": fp_ids["ENCRYPTED_PAT_ID"].tolist()})
    out_df.to_csv(OUT_IDS, index=False)

    print("Using summary:", merged_path)
    print("FP MRNs:", len(fp_mrns))
    print("Mapped ENCRYPTED_PAT_IDs:", len(out_df))
    print("Wrote (NO MRN column):", OUT_IDS)
    print("")

    # Run exporter
    cmd = [sys.executable, BATCH_EXPORTER, OUT_IDS]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
