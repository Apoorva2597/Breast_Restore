#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_fp_deid_bundles_from_mrn.py (Python 3.6.8 compatible)

Goal:
- Use MRN (from validation_merged_STAGE2_ANCHOR_FIXED.csv) to export de-id bundles,
  WITHOUT requiring the exporter to accept MRN directly.
- We build an MRN -> ENCRYPTED_PAT_ID map from:
    ./_staging_inputs/HPI11526 Operation Notes.csv
  (same bridge your validation uses)
- Then we feed ENCRYPTED_PAT_IDs into your existing batch exporter
  as the "patient_id" column.

Privacy:
- The only CSV we write for exporting contains *no MRN column*.
  It contains just: patient_id (which will be ENCRYPTED_PAT_ID values).
"""

from __future__ import print_function

import os
import sys
import subprocess
import pandas as pd


ROOT = os.path.abspath(".")

MERGED_PATH = os.path.join(
    ROOT,
    "_outputs",
    "stage2_rules_20260301_135501__patient_stage_summary.csv")  
  
OP_PATH = os.path.join(ROOT, "_staging_inputs", "HPI11526 Operation Notes.csv")

OUT_IDS = os.path.join(ROOT, "_outputs", "FP_deid_patient_ids.csv")  # NO MRN column
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
    if not os.path.isfile(MERGED_PATH):
        raise IOError("Missing merged validation file: {}".format(MERGED_PATH))
    if not os.path.isfile(OP_PATH):
        raise IOError("Missing op-notes bridge file: {}".format(OP_PATH))
    if not os.path.isfile(BATCH_EXPORTER):
        raise IOError("Missing batch exporter script: {}".format(BATCH_EXPORTER))

    merged = normalize_cols(read_csv_robust(MERGED_PATH, dtype=str, low_memory=False))
    op = normalize_cols(read_csv_robust(OP_PATH, dtype=str, low_memory=False))

    # --- Required cols for fp logic
    if "GOLD_HAS_STAGE2" not in merged.columns or "PRED_HAS_STAGE2" not in merged.columns:
        raise ValueError(
            "Merged file missing GOLD_HAS_STAGE2 or PRED_HAS_STAGE2. Found: {}".format(list(merged.columns))
        )

    # --- MRN column in merged
    merged_mrn_col = pick_first_existing(merged, ["MRN", "mrn"])
    if not merged_mrn_col:
        raise ValueError("Merged file missing MRN column. Found: {}".format(list(merged.columns)))

    merged["MRN"] = merged[merged_mrn_col].map(normalize_mrn)

    # --- Build MRN <-> ENCRYPTED_PAT_ID map from op notes (bridge)
    op_mrn_col = pick_first_existing(op, ["MRN", "mrn"])
    op_enc_col = pick_first_existing(op, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not op_mrn_col or not op_enc_col:
        raise ValueError(
            "Op notes must contain MRN and ENCRYPTED_PAT_ID. Found: {}".format(list(op.columns))
        )

    op["MRN"] = op[op_mrn_col].map(normalize_mrn)
    op["ENCRYPTED_PAT_ID"] = op[op_enc_col].map(normalize_id)
    id_map = op[["MRN", "ENCRYPTED_PAT_ID"]].drop_duplicates()
    id_map = id_map[(id_map["MRN"] != "") & (id_map["ENCRYPTED_PAT_ID"] != "")].copy()

    # --- fp mask
    gold = merged["GOLD_HAS_STAGE2"].map(to01)
    pred = merged["PRED_HAS_STAGE2"].map(to01)
    fp_mask = (gold == 0) & (pred == 1)

    fp_mrns = merged.loc[fp_mask, ["MRN"]].copy()
    fp_mrns = fp_mrns[fp_mrns["MRN"] != ""].drop_duplicates()

    # --- Map FP MRNs -> ENCRYPTED_PAT_IDs
    fp_ids = fp_mrns.merge(id_map, on="MRN", how="left")
    fp_ids["ENCRYPTED_PAT_ID"] = fp_ids["ENCRYPTED_PAT_ID"].fillna("").map(normalize_id)

    # Keep only mapped IDs
    fp_ids = fp_ids[fp_ids["ENCRYPTED_PAT_ID"] != ""].drop_duplicates(subset=["ENCRYPTED_PAT_ID"])

    # Write export list WITHOUT MRN column
    out_df = pd.DataFrame({"patient_id": fp_ids["ENCRYPTED_PAT_ID"].tolist()})
    out_df.to_csv(OUT_IDS, index=False)

    missing_ct = int((fp_ids["ENCRYPTED_PAT_ID"] == "").sum())  # should be 0 after filter, kept for clarity

    print("Using merged: {}".format(MERGED_PATH))
    print("Using op bridge: {}".format(OP_PATH))
    print("fp MRNs: {}".format(len(fp_mrns)))
    print("Mapped ENCRYPTED_PAT_IDs for export: {}".format(len(out_df)))
    print("Wrote (NO MRN column): {}".format(OUT_IDS))
    print("")

    # Run your existing batch exporter
    cmd = [sys.executable, BATCH_EXPORTER, OUT_IDS]
    print("Running: {}".format(" ".join(cmd)))
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
