#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_fp_deid_bundles_from_mrn.py

FP definition:
  GOLD = Stage2_Applicable  (from gold_cleaned_for_cedar.csv)   [you confirmed this]
  PRED = HIT_COUNT > 0      (from latest frozen stage2 summary)

FP mask:
  (GOLD == 0) & (PRED == 1)

Exports:
  _outputs/FP_deid_patient_ids.csv  (NO MRN column; only patient_id=ENCRYPTED_PAT_ID)
Then runs:
  batch_export_deid_note_bundles.py <that csv>
"""

from __future__ import print_function

import os
import sys
import glob
import subprocess
import pandas as pd


ROOT = os.path.abspath(".")
OUTPUTS_DIR = os.path.join(ROOT, "_outputs")

# ---- inputs
GOLD_PATH = os.path.join(ROOT, "gold_cleaned_for_cedar.csv")
OP_PATH   = os.path.join(ROOT, "_staging_inputs", "HPI11526 Operation Notes.csv")

# auto-find latest frozen summary (handles _patient_... and __patient_...)
SUMMARY_GLOBS = [
    os.path.join(OUTPUTS_DIR, "stage2_rules_*_patient_stage_summary.csv"),
    os.path.join(OUTPUTS_DIR, "stage2_rules_*__patient_stage_summary.csv"),
]

# ---- outputs
OUT_IDS = os.path.join(OUTPUTS_DIR, "FP_deid_patient_ids.csv")  # NO MRN column
BATCH_EXPORTER = os.path.join(ROOT, "batch_export_deid_note_bundles.py")


def find_latest_summary():
    candidates = []
    for g in SUMMARY_GLOBS:
        candidates.extend(glob.glob(g))
    candidates = sorted(set(candidates))
    if not candidates:
        return None
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


def pick_gold_stage2_applicable_col(gold_df):
    """
    You said GOLD should be Stage2_Applicable.
    This tries common variants so it won't break if the exact casing differs.
    """
    options = [
        "Stage2_Applicable", "STAGE2_APPLICABLE", "stage2_applicable",
        "Stage2Applicable", "STAGE2APPLICABLE"
    ]
    c = pick_first_existing(gold_df, options)
    if c:
        return c

    # last resort: fuzzy contains both 'stage2' and 'app'
    for col in gold_df.columns:
        u = col.upper()
        if "STAGE2" in u and ("APPLIC" in u or "APPL" in u):
            return col

    return None


def main():
    summary_path = find_latest_summary()
    if not summary_path:
        raise IOError("No frozen stage2 summary found in _outputs. Looked for: {}".format(SUMMARY_GLOBS))

    for req in [GOLD_PATH, OP_PATH, BATCH_EXPORTER]:
        if not os.path.isfile(req):
            raise IOError("Missing required file: {}".format(req))

    # ---- load
    summary = normalize_cols(read_csv_robust(summary_path, dtype=str, low_memory=False))
    gold    = normalize_cols(read_csv_robust(GOLD_PATH,    dtype=str, low_memory=False))
    op      = normalize_cols(read_csv_robust(OP_PATH,      dtype=str, low_memory=False))

    # ---- summary must have MRN, HIT_COUNT
    for col in ["MRN", "HIT_COUNT"]:
        if col not in summary.columns:
            raise ValueError("Summary missing required column {}. Found: {}".format(col, list(summary.columns)))

    summary["MRN"] = summary["MRN"].map(normalize_mrn)
    summary["HIT_COUNT"] = summary["HIT_COUNT"].fillna("").map(normalize_id)

    # ---- gold must have MRN + Stage2_Applicable
    gold_mrn_col = pick_first_existing(gold, ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"])
    if not gold_mrn_col:
        raise ValueError("Gold file missing MRN column. Found: {}".format(list(gold.columns)))

    gold_stage2_col = pick_gold_stage2_applicable_col(gold)
    if not gold_stage2_col:
        raise ValueError("Gold file missing Stage2_Applicable (or variant). Found: {}".format(list(gold.columns)))

    gold["MRN"] = gold[gold_mrn_col].map(normalize_mrn)
    gold["GOLD_STAGE2_APPLICABLE"] = gold[gold_stage2_col].map(to01)

    gold_small = gold[["MRN", "GOLD_STAGE2_APPLICABLE"]].drop_duplicates()

    # ---- merge gold + prediction summary on MRN
    m = summary.merge(gold_small, on="MRN", how="left")

    # keep only rows where we have a gold label (optional; but safer)
    m = m[m["GOLD_STAGE2_APPLICABLE"].notnull()].copy()
    m["GOLD_STAGE2_APPLICABLE"] = m["GOLD_STAGE2_APPLICABLE"].fillna(0).astype(int)

    # ---- PRED = HIT_COUNT > 0
    def hitcount_to_pred(x):
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return 0
        try:
            return 1 if int(float(s)) > 0 else 0
        except Exception:
            return 0

    m["PRED_HAS_STAGE2"] = m["HIT_COUNT"].map(hitcount_to_pred)

    # ---- FP mask (GOLD==0 & PRED==1)
    fp_mask = (m["GOLD_STAGE2_APPLICABLE"] == 0) & (m["PRED_HAS_STAGE2"] == 1)
    fp_mrns = m.loc[fp_mask, ["MRN"]].copy()
    fp_mrns = fp_mrns[fp_mrns["MRN"] != ""].drop_duplicates()

    # ---- build MRN -> ENCRYPTED_PAT_ID map from op-notes bridge
    op_mrn_col = pick_first_existing(op, ["MRN", "mrn"])
    op_enc_col = pick_first_existing(op, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not op_mrn_col or not op_enc_col:
        raise ValueError("Op notes must contain MRN and ENCRYPTED_PAT_ID. Found: {}".format(list(op.columns)))

    op["MRN"] = op[op_mrn_col].map(normalize_mrn)
    op["ENCRYPTED_PAT_ID"] = op[op_enc_col].map(normalize_id)
    id_map = op[["MRN", "ENCRYPTED_PAT_ID"]].drop_duplicates()
    id_map = id_map[(id_map["MRN"] != "") & (id_map["ENCRYPTED_PAT_ID"] != "")].copy()

    fp_ids = fp_mrns.merge(id_map, on="MRN", how="left")
    fp_ids["ENCRYPTED_PAT_ID"] = fp_ids["ENCRYPTED_PAT_ID"].fillna("").map(normalize_id)
    fp_ids = fp_ids[fp_ids["ENCRYPTED_PAT_ID"] != ""].drop_duplicates(subset=["ENCRYPTED_PAT_ID"])

    # ---- write export list WITHOUT MRN column
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_df = pd.DataFrame({"patient_id": fp_ids["ENCRYPTED_PAT_ID"].tolist()})
    out_df.to_csv(OUT_IDS, index=False)

    # ---- print quick audit
    print("Using summary: {}".format(summary_path))
    print("Using gold:    {}".format(GOLD_PATH))
    print("Gold col:      {}".format(gold_stage2_col))
    print("FP MRNs:       {}".format(len(fp_mrns)))
    print("Mapped IDs:    {}".format(len(out_df)))
    print("Wrote (NO MRN): {}".format(OUT_IDS))
    print("")

    # ---- run exporter
    cmd = [sys.executable, BATCH_EXPORTER, OUT_IDS]
    print("Running: {}".format(" ".join(cmd)))
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
