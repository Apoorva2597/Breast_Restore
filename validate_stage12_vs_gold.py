#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate Stage1/Stage2 presence vs gold (no dates).

Gold provides:
- MRN
- Stage2_Applicable  (boolean-ish)

Gold does NOT provide Stage1_Applicable.
We therefore validate Stage1 as:
- Treat ALL gold rows as stage1-eligible (gold_stage1=True for all rows)
- Report recall of pipeline has_expander within gold cohort.

Inputs (hardcoded):
- CROSSWALK: MRN <-> patient_id
- PIPELINE: MASTER__STAGING_PATHWAY__vNEW.csv
- GOLD: gold_cleaned_for_cedar.csv

Outputs:
- joined_gold_crosswalk_pipeline.csv
- gold_rows_missing_crosswalk_mapping.csv
- stage1_missing_expander_in_pipeline.csv
- stage2_FN.csv
- stage2_FP.csv
- summary printed to stdout
"""

from __future__ import print_function
import os
import pandas as pd

# ----------------------------
# HARD-CODED PATHS
# ----------------------------
BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"

CROSSWALK = os.path.join(BREAST_RESTORE_DIR, "CROSSWALK", "CROSSWALK__MRN_to_patient_id__vNEW.csv")
PIPELINE = os.path.join(BREAST_RESTORE_DIR, "MASTER__STAGING_PATHWAY__vNEW.csv")
GOLD = os.path.join(BREAST_RESTORE_DIR, "gold_cleaned_for_cedar.csv")

OUT_DIR = os.path.join(BREAST_RESTORE_DIR, "VALIDATION_STAGE12")
os.makedirs(OUT_DIR, exist_ok=True)

# Column names
GOLD_MRN_COL = "MRN"
GOLD_STAGE2_FLAG_COL = "Stage2_Applicable"

XW_MRN_COL = "MRN"
XW_PID_COL = "patient_id"

PIPE_PID_COL = "patient_id"
PIPE_STAGE1_COL = "has_expander"
PIPE_STAGE2_COL = "has_stage2_definitive"

# ----------------------------
# Helpers
# ----------------------------
def read_csv_robust(path):
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")

def norm_str(series):
    """Normalize to stripped string series (keeps NaN as 'nan' after astype(str))."""
    return series.astype(str).str.strip()

def is_mapped_patient_id(pid_series):
    """
    Return boolean Series for "has a real patient_id".
    Treats null/empty/"nan"/"none"/"null" as not mapped.
    """
    # do NOT call astype(str) before isnull() check (keep null semantics)
    not_null = ~pid_series.isnull()
    s = pid_series.astype(str).str.strip().str.lower()
    not_blank = s != ""
    not_bad = ~s.isin(["nan", "none", "null"])
    return not_null & not_blank & not_bad

def to_boolish(x):
    """
    Convert common representations to bool.
    Unknown non-empty values -> False (conservative).
    """
    s = "" if x is None else str(x).strip().lower()
    if s in ["1", "true", "t", "yes", "y"]:
        return True
    if s in ["0", "false", "f", "no", "n", ""]:
        return False
    return False

def confusion(df, gold_col, pred_col):
    tp = int(((df[gold_col] == True) & (df[pred_col] == True)).sum())
    tn = int(((df[gold_col] == False) & (df[pred_col] == False)).sum())
    fp = int(((df[gold_col] == False) & (df[pred_col] == True)).sum())
    fn = int(((df[gold_col] == True) & (df[pred_col] == False)).sum())
    return tp, fp, fn, tn

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

# ----------------------------
# Main
# ----------------------------
def main():
    # Load inputs
    if not os.path.exists(CROSSWALK):
        raise RuntimeError("CROSSWALK not found: {}".format(CROSSWALK))
    if not os.path.exists(PIPELINE):
        raise RuntimeError("PIPELINE not found: {}".format(PIPELINE))
    if not os.path.exists(GOLD):
        raise RuntimeError("GOLD not found: {}".format(GOLD))

    xw = read_csv_robust(CROSSWALK)
    pipe = read_csv_robust(PIPELINE)
    gold = read_csv_robust(GOLD)

    # Validate required columns exist
    for col in [XW_MRN_COL, XW_PID_COL]:
        if col not in xw.columns:
            raise RuntimeError("Crosswalk missing required column: {}".format(col))

    for col in [PIPE_PID_COL]:
        if col not in pipe.columns:
            raise RuntimeError("Pipeline missing required column: {}".format(col))

    for col in [GOLD_MRN_COL, GOLD_STAGE2_FLAG_COL]:
        if col not in gold.columns:
            raise RuntimeError("Gold missing required column: {}".format(col))

    # Normalize key fields
    xw[XW_MRN_COL] = norm_str(xw[XW_MRN_COL])
    xw[XW_PID_COL] = norm_str(xw[XW_PID_COL])

    gold[GOLD_MRN_COL] = norm_str(gold[GOLD_MRN_COL])
    pipe[PIPE_PID_COL] = norm_str(pipe[PIPE_PID_COL])

    # Join: gold -> crosswalk -> pipeline
    g = gold.merge(
        xw,
        left_on=GOLD_MRN_COL,
        right_on=XW_MRN_COL,
        how="left"
    )

    joined = g.merge(
        pipe,
        on=PIPE_PID_COL,
        how="left",
        suffixes=("_gold", "_pipe")
    )

    joined_path = os.path.join(OUT_DIR, "joined_gold_crosswalk_pipeline.csv")
    joined.to_csv(joined_path, index=False, encoding="utf-8")

    # Crosswalk coverage
    mapped_mask = is_mapped_patient_id(joined[PIPE_PID_COL])
    mapped = int(mapped_mask.sum())
    total_gold_rows = int(len(joined))

    missing_xw = joined[~mapped_mask].copy()
    missing_xw_path = os.path.join(OUT_DIR, "gold_rows_missing_crosswalk_mapping.csv")
    missing_xw.to_csv(missing_xw_path, index=False, encoding="utf-8")

    # Stage2 (gold provides Stage2_Applicable)
    joined["gold_stage2"] = joined[GOLD_STAGE2_FLAG_COL].apply(to_boolish)

    if PIPE_STAGE2_COL in joined.columns:
        joined["pipe_stage2"] = joined[PIPE_STAGE2_COL].apply(to_boolish)
    else:
        # if pipeline column is missing, everything becomes False
        joined["pipe_stage2"] = False

    s2_tp, s2_fp, s2_fn, s2_tn = confusion(joined, "gold_stage2", "pipe_stage2")
    s2_precision = safe_div(s2_tp, (s2_tp + s2_fp))
    s2_recall = safe_div(s2_tp, (s2_tp + s2_fn))

    # Stage1 (gold has no Stage1_Applicable; treat all gold rows as eligible)
    joined["gold_stage1"] = True
    if PIPE_STAGE1_COL in joined.columns:
        joined["pipe_stage1"] = joined[PIPE_STAGE1_COL].apply(to_boolish)
    else:
        joined["pipe_stage1"] = False

    s1_tp = int((joined["pipe_stage1"] == True).sum())
    s1_fn = int((joined["pipe_stage1"] == False).sum())
    s1_recall = safe_div(s1_tp, (s1_tp + s1_fn))

    # Write mismatch lists
    stage1_miss_path = os.path.join(OUT_DIR, "stage1_missing_expander_in_pipeline.csv")
    joined[joined["pipe_stage1"] == False].to_csv(stage1_miss_path, index=False, encoding="utf-8")

    stage2_fn_path = os.path.join(OUT_DIR, "stage2_FN.csv")
    joined[(joined["gold_stage2"] == True) & (joined["pipe_stage2"] == False)].to_csv(
        stage2_fn_path, index=False, encoding="utf-8"
    )

    stage2_fp_path = os.path.join(OUT_DIR, "stage2_FP.csv")
    joined[(joined["gold_stage2"] == False) & (joined["pipe_stage2"] == True)].to_csv(
        stage2_fp_path, index=False, encoding="utf-8"
    )

    # Print summary
    print("==== INPUTS ====")
    print("CROSSWALK:", CROSSWALK)
    print("PIPELINE :", PIPELINE)
    print("GOLD     :", GOLD)
    print("OUT_DIR  :", OUT_DIR)
    print("")

    print("==== CROSSWALK COVERAGE ====")
    print("Gold rows:", total_gold_rows)
    print("Gold rows mapped to patient_id:", mapped)
    print("Gold rows NOT mapped:", total_gold_rows - mapped)
    print("WROTE:", missing_xw_path)
    print("WROTE:", joined_path)
    print("")

    print("==== STAGE1 (gold has no Stage1_Applicable) ====")
    print("Gold rows assumed Stage1-eligible:", total_gold_rows)
    print("Pipeline has_expander True :", s1_tp)
    print("Pipeline has_expander False:", s1_fn)
    print("Recall (pipeline expander among gold):", round(s1_recall, 4))
    print("WROTE:", stage1_miss_path)
    print("")

    print("==== STAGE2 (Stage2_Applicable) ====")
    print("TP:", s2_tp, "FP:", s2_fp, "FN:", s2_fn, "TN:", s2_tn)
    print("Precision:", round(s2_precision, 4), "Recall:", round(s2_recall, 4))
    print("WROTE:", stage2_fn_path)
    print("WROTE:", stage2_fp_path)
    print("")
    print("Done.")

if __name__ == "__main__":
    main()
