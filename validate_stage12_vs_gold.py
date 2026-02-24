#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Stage1/Stage2 presence vs gold (no dates).

Gold provides:
- MRN
- Stage2_Applicable

Gold does NOT provide Stage1_Applicable.
We therefore validate Stage1 as: "all gold rows are stage1-eligible"
(i.e., compare pipeline has_expander against gold membership).
"""

from __future__ import print_function
import os
import pandas as pd

BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"

CROSSWALK = os.path.join(BREAST_RESTORE_DIR, "CROSSWALK", "CROSSWALK__MRN_to_patient_id__vNEW.csv")
PIPELINE = os.path.join(BREAST_RESTORE_DIR, "MASTER__STAGING_PATHWAY__vNEW.csv")

# TODO: set to your actual gold file path
GOLD = os.path.join(BREAST_RESTORE_DIR, "gold_cleaned_for_cedar.csv")

OUT_DIR = os.path.join(BREAST_RESTORE_DIR, "VALIDATION_STAGE12")
os.makedirs(OUT_DIR, exist_ok=True)

GOLD_MRN_COL = "MRN"
GOLD_STAGE2_FLAG_COL = "Stage2_Applicable"

def read_csv_robust(path):
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")

def to_boolish(x):
    s = "" if x is None else str(x).strip().lower()
    if s in ["1", "true", "t", "yes", "y"]:
        return True
    if s in ["0", "false", "f", "no", "n", ""]:
        return False
    # fallback: unknown non-empty -> False (conservative)
    return False

def confusion(df, gold_col, pred_col):
    tp = int(((df[gold_col] == True) & (df[pred_col] == True)).sum())
    tn = int(((df[gold_col] == False) & (df[pred_col] == False)).sum())
    fp = int(((df[gold_col] == False) & (df[pred_col] == True)).sum())
    fn = int(((df[gold_col] == True) & (df[pred_col] == False)).sum())
    return tp, fp, fn, tn

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def main():
    xw = read_csv_robust(CROSSWALK)
    pipe = read_csv_robust(PIPELINE)
    gold = read_csv_robust(GOLD)

    # normalize IDs
    xw["MRN"] = xw["MRN"].astype(str).str.strip()
    xw["patient_id"] = xw["patient_id"].astype(str).str.strip()

    gold[GOLD_MRN_COL] = gold[GOLD_MRN_COL].astype(str).str.strip()
    pipe["patient_id"] = pipe["patient_id"].astype(str).str.strip()

    # join gold -> crosswalk -> pipeline
    g = gold.merge(xw, left_on=GOLD_MRN_COL, right_on="MRN", how="left")
    joined = g.merge(pipe, on="patient_id", how="left", suffixes=("_gold", "_pipe"))

    joined.to_csv(os.path.join(OUT_DIR, "joined_gold_crosswalk_pipeline.csv"), index=False, encoding="utf-8")

    # Stage2 from gold
    if GOLD_STAGE2_FLAG_COL not in joined.columns:
        raise RuntimeError("Gold Stage2 flag column not found: {}".format(GOLD_STAGE2_FLAG_COL))

    joined["gold_stage2"] = joined[GOLD_STAGE2_FLAG_COL].apply(to_boolish)
    joined["pipe_stage2"] = joined.get("has_stage2_definitive", "").apply(to_boolish)

    # Stage1: gold cohort membership only (eligible = True for all gold rows)
    joined["gold_stage1"] = True
    joined["pipe_stage1"] = joined.get("has_expander", "").apply(to_boolish)

    # Stage2 confusion
    s2_tp, s2_fp, s2_fn, s2_tn = confusion(joined, "gold_stage2", "pipe_stage2")
    s2_precision = safe_div(s2_tp, (s2_tp + s2_fp))
    s2_recall = safe_div(s2_tp, (s2_tp + s2_fn))

    # Stage1: only recall among gold (since gold_stage1 always True)
    # Equivalent: FN = gold patients where pipeline did not find expander
    s1_fn = int((joined["pipe_stage1"] == False).sum())
    s1_tp = int((joined["pipe_stage1"] == True).sum())
    s1_recall = safe_div(s1_tp, (s1_tp + s1_fn))

    # write mismatch lists
    joined[(joined["pipe_stage1"] == False)].to_csv(os.path.join(OUT_DIR, "stage1_missing_expander_in_pipeline.csv"),
                                                    index=False, encoding="utf-8")

    joined[(joined["gold_stage2"] == True) & (joined["pipe_stage2"] == False)].to_csv(
        os.path.join(OUT_DIR, "stage2_FN.csv"), index=False, encoding="utf-8"
    )
    joined[(joined["gold_stage2"] == False) & (joined["pipe_stage2"] == True)].to_csv(
        os.path.join(OUT_DIR, "stage2_FP.csv"), index=False, encoding="utf-8"
    )

    # crosswalk coverage issues
    joined[joined["patient_id"].isnull() | (joined["patient_id"].astype(str).str.strip() == "")].to_csv(
        os.path.join(OUT_DIR, "gold_rows_missing_crosswalk_mapping.csv"), index=False, encoding="utf-8"
    )

    # print summary
    print("==== CROSSWALK COVERAGE ====")pid_str = joined["patient_id"].astype(str).str.strip()
    mapped = ((~joined["patient_id"].isnull()) & (pid_str != "") & (pid_str.str.lower() != "nan")).sum()
    mapped = int(mapped)
    print("Gold rows:", len(joined))
    print("Gold rows mapped to patient_id:", mapped)
    print("Gold rows NOT mapped:", len(joined) - mapped)
    print("")

    print("==== STAGE1 (gold has no Stage1_Applicable) ====")
    print("Gold rows assumed Stage1-eligible: {}".format(len(joined)))
    print("Pipeline has_expander True: {}".format(s1_tp))
    print("Pipeline has_expander False: {}".format(s1_fn))
    print("Recall (pipeline expander among gold):", round(s1_recall, 4))
    print("WROTE: stage1_missing_expander_in_pipeline.csv")
    print("")

    print("==== STAGE2 (Stage2_Applicable) ====")
    print("TP:", s2_tp, "FP:", s2_fp, "FN:", s2_fn, "TN:", s2_tn)
    print("Precision:", round(s2_precision, 4), "Recall:", round(s2_recall, 4))
    print("WROTE: stage2_FN.csv, stage2_FP.csv")
    print("")
    print("Artifacts in:", OUT_DIR)

if __name__ == "__main__":
    main()
