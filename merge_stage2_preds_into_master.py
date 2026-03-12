#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_stage2_preds_into_master.py

Run from:
    /home/apokol/Breast_Restore

Purpose:
- Merge selected Stage 2 staging columns from validation_merged.csv
- Into the master file
- Left join on MRN
- Write a new safe output file
- Python 3.6.8 compatible
"""

import os
import pandas as pd

BASE_DIR = os.getcwd()

MASTER_FILE = os.path.join(
    BASE_DIR,
    "_outputs",
    "master_abstraction_rule_FINAL_NO_GOLD.csv"
)

STAGE_FILE = os.path.join(
    BASE_DIR,
    "_frozen_stage2",
    "20260228_200052",
    "validation_merged.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "_outputs",
    "master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds.csv"
)

MERGE_KEY = "MRN"

STAGE_COLS = [
    "MRN",
    "PRED_HAS_STAGE2",
    "STAGE2_DATE",
    "WINDOW_START",
    "WINDOW_END",
    "Stage2_MinorComp_pred",
    "Stage2_Reoperation_pred",
    "Stage2_Rehospitalization_pred",
    "Stage2_MajorComp_pred",
    "Stage2_Failure_pred",
    "Stage2_Revision_pred",
]

def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                error_bad_lines=False,
                warn_bad_lines=True
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                on_bad_lines="skip"
            )
        except TypeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )

def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s

def normalize_mrn(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    found = None
    for k in key_variants:
        if k in df.columns:
            found = k
            break
    if found is None:
        raise RuntimeError("MRN column not found. Columns seen: {0}".format(list(df.columns)[:50]))
    if found != MERGE_KEY:
        df = df.rename(columns={found: MERGE_KEY})
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df

def ensure_stage_cols(stage_df, needed_cols):
    missing = [c for c in needed_cols if c not in stage_df.columns]
    if missing:
        raise RuntimeError("Missing required staging columns: {0}".format(missing))

def dedupe_stage(stage_df):
    value_cols = [c for c in STAGE_COLS if c != MERGE_KEY]

    def score_row(row):
        score = 0
        for c in value_cols:
            if clean_cell(row.get(c, "")):
                score += 1
        return score

    tmp = stage_df.copy()
    tmp["_stage_score_"] = tmp.apply(score_row, axis=1)
    tmp = tmp.sort_values(by=[MERGE_KEY, "_stage_score_"], ascending=[True, False])
    tmp = tmp.drop_duplicates(subset=[MERGE_KEY], keep="first")
    tmp = tmp.drop(columns=["_stage_score_"])
    return tmp

def main():
    print("Working directory:", BASE_DIR)
    print("Master file:", MASTER_FILE)
    print("Stage file:", STAGE_FILE)
    print("Output file:", OUTPUT_FILE)

    if not os.path.exists(MASTER_FILE):
        raise FileNotFoundError("Master file not found: {0}".format(MASTER_FILE))
    if not os.path.exists(STAGE_FILE):
        raise FileNotFoundError("Stage file not found: {0}".format(STAGE_FILE))

    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)

    stage = clean_cols(read_csv_robust(STAGE_FILE))
    stage = normalize_mrn(stage)

    ensure_stage_cols(stage, STAGE_COLS)

    master_rows_before = len(master)

    stage_subset = stage[STAGE_COLS].copy()
    stage_subset = dedupe_stage(stage_subset)

    merge_cols = [MERGE_KEY]
    added_cols = []

    for c in STAGE_COLS:
        if c == MERGE_KEY:
            continue
        if c in master.columns:
            print("Column already in master, skipping:", c)
        else:
            merge_cols.append(c)
            added_cols.append(c)

    stage_subset = stage_subset[merge_cols]

    merged = master.merge(stage_subset, on=MERGE_KEY, how="left")

    if len(merged) != master_rows_before:
        raise RuntimeError(
            "Row count changed after merge. Before={0}, After={1}".format(
                master_rows_before, len(merged)
            )
        )

    merged.to_csv(OUTPUT_FILE, index=False)

    print("Done.")
    print("Rows in master:", master_rows_before)
    print("Columns added:", added_cols)
    print("Wrote:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
