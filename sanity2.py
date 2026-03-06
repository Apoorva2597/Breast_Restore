#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qa_race_mismatches_simple.py

Simple QA script for Race:
- merges master vs gold on MRN
- keeps only rows where gold Race is non-missing
- outputs only MRN, Race_gold, Race_pred for mismatches
- also writes a frequency table of mismatch pairs

Python 3.6.8 compatible
"""

import os
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = os.path.join(BASE_DIR, "_outputs", "master_abstraction_rule_FINAL_NO_GOLD.csv")
GOLD_FILE = os.path.join(BASE_DIR, "gold_cleaned_for_cedar.csv")

OUT_MISMATCH_ROWS = os.path.join(BASE_DIR, "_outputs", "qa_race_mismatches_simple.csv")
OUT_MISMATCH_COUNTS = os.path.join(BASE_DIR, "_outputs", "qa_race_mismatch_counts.csv")

MRN = "MRN"


def safe_read_csv(path):
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=str, encoding="latin1")


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MRN:
                df = df.rename(columns={k: MRN})
            break
    if MRN not in df.columns:
        raise RuntimeError("MRN column not found. Columns seen: {0}".format(list(df.columns)[:40]))
    df[MRN] = df[MRN].astype(str).str.strip()
    return df


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


def main():
    print("Loading files...")
    master = clean_cols(safe_read_csv(MASTER_FILE))
    gold = clean_cols(safe_read_csv(GOLD_FILE))

    master = normalize_mrn(master)
    gold = normalize_mrn(gold)

    if "Race" not in master.columns:
        raise RuntimeError("Race column missing from master file")
    if "Race" not in gold.columns:
        raise RuntimeError("Race column missing from gold file")

    master = master[[MRN, "Race"]].drop_duplicates(subset=[MRN]).copy()
    gold = gold[[MRN, "Race"]].drop_duplicates(subset=[MRN]).copy()

    merged = pd.merge(
        master,
        gold,
        on=MRN,
        how="inner",
        suffixes=("_pred", "_gold")
    )

    merged["Race_pred"] = merged["Race_pred"].apply(clean_cell)
    merged["Race_gold"] = merged["Race_gold"].apply(clean_cell)

    # only compare rows where gold race exists
    compare = merged[merged["Race_gold"] != ""].copy()

    mism = compare[compare["Race_pred"] != compare["Race_gold"]].copy()

    counts = (
        mism.groupby(["Race_gold", "Race_pred"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "Race_gold", "Race_pred"], ascending=[False, True, True])
    )

    os.makedirs(os.path.join(BASE_DIR, "_outputs"), exist_ok=True)
    mism[[MRN, "Race_gold", "Race_pred"]].to_csv(OUT_MISMATCH_ROWS, index=False)
    counts.to_csv(OUT_MISMATCH_COUNTS, index=False)

    print("Total compared:", len(compare))
    print("Total mismatches:", len(mism))
    print("Wrote:", OUT_MISMATCH_ROWS)
    print("Wrote:", OUT_MISMATCH_COUNTS)

    if len(counts) > 0:
        print("\nTop mismatch pairs:")
        print(counts.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
