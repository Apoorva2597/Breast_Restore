#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
validate_abstraction.py

Validates abstraction output against gold labels using direct MRN merge.

Compatible with Python 3.6.8.
"""

import pandas as pd
import os
import sys

MASTER_FILE = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"

MRN = "MRN"

VARIABLES = [
    "Race",
    "Ethnicity",
    "Age",
    "BMI",
    "SmokingStatus",
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
    "PBS_Lumpectomy",
    "PBS_Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "Radiation",
    "Chemo"
]


# ---------------------------------------------------
# Safe CSV reader
# ---------------------------------------------------

def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="latin1")


# ---------------------------------------------------
# Normalize values
# ---------------------------------------------------

def normalize(series):
    series = series.fillna("NA")
    series = series.astype(str)
    series = series.str.strip().str.lower()
    return series


# ---------------------------------------------------
# Compute accuracy
# ---------------------------------------------------

def compute_metrics(pred, gold):
    pred = normalize(pred)
    gold = normalize(gold)

    total = len(gold)

    if total == 0:
        return 0, 0, 0

    matches = (pred == gold).sum()
    accuracy = float(matches) / float(total)

    return accuracy, matches, total


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    print("Loading files...")

    master = safe_read_csv(MASTER_FILE)
    gold = safe_read_csv(GOLD_FILE)

    print("Master rows:", len(master))
    print("Gold rows:", len(gold))

    if MRN not in master.columns:
        print("ERROR: master file missing MRN column")
        sys.exit(1)

    if MRN not in gold.columns:
        print("ERROR: gold file missing MRN column")
        sys.exit(1)

    master[MRN] = master[MRN].astype(str).str.strip()
    gold[MRN] = gold[MRN].astype(str).str.strip()

    # drop blank MRNs before merge
    master = master[master[MRN] != ""].copy()
    gold = gold[gold[MRN] != ""].copy()

    # optional: deduplicate by MRN to avoid unexpected row multiplication
    master = master.drop_duplicates(subset=[MRN])
    gold = gold.drop_duplicates(subset=[MRN])

    # ---------------------------------------------------
    # Merge directly on MRN
    # ---------------------------------------------------

    print("Merging directly on MRN...")

    merged = pd.merge(master, gold, on=MRN, how="inner", suffixes=("_pred", "_gold"))

    print("Merged rows:", len(merged))

    if len(merged) == 0:
        print("ERROR: No rows matched on MRN.")
        sys.exit(1)

    results = []

    # ---------------------------------------------------
    # Evaluate variables
    # ---------------------------------------------------

    for v in VARIABLES:
        pred_col = v + "_pred"
        gold_col = v + "_gold"

        if pred_col not in merged.columns or gold_col not in merged.columns:
            print("Skipping variable:", v)
            continue

        pred = merged[pred_col]
        goldv = merged[gold_col]

        acc, matches, total = compute_metrics(pred, goldv)

        results.append({
            "variable": v,
            "accuracy": acc,
            "matches": matches,
            "total_compared": total
        })

    df = pd.DataFrame(results)

    print("\nValidation Results\n")
    print(df)

    if not os.path.exists("_outputs"):
        os.makedirs("_outputs")

    df.to_csv("_outputs/validation_summary.csv", index=False)

    print("\nValidation complete.")
    print("Results saved to _outputs/validation_summary.csv")


if __name__ == "__main__":
    main()
