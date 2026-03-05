#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
validate_abstraction.py

Compares patient_master.csv against gold_cleaned_for_cedar.csv
and computes accuracy for each abstraction variable.

Python 3.6.8 compatible.
"""

import pandas as pd
import os

MASTER_FILE = "_outputs/patient_master.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"

PID = "ENCRYPTED_PAT_ID"

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
# Safe CSV reader (handles encoding issues)
# ---------------------------------------------------

def safe_read_csv(path):

    try:
        return pd.read_csv(path, encoding="utf-8")
    except:
        return pd.read_csv(path, encoding="latin1")


# ---------------------------------------------------
# Standardize values before comparison
# ---------------------------------------------------

def normalize(series):

    series = series.fillna("NA")
    series = series.astype(str)
    series = series.str.strip()
    series = series.str.lower()

    return series


# ---------------------------------------------------
# Compute accuracy
# ---------------------------------------------------

def compute_metrics(pred, gold):

    pred = normalize(pred)
    gold = normalize(gold)

    total = len(gold)

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

    merged = pd.merge(master, gold, on=PID, suffixes=("_pred", "_gold"))

    print("Merged rows:", len(merged))

    results = []

    for v in VARIABLES:

        pred_col = v + "_pred"
        gold_col = v + "_gold"

        if pred_col not in merged.columns or gold_col not in merged.columns:
            print("Skipping missing variable:", v)
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

    print("\nValidation results:\n")
    print(df)

    if not os.path.exists("_outputs"):
        os.makedirs("_outputs")

    df.to_csv("_outputs/validation_summary.csv", index=False)

    print("\nValidation complete.")
    print("Results saved to: _outputs/validation_summary.csv")


if __name__ == "__main__":
    main()
