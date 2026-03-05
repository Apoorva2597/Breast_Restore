#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
validate_abstraction.py

Validates abstraction output against gold labels using MRN→ENCRYPTED_PAT_ID mapping.

Compatible with Python 3.6.8.
"""

import pandas as pd
import os
import sys

MASTER_FILE = "_outputs/patient_master.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"

DATA_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

ENCOUNTER_FILES = [
    os.path.join(DATA_DIR, "HPI11526 Clinic Encounters.csv"),
    os.path.join(DATA_DIR, "HPI11526 Inpatient Encounters.csv"),
    os.path.join(DATA_DIR, "HPI11526 Operation Encounters.csv")
]

PID = "ENCRYPTED_PAT_ID"
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
    except:
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
# Build MRN → ENCRYPTED_PAT_ID mapping
# ---------------------------------------------------

def build_mapping():

    frames = []

    for f in ENCOUNTER_FILES:
        df = safe_read_csv(f)

        if MRN in df.columns and PID in df.columns:
            frames.append(df[[MRN, PID]])

    if len(frames) == 0:
        print("ERROR: Could not find MRN ↔ ENCRYPTED_PAT_ID mapping in encounter files.")
        sys.exit(1)

    mapping = pd.concat(frames)

    mapping = mapping.drop_duplicates()

    mapping[MRN] = mapping[MRN].astype(str)
    mapping[PID] = mapping[PID].astype(str)

    return mapping


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    print("Loading files...")

    master = safe_read_csv(MASTER_FILE)
    gold = safe_read_csv(GOLD_FILE)

    print("Master rows:", len(master))
    print("Gold rows:", len(gold))

    master[PID] = master[PID].astype(str)

    if MRN not in gold.columns:
        print("ERROR: gold file missing MRN column")
        sys.exit(1)

    gold[MRN] = gold[MRN].astype(str)

    # ---------------------------------------------------
    # Build mapping
    # ---------------------------------------------------

    print("Building MRN → ENCRYPTED_PAT_ID mapping...")

    mapping = build_mapping()

    print("Mapping rows:", len(mapping))

    # ---------------------------------------------------
    # Attach ENCRYPTED_PAT_ID to gold
    # ---------------------------------------------------

    gold = pd.merge(gold, mapping, on=MRN, how="left")

    print("Gold rows after mapping:", len(gold))

    # ---------------------------------------------------
    # Merge with master abstraction
    # ---------------------------------------------------

    merged = pd.merge(master, gold, on=PID, suffixes=("_pred", "_gold"))

    print("Merged rows:", len(merged))

    if len(merged) == 0:
        print("ERROR: No rows matched after mapping.")
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
