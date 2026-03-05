#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_abstraction.py

Compares patient_master.csv vs gold labels.
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


def compute_metrics(pred, gold):

    pred = pred.fillna("NA")
    gold = gold.fillna("NA")

    total = len(gold)
    match = (pred == gold).sum()

    accuracy = match / float(total)

    return accuracy


def main():

    master = pd.read_csv(MASTER_FILE)
    gold = pd.read_csv(GOLD_FILE)

    merged = pd.merge(master, gold, on=PID, suffixes=("_pred","_gold"))

    results = []

    for v in VARIABLES:

        pred = merged[v+"_pred"]
        goldv = merged[v+"_gold"]

        acc = compute_metrics(pred, goldv)

        results.append({
            "variable":v,
            "accuracy":acc
        })

    df = pd.DataFrame(results)

    print(df)

    df.to_csv("_outputs/validation_summary.csv",index=False)

    print("Validation complete")


if __name__ == "__main__":
    main()
