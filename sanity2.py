#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
confusion_complications.py

Compute confusion matrices for Stage1 and Stage2 complication variables.

Input:
    _outputs/validation_merged.csv

Outputs:
    printed confusion matrices
"""

import pandas as pd

FILE = "_outputs/validation_merged.csv"

COMPLICATION_VARS = [
    "Stage1_MinorComp",
    "Stage1_Reoperation",
    "Stage1_Rehospitalization",
    "Stage1_MajorComp",
    "Stage1_Failure",
    "Stage1_Revision",
    "Stage2_MinorComp",
    "Stage2_Reoperation",
    "Stage2_Rehospitalization",
    "Stage2_MajorComp",
    "Stage2_Failure",
    "Stage2_Revision",
]


def normalize_binary(x):
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    return 0


def confusion(pred, gold):
    TP = ((pred == 1) & (gold == 1)).sum()
    TN = ((pred == 0) & (gold == 0)).sum()
    FP = ((pred == 1) & (gold == 0)).sum()
    FN = ((pred == 0) & (gold == 1)).sum()
    return TP, FP, FN, TN


def main():

    df = pd.read_csv(FILE, dtype=str)

    print("\nConfusion Matrices\n")

    for var in COMPLICATION_VARS:

        pred_col = "PRED_" + var if "PRED_" + var in df.columns else var
        gold_col = "GOLD_" + var if "GOLD_" + var in df.columns else var

        if pred_col not in df.columns or gold_col not in df.columns:
            continue

        pred = df[pred_col].apply(normalize_binary)
        gold = df[gold_col].apply(normalize_binary)

        TP, FP, FN, TN = confusion(pred, gold)

        print("\n", var)
        print("--------------------------")
        print("TP:", TP)
        print("FP:", FP)
        print("FN:", FN)
        print("TN:", TN)

        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0

        print("Precision:", round(precision, 3))
        print("Recall:", round(recall, 3))


if __name__ == "__main__":
    main()
