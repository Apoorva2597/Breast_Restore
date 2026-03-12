#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        return pd.NA
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n", ""}:
        return 0
    return pd.NA

def confusion(pred, gold):
    tp = ((pred == 1) & (gold == 1)).sum()
    fp = ((pred == 1) & (gold == 0)).sum()
    fn = ((pred == 0) & (gold == 1)).sum()
    tn = ((pred == 0) & (gold == 0)).sum()
    return tp, fp, fn, tn

def main():
    df = pd.read_csv(FILE, dtype=str)

    print("\nConfusion Matrices\n")

    for var in COMPLICATION_VARS:
        pred_col = var + "_pred"
        gold_col = var + "_gold"

        if pred_col not in df.columns or gold_col not in df.columns:
            print("Skipping:", var)
            continue

        pred = df[pred_col].apply(normalize_binary)
        gold = df[gold_col].apply(normalize_binary)

        # only compare rows where gold is non-missing
        mask = gold.notna()
        pred = pred[mask].fillna(0)
        gold = gold[mask]

        tp, fp, fn, tn = confusion(pred, gold)

        precision = float(tp) / float(tp + fp) if (tp + fp) else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) else 0.0

        print("\n{0}".format(var))
        print("--------------------------")
        print("TP:", int(tp))
        print("FP:", int(fp))
        print("FN:", int(fn))
        print("TN:", int(tn))
        print("Precision:", round(precision, 3))
        print("Recall:", round(recall, 3))

if __name__ == "__main__":
    main()
