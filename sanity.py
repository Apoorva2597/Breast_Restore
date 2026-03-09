#!/usr/bin/env python3
# qa_smoking_mismatches.py
#
# QA script to investigate SmokingStatus validation mismatches.
# Uses the SAME file paths as your build and validation scripts.
#
# Checks:
# 1. Rows where gold SmokingStatus exists but prediction is missing
# 2. Rows where prediction exists but does not match gold
# 3. Whether prediction is empty / NaN / whitespace
# 4. Prints counts and saves detailed QA table
#
# Python 3.6.8 compatible

import pandas as pd
import numpy as np

MASTER_FILE = "/home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"

MRN = "MRN"
VAR = "SmokingStatus"

def normalize_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ["", "nan", "none", "null", "na"]:
        return None
    return s

print("\nLoading files...")
pred = pd.read_csv(MASTER_FILE, dtype=str)
gold = pd.read_csv(GOLD_FILE, dtype=str)

pred.columns = pred.columns.str.strip()
gold.columns = gold.columns.str.strip()

pred[MRN] = pred[MRN].astype(str).str.strip()
gold[MRN] = gold[MRN].astype(str).str.strip()

print("Pred rows:", len(pred))
print("Gold rows:", len(gold))

df = gold[[MRN, VAR]].merge(
    pred[[MRN, VAR]],
    on=MRN,
    how="left",
    suffixes=("_gold", "_pred")
)

df["gold_norm"] = df[VAR + "_gold"].apply(normalize_text)
df["pred_norm"] = df[VAR + "_pred"].apply(normalize_text)

df["pred_missing"] = df["pred_norm"].isna()
df["match"] = df["gold_norm"] == df["pred_norm"]

gold_present = df["gold_norm"].notna()

subset = df[gold_present].copy()

total = len(subset)
missing_pred = subset["pred_missing"].sum()
matches = subset["match"].sum()
mismatches = total - matches

print("\nQA SUMMARY")
print("----------------------------")
print("Gold rows:", total)
print("Pred missing:", missing_pred)
print("Matches:", matches)
print("Mismatches:", mismatches)
print("Accuracy:", matches / total if total > 0 else 0)

print("\nUnique predicted values:")
print(pred[VAR].dropna().value_counts())

print("\nUnique gold values:")
print(gold[VAR].dropna().value_counts())

qa_out = "/home/apokol/Breast_Restore/_outputs/qa_smoking_mismatches.csv"

subset.to_csv(qa_out, index=False)

print("\nDetailed QA file saved to:")
print(qa_out)
