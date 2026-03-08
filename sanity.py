#!/usr/bin/env python3

import pandas as pd
import numpy as np

MASTER_FILE = "/home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
OUTPUT_FILE = "/home/apokol/Breast_Restore/_outputs/bmi_gold_vs_pred.csv"

MRN_COL = "MRN"
GOLD_COL = "BMI"
PRED_COL = "BMI"

print("Loading files...")

master = pd.read_csv(MASTER_FILE, dtype=str)
gold = pd.read_csv(GOLD_FILE, dtype=str)

master.columns = master.columns.str.strip()
gold.columns = gold.columns.str.strip()

master[MRN_COL] = master[MRN_COL].astype(str).str.strip()
gold[MRN_COL] = gold[MRN_COL].astype(str).str.strip()

print("Merging gold and predictions...")

df = gold[[MRN_COL, GOLD_COL]].merge(
    master[[MRN_COL, PRED_COL]],
    on=MRN_COL,
    how="outer",
    suffixes=("_gold", "_pred")
)

df.rename(columns={
    GOLD_COL + "_gold": "BMI_gold",
    PRED_COL + "_pred": "BMI_pred"
}, inplace=True)

def to_float(x):
    try:
        return float(x)
    except:
        return np.nan

df["BMI_gold"] = df["BMI_gold"].apply(to_float)
df["BMI_pred"] = df["BMI_pred"].apply(to_float)

df["gold_missing"] = df["BMI_gold"].isna()
df["pred_missing"] = df["BMI_pred"].isna()

df["difference"] = abs(df["BMI_gold"] - df["BMI_pred"])

df["exact_match"] = df["BMI_gold"] == df["BMI_pred"]

df["integer_match"] = (
    df["BMI_gold"].round(0) == df["BMI_pred"].round(0)
)

print("\nSUMMARY\n")

total = len(df)
gold_missing = df["gold_missing"].sum()
pred_missing = df["pred_missing"].sum()
both_missing = ((df["gold_missing"]) & (df["pred_missing"])).sum()
exact_match = df["exact_match"].sum()
integer_match = df["integer_match"].sum()

print("Total rows:", total)
print("Gold missing:", gold_missing)
print("Pred missing:", pred_missing)
print("Both missing:", both_missing)
print("Exact matches:", exact_match)
print("Integer matches:", integer_match)

valid_diff = df["difference"].dropna()

if len(valid_diff) > 0:
    print("Mean absolute difference:", round(valid_diff.mean(), 2))
    print("Median difference:", round(valid_diff.median(), 2))
    print("Max difference:", round(valid_diff.max(), 2))

df.to_csv(OUTPUT_FILE, index=False)

print("\nQA table written to:")
print(OUTPUT_FILE)
