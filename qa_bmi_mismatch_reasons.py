#!/usr/bin/env python3

import pandas as pd
import numpy as np

MASTER_FILE = "/home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
OUTPUT_FILE = "/home/apokol/Breast_Restore/_outputs/bmi_mismatch_reasons.csv"

MRN_COL = "MRN"
BMI_COL = "BMI"


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df):
    for k in ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]:
        if k in df.columns:
            if k != MRN_COL:
                df = df.rename(columns={k: MRN_COL})
            break
    df[MRN_COL] = df[MRN_COL].astype(str).str.strip()
    return df


def is_missing_val(x):
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in ["", "nan", "none", "null", "na"]


def to_float(x):
    try:
        if is_missing_val(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def classify_reason(gold_bmi, pred_bmi):
    gold_missing = pd.isna(gold_bmi)
    pred_missing = pd.isna(pred_bmi)

    if gold_missing and pred_missing:
        return "BOTH_MISSING"

    if gold_missing and (not pred_missing):
        return "GOLD_MISSING"

    if (not gold_missing) and pred_missing:
        return "PRED_MISSING"

    diff = pred_bmi - gold_bmi
    abs_diff = abs(diff)

    if abs_diff == 0:
        return "EXACT_MATCH"

    if round(pred_bmi) == round(gold_bmi):
        return "ROUND_MATCH_ONLY"

    if abs_diff <= 0.5:
        if diff > 0:
            return "SMALL_DIFF_PRED_HIGHER"
        return "SMALL_DIFF_PRED_LOWER"

    if abs_diff <= 1.0:
        if diff > 0:
            return "MODERATE_DIFF_PRED_HIGHER"
        return "MODERATE_DIFF_PRED_LOWER"

    if abs_diff <= 2.0:
        if diff > 0:
            return "LARGER_DIFF_PRED_HIGHER"
        return "LARGER_DIFF_PRED_LOWER"

    if diff > 0:
        return "LARGE_DIFF_PRED_HIGHER"
    return "LARGE_DIFF_PRED_LOWER"


print("Loading files...")

master = pd.read_csv(MASTER_FILE, dtype=str)
gold = pd.read_csv(GOLD_FILE, dtype=str)

master = clean_cols(master)
gold = clean_cols(gold)

master = normalize_mrn(master)
gold = normalize_mrn(gold)

if BMI_COL not in gold.columns:
    raise RuntimeError("Gold file missing BMI column.")

if BMI_COL not in master.columns:
    master[BMI_COL] = ""

print("Merging gold and predictions...")

df = gold[[MRN_COL, BMI_COL]].merge(
    master[[MRN_COL, BMI_COL]],
    on=MRN_COL,
    how="left",
    suffixes=("_gold", "_pred")
)

df = df.rename(columns={
    BMI_COL + "_gold": "BMI_gold_raw",
    BMI_COL + "_pred": "BMI_pred_raw"
})

df["BMI_gold"] = df["BMI_gold_raw"].apply(to_float)
df["BMI_pred"] = df["BMI_pred_raw"].apply(to_float)

df["gold_missing"] = df["BMI_gold"].isna()
df["pred_missing"] = df["BMI_pred"].isna()

df["diff_signed"] = df["BMI_pred"] - df["BMI_gold"]
df["diff_abs"] = (df["BMI_pred"] - df["BMI_gold"]).abs()

df["gold_round"] = df["BMI_gold"].round(0)
df["pred_round"] = df["BMI_pred"].round(0)

df["exact_match"] = (df["BMI_gold"] == df["BMI_pred"])
df["round_match"] = (df["gold_round"] == df["pred_round"])

df["reason"] = df.apply(
    lambda r: classify_reason(r["BMI_gold"], r["BMI_pred"]),
    axis=1
)

df_out = df[[
    MRN_COL,
    "BMI_gold",
    "BMI_pred",
    "diff_signed",
    "diff_abs",
    "gold_missing",
    "pred_missing",
    "exact_match",
    "round_match",
    "reason"
]].copy()

df_out = df_out.sort_values(
    by=["reason", "diff_abs", MRN_COL],
    ascending=[True, False, True]
)

df_out.to_csv(OUTPUT_FILE, index=False)

print("")
print("SUMMARY")
print("")

reason_counts = df_out["reason"].value_counts(dropna=False)
for reason, count in reason_counts.items():
    print("{0}: {1}".format(reason, count))

print("")

valid_diffs = df_out["diff_abs"].dropna()
if len(valid_diffs) > 0:
    print("Mean absolute difference: {0}".format(round(valid_diffs.mean(), 3)))
    print("Median absolute difference: {0}".format(round(valid_diffs.median(), 3)))
    print("Max absolute difference: {0}".format(round(valid_diffs.max(), 3)))

print("")
print("CSV written to:")
print(OUTPUT_FILE)

print("")
print("Top 15 worst mismatches:")
worst = df_out[
    (~df_out["gold_missing"]) &
    (~df_out["pred_missing"])
].sort_values(by="diff_abs", ascending=False).head(15)

for _, row in worst.iterrows():
    print("{0} | gold={1} | pred={2} | abs_diff={3} | reason={4}".format(
        row[MRN_COL],
        row["BMI_gold"],
        row["BMI_pred"],
        row["diff_abs"],
        row["reason"]
    ))
