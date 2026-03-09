#!/usr/bin/env python3

import pandas as pd
import numpy as np

MASTER_FILE = "/home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
OUTPUT_FILE = "/home/apokol/Breast_Restore/_outputs/validation_summary.csv"

MRN_COL = "MRN"


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df):
    for k in ["MRN","mrn","Patient_MRN","PAT_MRN","PATIENT_MRN"]:
        if k in df.columns:
            if k != MRN_COL:
                df = df.rename(columns={k:MRN_COL})
            break

    df[MRN_COL] = df[MRN_COL].astype(str).str.strip()
    return df


def is_missing_val(x):
    if pd.isna(x):
        return True

    s = str(x).strip().lower()

    return s in ["","nan","none","null","na"]


def to_float(x):
    try:
        if is_missing_val(x):
            return np.nan
        return float(x)
    except:
        return np.nan


def generic_string_compare(gold_series,pred_series):

    gold_missing = gold_series.apply(is_missing_val)
    pred_missing = pred_series.apply(is_missing_val)

    comparable = (~gold_missing) & (~pred_missing)

    total = int(comparable.sum())

    if total == 0:
        return 0.0,0,0

    gold_clean = gold_series[comparable].astype(str).str.strip().str.lower()
    pred_clean = pred_series[comparable].astype(str).str.strip().str.lower()

    matches = int((gold_clean == pred_clean).sum())

    acc = matches / total

    return round(acc,6),matches,total


print("Loading files...")

master = pd.read_csv(MASTER_FILE,dtype=str)
gold = pd.read_csv(GOLD_FILE,dtype=str)

master = clean_cols(master)
gold = clean_cols(gold)

master = normalize_mrn(master)
gold = normalize_mrn(gold)

print("Master rows:",len(master))
print("Gold rows:",len(gold))

print("Merging directly on MRN...")

df = gold.merge(master,on=MRN_COL,how="left",suffixes=("_gold","_pred"))

print("Merged rows:",len(df))

results = []

variables = [
"Race",
"Ethnicity",
"SmokingStatus",
"Age",
"Diabetes",
"Hypertension",
"CardiacDisease",
"VenousThromboembolism",
"Steroid",
"PBS_Lumpectomy",
"Radiation",
"Chemo"
]

for var in variables:

    g = var+"_gold"
    p = var+"_pred"

    if g not in df.columns or p not in df.columns:
        continue

    acc,match,total = generic_string_compare(df[g],df[p])

    results.append({
        "variable":var,
        "accuracy":acc,
        "matches":match,
        "total_compared":total
    })


# -----------------------
# BMI ORIGINAL METRIC
# -----------------------

if "BMI_gold" in df.columns and "BMI_pred" in df.columns:

    df["BMI_gold_num"] = df["BMI_gold"].apply(to_float)
    df["BMI_pred_num"] = df["BMI_pred"].apply(to_float)

    comp = (~df["BMI_gold_num"].isna()) & (~df["BMI_pred_num"].isna())

    total = int(comp.sum())

    if total > 0:

        diff = (df.loc[comp,"BMI_gold_num"] - df.loc[comp,"BMI_pred_num"]).abs()

        exact = int((diff == 0).sum())

        acc = exact / total

        results.append({
        "variable":"BMI",
        "accuracy":round(acc,6),
        "matches":exact,
        "total_compared":total
        })


# -----------------------
# BMI CLOSE ±0.5
# -----------------------

        close = int((diff <= 0.5).sum())

        results.append({
        "variable":"BMI_close_0_5",
        "accuracy":round(close/total,6),
        "matches":close,
        "total_compared":total
        })


# -----------------------
# BMI INTEGER ROUND
# -----------------------

        g_round = df.loc[comp,"BMI_gold_num"].round(0)
        p_round = df.loc[comp,"BMI_pred_num"].round(0)

        round_matches = int((g_round == p_round).sum())

        results.append({
        "variable":"BMI_round_integer",
        "accuracy":round(round_matches/total,6),
        "matches":round_matches,
        "total_compared":total
        })


# -----------------------
# OBESITY FROM BMI
# -----------------------

        gold_ob = (df.loc[comp,"BMI_gold_num"] >= 30).astype(int)
        pred_ob = (df.loc[comp,"BMI_pred_num"] >= 30).astype(int)

        ob_match = int((gold_ob == pred_ob).sum())

        results.append({
        "variable":"Obesity_from_BMI",
        "accuracy":round(ob_match/total,6),
        "matches":ob_match,
        "total_compared":total
        })


summary = pd.DataFrame(results)

print("\nValidation Results\n")
print(summary)

summary.to_csv(OUTPUT_FILE,index=False)

print("\nValidation complete.")
print("Results saved to",OUTPUT_FILE)
