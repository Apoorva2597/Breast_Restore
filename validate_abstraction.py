#!/usr/bin/env python3

import pandas as pd
import numpy as np

MASTER_FILE = "/home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
OUTPUT_FILE = "/home/apokol/Breast_Restore/_outputs/validation_summary.csv"

MRN_COL = "MRN"

# -----------------------------------
# Helpers
# -----------------------------------
def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def normalize_mrn(df):
    for k in ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]:
        if k in df.columns:
            if k != MRN_COL:
                df = df.rename(columns={k: MRN_COL})
            break
    if MRN_COL not in df.columns:
        raise RuntimeError("MRN column not found.")
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

def to_binary01(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["1", "1.0", "true", "yes", "y"]:
        return 1
    if s in ["0", "0.0", "false", "no", "n"]:
        return 0
    try:
        f = float(s)
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
    except Exception:
        pass
    return np.nan

def obesity_from_bmi(x):
    if pd.isna(x):
        return np.nan
    return 1 if float(x) >= 30.0 else 0

def compare_generic(gold_series, pred_series):
    gold_missing = gold_series.apply(is_missing_val)
    pred_missing = pred_series.apply(is_missing_val)

    comparable = (~gold_missing) & (~pred_missing)
    total_compared = int(comparable.sum())

    if total_compared == 0:
        return 0.0, 0, 0

    gold_clean = gold_series[comparable].astype(str).str.strip().str.lower()
    pred_clean = pred_series[comparable].astype(str).str.strip().str.lower()

    matches = int((gold_clean == pred_clean).sum())
    acc = float(matches) / float(total_compared)

    return acc, matches, total_compared

# -----------------------------------
# Load data
# -----------------------------------
print("Loading files...")

master = pd.read_csv(MASTER_FILE, dtype=str)
gold = pd.read_csv(GOLD_FILE, dtype=str)

master = clean_cols(master)
gold = clean_cols(gold)

master = normalize_mrn(master)
gold = normalize_mrn(gold)

print("Master rows: {0}".format(len(master)))
print("Gold rows: {0}".format(len(gold)))

print("Merging directly on MRN...")
df = gold.merge(master, on=MRN_COL, how="left", suffixes=("_gold", "_pred"))
print("Merged rows: {0}".format(len(df)))

results = []

# -----------------------------------
# Standard direct-string variables
# -----------------------------------
direct_vars = [
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
    "Chemo",
]

for var in direct_vars:
    gold_col = var + "_gold"
    pred_col = var + "_pred"

    if gold_col not in df.columns or pred_col not in df.columns:
        continue

    acc, matches, total_compared = compare_generic(df[gold_col], df[pred_col])

    results.append({
        "variable": var,
        "accuracy": round(acc, 6),
        "matches": matches,
        "total_compared": total_compared
    })

# -----------------------------------
# BMI metrics
# -----------------------------------
if "BMI_gold" in df.columns and "BMI_pred" in df.columns:

    df["BMI_gold_num"] = df["BMI_gold"].apply(to_float)
    df["BMI_pred_num"] = df["BMI_pred"].apply(to_float)

    bmi_comp = (~df["BMI_gold_num"].isna()) & (~df["BMI_pred_num"].isna())
    bmi_total = int(bmi_comp.sum())

    if bmi_total > 0:
        diff_abs = (df.loc[bmi_comp, "BMI_pred_num"] - df.loc[bmi_comp, "BMI_gold_num"]).abs()

        # exact
        bmi_exact_matches = int((diff_abs == 0).sum())
        bmi_exact_acc = float(bmi_exact_matches) / float(bmi_total)

        # close tolerance ±0.5
        bmi_close_matches = int((diff_abs <= 0.5).sum())
        bmi_close_acc = float(bmi_close_matches) / float(bmi_total)

        # rounded integer match
        gold_round = df.loc[bmi_comp, "BMI_gold_num"].round(0)
        pred_round = df.loc[bmi_comp, "BMI_pred_num"].round(0)
        bmi_round_matches = int((gold_round == pred_round).sum())
        bmi_round_acc = float(bmi_round_matches) / float(bmi_total)

        results.append({
            "variable": "BMI",
            "accuracy": round(bmi_exact_acc, 6),
            "matches": bmi_exact_matches,
            "total_compared": bmi_total
        })

        results.append({
            "variable": "BMI_close_0_5",
            "accuracy": round(bmi_close_acc, 6),
            "matches": bmi_close_matches,
            "total_compared": bmi_total
        })

        results.append({
            "variable": "BMI_round_integer",
            "accuracy": round(bmi_round_acc, 6),
            "matches": bmi_round_matches,
            "total_compared": bmi_total
        })

        # obesity from BMI
        df.loc[bmi_comp, "Obesity_gold_from_BMI"] = df.loc[bmi_comp, "BMI_gold_num"].apply(obesity_from_bmi)
        df.loc[bmi_comp, "Obesity_pred_from_BMI"] = df.loc[bmi_comp, "BMI_pred_num"].apply(obesity_from_bmi)

        obesity_matches = int(
            (
                df.loc[bmi_comp, "Obesity_gold_from_BMI"] ==
                df.loc[bmi_comp, "Obesity_pred_from_BMI"]
            ).sum()
        )
        obesity_acc = float(obesity_matches) / float(bmi_total)

        results.append({
            "variable": "Obesity_from_BMI",
            "accuracy": round(obesity_acc, 6),
            "matches": obesity_matches,
            "total_compared": bmi_total
        })

# -----------------------------------
# Optional: compare existing Obesity column if gold has one
# -----------------------------------
if "Obesity_gold" in df.columns and "Obesity_pred" in df.columns:
    gold_ob = df["Obesity_gold"].apply(to_binary01)
    pred_ob = df["Obesity_pred"].apply(to_binary01)

    ob_comp = (~gold_ob.isna()) & (~pred_ob.isna())
    ob_total = int(ob_comp.sum())

    if ob_total > 0:
        ob_matches = int((gold_ob[ob_comp] == pred_ob[ob_comp]).sum())
        ob_acc = float(ob_matches) / float(ob_total)

        results.append({
            "variable": "Obesity_column_direct",
            "accuracy": round(ob_acc, 6),
            "matches": ob_matches,
            "total_compared": ob_total
        })

# -----------------------------------
# Output
# -----------------------------------
summary_df = pd.DataFrame(results)

print("\nValidation Results\n")
print(summary_df)

summary_df.to_csv(OUTPUT_FILE, index=False)

print("\nValidation complete.")
print("Results saved to {0}".format(OUTPUT_FILE))
