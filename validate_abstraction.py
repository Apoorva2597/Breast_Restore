#!/usr/bin/env python3

import pandas as pd
import numpy as np

MASTER_FILE = "/home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
OUTPUT_FILE = "/home/apokol/Breast_Restore/_outputs/validation_summary.csv"

MRN_COL = "MRN"
BMI_TOL = 0.5


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


def clean_string(x):
    if is_missing_val(x):
        return ""
    return str(x).strip().lower()


def to_float(x):
    try:
        if is_missing_val(x):
            return np.nan
        return float(str(x).strip())
    except Exception:
        return np.nan


def to_int_like(x):
    try:
        if is_missing_val(x):
            return np.nan
        return int(round(float(str(x).strip())))
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


# -----------------------------------
# Comparison functions
# -----------------------------------
def compare_string_var(df, gold_col, pred_col):
    gold_clean = df[gold_col].apply(clean_string)
    pred_clean = df[pred_col].apply(clean_string)

    comp = (gold_clean != "") & (pred_clean != "")
    total = int(comp.sum())

    if total == 0:
        return 0.0, 0, 0

    matches = int((gold_clean[comp] == pred_clean[comp]).sum())
    acc = float(matches) / float(total)
    return round(acc, 6), matches, total


def compare_numeric_exact(df, gold_col, pred_col):
    gold_num = df[gold_col].apply(to_float)
    pred_num = df[pred_col].apply(to_float)

    comp = (~gold_num.isna()) & (~pred_num.isna())
    total = int(comp.sum())

    if total == 0:
        return 0.0, 0, 0

    matches = int((gold_num[comp] == pred_num[comp]).sum())
    acc = float(matches) / float(total)
    return round(acc, 6), matches, total


def compare_binary_var(df, gold_col, pred_col):
    gold_bin = df[gold_col].apply(to_binary01)
    pred_bin = df[pred_col].apply(to_binary01)

    comp = (~gold_bin.isna()) & (~pred_bin.isna())
    total = int(comp.sum())

    if total == 0:
        return 0.0, 0, 0

    matches = int((gold_bin[comp] == pred_bin[comp]).sum())
    acc = float(matches) / float(total)
    return round(acc, 6), matches, total


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
# Finalized direct variables
# -----------------------------------
string_vars = [
    "Race",
    "Ethnicity",
    "SmokingStatus",
]

for var in string_vars:
    gold_col = var + "_gold"
    pred_col = var + "_pred"
    if gold_col in df.columns and pred_col in df.columns:
        acc, matches, total = compare_string_var(df, gold_col, pred_col)
        results.append({
            "variable": var,
            "accuracy": acc,
            "matches": matches,
            "total_compared": total
        })

# -----------------------------------
# Age as numeric exact
# -----------------------------------
if "Age_gold" in df.columns and "Age_pred" in df.columns:
    acc, matches, total = compare_numeric_exact(df, "Age_gold", "Age_pred")
    results.append({
        "variable": "Age",
        "accuracy": acc,
        "matches": matches,
        "total_compared": total
    })

# -----------------------------------
# Binary variables
# -----------------------------------
binary_vars = [
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
    "PBS_Lumpectomy",
    "Radiation",
    "Chemo",
]

for var in binary_vars:
    gold_col = var + "_gold"
    pred_col = var + "_pred"
    if gold_col in df.columns and pred_col in df.columns:
        acc, matches, total = compare_binary_var(df, gold_col, pred_col)
        results.append({
            "variable": var,
            "accuracy": acc,
            "matches": matches,
            "total_compared": total
        })

# -----------------------------------
# BMI + obesity metrics
# -----------------------------------
if "BMI_gold" in df.columns and "BMI_pred" in df.columns:
    df["BMI_gold_num"] = df["BMI_gold"].apply(to_float)
    df["BMI_pred_num"] = df["BMI_pred"].apply(to_float)

    bmi_comp = (~df["BMI_gold_num"].isna()) & (~df["BMI_pred_num"].isna())
    bmi_total = int(bmi_comp.sum())

    if bmi_total > 0:
        bmi_gold = df.loc[bmi_comp, "BMI_gold_num"]
        bmi_pred = df.loc[bmi_comp, "BMI_pred_num"]
        diff_abs = (bmi_pred - bmi_gold).abs()

        # Exact BMI
        bmi_exact_matches = int((diff_abs == 0).sum())
        bmi_exact_acc = float(bmi_exact_matches) / float(bmi_total)
        results.append({
            "variable": "BMI",
            "accuracy": round(bmi_exact_acc, 6),
            "matches": bmi_exact_matches,
            "total_compared": bmi_total
        })

        # Close BMI within ±0.5
        bmi_close_matches = int((diff_abs <= BMI_TOL).sum())
        bmi_close_acc = float(bmi_close_matches) / float(bmi_total)
        results.append({
            "variable": "BMI_close_0_5",
            "accuracy": round(bmi_close_acc, 6),
            "matches": bmi_close_matches,
            "total_compared": bmi_total
        })

        # Integer-rounded BMI match
        gold_round = bmi_gold.round(0)
        pred_round = bmi_pred.round(0)
        bmi_round_matches = int((gold_round == pred_round).sum())
        bmi_round_acc = float(bmi_round_matches) / float(bmi_total)
        results.append({
            "variable": "BMI_round_integer",
            "accuracy": round(bmi_round_acc, 6),
            "matches": bmi_round_matches,
            "total_compared": bmi_total
        })

        # Obesity from BMI exact threshold
        gold_ob = bmi_gold.apply(obesity_from_bmi)
        pred_ob = bmi_pred.apply(obesity_from_bmi)

        obesity_matches = int((gold_ob == pred_ob).sum())
        obesity_acc = float(obesity_matches) / float(bmi_total)
        results.append({
            "variable": "Obesity_from_BMI",
            "accuracy": round(obesity_acc, 6),
            "matches": obesity_matches,
            "total_compared": bmi_total
        })

        # Obesity from BMI with tolerance override
        obesity_tol_match = ((gold_ob == pred_ob) | (diff_abs <= BMI_TOL))
        obesity_tol_matches = int(obesity_tol_match.sum())
        obesity_tol_acc = float(obesity_tol_matches) / float(bmi_total)
        results.append({
            "variable": "Obesity_from_BMI_tol_0_5",
            "accuracy": round(obesity_tol_acc, 6),
            "matches": obesity_tol_matches,
            "total_compared": bmi_total
        })

# -----------------------------------
# Optional direct Obesity column comparison
# -----------------------------------
if "Obesity_gold" in df.columns and "Obesity_pred" in df.columns:
    acc, matches, total = compare_binary_var(df, "Obesity_gold", "Obesity_pred")
    results.append({
        "variable": "Obesity_column_direct",
        "accuracy": acc,
        "matches": matches,
        "total_compared": total
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
