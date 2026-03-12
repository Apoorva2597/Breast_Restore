#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
validate_abstraction.py

Validates abstraction output against gold labels using direct MRN merge.

Compares only rows where gold is non-missing.

Uses type-aware comparison for categorical, numeric, and binary fields.

Includes race normalization so builder-standardized race values can be
compared fairly against gold race labels.

UPDATE:
Added supplemental BMI / obesity validation metrics WITHOUT changing
existing finalized validation behavior:
- BMI_close_0_5
- BMI_close_1_0
- BMI_round_integer
- Obesity_from_BMI
- Obesity_from_BMI_tol_0_5

UPDATED PBS VALIDATION:
Added:
- PastBreastSurgery
- PBS_Lumpectomy
- PBS_Breast Reduction
- PBS_Mastopexy
- PBS_Augmentation
- PBS_Other

UPDATED CANCER / RECON VALIDATION:
Added:
- Mastectomy_Laterality
- Indication_Left
- Indication_Right
- LymphNode
- Radiation_Before
- Radiation_After
- Chemo_Before
- Chemo_After
- Recon_Laterality
- Recon_Type
- Recon_Classification
- Recon_Timing

UPDATED STAGE 2 VALIDATION:
Added:
- PRED_HAS_STAGE2
- Stage2_MinorComp
- Stage2_Reoperation
- Stage2_Rehospitalization
- Stage2_MajorComp
- Stage2_Failure
- Stage2_Revision

Reference only, not validated:
- STAGE2_DATE
- WINDOW_START
- WINDOW_END

Compatible with Python 3.6.8.
"""

import os
import sys
import pandas as pd

MASTER_FILE = "_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"

MRN = "MRN"

CATEGORICAL_VARS = [
    "Race",
    "Ethnicity",
    "SmokingStatus",
    "Mastectomy_Laterality",
    "Indication_Left",
    "Indication_Right",
    "LymphNode",
    "Recon_Laterality",
    "Recon_Type",
    "Recon_Classification",
    "Recon_Timing"
]

NUMERIC_VARS = [
    "Age",
    "BMI"
]

BINARY_VARS = [
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
    "PastBreastSurgery",
    "PBS_Lumpectomy",
    "PBS_Breast Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "PBS_Other",
    "Radiation",
    "Radiation_Before",
    "Radiation_After",
    "Chemo",
    "Chemo_Before",
    "Chemo_After",
    "PRED_HAS_STAGE2",
    "Stage2_MinorComp",
    "Stage2_Reoperation",
    "Stage2_Rehospitalization",
    "Stage2_MajorComp",
    "Stage2_Failure",
    "Stage2_Revision"
]

ALL_VARIABLES = CATEGORICAL_VARS + NUMERIC_VARS + BINARY_VARS


# ---------------------------------------------------
# Safe CSV reader
# ---------------------------------------------------

def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8", dtype=str)
    except Exception:
        return pd.read_csv(path, encoding="latin1", dtype=str)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def clean_string_series(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.copy()
    series = series.astype(str)
    series = series.str.strip()

    series = series.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "none": pd.NA,
        "NA": pd.NA,
        "na": pd.NA,
        "null": pd.NA,
        "Null": pd.NA
    })

    return series


def normalize_categorical(series):
    series = clean_string_series(series)
    series = series.astype("object")

    mask = series.notna()

    series.loc[mask] = (
        series.loc[mask]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return series


def normalize_binary(series):
    series = clean_string_series(series)

    def conv(x):
        if pd.isna(x):
            return pd.NA

        s = str(x).strip().lower()

        if s in ["1", "true", "t", "yes", "y"]:
            return 1

        if s in ["0", "false", "f", "no", "n"]:
            return 0

        return pd.NA

    return series.apply(conv)


def normalize_numeric(series):
    series = clean_string_series(series)
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------
# Race normalization
# ---------------------------------------------------

def normalize_race_token(token):
    s = str(token).strip().lower()

    if s in ["", "nan", "none", "null", "na"]:
        return ""

    if s in ["white", "white or caucasian", "caucasian"]:
        return "White"

    if s in ["black", "black or african american", "african american"]:
        return "Black or African American"

    if s in [
        "asian", "filipino", "other asian", "asian indian",
        "chinese", "japanese", "korean", "vietnamese"
    ]:
        return "Asian"

    if s == "american indian or alaska native":
        return "American Indian or Alaska Native"

    if s in [
        "native hawaiian",
        "pacific islander",
        "native hawaiian or other pacific islander"
    ]:
        return "Native Hawaiian or Other Pacific Islander"

    if s == "other":
        return "Other"

    if s in [
        "unknown",
        "declined",
        "refused",
        "patient refused",
        "choose not to disclose",
        "unknown / declined / not reported"
    ]:
        return "Unknown / Declined / Not Reported"

    return str(token).strip()


def collapse_race_value(x):
    if pd.isna(x):
        return pd.NA

    raw = str(x).strip()

    if raw == "":
        return pd.NA

    pieces = []
    tmp = raw.replace(";", ",")

    for part in tmp.split(","):
        p = part.strip()
        if p:
            pieces.append(p)

    if not pieces:
        return pd.NA

    real = []
    saw_unknown = False

    for p in pieces:
        norm = normalize_race_token(p)

        if not norm:
            continue

        if norm == "Unknown / Declined / Not Reported":
            saw_unknown = True
            continue

        if norm not in real:
            real.append(norm)

    if len(real) == 0:
        if saw_unknown:
            return "Unknown / Declined / Not Reported"
        return pd.NA

    if len(real) == 1:
        return real[0]

    return "Multiracial"


def normalize_race_series(series):
    series = clean_string_series(series)
    return series.apply(collapse_race_value)


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

def compute_categorical_metrics(pred, gold):
    pred = normalize_categorical(pred)
    gold = normalize_categorical(gold)

    mask = gold.notna()

    pred = pred[mask]
    gold = gold[mask]

    total = len(gold)

    if total == 0:
        return 0.0, 0, 0

    matches = (pred == gold).sum()
    accuracy = float(matches) / float(total)

    return accuracy, int(matches), int(total)


def compute_race_metrics(pred, gold):
    pred = normalize_race_series(pred)
    gold = normalize_race_series(gold)

    mask = gold.notna()

    pred = pred[mask]
    gold = gold[mask]

    total = len(gold)

    if total == 0:
        return 0.0, 0, 0

    matches = (pred == gold).sum()
    accuracy = float(matches) / float(total)

    return accuracy, int(matches), int(total)


def compute_binary_metrics(pred, gold):
    pred = normalize_binary(pred)
    gold = normalize_binary(gold)

    mask = gold.notna()

    pred = pred[mask]
    gold = gold[mask]

    total = len(gold)

    if total == 0:
        return 0.0, 0, 0

    matches = (pred == gold).sum()
    accuracy = float(matches) / float(total)

    return accuracy, int(matches), int(total)


def compute_numeric_metrics(pred, gold, tolerance=None):
    pred = normalize_numeric(pred)
    gold = normalize_numeric(gold)

    mask = gold.notna()

    pred = pred[mask]
    gold = gold[mask]

    total = len(gold)

    if total == 0:
        return 0.0, 0, 0

    if tolerance is None:
        matches = (pred == gold).sum()
    else:
        matches = ((pred - gold).abs() <= tolerance).sum()

    accuracy = float(matches) / float(total)

    return accuracy, int(matches), int(total)


def compute_age_floor_round_metrics(pred, gold):
    pred = normalize_numeric(pred)
    gold = normalize_numeric(gold)

    mask = gold.notna()

    pred = pred[mask]
    gold = gold[mask]

    total = len(gold)

    if total == 0:
        return 0.0, 0, 0

    matches = (
        (gold == pred) |
        (gold == pred - 1) |
        (gold == pred + 1)
    ).sum()

    accuracy = float(matches) / float(total)

    return accuracy, int(matches), int(total)


# ---------------------------------------------------
# BMI / Obesity supplemental metrics
# ---------------------------------------------------

def compute_bmi_round_integer_metrics(pred, gold):
    pred = normalize_numeric(pred)
    gold = normalize_numeric(gold)

    mask = gold.notna()

    pred = pred[mask]
    gold = gold[mask]

    total = len(gold)

    if total == 0:
        return 0.0, 0, 0

    matches = (pred.round(0) == gold.round(0)).sum()
    accuracy = float(matches) / float(total)

    return accuracy, int(matches), int(total)


def compute_obesity_from_bmi_metrics(pred, gold, tolerance=None):
    pred = normalize_numeric(pred)
    gold = normalize_numeric(gold)

    mask = gold.notna()

    pred = pred[mask]
    gold = gold[mask]

    total = len(gold)

    if total == 0:
        return 0.0, 0, 0

    gold_ob = (gold >= 30).astype(int)
    pred_ob = (pred >= 30).astype(int)

    if tolerance is None:
        matches = (pred_ob == gold_ob).sum()
    else:
        close_mask = (pred - gold).abs() <= tolerance
        matches = ((pred_ob == gold_ob) | close_mask).sum()

    accuracy = float(matches) / float(total)

    return accuracy, int(matches), int(total)


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    print("Loading files...")

    master = safe_read_csv(MASTER_FILE)
    gold = safe_read_csv(GOLD_FILE)

    print("Master rows:", len(master))
    print("Gold rows:", len(gold))

    if MRN not in master.columns:
        print("ERROR: master file missing MRN column")
        sys.exit(1)

    if MRN not in gold.columns:
        print("ERROR: gold file missing MRN column")
        sys.exit(1)

    master[MRN] = master[MRN].astype(str).str.strip()
    gold[MRN] = gold[MRN].astype(str).str.strip()

    master = master[master[MRN] != ""].copy()
    gold = gold[gold[MRN] != ""].copy()

    master = master.drop_duplicates(subset=[MRN])
    gold = gold.drop_duplicates(subset=[MRN])

    print("Merging directly on MRN...")

    merged = pd.merge(
        master,
        gold,
        on=MRN,
        how="inner",
        suffixes=("_pred", "_gold")
    )

    print("Merged rows:", len(merged))

    if len(merged) == 0:
        print("ERROR: No rows matched on MRN.")
        sys.exit(1)

    results = []

    for v in ALL_VARIABLES:
        pred_col = v + "_pred"
        gold_col = v + "_gold"

        if pred_col not in merged.columns or gold_col not in merged.columns:
            print("Skipping variable:", v)
            continue

        pred = merged[pred_col]
        goldv = merged[gold_col]

        if v == "Race":
            acc, matches, total = compute_race_metrics(pred, goldv)

        elif v in CATEGORICAL_VARS:
            acc, matches, total = compute_categorical_metrics(pred, goldv)

        elif v in BINARY_VARS:
            acc, matches, total = compute_binary_metrics(pred, goldv)

        elif v == "Age":
            acc, matches, total = compute_age_floor_round_metrics(pred, goldv)

        elif v == "BMI":
            acc, matches, total = compute_numeric_metrics(pred, goldv, tolerance=0.2)

        else:
            acc, matches, total = 0.0, 0, 0

        results.append({
            "variable": v,
            "accuracy": acc,
            "matches": matches,
            "total_compared": total
        })

    # -----------------------------------------------
    # Supplemental BMI / obesity validation
    # -----------------------------------------------

    if "BMI_pred" in merged.columns and "BMI_gold" in merged.columns:
        pred_bmi = merged["BMI_pred"]
        gold_bmi = merged["BMI_gold"]

        acc, matches, total = compute_numeric_metrics(pred_bmi, gold_bmi, tolerance=0.5)
        results.append({
            "variable": "BMI_close_0_5",
            "accuracy": acc,
            "matches": matches,
            "total_compared": total
        })

        acc, matches, total = compute_numeric_metrics(pred_bmi, gold_bmi, tolerance=1.0)
        results.append({
            "variable": "BMI_close_1_0",
            "accuracy": acc,
            "matches": matches,
            "total_compared": total
        })

        acc, matches, total = compute_bmi_round_integer_metrics(pred_bmi, gold_bmi)
        results.append({
            "variable": "BMI_round_integer",
            "accuracy": acc,
            "matches": matches,
            "total_compared": total
        })

        acc, matches, total = compute_obesity_from_bmi_metrics(pred_bmi, gold_bmi, tolerance=None)
        results.append({
            "variable": "Obesity_from_BMI",
            "accuracy": acc,
            "matches": matches,
            "total_compared": total
        })

        acc, matches, total = compute_obesity_from_bmi_metrics(pred_bmi, gold_bmi, tolerance=0.5)
        results.append({
            "variable": "Obesity_from_BMI_tol_0_5",
            "accuracy": acc,
            "matches": matches,
            "total_compared": total
        })

    df = pd.DataFrame(results)

    print("\nValidation Results\n")
    print(df)

    if not os.path.exists("_outputs"):
        os.makedirs("_outputs")

    out_path = "_outputs/validation_summary.csv"
    df.to_csv(out_path, index=False)

    print("\nValidation complete.")
    print("Results saved to", out_path)


if __name__ == "__main__":
    main()
