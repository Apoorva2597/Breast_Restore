#!/usr/bin/env python3
# validate_stage2_anchor_FIXED.py
# Updated: use MRN (or PatientID) when ENCRYPTED_PAT_ID is absent.

import os
import pandas as pd

# =========================
# CONFIG (EDIT PATHS)
# =========================
GOLD_PATH      = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
STAGE_PRED_PATH= "/home/apokol/Breast_Restore/_outputs/patient_stage_summary.csv"
# optional, only used for sanity checks / future extension
OP_NOTES_PATH  = "/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Operation Notes.csv"

# Column in GOLD that indicates true Stage2 (edit if your gold uses a different name)
GOLD_STAGE2_COL_CANDIDATES = ["HAS_STAGE2", "Stage2", "STAGE2", "STAGE2_ANCHOR", "Stage2_Applicable"]

# Column in STAGE_PRED that indicates predicted Stage2 (edit if needed)
PRED_STAGE2_COL_CANDIDATES = ["HAS_STAGE2", "Stage2", "STAGE2", "PRED_STAGE2"]

# Preferred patient key order
KEY_CANDIDATES = ["ENCRYPTED_PAT_ID", "MRN", "PatientID", "PATIENT_ID", "PAT_ID"]

# =========================
# IO helpers
# =========================
def read_csv_robust(path: str) -> pd.DataFrame:
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        # older pandas
        return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def find_first_existing_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def normalize_key(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    df[key_col] = df[key_col].astype(str).str.strip()
    return df

def to_binary(series: pd.Series) -> pd.Series:
    # Accept: 1/0, True/False, yes/no, Y/N, etc.
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"]).astype(int)

# =========================
# Main
# =========================
def main():
    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f"Missing GOLD_PATH: {GOLD_PATH}")
    if not os.path.exists(STAGE_PRED_PATH):
        raise FileNotFoundError(f"Missing STAGE_PRED_PATH: {STAGE_PRED_PATH}")

    print("Loading gold...")
    gold = clean_cols(read_csv_robust(GOLD_PATH))

    print("Loading stage predictions...")
    pred = clean_cols(read_csv_robust(STAGE_PRED_PATH))

    # pick join key
    gold_key = find_first_existing_col(gold, KEY_CANDIDATES)
    pred_key = find_first_existing_col(pred, KEY_CANDIDATES)

    if not gold_key:
        raise ValueError(f"Gold missing a usable key. Found columns: {list(gold.columns)}")
    if not pred_key:
        raise ValueError(f"Stage prediction file missing a usable key. Found columns: {list(pred.columns)}")

    # If keys differ but both exist, align to a single name
    JOIN_KEY = gold_key
    if pred_key != JOIN_KEY:
        pred = pred.rename(columns={pred_key: JOIN_KEY})

    gold = normalize_key(gold, JOIN_KEY)
    pred = normalize_key(pred, JOIN_KEY)

    # pick label cols
    gold_y_col = find_first_existing_col(gold, GOLD_STAGE2_COL_CANDIDATES)
    pred_y_col = find_first_existing_col(pred, PRED_STAGE2_COL_CANDIDATES)

    if not gold_y_col:
        raise ValueError(
            f"Gold missing Stage2 truth column. Tried {GOLD_STAGE2_COL_CANDIDATES}. "
            f"Found columns: {list(gold.columns)}"
        )
    if not pred_y_col:
        raise ValueError(
            f"Pred missing Stage2 prediction column. Tried {PRED_STAGE2_COL_CANDIDATES}. "
            f"Found columns: {list(pred.columns)}"
        )

    # merge
    merged = gold[[JOIN_KEY, gold_y_col]].merge(
        pred[[JOIN_KEY, pred_y_col]],
        on=JOIN_KEY,
        how="left",
        suffixes=("_GOLD", "_PRED"),
    )

    # convert to binary
    y_true = to_binary(merged[gold_y_col])
    y_pred = to_binary(merged[pred_y_col].fillna("0"))

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("\nValidation complete.")
    print("Stage2 Anchor:")
    print(f"  TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"  Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

if __name__ == "__main__":
    main()
