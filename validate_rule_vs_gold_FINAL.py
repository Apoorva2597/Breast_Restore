#!/usr/bin/env python3
# validate_rule_vs_gold_FINAL.py
#
# Validates RULE-based predictions vs GOLD.
# Fixes pandas compatibility (no on_bad_lines dependency).

import os
import pandas as pd
from typing import Dict, Tuple

BASE_DIR = "/home/apokol/Breast_Restore"
GOLD_PATH = f"{BASE_DIR}/gold_cleaned_for_cedar.csv"
PRED_PATH = f"{BASE_DIR}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"

OUT_REPORT = f"{BASE_DIR}/_outputs/rule_vs_gold_validation_report_FINAL.csv"
OUT_MISMATCH = f"{BASE_DIR}/_outputs/rule_vs_gold_mismatches_FINAL.csv"

KEY = "MRN"

# Validate these (add/remove as needed)
BINARY_VARS = [
    "Diabetes","Obesity","Hypertension","CardiacDisease","VenousThromboembolism","Steroid",
    "PastBreastSurgery","PBS_Lumpectomy","PBS_Breast Reduction","PBS_Mastopexy","PBS_Augmentation","PBS_Other",
    "Radiation","Chemo"
]

CATEGORICAL_VARS = [
    "SmokingStatus","Mastectomy_Laterality"
]

NUMERIC_VARS = [
    "Age","BMI"
]

def read_csv_robust(path: str) -> pd.DataFrame:
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except TypeError:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            try:
                return pd.read_csv(path, **common_kwargs, encoding="latin-1", error_bad_lines=False, warn_bad_lines=True)
            except TypeError:
                return pd.read_csv(path, **common_kwargs, encoding="latin-1", error_bad_lines=False)

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def norm_mrn(df: pd.DataFrame) -> pd.DataFrame:
    if KEY not in df.columns:
        # try common variants
        for k in ["mrn","Patient_MRN","PAT_MRN","PATIENT_MRN"]:
            if k in df.columns:
                df = df.rename(columns={k: KEY})
                break
    if KEY not in df.columns:
        raise RuntimeError(f"MRN column not found. Columns: {list(df.columns)[:50]}")
    df[KEY] = df[KEY].astype(str).str.strip()
    return df

def to01(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    if s in {"1","true","yes","y"}:
        return 1
    if s in {"0","false","no","n"}:
        return 0
    return pd.NA

def safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0

def confusion(y_true: pd.Series, y_pred: pd.Series) -> Tuple[int,int,int,int,int]:
    mask = (~y_true.isna()) & (~y_pred.isna())
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    tp = int(((yt==1) & (yp==1)).sum())
    fp = int(((yt==0) & (yp==1)).sum())
    fn = int(((yt==1) & (yp==0)).sum())
    tn = int(((yt==0) & (yp==0)).sum())
    n = int(mask.sum())
    return tp, fp, fn, tn, n

def main():
    print("Loading gold + predictions...")
    gold = norm_mrn(clean_cols(read_csv_robust(GOLD_PATH)))
    pred = norm_mrn(clean_cols(read_csv_robust(PRED_PATH)))

    merged = gold.merge(pred, on=KEY, how="inner", suffixes=("_gold","_pred"))
    print(f"Joined rows: {len(merged)} (inner join on MRN)")

    rows = []
    mismatch_rows = []

    # ---------- Numeric ----------
    for v in NUMERIC_VARS:
        gcol = v if v in gold.columns else None
        pcol = v if v in pred.columns else None
        if not gcol or not pcol:
            continue

        g = pd.to_numeric(merged[gcol], errors="coerce")
        p = pd.to_numeric(merged[pcol], errors="coerce")

        mask = (~g.isna()) & (~p.isna())
        n = int(mask.sum())
        if n == 0:
            continue

        abs_err = (g[mask] - p[mask]).abs()
        mae = float(abs_err.mean())
        pct_within_1 = float((abs_err <= 1.0).mean())
        pct_within_2 = float((abs_err <= 2.0).mean())

        rows.append({
            "variable": v,
            "type": "numeric",
            "n_compared": n,
            "mae": mae,
            "pct_within_1.0": pct_within_1,
            "pct_within_2.0": pct_within_2,
        })

    # ---------- Binary ----------
    for v in BINARY_VARS:
        if v not in gold.columns or v not in pred.columns:
            continue
        y_true = merged[v].apply(to01)
        y_pred = merged[v].apply(to01)

        tp, fp, fn, tn, n = confusion(y_true, y_pred)
        acc = safe_div(tp+tn, n)
        prec = safe_div(tp, tp+fp)
        rec = safe_div(tp, tp+fn)
        f1 = safe_div(2*prec*rec, prec+rec) if (prec+rec) else 0.0

        rows.append({
            "variable": v,
            "type": "binary",
            "n_compared": n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        })

        # mismatches
        mm = (~y_true.isna()) & (~y_pred.isna()) & (y_true != y_pred)
        for _, r in merged.loc[mm, [KEY, v]].iterrows():
            mismatch_rows.append({
                "MRN": r[KEY],
                "variable": v,
                "gold": r[v],
                "pred": merged.loc[merged[KEY]==r[KEY], v].iloc[0],
            })

    # ---------- Categorical ----------
    for v in CATEGORICAL_VARS:
        if v not in gold.columns or v not in pred.columns:
            continue
        g = merged[v].astype(str).str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA})
        p = merged[v].astype(str).str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA})
        mask = (~g.isna()) & (~p.isna())
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float((g[mask] == p[mask]).mean())
        rows.append({
            "variable": v,
            "type": "categorical",
            "n_compared": n,
            "accuracy": acc
        })

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_REPORT, index=False)
    pd.DataFrame(mismatch_rows).to_csv(OUT_MISMATCH, index=False)

    print("DONE.")
    print(f"- Report:     {OUT_REPORT}")
    print(f"- Mismatches: {OUT_MISMATCH}")
    print("Run:")
    print("  python validate_rule_vs_gold_FINAL.py")

if __name__ == "__main__":
    main()
