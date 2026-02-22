# validate_full_cohort_against_gold.py
# Python 3.6.8 compatible

import pandas as pd
import numpy as np

PRED_FILE = "cohort_all_patient_level_final.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"
BRIDGE_FILE = "pred_with_mrn.csv"   # must contain patient_id + MRN

OUT_FILE = "validation_metrics_summary.csv"


# -------------------------
# Utility
# -------------------------

def read_csv_safe(path):
    try:
        return pd.read_csv(path, dtype=object, encoding="utf-8", engine="python")
    except:
        return pd.read_csv(path, dtype=object, encoding="latin1", engine="python")

def to_binary(series):
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        ["1","true","yes","y"]
    ).astype(int)

def compute_metrics(df, gold_col, pred_col):
    g = to_binary(df[gold_col])
    p = to_binary(df[pred_col])

    tp = int(((g == 1) & (p == 1)).sum())
    tn = int(((g == 0) & (p == 0)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "Gold_Var": gold_col,
        "Pred_Var": pred_col,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "PPV": round(ppv, 4),
        "NPV": round(npv, 4),
    }


# -------------------------
# Main
# -------------------------

print("\n=== VALIDATION: FULL COHORT VS GOLD ===")

pred = read_csv_safe(PRED_FILE)
gold = read_csv_safe(GOLD_FILE)
bridge = read_csv_safe(BRIDGE_FILE)

# Ensure join columns exist
assert "patient_id" in pred.columns
assert "MRN" in gold.columns
assert "patient_id" in bridge.columns and "MRN" in bridge.columns

# Link pred -> MRN
pred = pred.merge(bridge[["patient_id","MRN"]], on="patient_id", how="left")

# Merge with gold
df = gold.merge(pred, on="MRN", how="left", suffixes=("_gold","_pred"))

print("Gold rows:", gold.shape[0])
print("Overlap rows:", df.shape[0])

# -------------------------
# Variable Mapping
# -------------------------

VARIABLE_MAP = [
    ("Stage1 MinorComp", "Stage1_MinorComp_pred"),
    ("Stage1 MajorComp", "Stage1_MajorComp_pred"),
    ("Stage1 Reoperation", "Stage1_Reoperation_pred"),
    ("Stage1 Rehospitalization", "Stage1_Rehospitalization_pred"),

    ("Stage2 MinorComp", "Stage2_MinorComp"),
    ("Stage2 MajorComp", "Stage2_MajorComp"),
    ("Stage2 Reoperation", "Stage2_Reoperation"),
    ("Stage2 Rehospitalization", "Stage2_Rehospitalization"),
    ("Stage2 Failure", "Stage2_Failure"),
    ("Stage2 Revision", "Stage2_Revision"),
]

results = []

for gold_var, pred_var in VARIABLE_MAP:
    if gold_var not in df.columns:
        print("Skip (missing gold col):", gold_var)
        continue
    if pred_var not in df.columns:
        print("Skip (missing pred col):", pred_var)
        continue

    metrics = compute_metrics(df, gold_var, pred_var)
    results.append(metrics)

out = pd.DataFrame(results)
out.to_csv(OUT_FILE, index=False)

print("\nWrote:", OUT_FILE)
print("\nPreview:\n")
print(out)
print("\nDone.\n")
