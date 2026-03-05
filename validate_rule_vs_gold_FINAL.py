#!/usr/bin/env python3
# validate_rule_vs_gold_FINAL.py
# Validates rule-based master output vs gold labels (where gold available).
# Outputs:
#   _outputs/rule_vs_gold_validation_report_FINAL.csv
#   _outputs/rule_vs_gold_confusions_FINAL.csv
#   _outputs/rule_vs_gold_mismatches_FINAL.csv

import os
import math
import pandas as pd

# ==============================
# FIXED PATHS (no user input)
# ==============================
BASE = "/home/apokol/Breast_Restore"
GOLD_PATH = os.path.join(BASE, "gold_cleaned_for_cedar.csv")
PRED_PATH = os.path.join(BASE, "_outputs", "master_abstraction_rule_FINAL.csv")

OUT_REPORT = os.path.join(BASE, "_outputs", "rule_vs_gold_validation_report_FINAL.csv")
OUT_CONF   = os.path.join(BASE, "_outputs", "rule_vs_gold_confusions_FINAL.csv")
OUT_MIS    = os.path.join(BASE, "_outputs", "rule_vs_gold_mismatches_FINAL.csv")

MERGE_KEY = "MRN"

# ==============================
# CONFIG: variables + types
# ==============================
# type:
#   - "binary"        : expects 0/1/True/False/Yes/No
#   - "categorical"   : string categories (case-insensitive compare)
#   - "numeric"       : numeric; reports MAE and within-tolerance rates
#
# NOTE: these names match your gold header screenshot.
VARS = [
    # numeric
    ("Age", "numeric", {"tol": 2.0}),      # tolerate +/-2 years (age-at-encounter vs age-at-surgery mismatch)
    ("BMI", "numeric", {"tol": 1.0}),      # tolerate +/-1.0 BMI

    # categorical
    ("Race", "categorical", {}),
    ("Ethnicity", "categorical", {}),
    ("SmokingStatus", "categorical", {}),

    # binary comorbidities + derived
    ("Diabetes", "binary", {}),
    ("Obesity", "binary", {}),
    ("Hypertension", "binary", {}),
    ("CardiacDisease", "binary", {}),
    ("VenousThromboembolism", "binary", {}),
    ("Steroid", "binary", {}),
    ("PastBreastSurgery", "binary", {}),

    # PBS subtypes
    ("PBS_Lumpectomy", "binary", {}),
    ("PBS_Breast Reduction", "binary", {}),
    ("PBS_Mastopexy", "binary", {}),
    ("PBS_Augmentation", "binary", {}),
    ("PBS_Other", "binary", {}),

    # laterality / treatment flags (often categorical or binary depending on your encoding)
    ("Mastectomy_Laterality", "categorical", {}),
    ("Indication_Left", "categorical", {}),
    ("Indication_Right", "categorical", {}),
    ("LymphNode", "categorical", {}),

    ("Radiation", "binary", {}),
    ("Radiation_Before", "binary", {}),
    ("Radiation_After", "binary", {}),
    ("Chemo", "binary", {}),
    ("Chemo_Before", "binary", {}),
    ("Chemo_After", "binary", {}),

    ("Recon_Laterality", "categorical", {}),
    ("Recon_Type", "categorical", {}),
    ("Recon_Classification", "categorical", {}),
    ("Recon_Timing", "categorical", {}),

    # stage outcomes (binary)
    ("Stage1_MinorComp", "binary", {}),
    ("Stage1_Reoperation", "binary", {}),
    ("Stage1_Rehospitalization", "binary", {}),
    ("Stage1_MajorComp", "binary", {}),
    ("Stage1_Failure", "binary", {}),
    ("Stage1_Revision", "binary", {}),

    ("Stage2_MinorComp", "binary", {}),
    ("Stage2_Reoperation", "binary", {}),
    ("Stage2_Rehospitalization", "binary", {}),
    ("Stage2_MajorComp", "binary", {}),
    ("Stage2_Failure", "binary", {}),
    ("Stage2_Revision", "binary", {}),
    ("Stage2_Applicable", "binary", {}),
]

# ==============================
# Helpers
# ==============================
def read_csv_robust(path: str) -> pd.DataFrame:
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def normalize_mrn(df: pd.DataFrame) -> pd.DataFrame:
    if MERGE_KEY not in df.columns:
        # try common variants
        for k in ["mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]:
            if k in df.columns:
                df = df.rename(columns={k: MERGE_KEY})
                break
    if MERGE_KEY not in df.columns:
        raise RuntimeError(f"MRN column not found in file. Columns: {list(df.columns)[:40]}")
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df

def to_binary(x):
    """Return 0/1 or None if missing/unknown."""
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return None
    if s in {"1", "true", "t", "yes", "y", "pos", "positive"}:
        return 1
    if s in {"0", "false", "f", "no", "n", "neg", "negative"}:
        return 0
    # some sheets use "performed"/"history" as yes-ish
    if s in {"performed", "history", "present"}:
        return 1
    if s in {"denied", "absent"}:
        return 0
    return None

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def norm_cat(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    # normalize whitespace + case
    s = " ".join(s.split())
    return s.lower()

def safe_div(a, b):
    return (a / b) if b else None

def f1(p, r):
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)

# ==============================
# Main
# ==============================
print("Loading gold + predictions...")
gold = normalize_mrn(clean_cols(read_csv_robust(GOLD_PATH)))
pred = normalize_mrn(clean_cols(read_csv_robust(PRED_PATH)))

# Merge (keep all gold rows)
df = gold.merge(pred, on=MERGE_KEY, how="left", suffixes=("_gold", "_pred"))

report_rows = []
conf_rows = []
mismatch_rows = []

# Suspicious similarity check (global)
shared_cols = set(gold.columns).intersection(set(pred.columns))
shared_cols.discard(MERGE_KEY)
if shared_cols:
    identical_counts = 0
    for c in list(shared_cols)[:50]:
        g = gold[c].astype(str).fillna("").str.strip()
        p = pred[c].astype(str).fillna("").str.strip()
        if (g == p).all():
            identical_counts += 1
    if identical_counts >= 5:
        print("WARNING: Many shared columns are identical between gold and master output.")
        print("         If master was initialized from gold, validation can be artificially inflated.")

for var, vtype, cfg in VARS:
    gcol = f"{var}_gold" if f"{var}_gold" in df.columns else var
    pcol = f"{var}_pred" if f"{var}_pred" in df.columns else var

    if gcol not in df.columns:
        report_rows.append({
            "variable": var, "type": vtype, "status": "SKIP (gold missing column)",
            "n_gold_nonmissing": 0
        })
        continue
    if pcol not in df.columns:
        report_rows.append({
            "variable": var, "type": vtype, "status": "SKIP (pred missing column)",
            "n_gold_nonmissing": int(df[gcol].notna().sum())
        })
        continue

    g_raw = df[gcol]
    p_raw = df[pcol]

    if vtype == "binary":
        g = g_raw.map(to_binary)
        p = p_raw.map(to_binary)

        mask = g.notna()  # only where gold exists
        g2 = g[mask]
        p2 = p[mask]

        # treat missing predictions as 0? NO — keep missing to avoid unfair TP/FP.
        # We'll compute metrics only where pred is non-missing, and also report coverage.
        mask2 = p2.notna()
        g3 = g2[mask2]
        p3 = p2[mask2]

        TP = int(((g3 == 1) & (p3 == 1)).sum())
        TN = int(((g3 == 0) & (p3 == 0)).sum())
        FP = int(((g3 == 0) & (p3 == 1)).sum())
        FN = int(((g3 == 1) & (p3 == 0)).sum())

        precision = safe_div(TP, TP + FP)
        recall    = safe_div(TP, TP + FN)
        f1score   = f1(precision, recall)
        acc       = safe_div(TP + TN, TP + TN + FP + FN)

        coverage = safe_div(int(mask2.sum()), int(mask.sum()))  # how often you produced a pred where gold exists

        report_rows.append({
            "variable": var,
            "type": "binary",
            "status": "OK",
            "n_gold_nonmissing": int(mask.sum()),
            "n_compared": int(mask2.sum()),
            "coverage_pred_given_gold": coverage,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1score,
        })

        conf_rows.append({
            "variable": var,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "n_gold_nonmissing": int(mask.sum()),
            "n_compared": int(mask2.sum()),
        })

        # sample mismatches
        mism = df.loc[mask].copy()
        mism["gold_bin"] = g
        mism["pred_bin"] = p
        mism2 = mism.loc[mism["pred_bin"].notna() & (mism["gold_bin"] != mism["pred_bin"])]
        for _, r in mism2.head(20).iterrows():
            mismatch_rows.append({
                "variable": var,
                "MRN": r[MERGE_KEY],
                "gold": r[gcol],
                "pred": r[pcol],
            })

    elif vtype == "categorical":
        g = g_raw.map(norm_cat)
        p = p_raw.map(norm_cat)

        mask = g.notna()
        g2 = g[mask]
        p2 = p[mask]

        mask2 = p2.notna()
        g3 = g2[mask2]
        p3 = p2[mask2]

        correct = int((g3 == p3).sum())
        total   = int(mask2.sum())
        acc     = safe_div(correct, total)
        coverage = safe_div(total, int(mask.sum()))

        report_rows.append({
            "variable": var,
            "type": "categorical",
            "status": "OK",
            "n_gold_nonmissing": int(mask.sum()),
            "n_compared": total,
            "coverage_pred_given_gold": coverage,
            "accuracy": acc,
        })

        mism2 = df.loc[mask].copy()
        mism2["gold_cat"] = g
        mism2["pred_cat"] = p
        mism3 = mism2.loc[mism2["pred_cat"].notna() & (mism2["gold_cat"] != mism2["pred_cat"])]
        for _, r in mism3.head(30).iterrows():
            mismatch_rows.append({
                "variable": var,
                "MRN": r[MERGE_KEY],
                "gold": r[gcol],
                "pred": r[pcol],
            })

    elif vtype == "numeric":
        tol = float(cfg.get("tol", 0.0))

        g = g_raw.map(to_float)
        p = p_raw.map(to_float)

        mask = g.notna()
        g2 = g[mask]
        p2 = p[mask]

        mask2 = p2.notna()
        g3 = g2[mask2]
        p3 = p2[mask2]

        if len(g3) == 0:
            report_rows.append({
                "variable": var,
                "type": "numeric",
                "status": "OK (no comparable rows)",
                "n_gold_nonmissing": int(mask.sum()),
                "n_compared": 0,
                "coverage_pred_given_gold": safe_div(0, int(mask.sum())),
            })
            continue

        abs_err = (p3 - g3).abs()
        mae = float(abs_err.mean())
        within = float((abs_err <= tol).mean()) if tol > 0 else None

        report_rows.append({
            "variable": var,
            "type": "numeric",
            "status": "OK",
            "n_gold_nonmissing": int(mask.sum()),
            "n_compared": int(mask2.sum()),
            "coverage_pred_given_gold": safe_div(int(mask2.sum()), int(mask.sum())),
            "mae": mae,
            f"pct_within_{tol}": within,
        })

        # sample mismatches (largest errors)
        tmp = df.loc[mask].copy()
        tmp["gold_num"] = g
        tmp["pred_num"] = p
        tmp = tmp.loc[tmp["pred_num"].notna()]
        tmp["abs_err"] = (tmp["pred_num"] - tmp["gold_num"]).abs()
        tmp = tmp.sort_values("abs_err", ascending=False).head(20)
        for _, r in tmp.iterrows():
            mismatch_rows.append({
                "variable": var,
                "MRN": r[MERGE_KEY],
                "gold": r[gcol],
                "pred": r[pcol],
                "abs_err": r["abs_err"],
            })

    else:
        report_rows.append({"variable": var, "type": vtype, "status": "SKIP (unknown type)"})

# Write outputs
os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)

pd.DataFrame(report_rows).to_csv(OUT_REPORT, index=False)
pd.DataFrame(conf_rows).to_csv(OUT_CONF, index=False)
pd.DataFrame(mismatch_rows).to_csv(OUT_MIS, index=False)

print("\nDONE.")
print(f"- Report:     {OUT_REPORT}")
print(f"- Confusions: {OUT_CONF}")
print(f"- Mismatches: {OUT_MIS}")
print("\nRun:")
print("  python validate_rule_vs_gold_FINAL.py")
