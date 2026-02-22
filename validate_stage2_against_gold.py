# validate_stage2_full_cohort_against_gold.py
# Python 3.6.8 compatible
#
# Purpose:
#   Validate Stage2 outcomes in the FULL cohort output against GOLD, using MRN as the join key.
#   FULL cohort is keyed by patient_id, so we bridge patient_id -> MRN using a cohort-wide mapping
#   built from encounter files (e.g., cohort_pid_to_mrn_from_encounters.csv).
#
# Inputs (edit in CONFIG):
#   - GOLD: gold_cleaned_for_cedar.csv   (MRN + Stage2 outcomes + Stage2_Applicable)
#   - COHORT: cohort_all_patient_level_final_gold_order.csv  (patient_id + predicted Stage2 outcomes)
#   - BRIDGE: cohort_pid_to_mrn_from_encounters.csv          (patient_id + MRN)
#
# Output:
#   - stage2_validation_confusion_by_var.csv
#   - stage2_validation_pairwise_rows.csv (optional row-level audit)
#
# Notes:
#   - Does NOT convert blanks to 0. Missing stays missing.
#   - Only scores on rows where GOLD Stage2_Applicable == 1 AND both gold & pred are scorable (0/1).
#   - Robust to GOLD column names containing spaces (e.g., "Stage2 MinorComp").
#
from __future__ import print_function
import os
import re
import pandas as pd

# -------------------------
# CONFIG: EDIT PATHS
# -------------------------
GOLD_CSV   = "gold_cleaned_for_cedar.csv"
COHORT_CSV = "cohort_all_patient_level_final_gold_order.csv"
BRIDGE_CSV = "cohort_pid_to_mrn_from_encounters.csv"

OUT_CONFUSION = "stage2_validation_confusion_by_var.csv"
OUT_PAIRWISE  = "stage2_validation_pairwise_rows.csv"  # row-level audit; can be large

# If GOLD uses a helper column to indicate Stage2 applicability:
GOLD_STAGE2_APPLICABLE_COL_CANDIDATES = ["Stage2_Applicable", "Stage2 Applicable", "stage2_applicable"]

# Map GOLD column names -> COHORT column names
STAGE2_VAR_MAP = {
    "Stage2 MinorComp":         "Stage2_MinorComp",
    "Stage2 MajorComp":         "Stage2_MajorComp",
    "Stage2 Reoperation":       "Stage2_Reoperation",
    "Stage2 Rehospitalization": "Stage2_Rehospitalization",
    "Stage2 Failure":           "Stage2_Failure",
    "Stage2 Revision":          "Stage2_Revision",
}

# Acceptable binary representations (strings)
TRUE_TOKENS  = set(["1", "true", "t", "yes", "y"])
FALSE_TOKENS = set(["0", "false", "f", "no", "n"])

# -------------------------
# Helpers
# -------------------------
def read_csv_safe(path):
    # Epic exports + mixed encodings: try utf-8 then latin1/cp1252-like fallback
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1", errors="replace")

def norm_str(x):
    if x is None:
        return ""
    s = str(x)
    try:
        s = s.replace("\xa0", " ")
    except Exception:
        pass
    s = s.strip()
    if s.lower() in ("", "nan", "none", "null"):
        return ""
    return s

def pick_col(cols, candidates):
    cols_u = {c: str(c).strip().upper() for c in cols}
    # exact match
    for want in candidates:
        w = str(want).strip().upper()
        for c in cols:
            if cols_u[c] == w:
                return c
    # contains match
    for want in candidates:
        w = str(want).strip().upper()
        for c in cols:
            if w in cols_u[c]:
                return c
    return None

def to_binary_or_missing(x):
    """
    Returns:
      1 or 0 if x can be interpreted as binary,
      None if missing / not interpretable.
    """
    s = norm_str(x)
    if s == "":
        return None
    sl = s.lower()
    if sl in TRUE_TOKENS:
        return 1
    if sl in FALSE_TOKENS:
        return 0
    # Sometimes columns come in as numeric-ish strings
    if re.match(r"^\d+(\.0+)?$", s):
        # "1", "0", "1.0", "0.0"
        try:
            v = int(float(s))
            if v in (0, 1):
                return v
        except Exception:
            pass
    return None

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def confusion_counts(gold_bin, pred_bin):
    """
    gold_bin, pred_bin are arrays/series of 0/1 integers
    """
    tp = int(((gold_bin == 1) & (pred_bin == 1)).sum())
    fp = int(((gold_bin == 0) & (pred_bin == 1)).sum())
    fn = int(((gold_bin == 1) & (pred_bin == 0)).sum())
    tn = int(((gold_bin == 0) & (pred_bin == 0)).sum())
    return tp, fp, fn, tn

def safe_div(n, d):
    return float(n) / float(d) if d else ""

# -------------------------
# Main
# -------------------------
print("\n=== VALIDATION: STAGE2 (FULL COHORT vs GOLD via bridge) ===")
print("GOLD  :", GOLD_CSV)
print("COHORT:", COHORT_CSV)
print("BRIDGE:", BRIDGE_CSV)

# Load files
gold = read_csv_safe(GOLD_CSV)
cohort = read_csv_safe(COHORT_CSV)
bridge = read_csv_safe(BRIDGE_CSV)

# Detect join columns
gold_mrn_col = pick_col(gold.columns, ["MRN"])
bridge_pid_col = pick_col(bridge.columns, ["patient_id", "PATIENT_ID", "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID"])
bridge_mrn_col = pick_col(bridge.columns, ["MRN"])
cohort_pid_col = pick_col(cohort.columns, ["patient_id", "PATIENT_ID", "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID"])

if not gold_mrn_col:
    raise RuntimeError("Could not find MRN column in GOLD.")
if not (bridge_pid_col and bridge_mrn_col):
    raise RuntimeError("Could not find (patient_id, MRN) columns in BRIDGE.")
if not cohort_pid_col:
    raise RuntimeError("Could not find patient_id column in COHORT.")

# Normalize join keys
gold["_MRN_"] = gold[gold_mrn_col].map(norm_str)
bridge["_PID_"] = bridge[bridge_pid_col].map(norm_str)
bridge["_MRN_"] = bridge[bridge_mrn_col].map(norm_str)
cohort["_PID_"] = cohort[cohort_pid_col].map(norm_str)

# Basic sanity on bridge uniqueness (should be 1-1)
amb_pid = bridge[bridge["_PID_"] != ""].groupby("_PID_")["_MRN_"].nunique()
amb_mrn = bridge[bridge["_MRN_"] != ""].groupby("_MRN_")["_PID_"].nunique()
n_pid_multi_mrn = int((amb_pid > 1).sum())
n_mrn_multi_pid = int((amb_mrn > 1).sum())

print("\nDetected columns:")
print("  GOLD MRN col   :", gold_mrn_col)
print("  BRIDGE pid col :", bridge_pid_col)
print("  BRIDGE MRN col :", bridge_mrn_col)
print("  COHORT pid col :", cohort_pid_col)

print("\nBridge ambiguity checks:")
print("  patient_id -> multiple MRNs:", n_pid_multi_mrn)
print("  MRN -> multiple patient_ids:", n_mrn_multi_pid)

if n_pid_multi_mrn or n_mrn_multi_pid:
    print("\nWARNING: Bridge mapping is not strictly 1-1. Proceeding, but validation may be impacted.")

# Attach MRN to cohort via bridge
bridge_slim = bridge[["_PID_", "_MRN_"]].drop_duplicates()
cohort2 = cohort.merge(bridge_slim, on="_PID_", how="left", suffixes=("", "_bridge"))

# Determine Stage2_Applicable in GOLD
gold_app_col = pick_col(gold.columns, GOLD_STAGE2_APPLICABLE_COL_CANDIDATES)
if not gold_app_col:
    raise RuntimeError("Could not find Stage2_Applicable column in GOLD (tried {}).".format(GOLD_STAGE2_APPLICABLE_COL_CANDIDATES))

gold["_Stage2_Applicable_"] = gold[gold_app_col].map(to_binary_or_missing)

# Build GOLD subset for scoring: only Stage2_Applicable == 1
gold_scoring = gold[(gold["_MRN_"] != "") & (gold["_Stage2_Applicable_"] == 1)].copy()

# Join GOLD scoring set to cohort2 via MRN
joined = gold_scoring.merge(
    cohort2,
    on="_MRN_",
    how="left",
    suffixes=("_gold", "_pred")
)

# Counts summary
n_gold_rows = int(len(gold))
n_gold_stage2_app = int((gold["_Stage2_Applicable_"] == 1).sum())
n_gold_mrn_nonblank = int((gold["_MRN_"] != "").sum())
n_cohort_rows = int(len(cohort2))
n_cohort_mrn_linked = int((cohort2["_MRN_"].map(norm_str) != "").sum())
overlap_mrn = int(set(gold["_MRN_"].tolist()) & set(cohort2["_MRN_"].tolist()) != set())

print("\nCounts:")
print("  GOLD rows:", n_gold_rows)
print("  GOLD MRN nonblank:", n_gold_mrn_nonblank)
print("  GOLD Stage2_Applicable==1:", n_gold_stage2_app)
print("  COHORT rows:", n_cohort_rows)
print("  COHORT rows w/ MRN linked:", n_cohort_mrn_linked)
print("  Joined rows for Stage2 scoring:", int(len(joined)))

# Validate each Stage2 variable
results = []
pair_rows = []

for gold_col, pred_col in STAGE2_VAR_MAP.items():
    gold_present = 1 if gold_col in joined.columns else 0
    pred_present = 1 if pred_col in joined.columns else 0

    row = {
        "var": pred_col,
        "gold_col": gold_col,
        "pred_col": pred_col,
        "status": "",
        "gold_col_present": gold_present,
        "pred_col_present": pred_present,
        "n_applicable_overlap": int(len(joined)),
        "n_scorable": 0,
        "gold_missing": "",
        "pred_missing": "",
        "TP": "",
        "FP": "",
        "FN": "",
        "TN": "",
        "sensitivity": "",
        "specificity": "",
        "ppv": "",
        "npv": "",
    }

    if not gold_present or not pred_present:
        row["status"] = "SKIP_missing_column"
        results.append(row)
        continue

    # Convert to binary/missing WITHOUT coercing blanks to 0
    gbin = joined[gold_col].map(to_binary_or_missing)
    pbin = joined[pred_col].map(to_binary_or_missing)

    # missingness counts (within applicable set)
    g_miss = int(gbin.isna().sum())
    p_miss = int(pbin.isna().sum())

    # scorable rows: both present as 0/1
    scorable_mask = (~gbin.isna()) & (~pbin.isna())
    g2 = gbin[scorable_mask].astype(int)
    p2 = pbin[scorable_mask].astype(int)

    n_scorable = int(len(g2))
    row["n_scorable"] = n_scorable
    row["gold_missing"] = g_miss
    row["pred_missing"] = p_miss

    if n_scorable == 0:
        row["status"] = "SKIP_no_scorable_rows"
        results.append(row)
        continue

    tp, fp, fn, tn = confusion_counts(g2, p2)

    row["status"] = "OK"
    row["TP"] = tp
    row["FP"] = fp
    row["FN"] = fn
    row["TN"] = tn

    row["sensitivity"] = safe_div(tp, tp + fn)
    row["specificity"] = safe_div(tn, tn + fp)
    row["ppv"] = safe_div(tp, tp + fp)
    row["npv"] = safe_div(tn, tn + fn)

    results.append(row)

    # Optional row-level audit output
    # Keep only scorable rows for compactness
    tmp = joined.loc[scorable_mask, ["_MRN_", "_PID_", gold_col, pred_col]].copy()
    tmp = tmp.rename(columns={
        "_MRN_": "MRN",
        "_PID_": "patient_id",
        gold_col: gold_col + "_gold",
        pred_col: pred_col + "_pred",
    })
    tmp["var"] = pred_col
    pair_rows.append(tmp)

# Write outputs
out_conf = pd.DataFrame(results)
out_conf.to_csv(OUT_CONFUSION, index=False, encoding="utf-8")
print("\nWrote:", OUT_CONFUSION)

if pair_rows:
    out_pairs = pd.concat(pair_rows, axis=0, ignore_index=True)
    out_pairs.to_csv(OUT_PAIRWISE, index=False, encoding="utf-8")
    print("Wrote:", OUT_PAIRWISE, "(rows={})".format(out_pairs.shape[0]))

print("\nDone.\n")
