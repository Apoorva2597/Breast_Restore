# validate_stage2_against_gold.py
# Python 3.6.8 compatible
#
# Validates Stage 2 outcome columns in COHORT vs GOLD.
# - Links via patient_id -> MRN bridge from encounters
# - Joins on MRN
# - Scores only where GOLD Stage2_Applicable == 1
# - DOES NOT coerce blanks to 0
#
# Outputs:
#   stage2_validation_summary.csv
#   stage2_validation_confusion_by_var.csv
#   stage2_joined_overlap_preview.csv

from __future__ import print_function
import os
import pandas as pd


# -------------------------
# CONFIG
# -------------------------
GOLD_CSV   = "gold_cleaned_for_cedar.csv"
COHORT_CSV = "cohort_all_patient_level_final_gold_order.csv"
BRIDGE_CSV = "cohort_pid_to_mrn_from_encounters.csv"

OUT_SUMMARY   = "stage2_validation_summary.csv"
OUT_CONFUSION = "stage2_validation_confusion_by_var.csv"
OUT_PREVIEW   = "stage2_joined_overlap_preview.csv"

# GOLD columns
GOLD_MRN_COL = "MRN"
GOLD_APPLICABLE_COL = "Stage2_Applicable"

# Stage2 outcomes to validate (must exist in BOTH files to score)
STAGE2_VAR_MAP = {
    # gold_col_name           : cohort_col_name
    "Stage2 MinorComp"        : "Stage2_MinorComp",
    "Stage2 MajorComp"        : "Stage2_MajorComp",
    "Stage2 Reoperation"      : "Stage2_Reoperation",
    "Stage2 Rehospitalization": "Stage2_Rehospitalization",
    "Stage2 Failure"          : "Stage2_Failure",
    "Stage2 Revision"         : "Stage2_Revision",
}
# COHORT columns
COHORT_PID_COL = "patient_id"  # in cohort
BRIDGE_PID_COL = "patient_id"  # in bridge
BRIDGE_MRN_COL = "MRN"         # in bridge (from encounters)


# -------------------------
# Helpers
# -------------------------
BLANK_TOKENS = set(["", "nan", "none", "null", "na", "n/a", ".", "-", "--", "nat", "unknown"])

def read_csv_safe(path, nrows=None):
    # Epic exports often include odd bytes; latin1+replace is safest for reading
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", dtype=object, nrows=nrows)
    finally:
        try:
            f.close()
        except Exception:
            pass

def norm_str(x):
    if x is None:
        return ""
    s = str(x)
    try:
        s = s.replace("\xa0", " ")
    except Exception:
        pass
    s = s.strip()
    if s.lower() in BLANK_TOKENS:
        return ""
    return s

def to_binary_or_na(x):
    """
    Convert a value to 0/1 if it looks like a binary, else return NaN.
    DOES NOT convert blanks to 0.
    """
    s = norm_str(x)
    if s == "":
        return pd.NA

    sl = s.lower()

    # common true/false variants
    if sl in ("1", "y", "yes", "true", "t"):
        return 1
    if sl in ("0", "n", "no", "false", "f"):
        return 0

    # sometimes values come as floats/ints serialized
    try:
        # e.g., "1.0", "0.0"
        fv = float(sl)
        if fv == 1.0:
            return 1
        if fv == 0.0:
            return 0
    except Exception:
        pass

    # otherwise treat as missing/unscorable
    return pd.NA

def safe_div(n, d):
    return float(n) / float(d) if d else 0.0

def fmt_pct(x):
    return round(100.0 * x, 2)


# -------------------------
# Main
# -------------------------
print("\n=== VALIDATE STAGE 2: COHORT vs GOLD (via patient_id -> MRN bridge) ===")
print("GOLD  :", GOLD_CSV)
print("COHORT:", COHORT_CSV)
print("BRIDGE:", BRIDGE_CSV)

for p in (GOLD_CSV, COHORT_CSV, BRIDGE_CSV):
    if not os.path.exists(p):
        raise RuntimeError("Missing required file: {}".format(p))

gold = read_csv_safe(GOLD_CSV)
cohort = read_csv_safe(COHORT_CSV)
bridge = read_csv_safe(BRIDGE_CSV)

# Ensure key cols exist
need_gold = [GOLD_MRN_COL, GOLD_APPLICABLE_COL]
for c in need_gold:
    if c not in gold.columns:
        raise RuntimeError("GOLD missing required col: {}".format(c))

need_cohort = [COHORT_PID_COL]
for c in need_cohort:
    if c not in cohort.columns:
        raise RuntimeError("COHORT missing required col: {}".format(c))

need_bridge = [BRIDGE_PID_COL, BRIDGE_MRN_COL]
for c in need_bridge:
    if c not in bridge.columns:
        raise RuntimeError("BRIDGE missing required col: {}".format(c))

# Clean keys
gold["_MRN_"] = gold[GOLD_MRN_COL].map(norm_str)
cohort["_PID_"] = cohort[COHORT_PID_COL].map(norm_str)
bridge["_PID_"] = bridge[BRIDGE_PID_COL].map(norm_str)
bridge["_MRN_"] = bridge[BRIDGE_MRN_COL].map(norm_str)

# De-duplicate bridge safely (should already be 1:1 per your ambiguity checks)
bridge2 = bridge[bridge["_PID_"] != ""].copy()
bridge2 = bridge2[bridge2["_MRN_"] != ""].copy()
bridge2 = bridge2.drop_duplicates(subset=["_PID_"], keep="first")

# Attach MRN to cohort
cohort_mrn = cohort.merge(
    bridge2[["_PID_", "_MRN_"]],
    on="_PID_",
    how="left",
    suffixes=("", "_bridge")
).copy()

cohort_mrn["_MRN_"] = cohort_mrn["_MRN_"].map(norm_str)

# Join GOLD vs COHORT on MRN
gold2 = gold[gold["_MRN_"] != ""].copy()
joined = gold2.merge(
    cohort_mrn,
    left_on="_MRN_",
    right_on="_MRN_",
    how="inner",
    suffixes=("_gold", "_cohort")
).copy()

print("\nCounts:")
print("  GOLD rows:", len(gold))
print("  GOLD unique MRN (nonblank):", int(gold2["_MRN_"].nunique()))
print("  COHORT rows:", len(cohort))
print("  COHORT patient_id nonblank:", int((cohort["_PID_"] != "").sum()))
print("  COHORT rows with MRN linked:", int((cohort_mrn["_MRN_"] != "").sum()))
print("  COHORT unique MRN linked:", int(cohort_mrn[cohort_mrn["_MRN_"] != ""]["_MRN_"].nunique()))
print("  OVERLAP rows (MRN inner join):", len(joined))
print("  OVERLAP unique MRN:", int(joined["_MRN_"].nunique()))

# Apply Stage2 applicability filter based on GOLD
# Note: GOLD Stage2_Applicable may be 0/1, "0"/"1", blank, etc.
joined["_gold_stage2_applicable"] = joined[GOLD_APPLICABLE_COL].map(to_binary_or_na)

# Keep only applicable==1 for scoring Stage2 outcomes
app = joined[joined["_gold_stage2_applicable"] == 1].copy()

print("\nStage2 applicability (from GOLD):")
print("  Overlap rows:", len(joined))
print("  Applicable==1 rows:", len(app))

# Validate variables
summary_rows = []
conf_rows = []

# ensure columns exist (if missing, mark as skipped)
present_gold = set(gold.columns)
present_cohort = set(cohort.columns)

for var_gold, var_pred in STAGE2_VAR_MAP.items():
    var_gold = var  # in GOLD
    var_pred = var  # in COHORT (you aligned to gold-friendly names)

    has_gold = (var_gold in app.columns)
    has_pred = (var_pred in app.columns)

    if not has_gold or not has_pred:
        summary_rows.append({
            "var": var,
            "status": "SKIP_missing_column",
            "gold_col_present": int(has_gold),
            "pred_col_present": int(has_pred),
            "n_applicable_overlap": len(app),
            "n_scorable": 0,
            "gold_missing": None,
            "pred_missing": None,
            "TP": None, "FP": None, "FN": None, "TN": None,
            "sensitivity": None, "specificity": None, "ppv": None, "npv": None,
            "accuracy": None,
        })
        continue

    # Parse to binary/NA (do not coerce blanks to 0)
    g = app[var_gold].map(to_binary_or_na)
    p = app[var_pred].map(to_binary_or_na)

    gold_missing = int(g.isna().sum())
    pred_missing = int(p.isna().sum())

    # Scorable rows: both present
    mask = (~g.isna()) & (~p.isna())
    n_scorable = int(mask.sum())

    if n_scorable == 0:
        summary_rows.append({
            "var": var,
            "status": "OK_no_scorable_rows",
            "gold_col_present": 1,
            "pred_col_present": 1,
            "n_applicable_overlap": len(app),
            "n_scorable": 0,
            "gold_missing": gold_missing,
            "pred_missing": pred_missing,
            "TP": 0, "FP": 0, "FN": 0, "TN": 0,
            "sensitivity": None, "specificity": None, "ppv": None, "npv": None,
            "accuracy": None,
        })
        continue

    g2 = g[mask].astype("int64")
    p2 = p[mask].astype("int64")

    TP = int(((g2 == 1) & (p2 == 1)).sum())
    TN = int(((g2 == 0) & (p2 == 0)).sum())
    FP = int(((g2 == 0) & (p2 == 1)).sum())
    FN = int(((g2 == 1) & (p2 == 0)).sum())

    sens = safe_div(TP, TP + FN)
    spec = safe_div(TN, TN + FP)
    ppv  = safe_div(TP, TP + FP)
    npv  = safe_div(TN, TN + FN)
    acc  = safe_div(TP + TN, TP + TN + FP + FN)

    summary_rows.append({
        "var": var,
        "status": "OK",
        "gold_col_present": 1,
        "pred_col_present": 1,
        "n_applicable_overlap": len(app),
        "n_scorable": n_scorable,
        "gold_missing": gold_missing,
        "pred_missing": pred_missing,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "sensitivity_pct": fmt_pct(sens),
        "specificity_pct": fmt_pct(spec),
        "ppv_pct": fmt_pct(ppv),
        "npv_pct": fmt_pct(npv),
        "accuracy_pct": fmt_pct(acc),
    })

    conf_rows.append({
        "var": var,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "n_scorable": n_scorable,
        "gold_missing_in_applicable": gold_missing,
        "pred_missing_in_applicable": pred_missing,
    })

# Write outputs
summary_df = pd.DataFrame(summary_rows)
conf_df = pd.DataFrame(conf_rows)

summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")
conf_df.to_csv(OUT_CONFUSION, index=False, encoding="utf-8")

# Save a small preview of the joined overlap for manual inspection
preview_cols = [
    "_MRN_",
    GOLD_APPLICABLE_COL,
]
for v in STAGE2_VARS:
    if v in joined.columns:
        preview_cols.append(v)
# add cohort patient_id if present in joined
if COHORT_PID_COL in joined.columns:
    preview_cols.append(COHORT_PID_COL)

joined[preview_cols].head(200).to_csv(OUT_PREVIEW, index=False, encoding="utf-8")

print("\nWrote:")
print("  -", OUT_SUMMARY)
print("  -", OUT_CONFUSION)
print("  -", OUT_PREVIEW)

print("\nDone.\n")
