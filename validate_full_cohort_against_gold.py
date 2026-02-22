# validate_full_cohort_against_gold.py
# Python 3.6.8 compatible
#
# Validates extracted cohort outputs (patient_id-based) against GOLD (MRN-based)
# by bridging patient_id -> MRN using pred_with_mrn.csv (or similar).

from __future__ import print_function
import re
import pandas as pd


# -------------------------
# CONFIG (edit if needed)
# -------------------------
GOLD_CSV   = "gold_cleaned_for_cedar.csv"
COHORT_CSV = "cohort_all_patient_level_final_gold_order.csv"
BRIDGE_CSV = "cohort_pid_to_mrn_from_encounters.csv"


# -------------------------
# Helpers
# -------------------------
def read_csv_safe(path):
    # your environment often needs latin1/cp1252 for Epic exports
    try:
        return pd.read_csv(path, dtype=object, encoding="utf-8", engine="python")
    except Exception:
        return pd.read_csv(path, dtype=object, encoding="latin1", engine="python", errors="replace")

def norm_colname(c):
    return str(c).strip()

def pick_col(cols, candidates):
    """
    Choose a column from `cols` matching candidates by:
      1) exact case-insensitive match
      2) substring match
    """
    cols_list = list(cols)
    cols_upper = {c: norm_colname(c).upper() for c in cols_list}

    # exact match
    for want in candidates:
        w = want.upper()
        for c in cols_list:
            if cols_upper[c] == w:
                return c

    # substring match
    for want in candidates:
        w = want.upper()
        for c in cols_list:
            if w in cols_upper[c]:
                return c

    return None

def norm_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("", "nan", "none", "null"):
        return ""
    return s


# -------------------------
# Main
# -------------------------
print("\n=== VALIDATION: FULL COHORT VS GOLD (via bridge) ===")
print("GOLD  :", GOLD_CSV)
print("COHORT:", COHORT_CSV)
print("BRIDGE:", BRIDGE_CSV)

gold = read_csv_safe(GOLD_CSV)
cohort = read_csv_safe(COHORT_CSV)
bridge = read_csv_safe(BRIDGE_CSV)

# Detect columns
gold_mrn_col = pick_col(gold.columns, ["MRN", "PAT_MRN_ID", "MRN_ID"])
bridge_pid_col = pick_col(bridge.columns, ["patient_id", "ENCRYPTED_PAT_ID", "PAT_ID", "PATIENT_ID"])
bridge_mrn_col = pick_col(bridge.columns, ["MRN", "MRN_FROM_ENCOUNTERS", "PAT_MRN_ID", "MRN_ID"])
cohort_pid_col = pick_col(cohort.columns, ["patient_id", "ENCRYPTED_PAT_ID", "PAT_ID", "PATIENT_ID"])

print("\nDetected columns:")
print("  GOLD MRN col      :", gold_mrn_col)
print("  BRIDGE patient_id :", bridge_pid_col)
print("  BRIDGE MRN        :", bridge_mrn_col)
print("  COHORT patient_id :", cohort_pid_col)

if gold_mrn_col is None:
    raise RuntimeError("Could not find MRN column in GOLD.")
if bridge_pid_col is None or bridge_mrn_col is None:
    raise RuntimeError("Could not find both patient_id and MRN columns in BRIDGE.")
if cohort_pid_col is None:
    raise RuntimeError("Could not find patient_id column in COHORT.")

# Normalize IDs
gold["_MRN_"] = gold[gold_mrn_col].map(norm_id)
bridge["_PID_"] = bridge[bridge_pid_col].map(norm_id)
bridge["_MRN_"] = bridge[bridge_mrn_col].map(norm_id)
cohort["_PID_"] = cohort[cohort_pid_col].map(norm_id)

# Keep only usable bridge rows
bridge2 = bridge[(bridge["_PID_"] != "") & (bridge["_MRN_"] != "")].copy()

# Check bridge ambiguity (should be 0/0 ideally)
pid_to_mrn_counts = bridge2.groupby("_PID_")["_MRN_"].nunique()
mrn_to_pid_counts = bridge2.groupby("_MRN_")["_PID_"].nunique()

n_pid_multi = int((pid_to_mrn_counts > 1).sum())
n_mrn_multi = int((mrn_to_pid_counts > 1).sum())

print("\nBridge ambiguity checks:")
print("  patient_id -> multiple MRNs:", n_pid_multi)
print("  MRN -> multiple patient_ids:", n_mrn_multi)

# Build mapping patient_id -> MRN (take first if duplicates exist)
pid_to_mrn = bridge2.drop_duplicates(subset=["_PID_"])[["_PID_", "_MRN_"]].set_index("_PID_")["_MRN_"].to_dict()

# Attach MRN to cohort
cohort["_MRN_"] = cohort["_PID_"].map(lambda x: pid_to_mrn.get(x, ""))

# Summary counts
gold_mrns = set([m for m in gold["_MRN_"].tolist() if m != ""])
cohort_mrns = set([m for m in cohort["_MRN_"].tolist() if m != ""])

print("\nCounts:")
print("  GOLD rows                :", int(len(gold)))
print("  GOLD unique MRNs         :", int(len(gold_mrns)))
print("  COHORT rows              :", int(len(cohort)))
print("  COHORT patient_id nonblank:", int((cohort["_PID_"] != "").sum()))
print("  COHORT rows w/ MRN linked:", int((cohort["_MRN_"] != "").sum()))
print("  COHORT unique MRNs linked:", int(len(cohort_mrns)))
print("  Overlap (GOLD ∩ COHORT MRN):", int(len(gold_mrns.intersection(cohort_mrns))))

# Write overlap report (useful for debugging)
overlap = gold_mrns.intersection(cohort_mrns)
only_gold = gold_mrns.difference(cohort_mrns)
only_cohort = cohort_mrns.difference(gold_mrns)

out_rep = pd.DataFrame({
    "metric": ["overlap_count", "only_gold_count", "only_cohort_count"],
    "value": [len(overlap), len(only_gold), len(only_cohort)]
})
out_rep.to_csv("validation_overlap_counts.csv", index=False, encoding="utf-8")

# Save row-level mapping for later joins/debug
cohort[["_PID_", "_MRN_"]].drop_duplicates().to_csv("cohort_pid_to_mrn.csv", index=False, encoding="utf-8")

print("\nWrote:")
print("  - validation_overlap_counts.csv")
print("  - cohort_pid_to_mrn.csv")
print("Done.\n")
