# sanity_link_gold_pred_via_encounters.py
# Python 3.6.8+ compatible
#
# Purpose:
#   Safely link GOLD (MRN-based) to PRED (patient_id-based) using Clinic Encounters bridge:
#     pred.patient_id  <->  clinic_encounters.ENCRYPTED_PAT_ID  -> MRN  <-> gold.MRN
#
# Outputs:
#   - Prints linkage QA to terminal
#   - Writes:
#       1) pred_with_mrn.csv            (adds MRN to pred rows where possible)
#       2) gold_pred_overlap_report.csv (summary table)
#
# NOTE:
#   We intentionally do NOT use gold.PatientID for linking (it's not ENCRYPTED_PAT_ID).

from __future__ import print_function

import os
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS)
# -------------------------
GOLD_CSV = "gold_cleaned_for_cedar.csv"

# Use the Clinic Encounters file you showed (contains MRN + ENCRYPTED_PAT_ID)
CLINIC_ENC_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11526 Clinic Encounters.csv"

# Your prediction spine (or any pred file with patient_id)
PRED_CSV = "pred_spine_stage1_stage2.csv"

OUT_PRED_WITH_MRN = "pred_with_mrn.csv"
OUT_REPORT = "gold_pred_overlap_report.csv"


# -------------------------
# Helpers
# -------------------------
def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", dtype=object, **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def norm_str(x):
    if x is None:
        return ""
    s = str(x).strip()
    return s


def pick_col(df_cols, candidates):
    cols = list(df_cols)
    upper = {c: str(c).strip().upper() for c in cols}
    for want in candidates:
        want_u = want.upper()
        for c in cols:
            if upper[c] == want_u:
                return c
    # fallback: contains match
    for want in candidates:
        want_u = want.upper()
        for c in cols:
            if want_u in upper[c]:
                return c
    return None


# -------------------------
# Main
# -------------------------
print("\n=== Sanity: Link GOLD <-> PRED via Clinic Encounters ===")
print("GOLD:", GOLD_CSV)
print("CLINIC_ENC:", CLINIC_ENC_CSV)
print("PRED:", PRED_CSV)

# 1) Load files
gold = read_csv_safe(GOLD_CSV)
enc = read_csv_safe(CLINIC_ENC_CSV)
pred = read_csv_safe(PRED_CSV)

# 2) Detect needed columns
gold_mrn_col = pick_col(gold.columns, ["MRN"])
enc_mrn_col = pick_col(enc.columns, ["MRN"])
enc_pid_col = pick_col(enc.columns, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "PAT_ID", "PATID"])
pred_pid_col = pick_col(pred.columns, ["patient_id", "ENCRYPTED_PAT_ID"])

if gold_mrn_col is None:
    raise RuntimeError("Could not find MRN column in GOLD: columns={}".format(list(gold.columns)))

if enc_mrn_col is None or enc_pid_col is None:
    raise RuntimeError(
        "Could not find MRN and ENCRYPTED_PAT_ID columns in Clinic Encounters.\n"
        "Found MRN col: {}\nFound patient col: {}\nColumns={}".format(enc_mrn_col, enc_pid_col, list(enc.columns))
    )

if pred_pid_col is None:
    raise RuntimeError(
        "Could not find patient_id (or ENCRYPTED_PAT_ID) in PRED.\nColumns={}".format(list(pred.columns))
    )

print("\nDetected columns:")
print("  GOLD MRN col:         {}".format(gold_mrn_col))
print("  ENC MRN col:          {}".format(enc_mrn_col))
print("  ENC ENCRYPTED col:    {}".format(enc_pid_col))
print("  PRED patient_id col:  {}".format(pred_pid_col))

# 3) Normalize IDs
gold["__MRN__"] = gold[gold_mrn_col].map(norm_str)
enc["__MRN__"] = enc[enc_mrn_col].map(norm_str)
enc["__PID__"] = enc[enc_pid_col].map(norm_str)
pred["__PID__"] = pred[pred_pid_col].map(norm_str)

gold = gold[gold["__MRN__"] != ""].copy()
enc = enc[(enc["__MRN__"] != "") & (enc["__PID__"] != "")].copy()
pred = pred[pred["__PID__"] != ""].copy()

# 4) Build bridge mapping (PID -> MRN)
#    First, measure ambiguity BEFORE collapsing.
pid_to_mrn_counts = enc.groupby("__PID__")["__MRN__"].nunique()
mrn_to_pid_counts = enc.groupby("__MRN__")["__PID__"].nunique()

n_pid_ambig = int((pid_to_mrn_counts > 1).sum())
n_mrn_ambig = int((mrn_to_pid_counts > 1).sum())

print("\nBridge ambiguity checks (Clinic Encounters):")
print("  Unique ENCRYPTED_PAT_ID: {}".format(int(enc["__PID__"].nunique())))
print("  Unique MRN:              {}".format(int(enc["__MRN__"].nunique())))
print("  ENCRYPTED_PAT_ID -> multiple MRNs: {}".format(n_pid_ambig))
print("  MRN -> multiple ENCRYPTED_PAT_IDs: {}".format(n_mrn_ambig))

# Collapse to one row per PID (take first MRN after drop_duplicates)
bridge = enc[["__PID__", "__MRN__"]].drop_duplicates(subset=["__PID__", "__MRN__"]).copy()
bridge_one = bridge.drop_duplicates(subset=["__PID__"], keep="first").copy()

# 5) Attach MRN to predictions
pred2 = pred.merge(bridge_one, on="__PID__", how="left")
pred2 = pred2.rename(columns={"__MRN__": "MRN_from_encounters"})

# 6) Overlap checks
gold_mrns = set(gold["__MRN__"].tolist())
pred_mrns = set([m for m in pred2["MRN_from_encounters"].map(norm_str).tolist() if m != ""])

n_gold = len(gold_mrns)
n_pred_pid = int(pred2["__PID__"].nunique())
n_pred_mrn = len(pred_mrns)

n_pred_linked = int(pred2["MRN_from_encounters"].notna().sum())
n_pred_total_rows = len(pred2)

overlap = gold_mrns.intersection(pred_mrns)

print("\nCounts:")
print("  GOLD unique MRNs:                    {}".format(n_gold))
print("  PRED unique patient_id:              {}".format(n_pred_pid))
print("  PRED rows:                           {}".format(n_pred_total_rows))
print("  PRED rows with MRN linked (non-null): {}".format(n_pred_linked))
print("  PRED unique MRNs after linking:      {}".format(n_pred_mrn))
print("  Overlap (GOLD MRN ∩ PRED MRN):       {}".format(len(overlap)))

# 7) Write outputs
pred2.to_csv(OUT_PRED_WITH_MRN, index=False, encoding="utf-8")

report = pd.DataFrame([{
    "gold_unique_mrns": n_gold,
    "pred_unique_patient_id": n_pred_pid,
    "pred_rows": n_pred_total_rows,
    "pred_rows_with_mrn_linked": n_pred_linked,
    "pred_unique_mrns_linked": n_pred_mrn,
    "mrn_overlap_gold_vs_pred": len(overlap),
    "pid_to_multiple_mrns_in_encounters": n_pid_ambig,
    "mrn_to_multiple_pids_in_encounters": n_mrn_ambig,
}])
report.to_csv(OUT_REPORT, index=False, encoding="utf-8")

print("\nWrote:")
print("  - {}".format(OUT_PRED_WITH_MRN))
print("  - {}".format(OUT_REPORT))
print("\nDone.\n")
