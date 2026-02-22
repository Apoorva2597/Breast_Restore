#!/usr/bin/env python3
# validate_stage2_full_cohort_against_gold.py
# Python 3.6.8 compatible
#
# Goal:
#   Validate Stage2 outcomes in your FULL cohort abstraction against GOLD,
#   using a deterministic MRN<->patient_id bridge built from encounters.
#
# Key fix vs your prior script:
#   - Resolve GOLD Stage2 columns from *gold.csv* only
#   - Resolve PRED Stage2 columns from *cohort.csv* only
#   - Handle merge suffixes explicitly (_gold / _pred) when they occur
#   - Convert blanks/NA -> 0 ONLY for Stage2 outcome scoring (does NOT rewrite your input files)
#
# Inputs (edit if needed):
#   GOLD   : gold_cleaned_for_cedar.csv
#   COHORT : cohort_all_patient_level_final_gold_order.csv
#   BRIDGE : cohort_pid_to_mrn_from_encounters.csv   (patient_id, MRN)
#
# Outputs:
#   - stage2_validation_confusion_by_var.csv
#   - stage2_validation_pairs_stage2.csv

from __future__ import print_function
import os
import re
import pandas as pd


# -------------------------
# CONFIG (edit if needed)
# -------------------------
GOLD_CSV   = "gold_cleaned_for_cedar.csv"
COHORT_CSV = "cohort_all_patient_level_final_gold_order.csv"
BRIDGE_CSV = "cohort_pid_to_mrn_from_encounters.csv"

OUT_BYVAR  = "stage2_validation_confusion_by_var.csv"
OUT_PAIRS  = "stage2_validation_pairs_stage2.csv"

# Stage2 variables to validate (canonical names used in reports)
STAGE2_VARS = [
    "Stage2_MinorComp",
    "Stage2_MajorComp",
    "Stage2_Reoperation",
    "Stage2_Rehospitalization",
    "Stage2_Failure",
    "Stage2_Revision",
]

# GOLD uses "Stage2 MinorComp" style (spaces). We support multiple aliases.
GOLD_ALIASES = {
    "Stage2_MinorComp":        ["Stage2 MinorComp", "Stage2_MinorComp", "Stage2-MinorComp"],
    "Stage2_MajorComp":        ["Stage2 MajorComp", "Stage2_MajorComp", "Stage2-MajorComp"],
    "Stage2_Reoperation":      ["Stage2 Reoperation", "Stage2_Reoperation", "Stage2-Reoperation"],
    "Stage2_Rehospitalization":["Stage2 Rehospitalization", "Stage2_Rehospitalization", "Stage2-Rehospitalization",
                                "Stage2 Rehospitalisation", "Stage2_Rehospitalisation"],
    "Stage2_Failure":          ["Stage2 Failure", "Stage2_Failure", "Stage2-Failure"],
    "Stage2_Revision":         ["Stage2 Revision", "Stage2_Revision", "Stage2-Revision"],
}

# COHORT often uses "_pred" suffix (e.g., Stage2_MinorComp_pred) but we support both.
PRED_ALIASES = {
    "Stage2_MinorComp":        ["Stage2_MinorComp_pred", "Stage2_MinorComp", "Stage2 MinorComp"],
    "Stage2_MajorComp":        ["Stage2_MajorComp_pred", "Stage2_MajorComp", "Stage2 MajorComp"],
    "Stage2_Reoperation":      ["Stage2_Reoperation_pred", "Stage2_Reoperation", "Stage2 Reoperation"],
    "Stage2_Rehospitalization":["Stage2_Rehospitalization_pred", "Stage2_Rehospitalization", "Stage2 Rehospitalization",
                                "Stage2_Rehospitalisation_pred", "Stage2_Rehospitalisation", "Stage2 Rehospitalisation"],
    "Stage2_Failure":          ["Stage2_Failure_pred", "Stage2_Failure", "Stage2 Failure"],
    "Stage2_Revision":         ["Stage2_Revision_pred", "Stage2_Revision", "Stage2 Revision"],
}

# GOLD applicability flag (must be 1 for Stage2 scoring)
GOLD_STAGE2_APPLICABLE_COL_CANDIDATES = ["Stage2_Applicable", "Stage2 Applicable", "stage2_applicable"]


# -------------------------
# Helpers
# -------------------------
def read_csv_safe(path, nrows=None):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", dtype=object, nrows=nrows)
    finally:
        try:
            f.close()
        except Exception:
            pass

def norm_colname(c):
    # canonicalize for matching: keep alnum only, lower
    s = str(c)
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

def resolve_column(df_cols, candidates):
    """
    Deterministic resolver:
      1) exact match (case sensitive)
      2) case-insensitive exact
      3) canonical normalized match (removes spaces/punct)
    """
    cols = list(df_cols)

    # 1) exact
    for want in candidates:
        if want in cols:
            return want

    # 2) case-insensitive exact
    cols_upper = {str(c).upper(): c for c in cols}
    for want in candidates:
        w = str(want).upper()
        if w in cols_upper:
            return cols_upper[w]

    # 3) canonical normalized
    norm_map = {}
    for c in cols:
        norm_map[norm_colname(c)] = c
    for want in candidates:
        nw = norm_colname(want)
        if nw in norm_map:
            return norm_map[nw]

    return None

def to01_series(s):
    """
    Convert a series to 0/1 for scoring.
    IMPORTANT: blanks/NA -> 0 (only used inside validation).
    Accepts: 1/0, True/False, yes/no, y/n, performed/denied (treated as 1/0),
             and strings containing '1' or '0'.
    """
    if s is None:
        return pd.Series([], dtype="int64")

    def _one(x):
        if x is None:
            return 0
        if isinstance(x, float) and pd.isna(x):
            return 0
        # keep ints/bools
        if isinstance(x, (int, bool)):
            return 1 if int(x) == 1 else 0
        # strings
        t = str(x).strip().lower()
        if t == "" or t in ("nan", "none", "null", "na", "n/a", ".", "-", "--"):
            return 0
        if t in ("1", "true", "t", "yes", "y", "positive", "pos", "performed", "present"):
            return 1
        if t in ("0", "false", "f", "no", "n", "negative", "neg", "denied", "absent"):
            return 0
        # last resort: if it contains a standalone 1/0
        if re.search(r"(^|[^0-9])1([^0-9]|$)", t):
            return 1
        if re.search(r"(^|[^0-9])0([^0-9]|$)", t):
            return 0
        # otherwise unknown -> 0 (conservative for prediction validation)
        return 0

    return s.map(_one).astype(int)

def confusion_counts(gold01, pred01):
    tp = int(((gold01 == 1) & (pred01 == 1)).sum())
    fp = int(((gold01 == 0) & (pred01 == 1)).sum())
    fn = int(((gold01 == 1) & (pred01 == 0)).sum())
    tn = int(((gold01 == 0) & (pred01 == 0)).sum())
    return tp, fp, fn, tn

def safe_div(n, d):
    if d is None or d == 0:
        return ""
    return float(n) / float(d)


# -------------------------
# Main
# -------------------------
print("\n=== VALIDATION: STAGE 2 (FULL COHORT vs GOLD via bridge) ===")
print("GOLD  :", GOLD_CSV)
print("COHORT:", COHORT_CSV)
print("BRIDGE:", BRIDGE_CSV)

for p in (GOLD_CSV, COHORT_CSV, BRIDGE_CSV):
    if not os.path.exists(p):
        raise RuntimeError("Missing input file: {}".format(p))

gold = read_csv_safe(GOLD_CSV)
cohort = read_csv_safe(COHORT_CSV)
bridge = read_csv_safe(BRIDGE_CSV)

# detect key columns
gold_mrn_col = resolve_column(gold.columns, ["MRN", "mrn"])
bridge_pid_col = resolve_column(bridge.columns, ["patient_id", "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "PATIENT_ID", "PAT_ID"])
bridge_mrn_col = resolve_column(bridge.columns, ["MRN", "mrn"])
cohort_pid_col = resolve_column(cohort.columns, ["patient_id", "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "PATIENT_ID", "PAT_ID"])

print("\nDetected columns:")
print("  GOLD MRN col   :", gold_mrn_col)
print("  BRIDGE pid col :", bridge_pid_col)
print("  BRIDGE MRN col :", bridge_mrn_col)
print("  COHORT pid col :", cohort_pid_col)

if not gold_mrn_col or not bridge_pid_col or not bridge_mrn_col or not cohort_pid_col:
    raise RuntimeError("Could not detect required key columns (MRN/patient_id).")

# Stage2 applicable col in GOLD
gold_stage2_app_col = resolve_column(gold.columns, GOLD_STAGE2_APPLICABLE_COL_CANDIDATES)
if not gold_stage2_app_col:
    raise RuntimeError("Could not find Stage2_Applicable column in GOLD. Checked: {}".format(GOLD_STAGE2_APPLICABLE_COL_CANDIDATES))

# normalize keys
gold["_MRN_"] = gold[gold_mrn_col].astype(str).str.strip()
bridge["_MRN_"] = bridge[bridge_mrn_col].astype(str).str.strip()
bridge["_PID_"] = bridge[bridge_pid_col].astype(str).str.strip()
cohort["_PID_"] = cohort[cohort_pid_col].astype(str).str.strip()

# clean empties
gold = gold[gold["_MRN_"] != ""].copy()
bridge = bridge[(bridge["_MRN_"] != "") & (bridge["_PID_"] != "")].copy()
cohort = cohort[cohort["_PID_"] != ""].copy()

# ambiguity checks on bridge
pid_to_mrn_counts = bridge.groupby("_PID_")["_MRN_"].nunique()
mrn_to_pid_counts = bridge.groupby("_MRN_")["_PID_"].nunique()
pid_multi_mrn = int((pid_to_mrn_counts > 1).sum())
mrn_multi_pid = int((mrn_to_pid_counts > 1).sum())

print("\nCounts:")
print("  GOLD rows                 :", int(len(gold)))
print("  GOLD Stage2_Applicable==1 :", int((to01_series(gold[gold_stage2_app_col]) == 1).sum()))
print("  COHORT rows               :", int(len(cohort)))
print("  BRIDGE rows               :", int(len(bridge)))

print("\nBridge ambiguity checks:")
print("  patient_id -> multiple MRNs:", pid_multi_mrn)
print("  MRN -> multiple patient_ids:", mrn_multi_pid)
if pid_multi_mrn or mrn_multi_pid:
    print("  WARNING: Bridge has ambiguity. Validation may be unreliable until resolved.")

# Use a de-duplicated bridge (pick first occurrence if duplicates exist)
bridge_dedup = bridge.drop_duplicates(subset=["_PID_"], keep="first").copy()

# Restrict GOLD to Stage2 applicable only
gold_app = gold[to01_series(gold[gold_stage2_app_col]) == 1].copy()

# Join: GOLD(MRN) -> BRIDGE(MRN->PID) -> COHORT(PID)
j1 = gold_app.merge(bridge_dedup[["_MRN_", "_PID_"]], on="_MRN_", how="left")
joined = j1.merge(cohort, on="_PID_", how="left", suffixes=("_gold", "_pred"))

# report join success
n_gold_app = int(len(gold_app))
n_join_pid = int(joined["_PID_"].notnull().sum())
n_join_cohort = int(joined[cohort.columns[0]].notnull().sum())  # crude but ok
# Better: count where we have at least one non-null cohort field (excluding _PID_)
cohort_nonnull = int(joined.drop(columns=[c for c in joined.columns if c in ("_MRN_", "_PID_")], errors="ignore").notnull().any(axis=1).sum())

print("\nJoined rows for Stage2 scoring (gold Stage2_applicable only):", int(len(joined)))
print("  with PID linked:", int(joined["_PID_"].notnull().sum()))
print("  with COHORT row linked (any non-null):", cohort_nonnull)

# Resolve GOLD and PRED Stage2 columns deterministically
resolved = []
print("\nResolved Stage2 column matches (source-based, deterministic):")
for v in STAGE2_VARS:
    g_col = resolve_column(gold.columns, GOLD_ALIASES.get(v, [v]))
    p_col = resolve_column(cohort.columns, PRED_ALIASES.get(v, [v]))

    # Determine the actual column names inside `joined` after merges:
    # - GOLD columns may appear as original name or "<name>_gold" depending on overlap
    # - COHORT columns may appear as original name or "<name>_pred"
    g_join = None
    p_join = None

    if g_col is not None:
        if g_col in joined.columns:
            g_join = g_col
        elif (g_col + "_gold") in joined.columns:
            g_join = g_col + "_gold"

    if p_col is not None:
        if p_col in joined.columns:
            p_join = p_col
        elif (p_col + "_pred") in joined.columns:
            p_join = p_col + "_pred"

    print("  {:<22} GOLD -> {:<28} | PRED -> {}".format(v, str(g_join), str(p_join)))
    resolved.append((v, g_col, p_col, g_join, p_join))

# Build per-var validation
rows = []
pair_rows = []

for (v, g_src, p_src, g_col_join, p_col_join) in resolved:
    gold_present = 1 if (g_col_join is not None and g_col_join in joined.columns) else 0
    pred_present = 1 if (p_col_join is not None and p_col_join in joined.columns) else 0

    n_applicable_overlap = int(len(joined))  # by construction: gold Stage2_applicable only

    if not gold_present or not pred_present:
        rows.append({
            "var": v,
            "gold_col": g_src if g_src else "",
            "pred_col": p_src if p_src else "",
            "status": "SKIP_missing_column",
            "gold_col_present": gold_present,
            "pred_col_present": pred_present,
            "n_applicable_overlap": n_applicable_overlap,
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
        })
        continue

    g_raw = joined[g_col_join]
    p_raw = joined[p_col_join]

    # Scorable rows: where we have a cohort row linked (PID exists in cohort merge)
    # We consider a row scorable if PRED column exists (it will) AND PID is present AND at least one cohort field is non-null.
    # Minimal requirement: PID exists and pred_raw notnull OR blank (we treat blank as 0), so we can score.
    # We'll score ALL joined rows, but keep diagnostics on how many were missing in source.
    g01 = to01_series(g_raw)  # blanks -> 0 for scoring
    p01 = to01_series(p_raw)

    # Missing diagnostics (before conversion)
    gold_missing = int(pd.isna(g_raw).sum()) + int((g_raw.astype(str).str.strip() == "").sum())
    pred_missing = int(pd.isna(p_raw).sum()) + int((p_raw.astype(str).str.strip() == "").sum())

    n_scorable = int(len(joined))

    tp, fp, fn, tn = confusion_counts(g01, p01)

    sens = safe_div(tp, (tp + fn))
    spec = safe_div(tn, (tn + fp))
    ppv  = safe_div(tp, (tp + fp))
    npv  = safe_div(tn, (tn + fn))

    rows.append({
        "var": v,
        "gold_col": g_col_join,
        "pred_col": p_col_join,
        "status": "OK",
        "gold_col_present": gold_present,
        "pred_col_present": pred_present,
        "n_applicable_overlap": n_applicable_overlap,
        "n_scorable": n_scorable,
        "gold_missing": gold_missing,
        "pred_missing": pred_missing,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
    })

    # Pairwise output (one row per patient in overlap, per var)
    tmp = pd.DataFrame({
        "MRN": joined["_MRN_"],
        "patient_id": joined["_PID_"],
        "var": v,
        "gold_raw": g_raw,
        "pred_raw": p_raw,
        "gold_01": g01,
        "pred_01": p01,
    })
    pair_rows.append(tmp)

# Write outputs
out_byvar = pd.DataFrame(rows)
out_byvar.to_csv(OUT_BYVAR, index=False, encoding="utf-8")

if pair_rows:
    out_pairs = pd.concat(pair_rows, axis=0, ignore_index=True)
else:
    out_pairs = pd.DataFrame(columns=["MRN", "patient_id", "var", "gold_raw", "pred_raw", "gold_01", "pred_01"])
out_pairs.to_csv(OUT_PAIRS, index=False, encoding="utf-8")

print("\nWrote:", OUT_BYVAR)
print("Wrote:", OUT_PAIRS)

# Print a compact terminal summary
print("\n=== Stage2 validation summary (by var) ===")
show_cols = ["var","status","n_applicable_overlap","n_scorable","TP","FP","FN","TN","sensitivity","specificity","ppv","npv"]
print(out_byvar[show_cols].to_string(index=False))

print("\nDone.\n")
