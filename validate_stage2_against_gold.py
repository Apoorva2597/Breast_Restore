# validate_stage2_full_cohort_against_gold_blank_as_zero.py
# Python 3.6.8 compatible
#
# Validates Stage2 outcomes by linking:
#   GOLD (MRN) <---> BRIDGE (patient_id <-> MRN) <---> COHORT (patient_id)
#
# Key behavior:
#   - Restrict scoring universe to GOLD Stage2_Applicable == 1
#   - Treat BLANK in GOLD Stage2 outcome cols as 0 (optional + default ON here)
#   - Does NOT modify the original gold CSV on disk

from __future__ import print_function
import os
import re
import numpy as np
import pandas as pd

# -------------------------
# CONFIG (edit if needed)
# -------------------------
GOLD_CSV   = "gold_cleaned_for_cedar.csv"
COHORT_CSV = "cohort_all_patient_level_final_gold_order.csv"
BRIDGE_CSV = "cohort_pid_to_mrn_from_encounters.csv"  # patient_id, MRN

OUT_BY_VAR = "stage2_validation_confusion_by_var.csv"
OUT_PAIRS  = "stage2_validation_pairs_stage2.csv"

# Which Stage2 vars to validate (canonical names)
STAGE2_VARS = [
    "Stage2_MinorComp",
    "Stage2_MajorComp",
    "Stage2_Reoperation",
    "Stage2_Rehospitalitalization",  # keep misspelling guard (we normalize)
    "Stage2_Rehospitalization",
    "Stage2_Failure",
    "Stage2_Revision",
]

# Gold columns might have spaces; pred typically uses underscores.
# We'll map by "canonicalized" versions anyway, but these help if you want explicit mapping.
GOLD_COL_ALIASES = {
    "Stage2_MinorComp": ["Stage2 MinorComp", "Stage2_MinorComp"],
    "Stage2_MajorComp": ["Stage2 MajorComp", "Stage2_MajorComp"],
    "Stage2_Reoperation": ["Stage2 Reoperation", "Stage2_Reoperation"],
    "Stage2_Rehospitalization": ["Stage2 Rehospitalitalization", "Stage2 Rehospitalization", "Stage2_Rehospitalization"],
    "Stage2_Failure": ["Stage2 Failure", "Stage2_Failure"],
    "Stage2_Revision": ["Stage2 Revision", "Stage2_Revision"],
}

PRED_COL_ALIASES = {
    "Stage2_MinorComp": ["Stage2_MinorComp", "Stage2_MinorComp_pred"],
    "Stage2_MajorComp": ["Stage2_MajorComp", "Stage2_MajorComp_pred"],
    "Stage2_Reoperation": ["Stage2_Reoperation", "Stage2_Reoperation_pred"],
    "Stage2_Rehospitalization": ["Stage2_Rehospitalization", "Stage2_Rehospitalization_pred"],
    "Stage2_Failure": ["Stage2_Failure", "Stage2_Failure_pred"],
    "Stage2_Revision": ["Stage2_Revision", "Stage2_Revision_pred"],
}

# IMPORTANT SWITCH:
# If True: within the Stage2_Applicable==1 set, blank/NA in GOLD outcome cols => 0
TREAT_GOLD_BLANK_AS_ZERO_FOR_STAGE2_OUTCOMES = True

# -------------------------
# Helpers
# -------------------------

BLANK_TOKENS = set(["", "nan", "none", "null", "na", "n/a", ".", "-", "--"])

def read_csv_safe(path, nrows=None):
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

def canon_colname(s):
    """
    Canonicalize column names so:
      "Stage2 MinorComp" -> "stage2minorcomp"
      "Stage2_MinorComp_pred" -> "stage2minorcomppred"
    """
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)  # remove spaces, underscores, punctuation
    return s

def pick_col_by_alias(df_cols, alias_list):
    """
    Return first matching col by exact alias, else by canonical match.
    """
    cols = list(df_cols)
    cols_set = set(cols)
    # exact first
    for a in alias_list:
        if a in cols_set:
            return a
    # canonical
    want_can = [canon_colname(a) for a in alias_list]
    cols_can = {c: canon_colname(c) for c in cols}
    for c in cols:
        if cols_can[c] in want_can:
            return c
    return None

def parse_binary_series(series):
    """
    Parse values into {0,1,NaN} with robust handling:
      - "1", 1, "yes", "y", "true" => 1
      - "0", 0, "no", "n", "false" => 0
      - blank/unknown => NaN
    """
    out = []
    for v in series.tolist():
        s = norm_str(v)
        if s == "":
            out.append(np.nan)
            continue
        sl = s.lower()
        if sl in ("1", "y", "yes", "true", "t", "pos", "positive", "present"):
            out.append(1)
            continue
        if sl in ("0", "n", "no", "false", "f", "neg", "negative", "absent"):
            out.append(0)
            continue
        # numeric-ish fallback
        try:
            fv = float(sl)
            if fv == 1.0:
                out.append(1)
            elif fv == 0.0:
                out.append(0)
            else:
                out.append(np.nan)
        except Exception:
            out.append(np.nan)
    return pd.Series(out, index=series.index)

def safe_div(num, den):
    try:
        num = float(num)
        den = float(den)
        if den == 0.0:
            return ""
        return num / den
    except Exception:
        return ""

# -------------------------
# Main
# -------------------------

print("\n=== VALIDATION: Stage2 (FULL COHORT VS GOLD; blank-as-zero on GOLD stage2 outcomes) ===")
print("GOLD  :", GOLD_CSV)
print("COHORT:", COHORT_CSV)
print("BRIDGE:", BRIDGE_CSV)

if not os.path.exists(GOLD_CSV):
    raise RuntimeError("Missing GOLD_CSV: {}".format(GOLD_CSV))
if not os.path.exists(COHORT_CSV):
    raise RuntimeError("Missing COHORT_CSV: {}".format(COHORT_CSV))
if not os.path.exists(BRIDGE_CSV):
    raise RuntimeError("Missing BRIDGE_CSV: {}".format(BRIDGE_CSV))

gold = read_csv_safe(GOLD_CSV)
coh  = read_csv_safe(COHORT_CSV)
br   = read_csv_safe(BRIDGE_CSV)

# Detect key columns
gold_mrn_col = pick_col_by_alias(gold.columns, ["MRN"])
coh_pid_col  = pick_col_by_alias(coh.columns,  ["patient_id", "PATIENT_ID"])
br_pid_col   = pick_col_by_alias(br.columns,   ["patient_id", "PATIENT_ID"])
br_mrn_col   = pick_col_by_alias(br.columns,   ["MRN"])

if gold_mrn_col is None:
    raise RuntimeError("Could not find GOLD MRN column.")
if coh_pid_col is None:
    raise RuntimeError("Could not find COHORT patient_id column.")
if br_pid_col is None or br_mrn_col is None:
    raise RuntimeError("Could not find BRIDGE patient_id and MRN columns.")

print("\nDetected columns:")
print("  GOLD MRN col :", gold_mrn_col)
print("  BRIDGE pid   :", br_pid_col)
print("  BRIDGE MRN   :", br_mrn_col)
print("  COHORT pid   :", coh_pid_col)

# Normalize IDs
gold["_MRN_"] = gold[gold_mrn_col].map(norm_str)
coh["_PID_"]  = coh[coh_pid_col].map(norm_str)
br["_PID_"]   = br[br_pid_col].map(norm_str)
br["_MRN_"]   = br[br_mrn_col].map(norm_str)

# Stage2_Applicable in gold
gold_stage2_app_col = pick_col_by_alias(gold.columns, ["Stage2_Applicable", "Stage2 Applicable", "stage2_applicable"])
if gold_stage2_app_col is None:
    raise RuntimeError("Could not find GOLD Stage2_Applicable column.")

# Coerce Stage2_Applicable to int-like
gold["_S2APP_"] = pd.to_numeric(gold[gold_stage2_app_col], errors="coerce").fillna(0).astype(int)

# Filter gold to Stage2 Applicable == 1
gold_s2 = gold[gold["_S2APP_"] == 1].copy()

print("\nCounts:")
print("  GOLD rows:", len(gold))
print("  GOLD Stage2_Applicable==1:", int((gold['_S2APP_'] == 1).sum()))
print("  COHORT rows:", len(coh))
print("  COHORT pid nonblank:", int((coh["_PID_"] != "").sum()))
print("  BRIDGE pid nonblank:", int((br["_PID_"] != "").sum()))
print("  BRIDGE MRN nonblank:", int((br["_MRN_"] != "").sum()))

# Bridge ambiguity checks
tmp = br[(br["_PID_"] != "") & (br["_MRN_"] != "")]
pid_to_mrn_mult = int(tmp.groupby("_PID_")["_MRN_"].nunique().gt(1).sum())
mrn_to_pid_mult = int(tmp.groupby("_MRN_")["_PID_"].nunique().gt(1).sum())
print("\nBridge ambiguity checks:")
print("  patient_id -> multiple MRNs:", pid_to_mrn_mult)
print("  MRN -> multiple patient_ids:", mrn_to_pid_mult)
if pid_to_mrn_mult != 0 or mrn_to_pid_mult != 0:
    print("  WARNING: bridge is not 1:1; results may be ambiguous.")

# Attach MRN onto cohort
coh_br = coh.merge(br[["_PID_", "_MRN_"]], on="_PID_", how="left", suffixes=("", "_br"))
coh_br = coh_br.rename(columns={"_MRN_": "MRN_from_bridge"})

# Join gold stage2-applicable to cohort via MRN
joined = gold_s2.merge(coh_br, left_on="_MRN_", right_on="MRN_from_bridge", how="inner", suffixes=("_gold", "_pred"))

print("\nJoined rows for Stage2 scoring (gold Stage2_applicable only):", len(joined))

# Resolve column mapping for Stage2 vars
def resolve_cols_for_var(varname):
    gold_alias = GOLD_COL_ALIASES.get(varname, [varname])
    pred_alias = PRED_COL_ALIASES.get(varname, [varname])
    gcol = pick_col_by_alias(joined.columns, gold_alias)  # joined has gold columns directly
    pcol = pick_col_by_alias(joined.columns, pred_alias)  # joined has cohort columns directly
    return gcol, pcol

resolved = []
for v in ["Stage2_MinorComp","Stage2_MajorComp","Stage2_Reoperation","Stage2_Rehospitalization","Stage2_Failure","Stage2_Revision"]:
    gcol, pcol = resolve_cols_for_var(v)
    resolved.append((v, gcol, pcol))

print("\nResolved Stage2 column matches:")
for v, gcol, pcol in resolved:
    print("  {:<22} GOLD -> {:<30} | PRED -> {}".format(v, str(gcol), str(pcol)))

# Build paired dataset for debugging + scoring
pairs = joined[["_MRN_", "_PID_", "MRN_from_bridge"]].copy()
pairs = pairs.rename(columns={"_MRN_": "MRN_gold", "_PID_": "patient_id_pred", "MRN_from_bridge": "MRN_pred"})

results = []

for var, gcol, pcol in resolved:
    row = {
        "var": var,
        "gold_col": gcol if gcol is not None else "",
        "pred_col": pcol if pcol is not None else "",
        "status": "",
        "gold_col_present": int(gcol is not None),
        "pred_col_present": int(pcol is not None),
        "n_applicable_overlap": int(len(joined)),
        "n_scorable": 0,
        "gold_missing": 0,
        "pred_missing": 0,
        "TP": "",
        "FP": "",
        "FN": "",
        "TN": "",
        "sensitivity": "",
        "specificity": "",
        "ppv": "",
        "npv": "",
    }

    if gcol is None or pcol is None:
        row["status"] = "SKIP_missing_column"
        results.append(row)
        continue

    gold_raw = joined[gcol]
    pred_raw = joined[pcol]

    # Parse to binary
    gold_bin = parse_binary_series(gold_raw)
    pred_bin = parse_binary_series(pred_raw)

    # Apply blank-as-zero ONLY to GOLD stage2 outcome columns if configured
    if TREAT_GOLD_BLANK_AS_ZERO_FOR_STAGE2_OUTCOMES:
        gold_bin = gold_bin.fillna(0)

    # missing counts (after parsing; gold after optional fill)
    row["gold_missing"] = int(gold_bin.isna().sum())
    row["pred_missing"] = int(pred_bin.isna().sum())

    # scorable = both sides non-missing
    mask = (~gold_bin.isna()) & (~pred_bin.isna())
    n_sc = int(mask.sum())
    row["n_scorable"] = n_sc

    # Add to pairs output for debugging
    pairs[var + "_gold"] = gold_bin
    pairs[var + "_pred"] = pred_bin

    if n_sc == 0:
        row["status"] = "SKIP_no_scorable_rows"
        results.append(row)
        continue

    g = gold_bin[mask].astype(int)
    p = pred_bin[mask].astype(int)

    TP = int(((g == 1) & (p == 1)).sum())
    FP = int(((g == 0) & (p == 1)).sum())
    FN = int(((g == 1) & (p == 0)).sum())
    TN = int(((g == 0) & (p == 0)).sum())

    row["TP"] = TP
    row["FP"] = FP
    row["FN"] = FN
    row["TN"] = TN

    sens = safe_div(TP, TP + FN)
    spec = safe_div(TN, TN + FP)
    ppv  = safe_div(TP, TP + FP)
    npv  = safe_div(TN, TN + FN)

    row["sensitivity"] = sens if sens != "" else ""
    row["specificity"] = spec if spec != "" else ""
    row["ppv"] = ppv if ppv != "" else ""
    row["npv"] = npv if npv != "" else ""

    row["status"] = "OK"
    results.append(row)

# Write outputs
out_df = pd.DataFrame(results)
out_df.to_csv(OUT_BY_VAR, index=False, encoding="utf-8")
pairs.to_csv(OUT_PAIRS, index=False, encoding="utf-8")

print("\nWrote:", OUT_BY_VAR)
print("Wrote:", OUT_PAIRS)

# Pretty print key lines
print("\n=== Stage2 validation summary (by var) ===")
show_cols = ["var","status","n_applicable_overlap","n_scorable","TP","FP","FN","TN","sensitivity","specificity","ppv","npv"]
try:
    print(out_df[show_cols].to_string(index=False))
except Exception:
    print(out_df[show_cols].head(20))

print("\nDone.\n")
