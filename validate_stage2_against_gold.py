# validate_stage2_full_cohort_against_gold.py
# Python 3.6.8 compatible
#
# Robust Stage2 validation (FULL cohort vs GOLD) via patient_id->MRN bridge.
# Fixes "missing column" issues caused by hidden header differences (BOM, NBSP, trailing spaces, etc.)
#
from __future__ import print_function
import re
import pandas as pd

# -------------------------
# CONFIG: EDIT PATHS
# -------------------------
GOLD_CSV   = "gold_cleaned_for_cedar.csv"
COHORT_CSV = "cohort_all_patient_level_final_gold_order.csv"
BRIDGE_CSV = "cohort_pid_to_mrn_from_encounters.csv"

OUT_CONFUSION = "stage2_validation_confusion_by_var.csv"
OUT_PAIRWISE  = "stage2_validation_pairwise_rows.csv"  # row-level audit

GOLD_STAGE2_APPLICABLE_COL_CANDIDATES = ["Stage2_Applicable", "Stage2 Applicable", "stage2_applicable"]

# Desired mapping (logical names)
# We will match these "targets" robustly (ignoring spaces/_/case/punct).
STAGE2_VAR_TARGETS = [
    ("Stage2 MinorComp",         "Stage2_MinorComp"),
    ("Stage2 MajorComp",         "Stage2_MajorComp"),
    ("Stage2 Reoperation",       "Stage2_Reoperation"),
    ("Stage2 Rehospitalization", "Stage2_Rehospitalization"),
    ("Stage2 Failure",           "Stage2_Failure"),
    ("Stage2 Revision",          "Stage2_Revision"),
]

TRUE_TOKENS  = set(["1", "true", "t", "yes", "y"])
FALSE_TOKENS = set(["0", "false", "f", "no", "n"])


# -------------------------
# Helpers
# -------------------------
def read_csv_safe(path):
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1", errors="replace")

def clean_header(s):
    # normalize BOM, NBSP, newlines, whitespace
    s = "" if s is None else str(s)
    s = s.replace("\ufeff", "")         # BOM
    s = s.replace("\xa0", " ")          # NBSP
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def canonical(s):
    # strip to alnum only, lowercase
    # Example: "Stage2 MinorComp" == "Stage2_MinorComp" == " stage2-minorcomp "
    s = clean_header(s).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def norm_str(x):
    if x is None:
        return ""
    s = str(x)
    try:
        s = s.replace("\xa0", " ")
    except Exception:
        pass
    s = s.replace("\ufeff", "")
    s = s.strip()
    if s.lower() in ("", "nan", "none", "null"):
        return ""
    return s

def to_binary_or_missing(x):
    s = norm_str(x)
    if s == "":
        return None
    sl = s.lower()
    if sl in TRUE_TOKENS:
        return 1
    if sl in FALSE_TOKENS:
        return 0
    if re.match(r"^\d+(\.0+)?$", s):
        try:
            v = int(float(s))
            if v in (0, 1):
                return v
        except Exception:
            pass
    return None

def pick_col_robust(cols, candidates):
    """
    Pick a column by matching canonical forms.
    """
    col_map = {canonical(c): c for c in cols}
    for want in candidates:
        w = canonical(want)
        if w in col_map:
            return col_map[w]
    # fallback: contains (canonical substring)
    for want in candidates:
        w = canonical(want)
        for ck, orig in col_map.items():
            if w and w in ck:
                return orig
    return None

def find_col_by_target(df_cols, target_name):
    """
    Return the actual column name in df_cols matching target_name by canonical equality.
    """
    target_key = canonical(target_name)
    for c in df_cols:
        if canonical(c) == target_key:
            return c
    return None

def confusion_counts(gold_bin, pred_bin):
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

gold = read_csv_safe(GOLD_CSV)
cohort = read_csv_safe(COHORT_CSV)
bridge = read_csv_safe(BRIDGE_CSV)

# Clean all headers up-front (THIS is the key fix)
gold.columns = [clean_header(c) for c in gold.columns]
cohort.columns = [clean_header(c) for c in cohort.columns]
bridge.columns = [clean_header(c) for c in bridge.columns]

# Detect join columns robustly
gold_mrn_col   = pick_col_robust(gold.columns,   ["MRN"])
bridge_pid_col = pick_col_robust(bridge.columns, ["patient_id", "PATIENT_ID", "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID"])
bridge_mrn_col = pick_col_robust(bridge.columns, ["MRN"])
cohort_pid_col = pick_col_robust(cohort.columns, ["patient_id", "PATIENT_ID", "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID"])

if not gold_mrn_col:
    raise RuntimeError("Could not find MRN column in GOLD after header cleaning.")
if not (bridge_pid_col and bridge_mrn_col):
    raise RuntimeError("Could not find (patient_id, MRN) columns in BRIDGE after header cleaning.")
if not cohort_pid_col:
    raise RuntimeError("Could not find patient_id column in COHORT after header cleaning.")

print("\nDetected columns:")
print("  GOLD MRN col   :", gold_mrn_col)
print("  BRIDGE pid col :", bridge_pid_col)
print("  BRIDGE MRN col :", bridge_mrn_col)
print("  COHORT pid col :", cohort_pid_col)

# Normalize keys
gold["_MRN_"]   = gold[gold_mrn_col].map(norm_str)
bridge["_PID_"] = bridge[bridge_pid_col].map(norm_str)
bridge["_MRN_"] = bridge[bridge_mrn_col].map(norm_str)
cohort["_PID_"] = cohort[cohort_pid_col].map(norm_str)

# Stage2_Applicable col
gold_app_col = pick_col_robust(gold.columns, GOLD_STAGE2_APPLICABLE_COL_CANDIDATES)
if not gold_app_col:
    raise RuntimeError("Could not find Stage2_Applicable column in GOLD after header cleaning.")
gold["_Stage2_Applicable_"] = gold[gold_app_col].map(to_binary_or_missing)

# Attach MRN to cohort via bridge (1 row per patient_id)
bridge_slim = bridge[["_PID_", "_MRN_"]].drop_duplicates()
cohort2 = cohort.merge(bridge_slim, on="_PID_", how="left")

# Filter GOLD to Stage2 applicable
gold_scoring = gold[(gold["_MRN_"] != "") & (gold["_Stage2_Applicable_"] == 1)].copy()

# Join on MRN
joined = gold_scoring.merge(cohort2, on="_MRN_", how="left", suffixes=("_gold", "_pred"))

print("\nCounts:")
print("  GOLD rows:", int(len(gold)))
print("  GOLD Stage2_Applicable==1:", int((gold["_Stage2_Applicable_"] == 1).sum()))
print("  COHORT rows:", int(len(cohort2)))
print("  COHORT rows with MRN linked:", int((cohort2["_MRN_"].map(norm_str) != "").sum()))
print("  Joined rows for Stage2 scoring:", int(len(joined)))

# Resolve actual column names by canonical match (critical)
resolved_pairs = []
print("\nResolved Stage2 column matches (after cleaning + canonical matching):")
for gold_target, pred_target in STAGE2_VAR_TARGETS:
    gold_actual = find_col_by_target(joined.columns, gold_target)
    pred_actual = find_col_by_target(joined.columns, pred_target)
    print("  GOLD target '{}' -> {}".format(gold_target, gold_actual if gold_actual else "(NOT FOUND)"))
    print("  PRED target '{}' -> {}".format(pred_target, pred_actual if pred_actual else "(NOT FOUND)"))
    resolved_pairs.append((gold_target, pred_target, gold_actual, pred_actual))

# Validate each variable
results = []
pair_rows = []

for gold_target, pred_target, gold_col, pred_col in resolved_pairs:
    gold_present = 1 if gold_col else 0
    pred_present = 1 if pred_col else 0

    row = {
        "var": pred_target,
        "gold_col": gold_target,
        "pred_col": pred_target,
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

    if not gold_col or not pred_col:
        row["status"] = "SKIP_missing_column"
        results.append(row)
        continue

    gbin = joined[gold_col].map(to_binary_or_missing)
    pbin = joined[pred_col].map(to_binary_or_missing)

    g_miss = int(gbin.isna().sum())
    p_miss = int(pbin.isna().sum())

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

    # Row-level audit for scorable rows
    tmp = joined.loc[scorable_mask, ["_MRN_", "_PID_", gold_col, pred_col]].copy()
    tmp = tmp.rename(columns={
        "_MRN_": "MRN",
        "_PID_": "patient_id",
        gold_col: gold_target + "_gold",
        pred_col: pred_target + "_pred",
    })
    tmp["var"] = pred_target
    pair_rows.append(tmp)

# Write outputs
out_conf = pd.DataFrame(results)
out_conf.to_csv(OUT_CONFUSION, index=False, encoding="utf-8")
print("\nWrote:", OUT_CONFUSION)

if pair_rows:
    out_pairs = pd.concat(pair_rows, axis=0, ignore_index=True)
    out_pairs.to_csv(OUT_PAIRWISE, index=False, encoding="utf-8")
    print("Wrote:", OUT_PAIRWISE, "(rows={})".format(out_pairs.shape[0]))
else:
    print("NOTE: No pairwise rows written (no scorable rows or all missing columns).")

print("\nDone.\n")
