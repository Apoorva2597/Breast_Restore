#!/usr/bin/env python3
# validate_stage2_full_cohort_against_gold.py
# Python 3.6.8 compatible
#
# Goal:
#   Validate Stage2 outcomes in your FULL cohort abstraction against GOLD,
#   joining GOLD->BRIDGE->COHORT deterministically (MRN -> patient_id -> cohort row).
#
# Key behavior (fixes + guardrails):
#   1) Resolve GOLD Stage2 columns from gold.csv only (no suffix confusion).
#   2) Resolve PRED Stage2 columns from cohort.csv only (supports *_pred).
#   3) After merge, handle suffixes explicitly when they occur (_gold / _pred).
#   4) Produce BOTH:
#        A) Coverage metrics (how often pred has values)
#        B) Accuracy conditional on pred having a value (recommended)
#      And optionally:
#        C) Accuracy if missing pred is treated as 0 (legacy behavior)
#
# Inputs (stored in /home/apokol/Breast_Restore per your layout):
#   - gold_cleaned_for_cedar.csv
#   - cohort_all_patient_level_final_gold_order.csv
#   - cohort_pid_to_mrn_from_encounters.csv
#
# Outputs (written next to this script by default):
#   - stage2_validation_coverage.csv
#   - stage2_validation_confusion_by_var_conditional.csv
#   - stage2_validation_confusion_by_var_missing_as_zero.csv   (optional)
#   - stage2_validation_pairs_stage2.csv
#   - stage2_validation_summary.txt
#
# Usage:
#   cd /home/apokol/Breast_Restore
#   python validate_stage2_full_cohort_against_gold.py
#
# Optional (create the extra “missing pred -> 0” report):
#   python validate_stage2_full_cohort_against_gold.py --missing_pred_as_zero

from __future__ import print_function
import os
import re
import argparse
import pandas as pd


# -------------------------
# CONFIG
# -------------------------
# Default to "same folder as this script" so you can run from anywhere safely.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# NOTE: filenames must match what's actually in /home/apokol/Breast_Restore
# (Your screenshot shows: cohort_all_patient_level_final_gold_order.csv and gold_cleaned_for_cedar.csv)
GOLD_CSV   = os.path.join(SCRIPT_DIR, "gold_cleaned_for_cedar.csv")
COHORT_CSV = os.path.join(SCRIPT_DIR, "cohort_all_patient_level_final_gold_order.csv")
BRIDGE_CSV = os.path.join(SCRIPT_DIR, "cohort_pid_to_mrn_from_encounters.csv")

OUT_COVER      = os.path.join(SCRIPT_DIR, "stage2_validation_coverage.csv")
OUT_BYVAR_COND = os.path.join(SCRIPT_DIR, "stage2_validation_confusion_by_var_conditional.csv")
OUT_BYVAR_MISS0 = os.path.join(SCRIPT_DIR, "stage2_validation_confusion_by_var_missing_as_zero.csv")
OUT_PAIRS      = os.path.join(SCRIPT_DIR, "stage2_validation_pairs_stage2.csv")
OUT_SUMMARY    = os.path.join(SCRIPT_DIR, "stage2_validation_summary.txt")

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
    "Stage2_MinorComp":         ["Stage2 MinorComp", "Stage2_MinorComp", "Stage2-MinorComp"],
    "Stage2_MajorComp":         ["Stage2 MajorComp", "Stage2_MajorComp", "Stage2-MajorComp"],
    "Stage2_Reoperation":       ["Stage2 Reoperation", "Stage2_Reoperation", "Stage2-Reoperation"],
    "Stage2_Rehospitalization": ["Stage2 Rehospitalization", "Stage2_Rehospitalization", "Stage2-Rehospitalization",
                                 "Stage2 Rehospitalisation", "Stage2_Rehospitalisation"],
    "Stage2_Failure":           ["Stage2 Failure", "Stage2_Failure", "Stage2-Failure"],
    "Stage2_Revision":          ["Stage2 Revision", "Stage2_Revision", "Stage2-Revision"],
}

# COHORT often uses "_pred" suffix (e.g., Stage2_MinorComp_pred) but we support both.
PRED_ALIASES = {
    "Stage2_MinorComp":         ["Stage2_MinorComp_pred", "Stage2_MinorComp", "Stage2 MinorComp"],
    "Stage2_MajorComp":         ["Stage2_MajorComp_pred", "Stage2_MajorComp", "Stage2 MajorComp"],
    "Stage2_Reoperation":       ["Stage2_Reoperation_pred", "Stage2_Reoperation", "Stage2 Reoperation"],
    "Stage2_Rehospitalization": ["Stage2_Rehospitalization_pred", "Stage2_Rehospitalization", "Stage2 Rehospitalization",
                                 "Stage2_Rehospitalisation_pred", "Stage2_Rehospitalisation", "Stage2 Rehospitalisation"],
    "Stage2_Failure":           ["Stage2_Failure_pred", "Stage2_Failure", "Stage2 Failure"],
    "Stage2_Revision":          ["Stage2_Revision_pred", "Stage2_Revision", "Stage2 Revision"],
}

# GOLD applicability flag (must be 1 for Stage2 scoring)
GOLD_STAGE2_APPLICABLE_COL_CANDIDATES = ["Stage2_Applicable", "Stage2 Applicable", "stage2_applicable"]

# Optional “coverage” fields in cohort (we will report coverage if present)
COHORT_STAGE2_COVERAGE_FIELDS = [
    "stage2_confirmed_flag",
    "stage2_date_final",
    "stage2_date",
    "Stage2_Confirmed",
    "Stage2_confirmed_flag",
]


# -------------------------
# Helpers
# -------------------------
def read_csv_safe(path, nrows=None):
    # Your environment + files sometimes have odd encodings; latin1 + replace is safest.
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", dtype=object, nrows=nrows)
    finally:
        try:
            f.close()
        except Exception:
            pass

def norm_colname(c):
    # canonicalize for matching: keep alnum+underscore only, lower, remove whitespace/punct
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

def normalize_mrn(x):
    # Normalize MRN for stable joins (strip, remove trailing .0)
    if x is None:
        return ""
    t = str(x).strip()
    if t.endswith(".0"):
        t = t[:-2]
    return t

def is_blank(x):
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    t = str(x).strip()
    if t == "":
        return True
    if t.lower() in ("nan", "none", "null", "na", "n/a", ".", "-", "--"):
        return True
    return False

def to01_series_allow_missing(s, missing_to_zero=False):
    """
    Convert a series to 0/1 for scoring.

    missing_to_zero=False:
      - blanks/NA -> NA

    missing_to_zero=True:
      - blanks/NA -> 0

    Accepts: 1/0, True/False, yes/no, y/n, positive/negative, performed/denied, present/absent.
    """
    if s is None:
        return pd.Series([], dtype="float64")

    def _conv(x):
        if is_blank(x):
            return 0 if missing_to_zero else None

        # ints/bools
        if isinstance(x, (int, bool)):
            return 1 if int(x) == 1 else 0

        t = str(x).strip().lower()

        if t in ("1", "true", "t", "yes", "y", "positive", "pos", "performed", "present"):
            return 1
        if t in ("0", "false", "f", "no", "n", "negative", "neg", "denied", "absent"):
            return 0

        # last resort: detect standalone 1 or 0 in text
        if re.search(r"(^|[^0-9])1([^0-9]|$)", t):
            return 1
        if re.search(r"(^|[^0-9])0([^0-9]|$)", t):
            return 0

        # unknown token:
        return 0 if missing_to_zero else None

    out = s.map(_conv)
    return out.astype("float64")

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

def pick_joined_col(joined_cols, base_col, side_suffix):
    """
    Find the actual column name in 'joined' for base_col, accounting for suffixes.
    side_suffix is expected to be '_gold' or '_pred' depending on merge.
    """
    if base_col is None:
        return None
    if base_col in joined_cols:
        return base_col
    cand = base_col + side_suffix
    if cand in joined_cols:
        return cand
    return None

def resolve_cols_in_joined(joined, cols_from_df, prefer_suffix):
    """
    Given a list of column names from an original dataframe (e.g., cohort.columns),
    return the list of *actual* column names present in joined after merge.

    prefer_suffix should be "_pred" for cohort-side, "_gold" for gold-side.

    Rules:
      - if col exists in joined, keep it
      - else if col+prefer_suffix exists in joined, use that
      - else skip
    """
    out = []
    joined_cols = set(joined.columns)
    for c in cols_from_df:
        if c in joined_cols:
            out.append(c)
        elif (c + prefer_suffix) in joined_cols:
            out.append(c + prefer_suffix)
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=GOLD_CSV)
    ap.add_argument("--cohort", default=COHORT_CSV)
    ap.add_argument("--bridge", default=BRIDGE_CSV)
    ap.add_argument("--missing_pred_as_zero", action="store_true",
                    help="Also compute a legacy accuracy table where missing pred values are treated as 0.")
    args = ap.parse_args()

    print("\n=== VALIDATION: STAGE 2 (FULL COHORT vs GOLD via bridge) ===")
    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("GOLD  :", args.gold)
    print("COHORT:", args.cohort)
    print("BRIDGE:", args.bridge)

    for p in (args.gold, args.cohort, args.bridge):
        if not os.path.exists(p):
            raise RuntimeError("Missing input file: {}".format(p))

    gold = read_csv_safe(args.gold)
    cohort = read_csv_safe(args.cohort)
    bridge = read_csv_safe(args.bridge)

    # detect key columns
    gold_mrn_col = resolve_column(gold.columns, ["MRN", "mrn"])
    bridge_pid_col = resolve_column(bridge.columns, ["patient_id", "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "PATIENT_ID", "PAT_ID"])
    bridge_mrn_col = resolve_column(bridge.columns, ["MRN", "mrn"])
    cohort_pid_col = resolve_column(cohort.columns, ["patient_id", "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "PATIENT_ID", "PAT_ID"])

    print("\nDetected key columns:")
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
    gold["_MRN_"] = gold[gold_mrn_col].map(normalize_mrn)
    bridge["_MRN_"] = bridge[bridge_mrn_col].map(normalize_mrn)
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

    gold_app01 = to01_series_allow_missing(gold[gold_stage2_app_col], missing_to_zero=True)

    print("\nCounts:")
    print("  GOLD rows                 :", int(len(gold)))
    print("  GOLD Stage2_Applicable==1 :", int((gold_app01 == 1).sum()))
    print("  COHORT rows               :", int(len(cohort)))
    print("  BRIDGE rows               :", int(len(bridge)))

    print("\nBridge ambiguity checks:")
    print("  patient_id -> multiple MRNs:", pid_multi_mrn)
    print("  MRN -> multiple patient_ids:", mrn_multi_pid)
    if pid_multi_mrn or mrn_multi_pid:
        print("  WARNING: Bridge has ambiguity. Validation may be unreliable until resolved.")

    # de-duplicate bridge
    bridge_dedup = bridge.drop_duplicates(subset=["_PID_"], keep="first").copy()
    bridge_mrn_dedup = bridge_dedup.drop_duplicates(subset=["_MRN_"], keep="first").copy()

    # Restrict GOLD to Stage2 applicable only
    gold_app = gold[(gold_app01 == 1)].copy()

    # Join: GOLD(MRN) -> BRIDGE(MRN->PID) -> COHORT(PID)
    j1 = gold_app.merge(bridge_mrn_dedup[["_MRN_", "_PID_"]], on="_MRN_", how="left", indicator=True)
    missing_pid = int((j1["_merge"] == "left_only").sum())
    j1 = j1.drop(columns=["_merge"])

    joined = j1.merge(cohort, on="_PID_", how="left", suffixes=("_gold", "_pred"), indicator=True)
    missing_cohort = int((joined["_merge"] == "left_only").sum())
    joined = joined.drop(columns=["_merge"])

    # -------------------------
    # Coverage: cohort merge success
    # -------------------------
    # IMPORTANT FIX:
    #   After merge, overlapping column names between GOLD and COHORT can be suffixed.
    #   So we must resolve what cohort columns are actually called inside `joined`.
    cohort_cols_excluding_pid = [c for c in cohort.columns if c not in ("_PID_",)]
    cohort_join_cols = resolve_cols_in_joined(joined, cohort_cols_excluding_pid, prefer_suffix="_pred")

    if cohort_join_cols:
        has_any_cohort = joined[cohort_join_cols].notnull().any(axis=1)
    else:
        # fallback: at least a PID exists
        has_any_cohort = joined["_PID_"].notnull()

    n_has_cohort = int(has_any_cohort.sum())

    print("\nJoined rows for Stage2 scoring (gold Stage2_applicable only):", int(len(joined)))
    print("  missing PID via bridge:", missing_pid)
    print("  missing COHORT via PID:", missing_cohort)
    print("  with COHORT row linked (any non-null):", n_has_cohort)

    # Resolve GOLD and PRED Stage2 columns deterministically
    resolved = []
    print("\nResolved Stage2 column matches (source-based, deterministic):")
    for v in STAGE2_VARS:
        g_base = resolve_column(gold.columns, GOLD_ALIASES.get(v, [v]))
        p_base = resolve_column(cohort.columns, PRED_ALIASES.get(v, [v]))

        g_join = pick_joined_col(joined.columns, g_base, "_gold")
        p_join = pick_joined_col(joined.columns, p_base, "_pred")

        print("  {:<22} GOLD -> {:<28} | PRED -> {}".format(v, str(g_join), str(p_join)))
        resolved.append((v, g_base, p_base, g_join, p_join))

    # -------------------------
    # Coverage report
    # -------------------------
    cover_rows = []
    n_eval = int(len(joined))

    cover_rows.append({"metric": "gold_stage2_applicable_rows", "count": n_eval, "pct": 1.0})

    cover_rows.append({
        "metric": "has_pid_via_bridge",
        "count": int(joined["_PID_"].notnull().sum()),
        "pct": safe_div(int(joined["_PID_"].notnull().sum()), n_eval)
    })

    cover_rows.append({
        "metric": "has_any_cohort_row_nonnull",
        "count": n_has_cohort,
        "pct": safe_div(n_has_cohort, n_eval)
    })

    # Cohort coverage fields (optional)
    for f in COHORT_STAGE2_COVERAGE_FIELDS:
        base = resolve_column(cohort.columns, [f])
        if base is None:
            continue
        jcol = pick_joined_col(joined.columns, base, "_pred")
        if jcol is None:
            continue
        nonblank = joined[jcol].map(lambda x: not is_blank(x))
        cover_rows.append({
            "metric": "pred_nonblank:{}".format(jcol),
            "count": int(nonblank.sum()),
            "pct": safe_div(int(nonblank.sum()), n_eval)
        })

    # Outcome coverage (pred nonblank)
    for (v, g_base, p_base, g_col_join, p_col_join) in resolved:
        if p_col_join is None or p_col_join not in joined.columns:
            cover_rows.append({"metric": "pred_missing_outcome_col:{}".format(v), "count": 0, "pct": 0.0})
            continue
        nonblank = joined[p_col_join].map(lambda x: not is_blank(x))
        cover_rows.append({
            "metric": "pred_nonblank_outcome:{}".format(v),
            "count": int(nonblank.sum()),
            "pct": safe_div(int(nonblank.sum()), n_eval)
        })

    cover_df = pd.DataFrame(cover_rows)
    cover_df.to_csv(OUT_COVER, index=False, encoding="utf-8")

    # -------------------------
    # Accuracy (conditional on pred coverage)
    # -------------------------
    rows_cond = []
    pair_rows = []

    for (v, g_base, p_base, g_col_join, p_col_join) in resolved:
        gold_present = 1 if (g_col_join is not None and g_col_join in joined.columns) else 0
        pred_present = 1 if (p_col_join is not None and p_col_join in joined.columns) else 0

        if not gold_present or not pred_present:
            rows_cond.append({
                "var": v,
                "gold_col": g_base if g_base else "",
                "pred_col": p_base if p_base else "",
                "status": "SKIP_missing_column",
                "gold_col_present": gold_present,
                "pred_col_present": pred_present,
                "n_eval_gold_stage2_applicable": n_eval,
                "n_scorable_conditional": 0,
                "gold_missing_raw": "",
                "pred_missing_raw": "",
                "TP": "", "FP": "", "FN": "", "TN": "",
                "sensitivity": "", "specificity": "", "ppv": "", "npv": ""
            })
            continue

        g_raw = joined[g_col_join]
        p_raw = joined[p_col_join]

        gold_missing = int(g_raw.map(is_blank).sum())
        pred_missing = int(p_raw.map(is_blank).sum())

        pred_has_value = (~p_raw.map(is_blank))
        mask = pred_has_value & has_any_cohort

        n_scorable = int(mask.sum())
        if n_scorable == 0:
            rows_cond.append({
                "var": v,
                "gold_col": g_col_join,
                "pred_col": p_col_join,
                "status": "SKIP_no_pred_coverage",
                "gold_col_present": gold_present,
                "pred_col_present": pred_present,
                "n_eval_gold_stage2_applicable": n_eval,
                "n_scorable_conditional": 0,
                "gold_missing_raw": gold_missing,
                "pred_missing_raw": pred_missing,
                "TP": "", "FP": "", "FN": "", "TN": "",
                "sensitivity": "", "specificity": "", "ppv": "", "npv": ""
            })
            continue

        g01 = to01_series_allow_missing(g_raw[mask], missing_to_zero=True)
        p01 = to01_series_allow_missing(p_raw[mask], missing_to_zero=True)

        g01 = g01.fillna(0).astype(int)
        p01 = p01.fillna(0).astype(int)

        tp, fp, fn, tn = confusion_counts(g01, p01)

        sens = safe_div(tp, (tp + fn))
        spec = safe_div(tn, (tn + fp))
        ppv  = safe_div(tp, (tp + fp))
        npv  = safe_div(tn, (tn + fn))

        rows_cond.append({
            "var": v,
            "gold_col": g_col_join,
            "pred_col": p_col_join,
            "status": "OK_conditional",
            "gold_col_present": gold_present,
            "pred_col_present": pred_present,
            "n_eval_gold_stage2_applicable": n_eval,
            "n_scorable_conditional": n_scorable,
            "gold_missing_raw": gold_missing,
            "pred_missing_raw": pred_missing,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv
        })

        tmp = pd.DataFrame({
            "MRN": joined.loc[mask, "_MRN_"],
            "patient_id": joined.loc[mask, "_PID_"],
            "var": v,
            "gold_raw": g_raw[mask],
            "pred_raw": p_raw[mask],
            "gold_01": g01.values,
            "pred_01": p01.values,
        })
        pair_rows.append(tmp)

    out_byvar_cond = pd.DataFrame(rows_cond)
    out_byvar_cond.to_csv(OUT_BYVAR_COND, index=False, encoding="utf-8")

    if pair_rows:
        out_pairs = pd.concat(pair_rows, axis=0, ignore_index=True)
    else:
        out_pairs = pd.DataFrame(columns=["MRN", "patient_id", "var", "gold_raw", "pred_raw", "gold_01", "pred_01"])
    out_pairs.to_csv(OUT_PAIRS, index=False, encoding="utf-8")

    # -------------------------
    # Optional: legacy scoring (missing pred -> 0)
    # -------------------------
    out_byvar_miss0 = None
    if args.missing_pred_as_zero:
        rows_m0 = []
        for (v, g_base, p_base, g_col_join, p_col_join) in resolved:
            gold_present = 1 if (g_col_join is not None and g_col_join in joined.columns) else 0
            pred_present = 1 if (p_col_join is not None and p_col_join in joined.columns) else 0

            if not gold_present or not pred_present:
                rows_m0.append({
                    "var": v,
                    "gold_col": g_base if g_base else "",
                    "pred_col": p_base if p_base else "",
                    "status": "SKIP_missing_column",
                    "gold_col_present": gold_present,
                    "pred_col_present": pred_present,
                    "n_eval_gold_stage2_applicable": n_eval,
                    "n_scorable_missing_as_zero": 0,
                    "TP": "", "FP": "", "FN": "", "TN": "",
                    "sensitivity": "", "specificity": "", "ppv": "", "npv": ""
                })
                continue

            g_raw = joined[g_col_join]
            p_raw = joined[p_col_join]

            mask = has_any_cohort
            n_scorable = int(mask.sum())
            if n_scorable == 0:
                rows_m0.append({
                    "var": v, "gold_col": g_col_join, "pred_col": p_col_join,
                    "status": "SKIP_no_cohort_rows",
                    "gold_col_present": gold_present, "pred_col_present": pred_present,
                    "n_eval_gold_stage2_applicable": n_eval,
                    "n_scorable_missing_as_zero": 0,
                    "TP": "", "FP": "", "FN": "", "TN": "",
                    "sensitivity": "", "specificity": "", "ppv": "", "npv": ""
                })
                continue

            g01 = to01_series_allow_missing(g_raw[mask], missing_to_zero=True).fillna(0).astype(int)
            p01 = to01_series_allow_missing(p_raw[mask], missing_to_zero=True).fillna(0).astype(int)

            tp, fp, fn, tn = confusion_counts(g01, p01)

            sens = safe_div(tp, (tp + fn))
            spec = safe_div(tn, (tn + fp))
            ppv  = safe_div(tp, (tp + fp))
            npv  = safe_div(tn, (tn + fn))

            rows_m0.append({
                "var": v,
                "gold_col": g_col_join,
                "pred_col": p_col_join,
                "status": "OK_missing_as_zero",
                "gold_col_present": gold_present,
                "pred_col_present": pred_present,
                "n_eval_gold_stage2_applicable": n_eval,
                "n_scorable_missing_as_zero": n_scorable,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv
            })

        out_byvar_miss0 = pd.DataFrame(rows_m0)
        out_byvar_miss0.to_csv(OUT_BYVAR_MISS0, index=False, encoding="utf-8")

    # -------------------------
    # Summary text
    # -------------------------
    with open(OUT_SUMMARY, "w") as f:
        f.write("Stage2 Validation Summary\n")
        f.write("=========================\n\n")
        f.write("Inputs:\n")
        f.write("  GOLD  : {}\n".format(args.gold))
        f.write("  COHORT: {}\n".format(args.cohort))
        f.write("  BRIDGE: {}\n\n".format(args.bridge))

        f.write("Eval set: GOLD Stage2_applicable == 1\n")
        f.write("  n_eval: {}\n".format(n_eval))
        f.write("  missing PID via bridge: {}\n".format(missing_pid))
        f.write("  missing COHORT via PID: {}\n".format(missing_cohort))
        f.write("  has cohort row (any non-null): {}\n\n".format(n_has_cohort))

        f.write("Outputs:\n")
        f.write("  Coverage              : {}\n".format(OUT_COVER))
        f.write("  Accuracy (conditional): {}\n".format(OUT_BYVAR_COND))
        f.write("  Pairs (conditional)   : {}\n".format(OUT_PAIRS))
        if args.missing_pred_as_zero:
            f.write("  Accuracy (miss->0)    : {}\n".format(OUT_BYVAR_MISS0))
        f.write("\n")

        f.write("Resolved columns:\n")
        for (v, g_base, p_base, g_col_join, p_col_join) in resolved:
            f.write("  {:<22} GOLD:{} | PRED:{}\n".format(v, str(g_col_join), str(p_col_join)))

    # -------------------------
    # Terminal prints
    # -------------------------
    print("\nWrote:", OUT_COVER)
    print("Wrote:", OUT_BYVAR_COND)
    print("Wrote:", OUT_PAIRS)
    if args.missing_pred_as_zero:
        print("Wrote:", OUT_BYVAR_MISS0)
    print("Wrote:", OUT_SUMMARY)

    print("\n=== Coverage (top) ===")
    print(cover_df.to_string(index=False))

    print("\n=== Stage2 validation summary (conditional; by var) ===")
    show_cols = ["var", "status", "n_eval_gold_stage2_applicable", "n_scorable_conditional",
                 "TP", "FP", "FN", "TN", "sensitivity", "specificity", "ppv", "npv"]
    show_cols = [c for c in show_cols if c in out_byvar_cond.columns]
    print(out_byvar_cond[show_cols].to_string(index=False))

    if args.missing_pred_as_zero and out_byvar_miss0 is not None:
        print("\n=== Stage2 validation summary (missing pred -> 0; by var) ===")
        show_cols2 = ["var", "status", "n_eval_gold_stage2_applicable", "n_scorable_missing_as_zero",
                      "TP", "FP", "FN", "TN", "sensitivity", "specificity", "ppv", "npv"]
        show_cols2 = [c for c in show_cols2 if c in out_byvar_miss0.columns]
        print(out_byvar_miss0[show_cols2].to_string(index=False))

    print("\nDone.\n")


if __name__ == "__main__":
    main()
