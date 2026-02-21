# evaluate_vs_gold.py
# Purpose: Compare prediction spine vs gold (patient-level) and print metrics + save error lists.
# Python 3.6+ (pandas)

from __future__ import print_function

import os
import re
import sys
import pandas as pd


# -----------------------
# EDIT THESE FILENAMES
# -----------------------
PRED_CSV = "pred_spine_stage1_stage2.csv"
GOLD_CSV = "gold_cleaned_for_cedar.csv"

OUT_METRICS = "eval_metrics.csv"
OUT_SUMMARY = "eval_summary.txt"
OUT_ERR_DIR = "eval_errors"


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def read_csv_safe(path):
    # try utf-8 first, then fallback latin1
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return pd.read_csv(path, dtype=str, encoding="latin1")


def find_col(df, candidates):
    # candidates: list of strings to match by normalized containment
    cols = list(df.columns)
    ncols = [(c, norm(c)) for c in cols]
    for cand in candidates:
        nc = norm(cand)
        # exact match first
        for c, n in ncols:
            if n == nc:
                return c
        # contains match
        for c, n in ncols:
            if nc in n:
                return c
    return None


def to01(series):
    # Convert common gold/pred values to 0/1 safely.
    # Treat NaN/blank as NaN (not 0) here; masking decides what to do.
    s = series.copy()

    # normalize strings
    s = s.fillna("").astype(str).str.strip()

    # empty -> NaN
    s = s.replace("", pd.NA)

    # common text forms
    s = s.replace({
        "nan": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "na": pd.NA,
        "n/a": pd.NA,
        "None": pd.NA,
        "none": pd.NA,
        "NULL": pd.NA,
        "null": pd.NA,
        "Yes": "1",
        "yes": "1",
        "Y": "1",
        "y": "1",
        "True": "1",
        "true": "1",
        "No": "0",
        "no": "0",
        "N": "0",
        "n": "0",
        "False": "0",
        "false": "0",
    })

    # numeric coercion
    out = pd.to_numeric(s, errors="coerce")

    # clamp weird values to 0/1 if they exist (e.g., 2 -> 1)
    out = out.apply(lambda x: pd.NA if pd.isnull(x) else (1 if x >= 1 else 0))
    return out


def confusion_counts(y_true, y_pred):
    # both are 0/1 numeric series (no NaN)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def safe_div(a, b):
    return (float(a) / float(b)) if b else 0.0


def main():
    if not os.path.exists(PRED_CSV):
        print("ERROR: Missing prediction file:", PRED_CSV)
        sys.exit(1)
    if not os.path.exists(GOLD_CSV):
        print("ERROR: Missing gold file:", GOLD_CSV)
        sys.exit(1)

    if not os.path.exists(OUT_ERR_DIR):
        os.makedirs(OUT_ERR_DIR)

    pred = read_csv_safe(PRED_CSV)
    gold = read_csv_safe(GOLD_CSV)

    # -----------------------
    # Detect patient_id columns
    # -----------------------
    pred_id_col = find_col(pred, ["patient_id"])
    if pred_id_col is None:
        print("ERROR: Could not find patient_id in prediction file.")
        print("Prediction columns:", list(pred.columns))
        sys.exit(1)

    gold_id_col = find_col(gold, ["patient_id", "patientid", "1. patientid", "patid", "encrypted_pat_id"])
    if gold_id_col is None:
        print("ERROR: Could not find PatientID column in gold file.")
        print("Gold columns:", list(gold.columns))
        sys.exit(1)

    pred[pred_id_col] = pred[pred_id_col].fillna("").astype(str).str.strip()
    gold[gold_id_col] = gold[gold_id_col].fillna("").astype(str).str.strip()

    # drop blank ids
    pred = pred[pred[pred_id_col] != ""].copy()
    gold = gold[gold[gold_id_col] != ""].copy()

    # -----------------------
    # Define endpoints
    #   pred_col is fixed (your spine)
    #   gold col is auto-detected by scanning
    # -----------------------
    endpoints = [
        # Stage 1
        ("Stage1_MinorComp", "Stage1_MinorComp_pred", ["stage1_minorcomp", "35 stage1 minorcomp", "35. stage1 minorcomp"]),
        ("Stage1_MajorComp", "Stage1_MajorComp_pred", ["stage1_majorcomp", "37 stage1 majorcomp", "37. stage1 majorcomp"]),
        ("Stage1_Reoperation", "Stage1_Reoperation_pred", ["stage1_reoperation", "36 stage1 reoperation", "36. stage1 reoperation"]),
        ("Stage1_Rehospitalization", "Stage1_Rehospitalization_pred", ["stage1_rehospitalization", "37 stage1 rehospitalization", "37. stage1 rehospitalization"]),

        # Stage 2 (pred spine has these names)
        ("Stage2_MinorComp", "Stage2_MinorComp", ["stage2_minorcomp", "40 stage2 minorcomp", "40. stage2 minorcomp"]),
        ("Stage2_MajorComp", "Stage2_MajorComp", ["stage2_majorcomp", "43 stage2 majorcomp", "43. stage2 majorcomp"]),
        ("Stage2_Reoperation", "Stage2_Reoperation", ["stage2_reoperation", "41 stage2 reoperation", "41. stage2 reoperation"]),
        ("Stage2_Rehospitalization", "Stage2_Rehospitalization", ["stage2_rehospitalization", "42 stage2 rehospitalization", "42. stage2 rehospitalization"]),
        ("Stage2_Failure", "Stage2_Failure", ["stage2_failure", "44 stage2 failure", "44. stage2 failure"]),
        ("Stage2_Revision", "Stage2_Revision", ["stage2_revision", "45 stage2 revision", "45. stage2 revision"]),
    ]

    # detect Stage2_Applicable in gold if present
    gold_stage2_app_col = find_col(gold, ["stage2_applicable", "stage2 applicable", "stage2applicable"])

    # Build merged table
    merged = pred.merge(
        gold,
        left_on=pred_id_col,
        right_on=gold_id_col,
        how="inner",
        suffixes=("_pred", "_gold")
    )

    n_pred = int(pred[pred_id_col].nunique())
    n_gold = int(gold[gold_id_col].nunique())
    n_merged = int(merged[pred_id_col].nunique())

    lines = []
    lines.append("=== Evaluation: Predictions vs Gold (patient-level) ===")
    lines.append("Pred file: {}".format(PRED_CSV))
    lines.append("Gold file: {}".format(GOLD_CSV))
    lines.append("Pred patients: {}".format(n_pred))
    lines.append("Gold patients: {}".format(n_gold))
    lines.append("Patients in intersection (evaluated): {}".format(n_merged))
    lines.append("")
    lines.append("ID columns:")
    lines.append("  pred: {}".format(pred_id_col))
    lines.append("  gold: {}".format(gold_id_col))
    if gold_stage2_app_col:
        lines.append("Gold Stage2 applicable column detected: {}".format(gold_stage2_app_col))
    else:
        lines.append("Gold Stage2 applicable column detected: (none)")
    lines.append("")

    metrics_rows = []

    # -----------------------
    # Evaluate each endpoint
    # -----------------------
    for name, pred_col, gold_candidates in endpoints:
        if pred_col not in merged.columns:
            lines.append("[SKIP] {}: missing pred column '{}'".format(name, pred_col))
            continue

        gold_col = find_col(merged, gold_candidates)
        if gold_col is None:
            lines.append("[SKIP] {}: could not find gold column (candidates: {})".format(name, gold_candidates))
            continue

        # mask for stage2 applicability if stage2 endpoint
        is_stage2 = name.startswith("Stage2_")
        mask = pd.Series([True] * len(merged))

        if is_stage2:
            if gold_stage2_app_col and gold_stage2_app_col in merged.columns:
                app = to01(merged[gold_stage2_app_col])
                # treat missing as not applicable
                mask = (app.fillna(0) == 1)
            else:
                # fallback: applicable if ANY stage2 gold fields for this endpoint is non-null
                # (most conservative: require this endpoint's gold to be non-null)
                gtmp = to01(merged[gold_col])
                mask = gtmp.notnull()

        # pull truth/preds
        y_pred_raw = to01(merged[pred_col])
        y_true_raw = to01(merged[gold_col])

        # apply mask
        y_pred = y_pred_raw[mask].copy()
        y_true = y_true_raw[mask].copy()

        # Keep only rows where gold is known (avoid scoring on missing gold)
        known = y_true.notnull()
        y_pred = y_pred[known].fillna(0)  # if pred missing, treat as 0
        y_true = y_true[known]

        n_eval = int(len(y_true))

        if n_eval == 0:
            lines.append("[SKIP] {}: no evaluable rows after masking (gold missing or not applicable)".format(name))
            continue

        tp, tn, fp, fn = confusion_counts(y_true, y_pred)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, (precision + recall)) if (precision + recall) else 0.0
        acc = safe_div(tp + tn, tp + tn + fp + fn)

        # Save FP/FN lists
        sub = merged.loc[y_true.index, [pred_id_col, pred_col, gold_col]].copy()
        sub = sub.rename(columns={pred_id_col: "patient_id", pred_col: "pred", gold_col: "gold"})
        sub["pred01"] = y_pred.values
        sub["gold01"] = y_true.values

        fp_df = sub[(sub["gold01"] == 0) & (sub["pred01"] == 1)].copy()
        fn_df = sub[(sub["gold01"] == 1) & (sub["pred01"] == 0)].copy()

        fp_path = os.path.join(OUT_ERR_DIR, "eval_fp_{}.csv".format(name))
        fn_path = os.path.join(OUT_ERR_DIR, "eval_fn_{}.csv".format(name))
        fp_df.to_csv(fp_path, index=False, encoding="utf-8")
        fn_df.to_csv(fn_path, index=False, encoding="utf-8")

        metrics_rows.append({
            "endpoint": name,
            "pred_col": pred_col,
            "gold_col": gold_col,
            "n_eval": n_eval,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "fp_file": fp_path,
            "fn_file": fn_path,
        })

        lines.append("Endpoint: {}".format(name))
        lines.append("  pred col: {}".format(pred_col))
        lines.append("  gold col: {}".format(gold_col))
        if is_stage2:
            lines.append("  Stage2 masking applied: {}".format("YES" if is_stage2 else "NO"))
        lines.append("  n_eval: {}".format(n_eval))
        lines.append("  TP={}  FP={}  FN={}  TN={}".format(tp, fp, fn, tn))
        lines.append("  Precision={:.3f}  Recall={:.3f}  F1={:.3f}  Acc={:.3f}".format(precision, recall, f1, acc))
        lines.append("  Wrote: {} (FP), {} (FN)".format(fp_path, fn_path))
        lines.append("")

    # Write metrics CSV + summary txt
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(OUT_METRICS, index=False, encoding="utf-8")
        lines.append("Wrote metrics: {}".format(OUT_METRICS))
        lines.append("Wrote error lists folder: {}".format(OUT_ERR_DIR))
    else:
        lines.append("No endpoints evaluated (missing columns or no overlap).")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
