#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate stage predictions vs gold.

Inputs (default locations):
- ./gold_cleaned_for_cedar.csv
- ./outputs/patient_stage_summary.csv
- Op notes CSV containing both MRN and ENCRYPTED_PAT_ID (auto-discovered)

Outputs:
- ./outputs/validation_merged.csv
- ./outputs/validation_mismatches.csv
- ./outputs/validation_metrics.txt
"""

from __future__ import print_function
import os
import sys
import glob
import pandas as pd


def _find_op_notes_csv():
    """
    Try common locations + name patterns.
    Adjust patterns if your filenames differ.
    """
    candidates = []

    # Common spots in your workflow
    candidates += glob.glob(os.path.join(".", "_staging_inputs", "*Operation Notes*.csv"))
    candidates += glob.glob(os.path.join(".", "_staging_inputs", "*Operation_Notes*.csv"))
    candidates += glob.glob(os.path.join(".", "*Operation Notes*.csv"))
    candidates += glob.glob(os.path.join(".", "*Operation_Notes*.csv"))

    # Also try nested data folders if you copied them into repo
    candidates += glob.glob(os.path.join(".", "**", "*Operation Notes*.csv"), recursive=True)
    candidates += glob.glob(os.path.join(".", "**", "*Operation_Notes*.csv"), recursive=True)

    # de-dupe while preserving order
    seen = set()
    uniq = []
    for c in candidates:
        c_abs = os.path.abspath(c)
        if c_abs not in seen and os.path.isfile(c_abs):
            uniq.append(c_abs)
            seen.add(c_abs)

    if not uniq:
        return None

    # Prefer a file inside _staging_inputs if available
    for p in uniq:
        if os.sep + "_staging_inputs" + os.sep in p:
            return p

    # Otherwise return first
    return uniq[0]


def _require_cols(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "{} is missing required columns: {}. Found columns: {}".format(
                label, missing, list(df.columns)[:50]
            )
        )


def main():
    root = os.path.abspath(".")
    gold_path = os.path.join(root, "gold_cleaned_for_cedar.csv")
    pred_path = os.path.join(root, "_outputs", "patient_stage_summary.csv")

    if not os.path.isfile(gold_path):
        raise IOError("Gold CSV not found at: {}".format(gold_path))
    if not os.path.isfile(pred_path):
        raise IOError("Prediction summary not found at: {}".format(pred_path))

    op_path = _find_op_notes_csv()
    if not op_path:
        raise IOError(
            "Could not auto-find Op Notes CSV. Put/copy it into ./_staging_inputs/ "
            "or rename it to include 'Operation Notes'."
        )

    # Load
    gold = pd.read_csv(gold_path, dtype=str, low_memory=False)
    pred = pd.read_csv(pred_path, dtype=str, low_memory=False)
    op = pd.read_csv(op_path, dtype=str, low_memory=False)

    # Required columns
    _require_cols(op, ["MRN", "ENCRYPTED_PAT_ID"], "Op Notes")
    _require_cols(pred, ["ENCRYPTED_PAT_ID"], "patient_stage_summary")
    _require_cols(gold, ["MRN"], "gold_cleaned_for_cedar")

    # Normalize IDs
    def norm(s):
        if s is None:
            return s
        return str(s).strip()

    for df, col in [(op, "MRN"), (op, "ENCRYPTED_PAT_ID"), (pred, "ENCRYPTED_PAT_ID"), (gold, "MRN")]:
        df[col] = df[col].map(norm)

    # Build crosswalk (one MRN per encrypted id; if multiple, keep the most frequent)
    x = op[["ENCRYPTED_PAT_ID", "MRN"]].dropna()
    x = x[(x["ENCRYPTED_PAT_ID"] != "") & (x["MRN"] != "")]
    # Resolve potential many-to-many by choosing the modal MRN per ENCRYPTED_PAT_ID
    counts = x.groupby(["ENCRYPTED_PAT_ID", "MRN"]).size().reset_index(name="n")
    counts = counts.sort_values(["ENCRYPTED_PAT_ID", "n"], ascending=[True, False])
    id_map = counts.drop_duplicates(subset=["ENCRYPTED_PAT_ID"], keep="first")[["ENCRYPTED_PAT_ID", "MRN"]]

    # Attach MRN to predictions
    pred_m = pred.merge(id_map, on="ENCRYPTED_PAT_ID", how="left")

    # Coerce pipeline prediction flags
    # HAS_STAGE2 is already in your summary; if missing, derive from STAGE2_DATE presence
    if "HAS_STAGE2" not in pred_m.columns:
        pred_m["HAS_STAGE2"] = pred_m["STAGE2_DATE"].notna().astype(int).astype(str)

    # Gold label column selection
    # Prefer Stage2_Applicable if present; otherwise fail loudly so you choose the right column.
    gold_label_col = None
    for c in ["Stage2_Applicable", "STAGE2_APPLICABLE", "stage2_applicable"]:
        if c in gold.columns:
            gold_label_col = c
            break
    if gold_label_col is None:
        raise ValueError(
            "Could not find Stage2 label column in gold. Expected 'Stage2_Applicable' (case-insensitive variants). "
            "Please tell me the exact gold column that represents Stage 2 presence."
        )

    # Standardize gold label to 0/1
    def to01(v):
        if v is None:
            return 0
        s = str(v).strip().lower()
        if s in ["1", "y", "yes", "true", "t"]:
            return 1
        if s in ["0", "n", "no", "false", "f", ""]:
            return 0
        # if weird values, treat non-zero numeric as 1
        try:
            return 1 if float(s) != 0.0 else 0
        except Exception:
            return 0

    gold["GOLD_HAS_STAGE2"] = gold[gold_label_col].map(to01).astype(int)
    pred_m["PRED_HAS_STAGE2"] = pred_m["HAS_STAGE2"].map(to01).astype(int)

    # Merge gold ↔ predictions by MRN
    merged = gold.merge(
        pred_m,
        on="MRN",
        how="left",
        suffixes=("_GOLD", "_PRED")
    )

    # Confusion matrix (Stage2)
    # If no prediction row found for an MRN, treat as predicted 0
    merged["PRED_HAS_STAGE2"] = merged["PRED_HAS_STAGE2"].fillna(0).astype(int)

    tp = int(((merged["GOLD_HAS_STAGE2"] == 1) & (merged["PRED_HAS_STAGE2"] == 1)).sum())
    fp = int(((merged["GOLD_HAS_STAGE2"] == 0) & (merged["PRED_HAS_STAGE2"] == 1)).sum())
    fn = int(((merged["GOLD_HAS_STAGE2"] == 1) & (merged["PRED_HAS_STAGE2"] == 0)).sum())
    tn = int(((merged["GOLD_HAS_STAGE2"] == 0) & (merged["PRED_HAS_STAGE2"] == 0)).sum())

    # Derived metrics with safe division
    def safe_div(a, b):
        return float(a) / float(b) if b else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    out_dir = os.path.join(root, "outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    merged_path = os.path.join(out_dir, "validation_merged.csv")
    mism_path = os.path.join(out_dir, "validation_mismatches.csv")
    metrics_path = os.path.join(out_dir, "validation_metrics.txt")

    merged.to_csv(merged_path, index=False)

    mism = merged[merged["GOLD_HAS_STAGE2"] != merged["PRED_HAS_STAGE2"]].copy()
    # Keep a compact set of columns helpful for review
    keep = []
    for c in [
        "__excel_row__", "MRN", "PatientID", "DOB",
        "GOLD_HAS_STAGE2", gold_label_col,
        "PRED_HAS_STAGE2", "ENCRYPTED_PAT_ID",
        "STAGE2_DATE", "STAGE2_NOTE_ID", "STAGE2_NOTE_TYPE", "STAGE2_MATCH_PATTERN", "STAGE2_HITS",
        "STAGE1_DATE", "STAGE1_NOTE_ID", "STAGE1_NOTE_TYPE", "STAGE1_MATCH_PATTERN", "STAGE1_HITS"
    ]:
        if c in mism.columns and c not in keep:
            keep.append(c)
    if keep:
        mism = mism[keep]
    mism.to_csv(mism_path, index=False)

    with open(metrics_path, "w") as f:
        f.write("Gold file: {}\n".format(gold_path))
        f.write("Pred file: {}\n".format(pred_path))
        f.write("Op notes file used for ID map: {}\n".format(op_path))
        f.write("Gold Stage2 label column: {}\n\n".format(gold_label_col))
        f.write("STAGE2 confusion matrix:\n")
        f.write("TP: {}\nFP: {}\nFN: {}\nTN: {}\n\n".format(tp, fp, fn, tn))
        f.write("Precision: {:.4f}\nRecall: {:.4f}\nF1: {:.4f}\n".format(precision, recall, f1))
        f.write("\nMerged rows: {}\nMismatches: {}\n".format(len(merged), len(mism)))

    print("OK: wrote")
    print(" - {}".format(merged_path))
    print(" - {}".format(mism_path))
    print(" - {}".format(metrics_path))
    print("Stage2: TP={} FP={} FN={} TN={} | Precision={:.3f} Recall={:.3f} F1={:.3f}".format(
        tp, fp, fn, tn, precision, recall, f1
    ))


if __name__ == "__main__":
    main()
