#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VALIDATE STAGE PIPELINE AGAINST GOLD
Python 3.6.8 compatible
"""

from __future__ import print_function
import os
import glob
import pandas as pd


# --------------------------------------------------
# Robust CSV reader (fixes UTF-8 / cp1252 issues)
# --------------------------------------------------

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))


# --------------------------------------------------
# Auto-find prediction summary
# --------------------------------------------------

def find_prediction_csv(root):
    candidates = []
    candidates += glob.glob(os.path.join(root, "_outputs", "*stage*summary*.csv"))
    candidates += glob.glob(os.path.join(root, "_outputs", "*patient*stage*.csv"))
    candidates += glob.glob(os.path.join(root, "**", "*stage*summary*.csv"), recursive=True)

    seen = set()
    uniq = []
    for c in candidates:
        if os.path.isfile(c):
            ab = os.path.abspath(c)
            if ab not in seen:
                uniq.append(ab)
                seen.add(ab)

    if not uniq:
        return None

    uniq.sort(key=lambda x: len(x))
    return uniq[0]


# --------------------------------------------------
# Auto-find op notes file
# --------------------------------------------------

def find_op_notes_csv(root):
    candidates = []
    candidates += glob.glob(os.path.join(root, "_staging_inputs", "*Operation Notes*.csv"))
    candidates += glob.glob(os.path.join(root, "**", "*Operation Notes*.csv"), recursive=True)

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    return None


# --------------------------------------------------
# Utility
# --------------------------------------------------

def normalize_id(x):
    if x is None:
        return ""
    return str(x).strip()


def to01(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except:
        return 0


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    root = os.path.abspath(".")

    gold_path = os.path.join(root, "gold_cleaned_for_cedar.csv")
    pred_path = find_prediction_csv(root)
    op_path = find_op_notes_csv(root)

    if not os.path.isfile(gold_path):
        raise IOError("Gold file not found: {}".format(gold_path))

    if not pred_path:
        raise IOError("Prediction summary not found under _outputs/")

    if not op_path:
        raise IOError("Operation Notes CSV not found.")

    print("Using:")
    print("  Gold:", gold_path)
    print("  Pred:", pred_path)
    print("  Op  :", op_path)
    print("")

    gold = read_csv_robust(gold_path, dtype=str, low_memory=False)
    pred = read_csv_robust(pred_path, dtype=str, low_memory=False)
    op = read_csv_robust(op_path, dtype=str, low_memory=False)

    # Clean non-breaking spaces
    gold.columns = [c.replace(u"\xa0", " ").strip() for c in gold.columns]
    pred.columns = [c.replace(u"\xa0", " ").strip() for c in pred.columns]
    op.columns = [c.replace(u"\xa0", " ").strip() for c in op.columns]

    # Required columns
    if "MRN" not in op.columns or "ENCRYPTED_PAT_ID" not in op.columns:
        raise ValueError("Op Notes must contain MRN and ENCRYPTED_PAT_ID")

    if "ENCRYPTED_PAT_ID" not in pred.columns:
        raise ValueError("Prediction summary missing ENCRYPTED_PAT_ID")

    if "MRN" not in gold.columns:
        raise ValueError("Gold file missing MRN")

    # Normalize IDs
    op["MRN"] = op["MRN"].map(normalize_id)
    op["ENCRYPTED_PAT_ID"] = op["ENCRYPTED_PAT_ID"].map(normalize_id)
    gold["MRN"] = gold["MRN"].map(normalize_id)
    pred["ENCRYPTED_PAT_ID"] = pred["ENCRYPTED_PAT_ID"].map(normalize_id)

    # Build ID map
    id_map = op[["ENCRYPTED_PAT_ID", "MRN"]].dropna()
    id_map = id_map.drop_duplicates()

    # Attach MRN to predictions
    pred = pred.merge(id_map, on="ENCRYPTED_PAT_ID", how="left")

    # Determine gold Stage2 label column
    gold_label = None
    for c in ["Stage2_Applicable", "STAGE2_APPLICABLE"]:
        if c in gold.columns:
            gold_label = c
            break

    if not gold_label:
        raise ValueError("Could not find Stage2_Applicable column in gold.")

    gold["GOLD_HAS_STAGE2"] = gold[gold_label].map(to01).astype(int)

    if "HAS_STAGE2" in pred.columns:
        pred["PRED_HAS_STAGE2"] = pred["HAS_STAGE2"].map(to01).astype(int)
    else:
        pred["PRED_HAS_STAGE2"] = pred["STAGE2_DATE"].notna().astype(int)

    # Merge
    merged = gold.merge(pred, on="MRN", how="left")

    merged["PRED_HAS_STAGE2"] = merged["PRED_HAS_STAGE2"].fillna(0).astype(int)

    tp = int(((merged["GOLD_HAS_STAGE2"] == 1) & (merged["PRED_HAS_STAGE2"] == 1)).sum())
    fp = int(((merged["GOLD_HAS_STAGE2"] == 0) & (merged["PRED_HAS_STAGE2"] == 1)).sum())
    fn = int(((merged["GOLD_HAS_STAGE2"] == 1) & (merged["PRED_HAS_STAGE2"] == 0)).sum())
    tn = int(((merged["GOLD_HAS_STAGE2"] == 0) & (merged["PRED_HAS_STAGE2"] == 0)).sum())

    def safe_div(a, b):
        return float(a) / float(b) if b else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    merged_path = os.path.join(out_dir, "validation_merged.csv")
    mism_path = os.path.join(out_dir, "validation_mismatches.csv")
    metrics_path = os.path.join(out_dir, "validation_metrics.txt")

    merged.to_csv(merged_path, index=False)

    mism = merged[merged["GOLD_HAS_STAGE2"] != merged["PRED_HAS_STAGE2"]]
    mism.to_csv(mism_path, index=False)

    with open(metrics_path, "w") as f:
        f.write("TP: {}\nFP: {}\nFN: {}\nTN: {}\n".format(tp, fp, fn, tn))
        f.write("Precision: {:.4f}\n".format(precision))
        f.write("Recall: {:.4f}\n".format(recall))
        f.write("F1: {:.4f}\n".format(f1))

    print("")
    print("Validation complete.")
    print("TP={} FP={} FN={} TN={}".format(tp, fp, fn, tn))
    print("Precision={:.3f} Recall={:.3f} F1={:.3f}".format(precision, recall, f1))
    print("")
    print("Files written to:", out_dir)


if __name__ == "__main__":
    main()
