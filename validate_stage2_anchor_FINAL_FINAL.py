#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_stage2_anchor_FIXED.py (Python 3.6.8 compatible)

Stage2 Anchor validation ONLY.

Run from: /home/apokol/Breast_Restore

Inputs (NO AUTO-DETECTION):
- ./gold_cleaned_for_cedar.csv
- ./_outputs/patient_stage_summary.csv
- ./_staging_inputs/HPI11526 Operation Notes.csv   (MRN<->ENCRYPTED_PAT_ID bridge)

Outputs:
- ./_outputs/validation_merged_STAGE2_ANCHOR_FIXED.csv
- ./_outputs/validation_mismatches_STAGE2_ANCHOR_FIXED.csv
- ./_outputs/validation_metrics_STAGE2_ANCHOR_FIXED.txt
"""

from __future__ import print_function
import os
import pandas as pd

# -------------------------
# Helpers
# -------------------------

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))

def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df

def normalize_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    if s.lower() == "nan":
        return ""
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit():
            return head
    return s

def normalize_mrn(x):
    return normalize_id(x)

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
    except Exception:
        return 0

def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def compute_binary_metrics(df, gold_col, pred_col):
    tp = int(((df[gold_col] == 1) & (df[pred_col] == 1)).sum())
    fp = int(((df[gold_col] == 0) & (df[pred_col] == 1)).sum())
    fn = int(((df[gold_col] == 1) & (df[pred_col] == 0)).sum())
    tn = int(((df[gold_col] == 0) & (df[pred_col] == 0)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    acc = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": acc
    }

# -------------------------
# Main
# -------------------------

def main():
    root = os.path.abspath(".")
    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    gold_path = os.path.join(root, "gold_cleaned_for_cedar.csv")
    stage_pred_path = os.path.join(root, "_outputs", "patient_stage_summary.csv")
    op_path = os.path.join(root, "_staging_inputs", "HPI11526 Operation Notes.csv")

    if not os.path.isfile(gold_path):
        raise IOError("Gold file not found: {}".format(gold_path))
    if not os.path.isfile(stage_pred_path):
        raise IOError("Stage prediction file not found: {}".format(stage_pred_path))
    if not os.path.isfile(op_path):
        raise IOError("Operation Notes CSV not found: {}".format(op_path))

    print("Using:")
    print("  Gold      :", gold_path)
    print("  Stage Pred:", stage_pred_path)
    print("  Op Notes  :", op_path)
    print("")

    gold = normalize_cols(read_csv_robust(gold_path, dtype=str, low_memory=False))
    stage_pred = normalize_cols(read_csv_robust(stage_pred_path, dtype=str, low_memory=False))
    op = normalize_cols(read_csv_robust(op_path, dtype=str, low_memory=False))

    # --- Build MRN <-> ENCRYPTED_PAT_ID map from op notes
    op_mrn_col = pick_first_existing(op, ["MRN", "mrn"])
    op_encpat_col = pick_first_existing(op, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not op_mrn_col or not op_encpat_col:
        raise ValueError("Op notes must contain MRN and ENCRYPTED_PAT_ID. Found: {}".format(list(op.columns)))

    op["MRN"] = op[op_mrn_col].map(normalize_mrn)
    op["ENCRYPTED_PAT_ID"] = op[op_encpat_col].map(normalize_id)
    id_map = op[["ENCRYPTED_PAT_ID", "MRN"]].drop_duplicates()
    id_map = id_map[(id_map["ENCRYPTED_PAT_ID"] != "") & (id_map["MRN"] != "")].copy()

    # --- Gold MRN + Stage2 applicable (gold label)
    gold_mrn_col = pick_first_existing(gold, ["MRN", "mrn"])
    if not gold_mrn_col:
        raise ValueError("Gold missing MRN column.")
    gold[gold_mrn_col] = gold[gold_mrn_col].map(normalize_mrn)

    gold_stage2_app_col = pick_first_existing(gold, ["Stage2_Applicable", "STAGE2_APPLICABLE"])
    if not gold_stage2_app_col:
        raise ValueError("Gold missing Stage2_Applicable (or STAGE2_APPLICABLE).")
    gold["GOLD_HAS_STAGE2"] = gold[gold_stage2_app_col].map(to01).astype(int)

    # --- Stage Pred: must have ENCRYPTED_PAT_ID and HAS_STAGE2 (or STAGE2_DATE)
    pred_enc_col = pick_first_existing(stage_pred, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not pred_enc_col:
        raise ValueError("Stage prediction file missing ENCRYPTED_PAT_ID. Found: {}".format(list(stage_pred.columns)))

    if pred_enc_col != "ENCRYPTED_PAT_ID":
        stage_pred = stage_pred.rename(columns={pred_enc_col: "ENCRYPTED_PAT_ID"})
    stage_pred["ENCRYPTED_PAT_ID"] = stage_pred["ENCRYPTED_PAT_ID"].map(normalize_id)

    # Map to MRN via op-notes id_map
    stage_pred = stage_pred.merge(id_map, on="ENCRYPTED_PAT_ID", how="left")
    stage_pred["MRN"] = stage_pred["MRN"].fillna("").map(normalize_mrn)

    # Pred stage2 signal
    if "HAS_STAGE2" in stage_pred.columns:
        stage_pred["PRED_HAS_STAGE2"] = stage_pred["HAS_STAGE2"].map(to01).astype(int)
    elif "STAGE2_DATE" in stage_pred.columns:
        stage_pred["PRED_HAS_STAGE2"] = stage_pred["STAGE2_DATE"].fillna("").map(lambda x: 1 if str(x).strip() else 0).astype(int)
    else:
        raise ValueError("Stage prediction file missing HAS_STAGE2 or STAGE2_DATE. Found: {}".format(list(stage_pred.columns)))

    # Collapse to one row per MRN
    stage_pred = stage_pred[stage_pred["MRN"] != ""].copy()
    stage_pred = stage_pred.groupby("MRN", as_index=False)["PRED_HAS_STAGE2"].max()

    # Merge gold + pred
    merged = gold.merge(
        stage_pred,
        left_on=gold_mrn_col,
        right_on="MRN",
        how="left",
        suffixes=("", "_pred")
    )
    merged["PRED_HAS_STAGE2"] = merged["PRED_HAS_STAGE2"].fillna(0).astype(int)

    # Metrics
    metrics = compute_binary_metrics(merged, "GOLD_HAS_STAGE2", "PRED_HAS_STAGE2")

    # Outputs
    merged_path = os.path.join(out_dir, "validation_merged_STAGE2_ANCHOR_FIXED.csv")
    mism_path = os.path.join(out_dir, "validation_mismatches_STAGE2_ANCHOR_FIXED.csv")
    metrics_path = os.path.join(out_dir, "validation_metrics_STAGE2_ANCHOR_FIXED.txt")

    merged.to_csv(merged_path, index=False)
    merged[merged["GOLD_HAS_STAGE2"] != merged["PRED_HAS_STAGE2"]].to_csv(mism_path, index=False)

    with open(metrics_path, "w") as f:
        f.write("=== Stage2 Anchor (Applicable) ===\n")
        f.write("TP: {TP}\nFP: {FP}\nFN: {FN}\nTN: {TN}\n".format(**metrics))
        f.write("Precision: {:.4f}\n".format(metrics["precision"]))
        f.write("Recall: {:.4f}\n".format(metrics["recall"]))
        f.write("F1: {:.4f}\n".format(metrics["f1"]))
        f.write("Accuracy: {:.4f}\n".format(metrics["accuracy"]))

    print("")
    print("Validation complete.")
    print("Stage2 Anchor:")
    print("  TP={TP} FP={FP} FN={FN} TN={TN}".format(**metrics))
    print("  Precision={:.3f} Recall={:.3f} F1={:.3f}".format(
        metrics["precision"], metrics["recall"], metrics["f1"]
    ))
    print("")
    print("Wrote:")
    print("  ", merged_path)
    print("  ", mism_path)
    print("  ", metrics_path)

if __name__ == "__main__":
    main()
