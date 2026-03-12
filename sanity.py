#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_complications_confusion_matrix.py

Purpose:
- Print confusion matrix counts for focused complication variables
- Use validation/QA CSV with gold + predicted columns
- No sklearn required

Focused fields:
- Stage1_MinorComp
- Stage1_Revision
- Stage2_MinorComp
- Stage2_Revision

Python 3.6.8 compatible.
"""

import os
import pandas as pd

INPUT_FILE = input("Enter path to validation CSV: ").strip()

FIELD_MAP = [
    {
        "label": "Stage1_MinorComp",
        "gold_col": "Stage1_MinorComp",
        "pred_col": "Stage1_MinorComp_pred",
    },
    {
        "label": "Stage1_Revision",
        "gold_col": "Stage1_Revision",
        "pred_col": "Stage1_Revision_pred",
    },
    {
        "label": "Stage2_MinorComp",
        "gold_col": "GOLD_Stage2_MinorComp",
        "pred_col": "Stage2_MinorComp_pred",
    },
    {
        "label": "Stage2_Revision",
        "gold_col": "GOLD_Stage2_Revision",
        "pred_col": "Stage2_Revision_pred",
    },
]


def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                error_bad_lines=False,
                warn_bad_lines=True
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                on_bad_lines="skip"
            )
        except TypeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


def to_binary_01(x):
    s = clean_cell(x).lower()
    if s in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    return 0


def compute_confusion(gold_series, pred_series):
    g = gold_series.apply(to_binary_01)
    p = pred_series.apply(to_binary_01)

    tn = int(((g == 0) & (p == 0)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())
    tp = int(((g == 1) & (p == 1)).sum())

    total = tn + fp + fn + tp
    acc = float(tn + tp) / total if total > 0 else 0.0
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_matrix(label, gold_col, pred_col, stats):
    print("\n==================================================")
    print("Field      : {0}".format(label))
    print("Gold col   : {0}".format(gold_col))
    print("Pred col   : {0}".format(pred_col))
    print("==================================================")
    print("                 Predicted")
    print("               0        1")
    print("Actual 0     {:<8} {:<8}".format(stats["tn"], stats["fp"]))
    print("Actual 1     {:<8} {:<8}".format(stats["fn"], stats["tp"]))
    print("")
    print("TN: {0}".format(stats["tn"]))
    print("FP: {0}".format(stats["fp"]))
    print("FN: {0}".format(stats["fn"]))
    print("TP: {0}".format(stats["tp"]))
    print("Total: {0}".format(stats["total"]))
    print("Accuracy : {:.4f}".format(stats["accuracy"]))
    print("Precision: {:.4f}".format(stats["precision"]))
    print("Recall   : {:.4f}".format(stats["recall"]))
    print("F1       : {:.4f}".format(stats["f1"]))


def main():
    if not INPUT_FILE:
        raise RuntimeError("No input file provided.")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("Input file not found: {0}".format(INPUT_FILE))

    print("Loading file...")
    df = clean_cols(read_csv_robust(INPUT_FILE))

    print("\nColumns found:")
    for c in df.columns:
        print(c)

    for item in FIELD_MAP:
        label = item["label"]
        gold_col = item["gold_col"]
        pred_col = item["pred_col"]

        if gold_col not in df.columns:
            print("\nWARNING: missing gold column for {0}: {1}".format(label, gold_col))
            continue

        if pred_col not in df.columns:
            print("\nWARNING: missing pred column for {0}: {1}".format(label, pred_col))
            continue

        stats = compute_confusion(df[gold_col], df[pred_col])
        print_matrix(label, gold_col, pred_col, stats)

    print("\nDONE.")


if __name__ == "__main__":
    main()
