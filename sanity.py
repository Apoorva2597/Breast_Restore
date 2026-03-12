#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

FILE = input("Enter CSV path: ").strip()
df = pd.read_csv(FILE)

pairs = [
    ("Stage1_MinorComp", "Stage1_MinorComp_pred"),
    ("Stage1_Reoperation", "Stage1_Reoperation_pred"),
    ("Stage1_Rehospitalization", "Stage1_Rehospitalization_pred"),
    ("Stage1_MajorComp", "Stage1_MajorComp_pred"),
    ("Stage1_Failure", "Stage1_Failure_pred"),
    ("Stage1_Revision", "Stage1_Revision_pred"),

    ("GOLD_Stage2_MinorComp", "Stage2_MinorComp_pred"),
    ("GOLD_Stage2_Reoperation", "Stage2_Reoperation_pred"),
    ("GOLD_Stage2_Rehospitalization", "Stage2_Rehospitalization_pred"),
    ("GOLD_Stage2_MajorComp", "Stage2_MajorComp_pred"),
    ("GOLD_Stage2_Failure", "Stage2_Failure_pred"),
    ("GOLD_Stage2_Revision", "Stage2_Revision_pred"),
]

for gold_col, pred_col in pairs:

    # fallback for Stage1 in case pred columns are not present
    if gold_col.startswith("Stage1_") and pred_col not in df.columns:
        pred_col = gold_col

    if gold_col not in df.columns or pred_col not in df.columns:
        print("\nSkipping:", gold_col, "/", pred_col)
        continue

    g = pd.to_numeric(df[gold_col], errors="coerce").fillna(0).astype(int)
    p = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int)

    tn = ((g == 0) & (p == 0)).sum()
    fp = ((g == 0) & (p == 1)).sum()
    fn = ((g == 1) & (p == 0)).sum()
    tp = ((g == 1) & (p == 1)).sum()

    total = tn + fp + fn + tp
    acc = float(tn + tp) / total if total > 0 else 0.0

    print("\n====================================")
    print("Gold :", gold_col)
    print("Pred :", pred_col)
    print("====================================")
    print("                 Predicted")
    print("               0        1")
    print("Actual 0     {:<8} {:<8}".format(tn, fp))
    print("Actual 1     {:<8} {:<8}".format(fn, tp))
    print("")
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("Accuracy: {:.4f}".format(acc))
