#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sanity_confusion_matrix.py

Print confusion matrices for abstraction variables
WITHOUT sklearn.

Python 3.6.8 compatible
"""

import pandas as pd

FILE = "/_outputs/validation_summary.csv"  

variables = [
    "Stage1_MinorComp",
    "Stage1_Reoperation",
    "Stage1_Rehospitalization",
    "Stage1_MajorComp",
    "Stage1_Failure",
    "Stage1_Revision",
    "Stage2_MinorComp",
    "Stage2_Reoperation",
    "Stage2_Rehospitalization",
    "Stage2_MajorComp",
    "Stage2_Failure",
    "Stage2_Revision"
]

df = pd.read_csv(FILE)

for var in variables:

    gold = "GOLD_" + var
    pred = "PRED_" + var

    if gold not in df.columns or pred not in df.columns:
        print("\nSkipping:", var)
        continue

    g = df[gold].fillna(0).astype(int)
    p = df[pred].fillna(0).astype(int)

    tn = ((g == 0) & (p == 0)).sum()
    fp = ((g == 0) & (p == 1)).sum()
    fn = ((g == 1) & (p == 0)).sum()
    tp = ((g == 1) & (p == 1)).sum()

    print("\n===================================")
    print("Variable:", var)
    print("===================================")

    print("                 Predicted")
    print("               0        1")
    print("Actual 0     {:<8} {:<8}".format(tn, fp))
    print("Actual 1     {:<8} {:<8}".format(fn, tp))

    total = tn + fp + fn + tp
    acc = float(tn + tp) / total if total else 0

    print("\nTN:", tn, " FP:", fp)
    print("FN:", fn, " TP:", tp)
    print("Accuracy:", round(acc, 4))
