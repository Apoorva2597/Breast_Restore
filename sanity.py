#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
print_confusion_matrices.py

Purpose
-------
Print confusion matrices for each complication abstraction variable.

Input
-----
Validation CSV containing:
    GOLD_<variable>
    PRED_<variable>

Output
------
Printed confusion matrix for each variable

Python 3.6.8 compatible.
"""

import pandas as pd
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------
# FILE
# -------------------------------------------------------

FILE = "validation_results.csv"   # change if needed

# -------------------------------------------------------
# VARIABLES
# -------------------------------------------------------

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

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

df = pd.read_csv(FILE)

# -------------------------------------------------------
# PRINT MATRICES
# -------------------------------------------------------

for var in variables:

    gold = "GOLD_" + var
    pred = "PRED_" + var

    if gold not in df.columns or pred not in df.columns:
        print("\nSkipping {} (columns missing)".format(var))
        continue

    y_true = df[gold].fillna(0).astype(int)
    y_pred = df[pred].fillna(0).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n======================================")
    print("Variable:", var)
    print("======================================")

    print("Confusion Matrix")

    print("                 Predicted")
    print("               0        1")
    print("Actual 0     {:<8} {:<8}".format(tn, fp))
    print("Actual 1     {:<8} {:<8}".format(fn, tp))

    print("\nCounts")
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    accuracy = float(tn + tp) / (tn + fp + fn + tp)

    print("Accuracy: {:.4f}".format(accuracy))
