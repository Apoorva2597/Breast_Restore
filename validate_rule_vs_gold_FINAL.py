#!/usr/bin/env python3
# validate_rule_vs_gold_FINAL_NO_GOLD.py
#
# Validates NO-GOLD master predictions against gold_cleaned_for_cedar.csv
# WITHOUT overwriting either.
#
# Outputs:
#  - /home/apokol/Breast_Restore/_outputs/rule_vs_gold_report_FINAL_NO_GOLD.csv
#  - /home/apokol/Breast_Restore/_outputs/rule_vs_gold_mismatches_FINAL_NO_GOLD.csv
#  - /home/apokol/Breast_Restore/_outputs/rule_vs_gold_confusions_FINAL_NO_GOLD.csv

import os
from typing import Dict, Tuple

import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"
GOLD_PATH = f"{BASE_DIR}/gold_cleaned_for_cedar.csv"
PRED_PATH = f"{BASE_DIR}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"

OUT_REPORT = f"{BASE_DIR}/_outputs/rule_vs_gold_report_FINAL_NO_GOLD.csv"
OUT_MISM   = f"{BASE_DIR}/_outputs/rule_vs_gold_mismatches_FINAL_NO_GOLD.csv"
OUT_CONF   = f"{BASE_DIR}/_outputs/rule_vs_gold_confusions_FINAL_NO_GOLD.csv"

MERGE_KEY = "MRN"

# Variables you said you care about first (rule-based now)
NUMERIC_VARS = ["Age", "BMI"]
CAT_VARS = ["SmokingStatus"]
BINARY_VARS = [
    "Diabetes", "Obesity", "Hypertension", "CardiacDisease", "VenousThromboembolism", "Steroid",
    "PastBreastSurgery", "PBS_Lumpectomy", "PBS_Breast Reduction", "PBS_Mastopexy",
    "PBS_Augmentation", "PBS_Other", "Radiation", "Chemo"
]

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    if MERGE_KEY not in df.columns:
        raise RuntimeError(f"{path} missing MRN column. Columns={list(df.columns)[:40]}")
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df

def to_binary(x):
    if pd.isna(x):
        return pd.NA
    t = str(x).strip().lower()
    if t in {"1", "true", "yes", "y"}:
        return 1
    if t in {"0", "false", "no", "n"}:
        return 0
    return pd.NA

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def confusion(y_true: pd.Series, y_pred: pd.Series) -> Tuple[int,int,int,int,int]:
    mask = (~y_true.isna()) & (~y_pred.isna())
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    tp = int(((yt==1) & (yp==1)).sum())
    fp = int(((yt==0) & (yp==1)).sum())
    fn = int(((yt==1) & (yp==0)).sum())
    tn = int(((yt==0) & (yp==0)).sum())
    n = int(mask.sum())
    return tp, fp, fn, tn, n

def metrics(tp, fp, fn, tn):
    acc = safe_div(tp+tn, tp+tn+fp+fn)
    prec = safe_div(tp, tp+fp)
    rec = safe_div(tp, tp+fn)
    f1 = safe_div(2*prec*rec, prec+rec) if (prec+rec) else 0.0
    return acc, prec, rec, f1

def main():
    gold = read_csv(GOLD_PATH)
    pred = read_csv(PRED_PATH)

    merged = gold.merge(pred, on=MERGE_KEY, how="inner", suffixes=("_gold", "_pred"))

    rows_report = []
    rows_conf = []
    rows_mism = []

    # ---- numeric
    for v in NUMERIC_VARS:
        gcol = f"{v}_gold"
        pcol = f"{v}_pred"
        if gcol not in merged.columns or pcol not in merged.columns:
            continue

        g = pd.to_numeric(merged[gcol], errors="coerce")
        p = pd.to_numeric(merged[pcol], errors="coerce")

        mask = (~g.isna()) & (~p.isna())
        n_gold = int((~g.isna()).sum())
        n_cmp = int(mask.sum())
        mae = float((g[mask] - p[mask]).abs().mean()) if n_cmp else None
        pct_within_1 = float(((g[mask] - p[mask]).abs() <= 1.0).mean()) if n_cmp else None
        pct_within_2 = float(((g[mask] - p[mask]).abs() <= 2.0).mean()) if n_cmp else None

        rows_report.append({
            "variable": v, "type": "numeric",
            "n_gold_nonmissing": n_gold, "n_compared": n_cmp,
            "mae": mae, "pct_within_1.0": pct_within_1, "pct_within_2.0": pct_within_2
        })

        # mismatches (only when both exist)
        if n_cmp:
            mm = merged.loc[mask & ((g - p).abs() > 1e-9), [MERGE_KEY, gcol, pcol]].copy()
            mm["variable"] = v
            mm = mm.rename(columns={gcol: "gold", pcol: "pred"})
            rows_mism.extend(mm.to_dict("records"))

    # ---- categorical (simple exact match)
    for v in CAT_VARS:
        gcol = f"{v}_gold"
        pcol = f"{v}_pred"
        if gcol not in merged.columns or pcol not in merged.columns:
            continue
        g = merged[gcol].astype(str).str.strip()
        p = merged[pcol].astype(str).str.strip()

        mask = (g != "nan") & (p != "nan") & (g != "") & (p != "")
        n_gold = int(((merged[gcol].notna()) & (merged[gcol].astype(str).str.strip()!="")).sum())
        n_cmp = int(mask.sum())
        acc = float((g[mask] == p[mask]).mean()) if n_cmp else None

        rows_report.append({
            "variable": v, "type": "categorical",
            "n_gold_nonmissing": n_gold, "n_compared": n_cmp,
            "accuracy": acc
        })

        if n_cmp:
            mm = merged.loc[mask & (g != p), [MERGE_KEY, gcol, pcol]].copy()
            mm["variable"] = v
            mm = mm.rename(columns={gcol: "gold", pcol: "pred"})
            rows_mism.extend(mm.to_dict("records"))

    # ---- binary
    for v in BINARY_VARS:
        gcol = f"{v}_gold"
        pcol = f"{v}_pred"
        if gcol not in merged.columns or pcol not in merged.columns:
            continue

        yg = merged[gcol].apply(to_binary)
        yp = merged[pcol].apply(to_binary)

        tp, fp, fn, tn, n = confusion(yg, yp)
        acc, prec, rec, f1 = metrics(tp, fp, fn, tn)

        rows_report.append({
            "variable": v, "type": "binary",
            "n_gold_nonmissing": int((~yg.isna()).sum()),
            "n_compared": n,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        })
        rows_conf.append({"variable": v, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n_compared": n})

        # mismatches
        mask = (~yg.isna()) & (~yp.isna()) & (yg.astype(int) != yp.astype(int))
        if int(mask.sum()):
            mm = merged.loc[mask, [MERGE_KEY, gcol, pcol]].copy()
            mm["variable"] = v
            mm = mm.rename(columns={gcol: "gold", pcol: "pred"})
            rows_mism.extend(mm.to_dict("records"))

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    pd.DataFrame(rows_report).to_csv(OUT_REPORT, index=False)
    pd.DataFrame(rows_conf).to_csv(OUT_CONF, index=False)
    pd.DataFrame(rows_mism).to_csv(OUT_MISM, index=False)

    print("DONE.")
    print(f"- Report:     {OUT_REPORT}")
    print(f"- Confusions: {OUT_CONF}")
    print(f"- Mismatches: {OUT_MISM}")
    print("\nRun:")
    print("  python validate_rule_vs_gold_FINAL_NO_GOLD.py")

if __name__ == "__main__":
    main()
