#!/usr/bin/env python3
# validate_bart_vs_rule_stage2_v3.py
# Uses gold label Stage2_Applicable (preferred) with fallback auto-detection.

import os
import math
import pandas as pd

BASE = "/home/apokol/Breast_Restore"
GOLD = os.path.join(BASE, "gold_cleaned_for_cedar.csv")

RULE_SUMMARY = os.path.join(BASE, "_outputs", "patient_stage_summary.csv")
BART_PATIENT = os.path.join(BASE, "_outputs_bart", "bart_stage2_hit_verifier_patient_summary.csv")

PREFERRED_GOLD_LABEL = "Stage2_Applicable"

def safe_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing: %s" % path)
    return pd.read_csv(path, dtype=str, low_memory=False)

def to_int_series(s, default=0):
    s = s.fillna("").astype(str).str.strip()
    out = []
    for v in s.tolist():
        if v == "" or v.lower() == "nan":
            out.append(default)
        else:
            # accept 0/1, True/False, yes/no
            vl = v.lower()
            if vl in ["true", "t", "yes", "y"]:
                out.append(1)
                continue
            if vl in ["false", "f", "no", "n"]:
                out.append(0)
                continue
            try:
                out.append(int(float(v)))
            except Exception:
                out.append(default)
    return pd.Series(out)

def pick_mrn_col(df):
    candidates = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN", "PATIENTID", "PAT_ID"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "MRN" in c.upper():
            return c
    return None

def pick_gold_label_col(df):
    # First: preferred
    if PREFERRED_GOLD_LABEL in df.columns:
        return PREFERRED_GOLD_LABEL

    # Fallback: search candidates
    cols = list(df.columns)
    upper_map = {c.upper(): c for c in cols}
    exact = [
        "HAS_STAGE2", "HAS_STAGE_2",
        "STAGE2", "STAGE_2",
        "STAGE2_APPLICABLE", "STAGE_2_APPLICABLE",
        "STAGE2_PRESENT", "STAGE_2_PRESENT",
        "STAGE2_EVENT", "STAGE_2_EVENT",
        "Y_STAGE2", "Y_STAGE_2",
        "LABEL_STAGE2", "LABEL_STAGE_2",
        "GOLD_STAGE2", "GOLD_STAGE_2",
    ]
    for k in exact:
        if k in upper_map:
            return upper_map[k]

    for c in cols:
        cu = c.upper().replace(" ", "").replace("-", "_")
        if ("STAGE" in cu and "2" in cu) and ("APPLICABLE" in cu or "HAS" in cu or "LABEL" in cu or "GOLD" in cu or "PRESENT" in cu):
            return c

    return None

def confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn

def metrics(tp, fp, fn, tn):
    def div(a, b):
        return a / b if b else float("nan")

    acc  = div(tp + tn, tp + fp + fn + tn)
    prec = div(tp, tp + fp)
    rec  = div(tp, tp + fn)
    spec = div(tn, tn + fp)
    npv  = div(tn, tn + fn)
    f1   = div(2 * tp, 2 * tp + fp + fn)
    bal  = div((rec + spec), 2)

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else float("nan")

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": acc, "Precision": prec, "Recall": rec, "Specificity": spec,
        "NPV": npv, "F1": f1, "BalancedAcc": bal, "MCC": mcc
    }

def print_block(title, m):
    print("\n=== %s ===" % title)
    print("TP=%d FP=%d FN=%d TN=%d" % (m["TP"], m["FP"], m["FN"], m["TN"]))
    print("Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f" %
          (m["Accuracy"], m["Precision"], m["Recall"], m["F1"]))
    print("Specificity=%.4f  NPV=%.4f  BalancedAcc=%.4f  MCC=%.4f" %
          (m["Specificity"], m["NPV"], m["BalancedAcc"], m["MCC"]))

def main():
    gold = safe_read(GOLD)
    mrn_gold = pick_mrn_col(gold)
    if mrn_gold is None:
        raise RuntimeError("Could not find MRN col in GOLD. Columns: %s" % list(gold.columns)[:80])

    gold_label = pick_gold_label_col(gold)
    if gold_label is None:
        print("\nCould not find gold Stage2 label column. Columns include (first 120):")
        for c in list(gold.columns)[:120]:
            print(" -", c)
        raise RuntimeError("Gold Stage2 label not found.")
    print("Gold MRN col:", mrn_gold)
    print("Gold label col:", gold_label)

    gold[mrn_gold] = gold[mrn_gold].astype(str).str.strip()
    y_true = to_int_series(gold[gold_label], default=0)

    rule = safe_read(RULE_SUMMARY)
    mrn_rule = pick_mrn_col(rule)
    if mrn_rule is None:
        raise RuntimeError("Could not find MRN col in RULE summary.")
    if "HAS_STAGE2" not in rule.columns:
        raise RuntimeError("RULE summary missing HAS_STAGE2. Columns: %s" % list(rule.columns)[:80])
    rule[mrn_rule] = rule[mrn_rule].astype(str).str.strip()

    bart = safe_read(BART_PATIENT)
    mrn_bart = pick_mrn_col(bart)
    if mrn_bart is None:
        raise RuntimeError("Could not find MRN col in BART patient summary.")

    bart_pred_col = None
    for c in bart.columns:
        if c.strip().lower() == "bart_any_hit_pred_stage2":
            bart_pred_col = c
            break
    if bart_pred_col is None:
        for c in bart.columns:
            cu = c.lower()
            if "pred" in cu and "stage2" in cu:
                bart_pred_col = c
                break
    if bart_pred_col is None:
        raise RuntimeError("Could not find BART pred col. Columns: %s" % list(bart.columns)[:80])

    bart[mrn_bart] = bart[mrn_bart].astype(str).str.strip()

    # Merge
    df = gold[[mrn_gold, gold_label]].copy().rename(columns={mrn_gold: "MRN", gold_label: "GOLD_STAGE2"})
    df = df.merge(rule[[mrn_rule, "HAS_STAGE2"]].rename(columns={mrn_rule: "MRN", "HAS_STAGE2": "RULE_STAGE2"}),
                  on="MRN", how="left")
    df = df.merge(bart[[mrn_bart, bart_pred_col]].rename(columns={mrn_bart: "MRN", bart_pred_col: "BART_STAGE2"}),
                  on="MRN", how="left")

    df["GOLD_STAGE2"] = to_int_series(df["GOLD_STAGE2"], default=0)
    df["RULE_STAGE2"] = to_int_series(df["RULE_STAGE2"], default=0)
    df["BART_STAGE2"] = to_int_series(df["BART_STAGE2"], default=0)

    y_true = df["GOLD_STAGE2"]
    y_rule = df["RULE_STAGE2"]
    y_bart = df["BART_STAGE2"]
    y_hybrid = ((y_rule == 1) | (y_bart == 1)).astype(int)

    print_block("Rule-only vs Gold", metrics(*confusion(y_true, y_rule)))
    print_block("BART-only vs Gold", metrics(*confusion(y_true, y_bart)))
    print_block("Hybrid (Rule OR BART) vs Gold", metrics(*confusion(y_true, y_hybrid)))

    print("\n=== Counts ===")
    print("Gold positives:", int((y_true == 1).sum()))
    print("Rule positives:", int((y_rule == 1).sum()))
    print("BART positives:", int((y_bart == 1).sum()))
    print("Hybrid positives:", int((y_hybrid == 1).sum()))

if __name__ == "__main__":
    main()
