#!/usr/bin/env python3
# validate_bart_vs_rule_stage2_v2.py
# Robust validation: auto-detects MRN + gold label column.

import os
import math
import pandas as pd

BASE = "/home/apokol/Breast_Restore"
GOLD = os.path.join(BASE, "gold_cleaned_for_cedar.csv")

RULE_SUMMARY = os.path.join(BASE, "_outputs", "patient_stage_summary.csv")
BART_PATIENT = os.path.join(BASE, "_outputs_bart", "bart_stage2_hit_verifier_patient_summary.csv")

# ----------------------------
# Helpers
# ----------------------------

def safe_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing: %s" % path)
    return pd.read_csv(path, dtype=str, low_memory=False)

def norm(s):
    return str(s).strip()

def to_int_series(s, default=0):
    s = s.fillna("").astype(str).str.strip()
    out = []
    for v in s.tolist():
        if v == "" or v.lower() == "nan":
            out.append(default)
        else:
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
    # fallback: any column containing 'MRN'
    for c in df.columns:
        if "MRN" in c.upper():
            return c
    return None

def pick_gold_label_col(df):
    """
    Try multiple common variants. If still not found, return None.
    """
    cols = list(df.columns)
    upper_map = {c.upper(): c for c in cols}

    # High-confidence exact matches
    exact = [
        "HAS_STAGE2", "HAS_STAGE_2",
        "STAGE2", "STAGE_2",
        "STAGE2_PRESENT", "STAGE_2_PRESENT",
        "STAGE2_EVENT", "STAGE_2_EVENT",
        "Y_STAGE2", "Y_STAGE_2",
        "LABEL_STAGE2", "LABEL_STAGE_2",
        "GOLD_STAGE2", "GOLD_STAGE_2",
    ]
    for k in exact:
        if k in upper_map:
            return upper_map[k]

    # More flexible: contains both STAGE and 2, and looks boolean-ish
    for c in cols:
        cu = c.upper().replace(" ", "").replace("-", "_")
        if ("STAGE" in cu and "2" in cu) and ("HAS" in cu or "LABEL" in cu or "GOLD" in cu or "PRESENT" in cu or "EVENT" in cu):
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

# ----------------------------
# Main
# ----------------------------

def main():
    gold = safe_read(GOLD)
    mrn_col_gold = pick_mrn_col(gold)
    if mrn_col_gold is None:
        raise RuntimeError("Could not find MRN column in GOLD. Columns: %s" % list(gold.columns)[:80])

    gold_label_col = pick_gold_label_col(gold)
    if gold_label_col is None:
        print("\nERROR: Could not auto-detect gold Stage2 label column.")
        print("Here are the first 80 columns in gold_cleaned_for_cedar.csv:\n")
        for c in list(gold.columns)[:80]:
            print(" -", c)
        print("\nFix: tell me which column is the true gold label for Stage2 (0/1).")
        raise RuntimeError("Gold file must contain a Stage2 label column (not found automatically).")

    print("Gold MRN col:", mrn_col_gold)
    print("Gold label col:", gold_label_col)

    gold[mrn_col_gold] = gold[mrn_col_gold].astype(str).str.strip()
    y_true = to_int_series(gold[gold_label_col], default=0)

    # Rule summary
    rule = safe_read(RULE_SUMMARY)
    mrn_col_rule = pick_mrn_col(rule)
    if mrn_col_rule is None:
        raise RuntimeError("Could not find MRN column in RULE summary. Columns: %s" % list(rule.columns)[:80])

    if "HAS_STAGE2" not in rule.columns:
        # some versions might name it differently
        # fallback: find a column containing STAGE2
        stage2_cols = [c for c in rule.columns if "STAGE2" in c.upper()]
        if stage2_cols:
            rule_stage2_col = stage2_cols[0]
        else:
            raise RuntimeError("Could not find HAS_STAGE2 in rule summary. Columns: %s" % list(rule.columns)[:80])
    else:
        rule_stage2_col = "HAS_STAGE2"

    rule[mrn_col_rule] = rule[mrn_col_rule].astype(str).str.strip()

    # BART patient
    bart = safe_read(BART_PATIENT)
    mrn_col_bart = pick_mrn_col(bart)
    if mrn_col_bart is None:
        raise RuntimeError("Could not find MRN column in BART patient summary. Columns: %s" % list(bart.columns)[:80])

    bart[mrn_col_bart] = bart[mrn_col_bart].astype(str).str.strip()

    bart_pred_col = None
    for c in bart.columns:
        if c.strip().lower() == "bart_any_hit_pred_stage2":
            bart_pred_col = c
            break
    if bart_pred_col is None:
        # loose match
        for c in bart.columns:
            cu = c.lower()
            if "pred" in cu and "stage2" in cu:
                bart_pred_col = c
                break
    if bart_pred_col is None:
        raise RuntimeError("Could not find BART pred column. Columns: %s" % list(bart.columns)[:80])

    print("Rule MRN col:", mrn_col_rule, " Rule label col:", rule_stage2_col)
    print("BART MRN col:", mrn_col_bart, " BART pred col:", bart_pred_col)

    # Merge
    df = gold[[mrn_col_gold, gold_label_col]].copy()
    df = df.rename(columns={mrn_col_gold: "MRN_GOLD"})
    df["MRN"] = df["MRN_GOLD"]

    df = df.merge(rule[[mrn_col_rule, rule_stage2_col]].rename(columns={mrn_col_rule: "MRN", rule_stage2_col: "HAS_STAGE2_rule"}),
                  on="MRN", how="left")
    df = df.merge(bart[[mrn_col_bart, bart_pred_col]].rename(columns={mrn_col_bart: "MRN", bart_pred_col: "HAS_STAGE2_bart"}),
                  on="MRN", how="left")

    df["HAS_STAGE2_rule"] = to_int_series(df["HAS_STAGE2_rule"], default=0)
    df["HAS_STAGE2_bart"] = to_int_series(df["HAS_STAGE2_bart"], default=0)

    y_rule = df["HAS_STAGE2_rule"]
    y_bart = df["HAS_STAGE2_bart"]
    y_hybrid = ((y_rule == 1) | (y_bart == 1)).astype(int)

    # Metrics
    tp, fp, fn, tn = confusion(y_true, y_rule)
    print_block("Rule-only vs Gold", metrics(tp, fp, fn, tn))

    tp, fp, fn, tn = confusion(y_true, y_bart)
    print_block("BART-only (patients not scored -> 0) vs Gold", metrics(tp, fp, fn, tn))

    tp, fp, fn, tn = confusion(y_true, y_hybrid)
    print_block("Hybrid (Rule OR BART) vs Gold", metrics(tp, fp, fn, tn))

    print("\n=== Counts ===")
    print("Gold positives:", int((y_true == 1).sum()))
    print("Rule positives:", int((y_rule == 1).sum()))
    print("BART positives:", int((y_bart == 1).sum()))
    print("Hybrid positives:", int((y_hybrid == 1).sum()))

if __name__ == "__main__":
    main()
