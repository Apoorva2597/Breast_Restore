#!/usr/bin/env python3
import os
import pandas as pd
import math

BASE = "/home/apokol/Breast_Restore"
GOLD = os.path.join(BASE, "gold_cleaned_for_cedar.csv")

RULE_SUMMARY = os.path.join(BASE, "_outputs", "patient_stage_summary.csv")
BART_PATIENT = os.path.join(BASE, "_outputs_bart", "bart_stage2_hit_verifier_patient_summary.csv")

MERGE_KEY = "MRN"

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
            try:
                out.append(int(float(v)))
            except Exception:
                out.append(default)
    return pd.Series(out)

def confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn

def metrics(tp, fp, fn, tn):
    def div(a,b):
        return a/b if b else float("nan")
    acc  = div(tp+tn, tp+fp+fn+tn)
    prec = div(tp, tp+fp)
    rec  = div(tp, tp+fn)
    spec = div(tn, tn+fp)
    npv  = div(tn, tn+fn)
    f1   = div(2*tp, 2*tp+fp+fn)
    bal  = div((rec + spec), 2)
    # MCC
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = ((tp*tn - fp*fn)/denom) if denom else float("nan")
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
    if MERGE_KEY not in gold.columns:
        raise RuntimeError("Gold missing MRN column. Columns: %s" % list(gold.columns)[:40])

    # infer gold label column
    gold_cols_upper = {c.upper(): c for c in gold.columns}
    if "HAS_STAGE2" in gold_cols_upper:
        gold_label_col = gold_cols_upper["HAS_STAGE2"]
    else:
        raise RuntimeError("Gold file must contain HAS_STAGE2 column.")

    gold[MERGE_KEY] = gold[MERGE_KEY].astype(str).str.strip()
    y_true = to_int_series(gold[gold_label_col], default=0)

    # Rule summary
    rule = safe_read(RULE_SUMMARY)
    rule[MERGE_KEY] = rule[MERGE_KEY].astype(str).str.strip()

    # BART patient
    bart = safe_read(BART_PATIENT)
    bart[MERGE_KEY] = bart[MERGE_KEY].astype(str).str.strip()

    # Identify bart prediction column robustly
    # Prefer bart_any_hit_pred_stage2; fall back to similar names
    bart_pred_col = None
    for c in bart.columns:
        cu = c.strip().lower()
        if cu == "bart_any_hit_pred_stage2":
            bart_pred_col = c
            break
    if bart_pred_col is None:
        # try a loose match
        for c in bart.columns:
            if "pred" in c.lower() and "stage2" in c.lower():
                bart_pred_col = c
                break
    if bart_pred_col is None:
        raise RuntimeError("Could not find BART pred column in %s. Columns: %s" %
                           (BART_PATIENT, list(bart.columns)))

    # Merge all
    df = gold[[MERGE_KEY, gold_label_col]].copy()
    df = df.merge(rule[[MERGE_KEY, "HAS_STAGE2"]], on=MERGE_KEY, how="left")
    df = df.merge(bart[[MERGE_KEY, bart_pred_col]], on=MERGE_KEY, how="left")

    df["HAS_STAGE2_rule"] = to_int_series(df["HAS_STAGE2"], default=0)
    # For BART: patients not in bart file (no hits) -> default 0 (conservative)
    df["HAS_STAGE2_bart"] = to_int_series(df[bart_pred_col], default=0)

    y_rule = df["HAS_STAGE2_rule"]
    y_bart = df["HAS_STAGE2_bart"]

    # Hybrid: OR (maximize recall; safe baseline)
    y_hybrid = ((y_rule == 1) | (y_bart == 1)).astype(int)

    # Compute metrics
    tp, fp, fn, tn = confusion(y_true, y_rule)
    print_block("Rule-only vs Gold", metrics(tp, fp, fn, tn))

    tp, fp, fn, tn = confusion(y_true, y_bart)
    print_block("BART-only (conservative: missing patients -> 0) vs Gold", metrics(tp, fp, fn, tn))

    tp, fp, fn, tn = confusion(y_true, y_hybrid)
    print_block("Hybrid (Rule OR BART) vs Gold", metrics(tp, fp, fn, tn))

    # Also show overlap counts
    print("\n=== Overlap diagnostics ===")
    print("Gold positives:", int((y_true == 1).sum()))
    print("Rule positives:", int((y_rule == 1).sum()))
    print("BART positives:", int((y_bart == 1).sum()))
    print("Hybrid positives:", int((y_hybrid == 1).sum()))

    both_pos = int(((y_rule == 1) & (y_bart == 1)).sum())
    rule_only = int(((y_rule == 1) & (y_bart == 0)).sum())
    bart_only = int(((y_rule == 0) & (y_bart == 1)).sum())
    print("Rule & BART both positive:", both_pos)
    print("Rule only positive:", rule_only)
    print("BART only positive:", bart_only)

if __name__ == "__main__":
    main()
