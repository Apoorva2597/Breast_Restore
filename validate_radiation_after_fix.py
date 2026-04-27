#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_radiation_after_fix.py

STANDALONE validation for Radiation_After fix only.

Compares ONLY Radiation, Radiation_Before, Radiation_After between:
  - New fixed master: master_abstraction_rule_FINAL_NO_GOLD_RAD_FIXED.csv
  - Gold:             gold_cleaned_for_cedar.csv

Does NOT touch or compare any other variables.
Does NOT overwrite any existing validation outputs.

Writes to:
  _outputs/validation_radiation_after_fix.csv

Python 3.6.8 compatible.
"""

import os
import sys
import pandas as pd

FIXED_MASTER = "_outputs/master_abstraction_rule_FINAL_NO_GOLD_RAD_FIXED.csv"
GOLD_FILE    = "gold_cleaned_for_cedar.csv"
OUTPUT_CSV   = "_outputs/validation_radiation_after_fix.csv"

MRN = "MRN"
TARGET_VARS  = ["Radiation", "Radiation_Before", "Radiation_After"]


def safe_read(path):
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=str, encoding="latin1")


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def norm_bin(series):
    def conv(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip().lower()
        if s in ["1", "true", "t", "yes", "y"]:
            return 1
        if s in ["0", "false", "f", "no", "n"]:
            return 0
        return pd.NA
    return series.apply(conv)


def wilson_ci(p, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 0.0)
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (round(max(0.0, center - margin), 3),
            round(min(1.0, center + margin), 3))


def compute_binary(pred, gold):
    mask  = gold.notna()
    pred  = pred[mask]
    gold  = gold[mask]
    valid = pred.notna() & gold.notna()
    pred  = pred[valid].astype(int)
    gold  = gold[valid].astype(int)

    n  = len(gold)
    if n == 0:
        return {}

    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    tn = int(((pred == 0) & (gold == 0)).sum())

    acc  = (tp + tn) / float(n)
    sens = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / float(tn + fn) if (tn + fn) > 0 else 0.0
    f1   = 2 * tp / float(2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0

    return {
        "n": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy":    round(acc,  3),
        "sensitivity": round(sens, 3), "sensitivity_CI": wilson_ci(sens, tp + fn),
        "specificity": round(spec, 3), "specificity_CI": wilson_ci(spec, tn + fp),
        "PPV":         round(ppv,  3), "PPV_CI":         wilson_ci(ppv,  tp + fp),
        "NPV":         round(npv,  3), "NPV_CI":         wilson_ci(npv,  tn + fn),
        "F1":          round(f1,   3),
    }


def main():
    if not os.path.exists(FIXED_MASTER):
        print("ERROR: Fixed master not found: {0}".format(FIXED_MASTER))
        print("Run fix_radiation_after_only.py first.")
        sys.exit(1)

    print("Loading fixed master: {0}".format(FIXED_MASTER))
    master = clean_cols(safe_read(FIXED_MASTER))
    master[MRN] = master[MRN].astype(str).str.strip()

    print("Loading gold: {0}".format(GOLD_FILE))
    gold = clean_cols(safe_read(GOLD_FILE))
    gold[MRN] = gold[MRN].astype(str).str.strip()

    merged = pd.merge(master, gold, on=MRN, how="inner", suffixes=("_pred", "_gold"))
    print("Merged rows: {0}".format(len(merged)))

    if len(merged) == 0:
        print("ERROR: No rows matched on MRN.")
        sys.exit(1)

    results = []
    print()
    print("=== Radiation_After Fix Validation ===")
    print()

    for v in TARGET_VARS:
        pred_col = v + "_pred"
        gold_col = v + "_gold"

        if pred_col not in merged.columns:
            print("SKIP {0}: pred column missing".format(v))
            continue
        if gold_col not in merged.columns:
            print("SKIP {0}: gold column missing".format(v))
            continue

        pred = norm_bin(merged[pred_col])
        gld  = norm_bin(merged[gold_col])
        m    = compute_binary(pred, gld)

        if not m:
            print("{0}: no valid comparisons".format(v))
            continue

        print("{v}: n={n}  TP={TP}  FP={FP}  FN={FN}  TN={TN}".format(v=v, **m))
        print("  Accuracy={accuracy}  Sensitivity={sensitivity} {sensitivity_CI}".format(**m))
        print("  Specificity={specificity} {specificity_CI}".format(**m))
        print("  PPV={PPV} {PPV_CI}  NPV={NPV} {NPV_CI}  F1={F1}".format(**m))
        print()

        results.append({
            "variable":    v,
            "n":           m["n"],
            "TP":          m["TP"],
            "FP":          m["FP"],
            "FN":          m["FN"],
            "TN":          m["TN"],
            "accuracy":    m["accuracy"],
            "sensitivity": m["sensitivity"],
            "sens_CI_lo":  m["sensitivity_CI"][0],
            "sens_CI_hi":  m["sensitivity_CI"][1],
            "specificity": m["specificity"],
            "spec_CI_lo":  m["specificity_CI"][0],
            "spec_CI_hi":  m["specificity_CI"][1],
            "PPV":         m["PPV"],
            "PPV_CI_lo":   m["PPV_CI"][0],
            "PPV_CI_hi":   m["PPV_CI"][1],
            "NPV":         m["NPV"],
            "NPV_CI_lo":   m["NPV_CI"][0],
            "NPV_CI_hi":   m["NPV_CI"][1],
            "F1":          m["F1"],
        })

    # Also print NLP cohort counts for Table 1
    print("=== NLP Cohort Radiation_After Count (for Table 1) ===")
    ra_col = "Radiation_After"
    if ra_col in master.columns:
        counts = master[ra_col].value_counts(dropna=False)
        print(counts.to_dict())
        n_total = len(master)
        n_pos   = int((master[ra_col].astype(str).str.strip().isin({"1", "True", "true"})).sum())
        print("Radiation_After=1: {0}/{1} ({2}%)".format(
            n_pos, n_total, round(100.0 * n_pos / n_total, 1)))

    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print("\nResults saved to: {0}".format(OUTPUT_CSV))
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
