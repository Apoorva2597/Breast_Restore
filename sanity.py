#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import pandas as pd

ROOT = os.path.abspath(".")
MIS = os.path.join(ROOT, "_outputs", "validation_mismatches.csv")

def pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def main():
    df = pd.read_csv(MIS, dtype=str, low_memory=False)
    # best guess column names
    gold_fail = pick_col(df, ["GOLD_Stage2_Failure", "Stage2_Failure", "Stage2_Failure_gold"])
    pred_fail = pick_col(df, ["Stage2_Failure_pred", "PRED_Stage2_Failure_pred"])
    fail_snip = pick_col(df, ["failure_evidence_snippet", "failure_snippet"])
    fail_src  = pick_col(df, ["failure_evidence_source", "failure_source"])
    fail_pat  = pick_col(df, ["failure_evidence_pattern", "failure_pattern"])
    mrn_col   = pick_col(df, ["MRN", "mrn"])
    pid_col   = pick_col(df, ["ENCRYPTED_PAT_ID", "encrypted_pat_id"])

    print("Loaded:", MIS)
    print("Columns:", len(df.columns))

    if not gold_fail or not pred_fail:
        print("Could not find required columns for failure in mismatches.")
        print("Found gold_fail:", gold_fail, "pred_fail:", pred_fail)
        return

    # Failure FNs: gold=1 pred=0
    fn = df[(df[gold_fail].astype(str) == "1") & (df[pred_fail].astype(str) == "0")].copy()
    print("\nFAILURE FNs (gold=1, pred=0):", len(fn))
    show_cols = [c for c in [mrn_col, pid_col, gold_fail, pred_fail, fail_src, fail_pat, fail_snip] if c]
    if len(fn) > 0:
        print(fn[show_cols].head(50).to_string(index=False))

    # For debugging: top FP sources for reop/rehosp if present
    for outcome, gold_cands, pred_cands, snip_cands in [
        ("Reoperation", ["GOLD_Stage2_Reoperation"], ["Stage2_Reoperation_pred"], ["reop_evidence_snippet"]),
        ("Rehospitalization", ["GOLD_Stage2_Rehospitalization"], ["Stage2_Rehospitalization_pred"], ["rehosp_evidence_snippet"]),
    ]:
        g = pick_col(df, gold_cands)
        p = pick_col(df, pred_cands)
        sn = pick_col(df, snip_cands)
        if g and p:
            fp = df[(df[g].astype(str) == "0") & (df[p].astype(str) == "1")].copy()
            print("\n{} FPs (gold=0, pred=1): {}".format(outcome, len(fp)))
            cols = [c for c in [mrn_col, pid_col, g, p, sn] if c]
            if len(fp) > 0:
                print(fp[cols].head(25).to_string(index=False))

if __name__ == "__main__":
    main()
