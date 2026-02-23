#!/usr/bin/env python3
# stage2_quick_counts.py
# Python 3.6.8 compatible
#
# Purpose:
#   Quick, copy-typed-friendly counts to diagnose Stage2 coverage in the FINAL cohort file:
#     - How many have stage2_confirmed_flag?
#     - How many have stage2_date_final?
#     - How many have Stage2_* outcomes (Minor/Major/Reop/Rehosp/Failure/Revision)?
#     - Among confirmed, how many are missing date_final?
#     - Among date_final present, are Failure/Revision still blank?
#
# Usage:
#   cd /home/apokol/Breast_Restore
#   python stage2_quick_counts.py
#
# Optional:
#   python stage2_quick_counts.py --cohort /path/to/your.csv

from __future__ import print_function
import os
import argparse
import pandas as pd


def read_csv_safe(path):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", dtype=object)
    finally:
        try:
            f.close()
        except Exception:
            pass


def is_blank(x):
    if x is None:
        return True
    try:
        # pandas NA
        if pd.isna(x):
            return True
    except Exception:
        pass
    t = str(x).strip()
    if t == "":
        return True
    if t.lower() in ("nan", "none", "null", "na", "n/a", ".", "-", "--"):
        return True
    return False


def to01(x):
    # Minimal conversion for flags: blank->0; yes/true/1->1; otherwise 0.
    if is_blank(x):
        return 0
    if isinstance(x, (int, bool)):
        return 1 if int(x) == 1 else 0
    t = str(x).strip().lower()
    if t in ("1", "true", "t", "yes", "y", "positive", "pos", "present"):
        return 1
    if t in ("0", "false", "f", "no", "n", "negative", "neg", "absent"):
        return 0
    # last resort: treat any nonblank as 1 for a "flag" field
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cohort",
        default="/home/apokol/Breast_Restore/cohort_all_patient_level_final_gold_order.csv",
        help="Path to cohort_all_patient_level_final_gold_order.csv",
    )
    args = ap.parse_args()

    cohort_path = args.cohort
    if not os.path.exists(cohort_path):
        raise RuntimeError("Missing cohort file: {}".format(cohort_path))

    df = read_csv_safe(cohort_path)
    print("\nLoaded:", cohort_path)
    print("Rows:", len(df))
    print("Cols:", len(df.columns))

    # Columns (based on the header you showed)
    col_confirmed = "stage2_confirmed_flag"
    col_datefinal = "stage2_date_final"

    stage2_outcomes = [
        "Stage2_MinorComp",
        "Stage2_MajorComp",
        "Stage2_Reoperation",
        "Stage2_Rehospitalization",
        "Stage2_Failure",
        "Stage2_Revision",
    ]

    # Helper to safely count nonblank for a column if it exists
    def count_nonblank(col):
        if col not in df.columns:
            return None
        return int((~df[col].map(is_blank)).sum())

    # Basic availability check
    print("\nColumn existence:")
    for c in [col_confirmed, col_datefinal] + stage2_outcomes:
        print("  {:<24} {}".format(c, "YES" if c in df.columns else "NO"))

    # Basic nonblank counts
    nb_confirmed = count_nonblank(col_confirmed)
    nb_datefinal = count_nonblank(col_datefinal)

    print("\nNonblank counts:")
    print("  {:<24} {}".format(col_confirmed, nb_confirmed if nb_confirmed is not None else "MISSING_COL"))
    print("  {:<24} {}".format(col_datefinal, nb_datefinal if nb_datefinal is not None else "MISSING_COL"))

    for c in stage2_outcomes:
        n = count_nonblank(c)
        print("  {:<24} {}".format(c, n if n is not None else "MISSING_COL"))

    # Build masks (only if the required columns exist)
    if col_confirmed in df.columns:
        confirmed01 = df[col_confirmed].map(to01)
        mask_confirmed = (confirmed01 == 1)
        n_confirmed = int(mask_confirmed.sum())
    else:
        mask_confirmed = None
        n_confirmed = None

    if col_datefinal in df.columns:
        mask_datefinal = (~df[col_datefinal].map(is_blank))
        n_datefinal = int(mask_datefinal.sum())
    else:
        mask_datefinal = None
        n_datefinal = None

    # Confirmed but missing date_final
    print("\nKey diagnostics:")
    if mask_confirmed is not None and mask_datefinal is not None:
        n_conf_missing_date = int((mask_confirmed & (~mask_datefinal)).sum())
        print("  confirmed==1 total               :", n_confirmed)
        print("  date_final nonblank total        :", n_datefinal)
        print("  confirmed==1 BUT date_final blank:", n_conf_missing_date)
    else:
        print("  (Need both stage2_confirmed_flag and stage2_date_final to compute confirmed-vs-date stats.)")

    # Among date_final present: are Failure/Revision still blank?
    if mask_datefinal is not None:
        for c in ["Stage2_Failure", "Stage2_Revision"]:
            if c not in df.columns:
                print("  Among date_final present, {} nonblank: MISSING_COL".format(c))
                continue
            n_nonblank_in_datefinal = int((mask_datefinal & (~df[c].map(is_blank))).sum())
            print("  Among date_final present, {} nonblank: {}".format(c, n_nonblank_in_datefinal))
    else:
        print("  (Need stage2_date_final to check Failure/Revision within date-finalized subset.)")

    # Optional: Among confirmed patients only, how many have outcomes filled?
    if mask_confirmed is not None:
        for c in stage2_outcomes:
            if c not in df.columns:
                continue
            n_nonblank_in_confirmed = int((mask_confirmed & (~df[c].map(is_blank))).sum())
            print("  Among confirmed==1, {:<18} nonblank: {}".format(c, n_nonblank_in_confirmed))

    print("\nDone.\n")


if __name__ == "__main__":
    main()
