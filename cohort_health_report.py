#!/usr/bin/env python3
# cohort_health_report.py
# Python 3.6.8 compatible
#
# Purpose:
#   Rapid sanity-check of the FINAL cohort output for "is anything fundamentally broken?"
#
# Usage:
#   cd /home/apokol/Breast_Restore
#   python cohort_health_report.py
#
# Optional:
#   python cohort_health_report.py --cohort /path/to/cohort.csv --gold /path/to/gold.csv --bridge /path/to/bridge.csv

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


def normalize_mrn(x):
    if x is None:
        return ""
    t = str(x).strip()
    if t.endswith(".0"):
        t = t[:-2]
    return t


def to01(x):
    if is_blank(x):
        return 0
    if isinstance(x, (int, bool)):
        return 1 if int(x) == 1 else 0
    t = str(x).strip().lower()
    if t in ("1", "true", "t", "yes", "y", "positive", "pos", "present"):
        return 1
    if t in ("0", "false", "f", "no", "n", "negative", "neg", "absent"):
        return 0
    # if it's some nonblank token in a "flag" column, treat as 1
    return 1


def top_values(series, k=10):
    # show top nonblank values
    s = series.dropna().map(lambda x: str(x).strip())
    s = s[s.map(lambda x: x != "" and x.lower() not in ("nan","none","null","na","n/a",".","-","--"))]
    if len(s) == 0:
        return []
    vc = s.value_counts().head(k)
    return list(zip(vc.index.tolist(), vc.values.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="/home/apokol/Breast_Restore/cohort_all_patient_level_final_gold_order.csv")
    ap.add_argument("--gold", default="/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv")
    ap.add_argument("--bridge", default="/home/apokol/Breast_Restore/cohort_pid_to_mrn_from_encounters.csv")
    ap.add_argument("--skip_gold", action="store_true", help="Skip gold/bridge overlap checks.")
    args = ap.parse_args()

    if not os.path.exists(args.cohort):
        raise RuntimeError("Missing cohort file: {}".format(args.cohort))

    df = read_csv_safe(args.cohort)

    print("\n=== COHORT HEALTH REPORT ===")
    print("COHORT:", args.cohort)
    print("Rows:", len(df))
    print("Cols:", len(df.columns))

    # Required-ish
    pid_col = "patient_id" if "patient_id" in df.columns else None
    if pid_col is None:
        # try common variants
        for c in df.columns:
            if str(c).lower().strip() in ("encrypted_pat_id","pat_id","patientid"):
                pid_col = c
                break

    if pid_col is None:
        print("\nWARNING: Could not find patient_id column.")
    else:
        n_unique = df[pid_col].astype(str).map(lambda x: x.strip()).nunique()
        print("\nUnique {}: {}".format(pid_col, n_unique))

    # Key fields to summarize (only if present)
    key_cols = [
        "Race", "Ethnicity", "Age_DOS", "BMI",
        "CardiacDisease", "DiabetesMellitus", "Hypertension", "SmokingStatus",
        "Mastectomy_Performed", "Mastectomy_Type", "Mastectomy_Laterality",
        "Recon_Type_op_enc", "Recon_has_expander_op_enc", "Recon_has_implant_op_enc", "Recon_has_flap_op_enc",
        "stage2_confirmed_flag", "stage2_date_final",
        "Stage1_MinorComp_pred", "Stage1_MajorComp_pred", "Stage1_Reoperation_pred", "Stage1_Rehospitalization_pred",
        "Stage2_MinorComp", "Stage2_MajorComp", "Stage2_Reoperation", "Stage2_Rehospitalization",
        "Stage2_Failure", "Stage2_Revision"
    ]

    print("\n--- Missingness + top values (selected columns) ---")
    for c in key_cols:
        if c not in df.columns:
            continue
        nonblank = int((~df[c].map(is_blank)).sum())
        print("\n{}: nonblank {}/{} ({:.1f}%)".format(
            c, nonblank, len(df), 100.0 * float(nonblank) / float(len(df))
        ))
        tv = top_values(df[c], k=8)
        if tv:
            print("  top:", tv[:8])

    # Quick plausibility checks (no fancy stats; just flags)
    print("\n--- Contradiction checks (high-yield) ---")
    issues = []

    # Stage2 confirmed but missing date_final
    if "stage2_confirmed_flag" in df.columns and "stage2_date_final" in df.columns:
        conf1 = df["stage2_confirmed_flag"].map(to01) == 1
        date_blank = df["stage2_date_final"].map(is_blank)
        n = int((conf1 & date_blank).sum())
        issues.append(("stage2_confirmed_flag==1 but stage2_date_final blank", n))

    # Recon type vs flap/implant/expander flags
    if "Recon_Type_op_enc" in df.columns:
        rt = df["Recon_Type_op_enc"].fillna("").astype(str).str.lower()
        if "Recon_has_flap_op_enc" in df.columns:
            flap0 = df["Recon_has_flap_op_enc"].map(to01) == 0
            n = int((rt.str.contains("flap") & flap0).sum())
            issues.append(("Recon_Type mentions flap but Recon_has_flap_op_enc==0", n))
        if "Recon_has_implant_op_enc" in df.columns:
            imp0 = df["Recon_has_implant_op_enc"].map(to01) == 0
            n = int((rt.str.contains("implant") & imp0).sum())
            issues.append(("Recon_Type mentions implant but Recon_has_implant_op_enc==0", n))
        if "Recon_has_expander_op_enc" in df.columns:
            te0 = df["Recon_has_expander_op_enc"].map(to01) == 0
            n = int((rt.str.contains("expander") & te0).sum())
            issues.append(("Recon_Type mentions expander but Recon_has_expander_op_enc==0", n))

    # Mastectomy performed but missing type
    if "Mastectomy_Performed" in df.columns and "Mastectomy_Type" in df.columns:
        mp1 = df["Mastectomy_Performed"].map(to01) == 1
        mt_blank = df["Mastectomy_Type"].map(is_blank)
        n = int((mp1 & mt_blank).sum())
        issues.append(("Mastectomy_Performed==1 but Mastectomy_Type blank", n))

    for name, n in issues:
        print("  {:<60} {}".format(name, n))

    # Gold overlap check (optional)
    if not args.skip_gold:
        if os.path.exists(args.gold) and os.path.exists(args.bridge):
            gold = read_csv_safe(args.gold)
            bridge = read_csv_safe(args.bridge)

            # detect columns
            gold_mrn = "MRN" if "MRN" in gold.columns else None
            bridge_mrn = "MRN" if "MRN" in bridge.columns else None
            bridge_pid = "patient_id" if "patient_id" in bridge.columns else None

            if gold_mrn and bridge_mrn and bridge_pid and pid_col:
                gold_mrns = set(gold[gold_mrn].map(normalize_mrn))
                bridge_mrns = set(bridge[bridge_mrn].map(normalize_mrn))
                overlap = len([m for m in gold_mrns if m in bridge_mrns and m != ""])
                print("\n--- Gold overlap via bridge ---")
                print("Gold MRNs:", len([m for m in gold_mrns if m != \"\"]))
                print("Bridge MRNs:", len([m for m in bridge_mrns if m != \"\"]))
                print("Overlap (gold ∩ bridge):", overlap)
            else:
                print("\n(Skipping gold overlap: could not detect MRN/patient_id columns.)")
        else:
            print("\n(Skipping gold overlap: gold or bridge file missing.)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
