# stage2_derive_outcomes_ab.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Take the Stage2 AB patient-level complication file (with S2_Comp1/2/3 fields)
#   and derive Stage2 outcome flags:
#     - Stage2_MinorComp (40)
#     - Stage2_Reoperation (41)
#     - Stage2_Rehospitalization (42)
#     - Stage2_MajorComp (43)
#   (Failure/Revision are NOT derived here—needs separate detectors.)
#
# Inputs:
#   - stage2_ab_complications_patient_level.csv   (default)
#
# Outputs:
#   - stage2_ab_outcomes_patient_level.csv
#   - stage2_ab_outcomes_summary.txt
#
# Notes:
#   - Reads with latin1(errors=replace) to avoid UnicodeDecodeError (0xA0, etc.)
#   - Writes utf-8
#   - Conservative logic: UNKNOWN-only does not count as Minor/Major

from __future__ import print_function

import sys
import re
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
IN_PATIENT_LEVEL = "stage2_ab_complications_patient_level.csv"
OUT_PATIENT_LEVEL = "stage2_ab_outcomes_patient_level.csv"
OUT_SUMMARY = "stage2_ab_outcomes_summary.txt"


# -------------------------
# Robust CSV reading (Python 3.6 safe)
# -------------------------
def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def norm_str(x):
    if x is None:
        return ""
    s = str(x)
    try:
        s = s.replace(u"\xa0", " ")
    except Exception:
        pass
    s = s.strip()
    return s


def norm_upper(x):
    return norm_str(x).upper()


def is_nonempty(x):
    return bool(norm_str(x))


def detect_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_treatment(x):
    """
    Normalize to one of:
      NO TREATMENT, NON-OPERATIVE, REOPERATION, REHOSPITALIZATION, UNKNOWN, ""
    """
    s = norm_upper(x)
    if not s:
        return ""
    # common variants
    s = re.sub(r"\s+", " ", s)

    if "REHOSP" in s or "RE-HOSP" in s or "RE HOSP" in s:
        return "REHOSPITALIZATION"
    if "REOP" in s or "RE-OP" in s or "RE OP" in s or "RETURN TO OR" in s:
        return "REOPERATION"
    if "NON" in s and "OPER" in s:
        return "NON-OPERATIVE"
    if s in ("NO TREATMENT", "NONE", "NO TX", "NO-TREATMENT"):
        return "NO TREATMENT"
    if "NO TREAT" in s:
        return "NO TREATMENT"
    if "UNKNOWN" in s or "UNK" == s:
        return "UNKNOWN"

    # if it doesn't map cleanly, keep UNKNOWN to be safe
    return "UNKNOWN"


def normalize_classification(x):
    """
    Normalize to one of: MINOR, MAJOR, UNKNOWN, ""
    """
    s = norm_upper(x)
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)

    if "MAJOR" in s:
        return "MAJOR"
    if "MINOR" in s:
        return "MINOR"
    if "UNKNOWN" in s or "UNK" == s:
        return "UNKNOWN"

    return "UNKNOWN"


def any_true(vals):
    for v in vals:
        if bool(v):
            return True
    return False


def main():
    df = read_csv_safe(IN_PATIENT_LEVEL)
    if df is None or df.empty:
        raise RuntimeError("Input file is empty or unreadable: {}".format(IN_PATIENT_LEVEL))

    # Required base fields
    required = ["patient_id"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError("Missing required column '{}' in {}".format(c, IN_PATIENT_LEVEL))

    # Stage2 date field
    stage2_date_col = detect_existing_col(df, ["stage2_dt", "STAGE2_DT", "Stage2_DT", "stage2_date", "Stage2_Date"])
    if stage2_date_col is None:
        # sometimes it is named like stage2_date_final in your pipeline
        stage2_date_col = detect_existing_col(df, ["stage2_date_final", "stage2_date_best", "stage2_event_dt_best"])

    if stage2_date_col is None:
        raise RuntimeError("Could not find a Stage2 date column in {}.".format(IN_PATIENT_LEVEL))

    # Verify the S2_Comp fields exist (at least Comp1)
    comp_cols = {
        1: {
            "comp": "S2_Comp1",
            "tx": "S2_Comp1_Treatment",
            "cls": "S2_Comp1_Classification",
            "dt": "S2_Comp1_Date",
        },
        2: {
            "comp": "S2_Comp2",
            "tx": "S2_Comp2_Treatment",
            "cls": "S2_Comp2_Classification",
            "dt": "S2_Comp2_Date",
        },
        3: {
            "comp": "S2_Comp3",
            "tx": "S2_Comp3_Treatment",
            "cls": "S2_Comp3_Classification",
            "dt": "S2_Comp3_Date",
        },
    }

    missing_all = True
    for k in (1, 2, 3):
        if comp_cols[k]["comp"] in df.columns:
            missing_all = False
            break
    if missing_all:
        raise RuntimeError("Could not find any S2_CompX columns (S2_Comp1/2/3) in {}".format(IN_PATIENT_LEVEL))

    # Normalize treatment/classification columns (create safe columns even if missing)
    for k in (1, 2, 3):
        tx_col = comp_cols[k]["tx"]
        cls_col = comp_cols[k]["cls"]
        comp_col = comp_cols[k]["comp"]

        if comp_col not in df.columns:
            df[comp_col] = ""
        if tx_col not in df.columns:
            df[tx_col] = ""
        if cls_col not in df.columns:
            df[cls_col] = ""

        df[tx_col] = df[tx_col].apply(normalize_treatment)
        df[cls_col] = df[cls_col].apply(normalize_classification)

    # A complication is "present" if S2_CompX has something non-empty
    # (We do NOT require a date—some notes may not give exact date field.)
    comp_present = []
    for k in (1, 2, 3):
        comp_present.append(df[comp_cols[k]["comp"]].apply(is_nonempty))

    df["_any_comp_present"] = comp_present[0] | comp_present[1] | comp_present[2]

    # Derive signals per comp slot
    minor_sig = []
    major_sig = []
    reop_sig = []
    rehosp_sig = []

    for k in (1, 2, 3):
        tx_col = comp_cols[k]["tx"]
        cls_col = comp_cols[k]["cls"]
        present = df[comp_cols[k]["comp"]].apply(is_nonempty)

        tx = df[tx_col]
        cls = df[cls_col]

        # Minor: classification MINOR OR tx in {NO TREATMENT, NON-OPERATIVE}, but only if comp present
        minor = present & (
            (cls == "MINOR") |
            (tx.isin(["NO TREATMENT", "NON-OPERATIVE"]))
        )

        # Reop/Rehosp: by tx, only if comp present
        reop = present & (tx == "REOPERATION")
        rehosp = present & (tx == "REHOSPITALIZATION")

        # Major: classification MAJOR OR tx in {REOPERATION, REHOSPITALIZATION}, only if comp present
        major = present & (
            (cls == "MAJOR") |
            (tx.isin(["REOPERATION", "REHOSPITALIZATION"]))
        )

        minor_sig.append(minor)
        major_sig.append(major)
        reop_sig.append(reop)
        rehosp_sig.append(rehosp)

    # Patient-level outcomes (any across 1..3)
    df["Stage2_Reoperation"] = (reop_sig[0] | reop_sig[1] | reop_sig[2]).astype(int)
    df["Stage2_Rehospitalization"] = (rehosp_sig[0] | rehosp_sig[1] | rehosp_sig[2]).astype(int)
    df["Stage2_MajorComp"] = (major_sig[0] | major_sig[1] | major_sig[2]).astype(int)

    # MinorComp: only if NOT major (to keep mutually exclusive) and has minor evidence
    any_minor = (minor_sig[0] | minor_sig[1] | minor_sig[2])
    df["Stage2_MinorComp"] = (any_minor & (df["Stage2_MajorComp"] == 0)).astype(int)

    # Placeholders (not derived here)
    if "Stage2_Failure" not in df.columns:
        df["Stage2_Failure"] = ""
    if "Stage2_Revision" not in df.columns:
        df["Stage2_Revision"] = ""

    # Basic sanity counts
    n = len(df)
    n_has_stage2_dt = int(pd.to_datetime(df[stage2_date_col], errors="coerce").notnull().sum())
    n_any_comp = int(df["_any_comp_present"].sum())

    n_minor = int(df["Stage2_MinorComp"].sum())
    n_major = int(df["Stage2_MajorComp"].sum())
    n_reop = int(df["Stage2_Reoperation"].sum())
    n_rehosp = int(df["Stage2_Rehospitalization"].sum())

    # Write outputs
    # Drop internal helper
    out_df = df.drop(columns=["_any_comp_present"], errors="ignore")
    out_df.to_csv(OUT_PATIENT_LEVEL, index=False, encoding="utf-8")

    lines = []
    lines.append("=== Stage2 Outcomes Derivation (AB) ===")
    lines.append("Python: 3.6.8 compatible | Read encoding: latin1(errors=replace) | Write: utf-8")
    lines.append("Input: {}".format(IN_PATIENT_LEVEL))
    lines.append("Detected Stage2 date column: {}".format(stage2_date_col))
    lines.append("")
    lines.append("Total patients (rows): {}".format(n))
    lines.append("Patients with non-null Stage2 date: {} ({:.1f}%)".format(
        n_has_stage2_dt, (100.0 * n_has_stage2_dt / n) if n else 0.0
    ))
    lines.append("Patients with >=1 Stage2 complication label present (S2_Comp1/2/3 non-empty): {} ({:.1f}%)".format(
        n_any_comp, (100.0 * n_any_comp / n) if n else 0.0
    ))
    lines.append("")
    lines.append("Derived Stage2 outcomes (mutually exclusive Minor vs Major):")
    lines.append("  Stage2_MinorComp: {} ({:.1f}%)".format(n_minor, (100.0 * n_minor / n) if n else 0.0))
    lines.append("  Stage2_MajorComp: {} ({:.1f}%)".format(n_major, (100.0 * n_major / n) if n else 0.0))
    lines.append("  Stage2_Reoperation: {} ({:.1f}%)".format(n_reop, (100.0 * n_reop / n) if n else 0.0))
    lines.append("  Stage2_Rehospitalization: {} ({:.1f}%)".format(n_rehosp, (100.0 * n_rehosp / n) if n else 0.0))
    lines.append("")
    lines.append("Notes:")
    lines.append("  - MinorComp = any MINOR classification OR (NO TREATMENT/NON-OPERATIVE), excluding anyone flagged MajorComp.")
    lines.append("  - MajorComp = any MAJOR classification OR (REOPERATION/REHOSPITALIZATION).")
    lines.append("  - Failure/Revision are left blank here (requires separate detectors).")
    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_PATIENT_LEVEL))
    lines.append("  - {}".format(OUT_SUMMARY))

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
