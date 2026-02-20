# qa_stage2_failure_revision.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Sanity-check Stage2 failure/revision detector outputs:
#     - patient-level counts (failure/revision)
#     - row-level counts and top rules
#     - optional cross-tabs vs stage2 outcomes if outcomes file present
#
# Inputs:
#   - stage2_ab_failure_revision_row_hits.csv
#   - stage2_ab_failure_revision_patient_level.csv
# Optional:
#   - stage2_ab_outcomes_patient_level.csv
#
# Outputs:
#   - qa_stage2_failure_revision_summary.txt
#   - qa_stage2_revision_rule_counts.csv
#   - qa_stage2_failure_rule_counts.csv

from __future__ import print_function
import os
import sys
import pandas as pd

ROW_HITS = "stage2_ab_failure_revision_row_hits.csv"
PATIENT_LEVEL = "stage2_ab_failure_revision_patient_level.csv"
OUT_SUMMARY = "qa_stage2_failure_revision_summary.txt"
OUT_REV_RULES = "qa_stage2_revision_rule_counts.csv"
OUT_FAIL_RULES = "qa_stage2_failure_rule_counts.csv"

OPTIONAL_OUTCOMES = "stage2_ab_outcomes_patient_level.csv"


def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def normalize_columns(df):
    # strip whitespace and preserve mapping
    new_cols = []
    for c in df.columns:
        try:
            new_cols.append(str(c).strip())
        except Exception:
            new_cols.append(c)
    df.columns = new_cols
    return df


def pick_col(df_cols, candidates):
    """
    Case-insensitive column pick.
    Returns actual column name or None.
    """
    lower_map = {str(c).strip().lower(): str(c).strip() for c in df_cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def main():
    if not os.path.exists(ROW_HITS):
        raise RuntimeError("Missing required file: {}".format(ROW_HITS))
    if not os.path.exists(PATIENT_LEVEL):
        raise RuntimeError("Missing required file: {}".format(PATIENT_LEVEL))

    rows = read_csv_safe(ROW_HITS)
    pats = read_csv_safe(PATIENT_LEVEL)

    rows = normalize_columns(rows)
    pats = normalize_columns(pats)

    # ---- detect key columns ----
    if "patient_id" not in pats.columns:
        # try to find a close alternative (case-insensitive)
        pid_alt = pick_col(pats.columns, ["PATIENT_ID", "patientid", "mrn", "MRN"])
        if pid_alt:
            pats = pats.rename(columns={pid_alt: "patient_id"})
        else:
            raise RuntimeError("patient_level file missing patient_id (or recognizable alternative).")

    if not rows.empty and "patient_id" not in rows.columns:
        pid_alt = pick_col(rows.columns, ["PATIENT_ID", "patientid", "mrn", "MRN"])
        if pid_alt:
            rows = rows.rename(columns={pid_alt: "patient_id"})
        else:
            raise RuntimeError("row_hits file missing patient_id (or recognizable alternative).")

    # patient-level failure/revision cols (robust)
    fail_col = pick_col(pats.columns, [
        "Stage2_Failure", "stage2_failure", "S2_Failure", "s2_failure",
        "FAILURE", "failure"
    ])
    rev_col = pick_col(pats.columns, [
        "Stage2_Revision", "stage2_revision", "S2_Revision", "s2_revision",
        "REVISION", "revision"
    ])

    # ---- start summary ----
    lines = []
    lines.append("=== QA: Stage2 Failure/Revision Detector ===")
    lines.append("Files:")
    lines.append("  Row hits:      {}".format(ROW_HITS))
    lines.append("  Patient level: {}".format(PATIENT_LEVEL))
    lines.append("")

    # ---- patient-level counts ----
    n_pat = int(pats["patient_id"].nunique())
    lines.append("Patient-level:")
    lines.append("  Unique patients: {}".format(n_pat))

    if fail_col:
        n_fail = int(pd.to_numeric(pats[fail_col], errors="coerce").fillna(0).astype(int).sum())
        lines.append("  Failure column used: {}".format(fail_col))
        lines.append("  Stage2_Failure: {} ({:.1f}%)".format(n_fail, 100.0 * n_fail / n_pat if n_pat else 0.0))
    else:
        lines.append("  Stage2_Failure column NOT FOUND (skipping failure patient count).")

    if rev_col:
        n_rev = int(pd.to_numeric(pats[rev_col], errors="coerce").fillna(0).astype(int).sum())
        lines.append("  Revision column used: {}".format(rev_col))
        lines.append("  Stage2_Revision: {} ({:.1f}%)".format(n_rev, 100.0 * n_rev / n_pat if n_pat else 0.0))
    else:
        lines.append("  Stage2_Revision column NOT FOUND (skipping revision patient count).")

    lines.append("")

    # ---- row-level counts + rule counts ----
    if rows.empty:
        lines.append("Row hits file is empty (no detected rows).")
        lines.append("")
    else:
        # flags (robust)
        fail_flag_col = pick_col(rows.columns, ["S2_Failure_Flag", "s2_failure_flag", "Failure_Flag", "failure_flag"])
        rev_flag_col = pick_col(rows.columns, ["S2_Revision_Flag", "s2_revision_flag", "Revision_Flag", "revision_flag"])

        if not fail_flag_col or not rev_flag_col:
            raise RuntimeError("row_hits missing required flag columns (failure/revision). Found columns: {}".format(
                ", ".join(rows.columns.tolist())
            ))

        rows[fail_flag_col] = rows[fail_flag_col].fillna(False).astype(bool)
        rows[rev_flag_col] = rows[rev_flag_col].fillna(False).astype(bool)

        n_rows = int(len(rows))
        n_fail_rows = int(rows[fail_flag_col].sum())
        n_rev_rows = int(rows[rev_flag_col].sum())
        n_both_rows = int((rows[fail_flag_col] & rows[rev_flag_col]).sum())

        lines.append("Row-level:")
        lines.append("  Total hit rows: {}".format(n_rows))
        lines.append("  Failure-hit rows: {}".format(n_fail_rows))
        lines.append("  Revision-hit rows: {}".format(n_rev_rows))
        lines.append("  Both flags rows: {}".format(n_both_rows))
        lines.append("")

        # rules
        rev_rule_col = pick_col(rows.columns, ["S2_Revision_Rule", "s2_revision_rule", "Revision_Rule", "revision_rule"])
        fail_rule_col = pick_col(rows.columns, ["S2_Failure_Rule", "s2_failure_rule", "Failure_Rule", "failure_rule"])

        if rev_rule_col:
            rev_rule_counts = rows.loc[rows[rev_flag_col], rev_rule_col].fillna("MISSING_RULE").value_counts()
            rev_rule_counts.to_csv(OUT_REV_RULES, header=["count"])
            lines.append("Top revision rules (from {}):".format(rev_rule_col))
            for k, v in rev_rule_counts.head(10).items():
                lines.append("  - {}: {}".format(k, int(v)))
            lines.append("")
        else:
            lines.append("No revision rule column found (skipping rule counts).")
            lines.append("")

        if fail_rule_col:
            fail_rule_counts = rows.loc[rows[fail_flag_col], fail_rule_col].fillna("MISSING_RULE").value_counts()
            fail_rule_counts.to_csv(OUT_FAIL_RULES, header=["count"])
            lines.append("Top failure rules (from {}):".format(fail_rule_col))
            for k, v in fail_rule_counts.head(10).items():
                lines.append("  - {}: {}".format(k, int(v)))
            lines.append("")
        else:
            lines.append("No failure rule column found (skipping rule counts).")
            lines.append("")

    # ---- Optional cross-tabs vs outcomes ----
    if os.path.exists(OPTIONAL_OUTCOMES):
        outc = read_csv_safe(OPTIONAL_OUTCOMES)
        outc = normalize_columns(outc)

        pid_alt = None
        if "patient_id" not in outc.columns:
            pid_alt = pick_col(outc.columns, ["PATIENT_ID", "patientid", "mrn", "MRN"])
            if pid_alt:
                outc = outc.rename(columns={pid_alt: "patient_id"})

        if "patient_id" in outc.columns:
            # detect outcomes columns flexibly
            major_col = pick_col(outc.columns, ["Stage2_MajorComp", "stage2_majorcomp", "S2_MajorComp", "s2_majorcomp"])
            minor_col = pick_col(outc.columns, ["Stage2_MinorComp", "stage2_minorcomp", "S2_MinorComp", "s2_minorcomp"])
            reop_col  = pick_col(outc.columns, ["Stage2_Reoperation", "stage2_reoperation", "S2_Reoperation", "s2_reoperation"])
            rehosp_col = pick_col(outc.columns, ["Stage2_Rehospitalization", "stage2_rehospitalization", "S2_Rehospitalization", "s2_rehospitalization"])

            # Build minimal patient flags df from patient-level file using detected names
            pats_flags = pats[["patient_id"]].copy()
            if fail_col:
                pats_flags["Stage2_Failure_QA"] = pd.to_numeric(pats[fail_col], errors="coerce").fillna(0).astype(int)
            else:
                pats_flags["Stage2_Failure_QA"] = 0
            if rev_col:
                pats_flags["Stage2_Revision_QA"] = pd.to_numeric(pats[rev_col], errors="coerce").fillna(0).astype(int)
            else:
                pats_flags["Stage2_Revision_QA"] = 0

            m = outc.merge(pats_flags, on="patient_id", how="left")

            lines.append("Cross-tabs vs outcomes (counts):")

            if major_col:
                a = pd.to_numeric(m[major_col], errors="coerce").fillna(0).astype(int)
                b = m["Stage2_Failure_QA"].fillna(0).astype(int)
                tab = pd.crosstab(a, b)
                lines.append("  {} x Failure (rows={}, cols=Failure):".format(major_col, major_col))
                lines.append(str(tab))
                lines.append("")
                b2 = m["Stage2_Revision_QA"].fillna(0).astype(int)
                tab2 = pd.crosstab(a, b2)
                lines.append("  {} x Revision (rows={}, cols=Revision):".format(major_col, major_col))
                lines.append(str(tab2))
                lines.append("")
            else:
                lines.append("  Stage2_MajorComp column not found in outcomes file; skipping MajorComp tabs.")
                lines.append("")

            if reop_col:
                a = pd.to_numeric(m[reop_col], errors="coerce").fillna(0).astype(int)
                b = m["Stage2_Revision_QA"].fillna(0).astype(int)
                tab = pd.crosstab(a, b)
                lines.append("  {} x Revision (rows={}, cols=Revision):".format(reop_col, reop_col))
                lines.append(str(tab))
                lines.append("")
            if rehosp_col:
                a = pd.to_numeric(m[rehosp_col], errors="coerce").fillna(0).astype(int)
                b = m["Stage2_Revision_QA"].fillna(0).astype(int)
                tab = pd.crosstab(a, b)
                lines.append("  {} x Revision (rows={}, cols=Revision):".format(rehosp_col, rehosp_col))
                lines.append(str(tab))
                lines.append("")
        else:
            lines.append("Outcomes file present, but no patient_id column detected; skipping cross-tabs.")
            lines.append("")

    # ---- write outputs ----
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nWrote:")
    print("  - {}".format(OUT_SUMMARY))
    if os.path.exists(OUT_REV_RULES):
        print("  - {}".format(OUT_REV_RULES))
    if os.path.exists(OUT_FAIL_RULES):
        print("  - {}".format(OUT_FAIL_RULES))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
