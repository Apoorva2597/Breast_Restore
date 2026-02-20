# qa_stage2_failure_revision.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Sanity-check Stage2 failure/revision detector outputs:
#     - basic counts
#     - top rules
#     - revision-only vs failure-only vs both
#     - (optional) compare to stage2 outcomes if available
#
# Inputs:
#   - stage2_ab_failure_revision_row_hits.csv
#   - stage2_ab_failure_revision_patient_level.csv
# Optional:
#   - stage2_ab_outcomes_patient_level.csv (if present)
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


def main():
    if not os.path.exists(ROW_HITS):
        raise RuntimeError("Missing required file: {}".format(ROW_HITS))
    if not os.path.exists(PATIENT_LEVEL):
        raise RuntimeError("Missing required file: {}".format(PATIENT_LEVEL))

    rows = read_csv_safe(ROW_HITS)
    pats = read_csv_safe(PATIENT_LEVEL)

    lines = []
    lines.append("=== QA: Stage2 Failure/Revision Detector ===")
    lines.append("Files:")
    lines.append("  Row hits:      {}".format(ROW_HITS))
    lines.append("  Patient level: {}".format(PATIENT_LEVEL))
    lines.append("")

    # patient-level counts
    if "patient_id" not in pats.columns:
        raise RuntimeError("patient_level missing patient_id column")

    n_pat = int(pats["patient_id"].nunique())
    n_fail = int(pats["Stage2_Failure"].fillna(0).sum()) if "Stage2_Failure" in pats.columns else -1
    n_rev = int(pats["Stage2_Revision"].fillna(0).sum()) if "Stage2_Revision" in pats.columns else -1

    lines.append("Patient-level:")
    lines.append("  Unique patients: {}".format(n_pat))
    if n_fail >= 0:
        lines.append("  Stage2_Failure: {} ({:.1f}%)".format(n_fail, 100.0 * n_fail / n_pat if n_pat else 0.0))
    if n_rev >= 0:
        lines.append("  Stage2_Revision: {} ({:.1f}%)".format(n_rev, 100.0 * n_rev / n_pat if n_pat else 0.0))
    lines.append("")

    # row-level counts
    if rows.empty:
        lines.append("Row hits file is empty (no detected rows).")
    else:
        req_cols = ["patient_id", "S2_Failure_Flag", "S2_Revision_Flag"]
        for c in req_cols:
            if c not in rows.columns:
                raise RuntimeError("row_hits missing required column: {}".format(c))

        rows["S2_Failure_Flag"] = rows["S2_Failure_Flag"].fillna(False).astype(bool)
        rows["S2_Revision_Flag"] = rows["S2_Revision_Flag"].fillna(False).astype(bool)

        n_rows = int(len(rows))
        n_fail_rows = int(rows["S2_Failure_Flag"].sum())
        n_rev_rows = int(rows["S2_Revision_Flag"].sum())
        n_both_rows = int((rows["S2_Failure_Flag"] & rows["S2_Revision_Flag"]).sum())

        lines.append("Row-level:")
        lines.append("  Total hit rows: {}".format(n_rows))
        lines.append("  Failure-hit rows: {}".format(n_fail_rows))
        lines.append("  Revision-hit rows: {}".format(n_rev_rows))
        lines.append("  Both flags rows: {}".format(n_both_rows))
        lines.append("")

        # rule counts
        if "S2_Revision_Rule" in rows.columns:
            rev_rule_counts = rows.loc[rows["S2_Revision_Flag"], "S2_Revision_Rule"].fillna("MISSING_RULE").value_counts()
            rev_rule_counts.to_csv(OUT_REV_RULES, header=["count"])
            lines.append("Top revision rules:")
            for k, v in rev_rule_counts.head(10).items():
                lines.append("  - {}: {}".format(k, int(v)))
            lines.append("")
        else:
            lines.append("No S2_Revision_Rule column found.")
            lines.append("")

        if "S2_Failure_Rule" in rows.columns:
            fail_rule_counts = rows.loc[rows["S2_Failure_Flag"], "S2_Failure_Rule"].fillna("MISSING_RULE").value_counts()
            fail_rule_counts.to_csv(OUT_FAIL_RULES, header=["count"])
            lines.append("Top failure rules:")
            for k, v in fail_rule_counts.head(10).items():
                lines.append("  - {}: {}".format(k, int(v)))
            lines.append("")
        else:
            lines.append("No S2_Failure_Rule column found.")
            lines.append("")

    # Optional cross-check with outcomes file if present
    if os.path.exists(OPTIONAL_OUTCOMES):
        outc = read_csv_safe(OPTIONAL_OUTCOMES)
        if "patient_id" in outc.columns:
            m = outc.merge(pats[["patient_id", "Stage2_Failure", "Stage2_Revision"]], on="patient_id", how="left")
            # choose columns if present
            cols = []
            for c in ["Stage2_MajorComp", "Stage2_MinorComp", "Stage2_Reoperation", "Stage2_Rehospitalization"]:
                if c in m.columns:
                    cols.append(c)

            if cols:
                lines.append("Cross-tabs vs outcomes (counts):")
                # print a few useful relationships if columns exist
                if "Stage2_MajorComp" in m.columns:
                    tab = pd.crosstab(m["Stage2_MajorComp"].fillna(0).astype(int), m["Stage2_Failure"].fillna(0).astype(int))
                    lines.append("  MajorComp x Failure (rows=MajorComp, cols=Failure):")
                    lines.append(str(tab))
                    lines.append("")
                if "Stage2_MajorComp" in m.columns:
                    tab = pd.crosstab(m["Stage2_MajorComp"].fillna(0).astype(int), m["Stage2_Revision"].fillna(0).astype(int))
                    lines.append("  MajorComp x Revision (rows=MajorComp, cols=Revision):")
                    lines.append(str(tab))
                    lines.append("")
            else:
                lines.append("Outcomes file present, but expected Stage2_* outcome columns not found for cross-tabs.")
        else:
            lines.append("Outcomes file present, but missing patient_id. Skipping cross-tabs.")

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
