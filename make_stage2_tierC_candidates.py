# make_stage2_tierC_candidates.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Create a Tier C candidate list from stage2_from_notes_patient_level.csv
#   for later QA / gold comparison, without contaminating the conservative A/B-only final.
#
# Input:
#   - stage2_from_notes_patient_level.csv
#
# Outputs:
#   - stage2_candidates_tierC_patient_level.csv
#   - stage2_candidates_tierC_summary.txt
#
# Notes:
#   - Reads with latin1(errors=replace) to avoid UnicodeDecodeError issues on WVD.
#   - Assumes your extractor output column names from the latest stage2 OP-notes script.

from __future__ import print_function

import sys
import pandas as pd


INFILE = "stage2_from_notes_patient_level.csv"

OUT_CSV = "stage2_candidates_tierC_patient_level.csv"
OUT_SUMMARY = "stage2_candidates_tierC_summary.txt"


def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def to_bool_series(s):
    # handles True/False, 1/0, yes/no, "TRUE"/"FALSE"
    if s is None:
        return pd.Series([], dtype=bool)
    def _one(x):
        if pd.isnull(x):
            return False
        v = str(x).strip().lower()
        return v in ["true", "1", "yes", "y", "t"]
    return s.apply(_one)


def bin_label(delta_days):
    if delta_days is None or pd.isnull(delta_days):
        return None
    try:
        x = int(delta_days)
    except Exception:
        return None
    if x <= 30:
        return "0-30d"
    if x <= 90:
        return "31-90d"
    if x <= 180:
        return "91-180d"
    if x <= 365:
        return "181-365d"
    return ">365d"


def main():
    df = read_csv_safe(INFILE)

    # Expected columns from your stage2 extractor outputs
    required = [
        "patient_id",
        "stage2_tier_best",
        "stage2_after_index",
        "stage2_event_dt_best",
        "stage2_delta_days_from_stage1",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Missing required columns in {}: {}".format(INFILE, missing))

    # Normalize tier
    df["stage2_tier_best_norm"] = (
        df["stage2_tier_best"].fillna("").astype(str).str.strip().str.upper()
    )

    # Normalize after-index flag
    df["stage2_after_index_bool"] = to_bool_series(df["stage2_after_index"])

    # Candidate definition:
    #   Tier C AND after-index AND has a date
    candidates = df[
        (df["stage2_tier_best_norm"] == "C") &
        (df["stage2_after_index_bool"]) &
        (df["stage2_event_dt_best"].notnull())
    ].copy()

    # Light cleanup / ordering
    keep_cols = [
        "patient_id",
        "mrn_from_staging",
        "mrn_from_notes",
        "expander_bucket",
        "stage1_dt",
        "stage2_event_dt_best",
        "stage2_tier_best",
        "stage2_rule_best",
        "stage2_after_index",
        "stage2_delta_days_from_stage1",
        "best_note_type",
        "best_note_id",
        "best_note_dos",
        "best_note_op_date",
        "best_has_proc_section",
        "best_has_implants_section",
        "snippet",
    ]
    keep_cols = [c for c in keep_cols if c in candidates.columns]

    candidates = candidates[keep_cols].copy()

    # Timing bins summary
    timing_bins = None
    if "stage2_delta_days_from_stage1" in candidates.columns:
        d = candidates["stage2_delta_days_from_stage1"].dropna()
        if not d.empty:
            timing_bins = d.apply(bin_label).value_counts()

    # Write outputs
    candidates.to_csv(OUT_CSV, index=False, encoding="utf-8")

    lines = []
    lines.append("=== Stage 2 Tier C Candidates (for later QA) ===")
    lines.append("Python: 3.6.8 compatible | Read encoding: latin1(errors=replace) | Write: utf-8")
    lines.append("Input: {}".format(INFILE))
    lines.append("Output: {}".format(OUT_CSV))
    lines.append("")
    lines.append("Total rows in input: {}".format(len(df)))
    lines.append("Tier C candidates (after-index + date present): {} ({:.1f}%)".format(
        len(candidates),
        (100.0 * len(candidates) / len(df)) if len(df) else 0.0
    ))

    if timing_bins is not None:
        lines.append("")
        lines.append("Timing bins among Tier C candidates (delta days from Stage1):")
        for k, v in timing_bins.items():
            lines.append("  {}: {}".format(k, int(v)))

    lines.append("")
    lines.append("Wrote summary: {}".format(OUT_SUMMARY))

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
