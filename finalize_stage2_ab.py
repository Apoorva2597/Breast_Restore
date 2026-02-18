# finalize_stage2_ab.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Finalize Stage 2 dates conservatively using ONLY Tier A/B hits
#   from stage2_from_notes_patient_level.csv (your current extractor output).
#
# Definition (conservative):
#   stage2_confirmed_flag = 1 if:
#       - stage2_after_index == True
#       - stage2_tier_best in {"A","B"}
#       - stage2_event_dt_best is not null
#   stage2_date_final = stage2_event_dt_best when confirmed, else blank
#
# Inputs:
#   - stage2_from_notes_patient_level.csv
#
# Outputs:
#   - stage2_final_ab_patient_level.csv
#   - stage2_final_ab_summary.txt
#
# Notes:
#   - Reads using latin1(errors=replace) for robustness on WVD
#   - Writes UTF-8
#
from __future__ import print_function

import sys
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
IN_PATIENT_LEVEL = "stage2_from_notes_patient_level.csv"

OUT_FINAL = "stage2_final_ab_patient_level.csv"
OUT_SUMMARY = "stage2_final_ab_summary.txt"


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


def to_bool(x):
    # Accept True/False, 1/0, yes/no, y/n
    s = "" if x is None else str(x).strip().lower()
    return s in ["true", "1", "yes", "y"]


def to_dt(x):
    return pd.to_datetime(x, errors="coerce")


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
    df = read_csv_safe(IN_PATIENT_LEVEL)

    # Required columns from your stage2 extractor output
    required = [
        "patient_id",
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
        "snippet",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Missing required columns in {}: {}".format(IN_PATIENT_LEVEL, missing))

    # Normalize
    df["patient_id"] = df["patient_id"].fillna("").astype(str)
    df["stage1_dt"] = to_dt(df["stage1_dt"])
    df["stage2_event_dt_best"] = to_dt(df["stage2_event_dt_best"])

    df["stage2_tier_best"] = df["stage2_tier_best"].fillna("").astype(str).str.strip().str.upper()
    df["stage2_after_index_bool"] = df["stage2_after_index"].apply(to_bool)

    # Conservative confirmation rule: AFTER-index AND Tier A/B AND date present
    df["stage2_confirmed_flag"] = (
        df["stage2_after_index_bool"]
        & df["stage2_tier_best"].isin(["A", "B"])
        & df["stage2_event_dt_best"].notnull()
    ).astype(int)

    # Final date
    df["stage2_date_final"] = pd.NaT
    df.loc[df["stage2_confirmed_flag"] == 1, "stage2_date_final"] = df.loc[
        df["stage2_confirmed_flag"] == 1, "stage2_event_dt_best"
    ]

    # Keep “used” evidence fields only when confirmed (prevents confusion downstream)
    evidence_cols = [
        "stage2_tier_best",
        "stage2_rule_best",
        "stage2_delta_days_from_stage1",
        "best_note_type",
        "best_note_id",
        "best_note_dos",
        "best_note_op_date",
        "snippet",
    ]
    for c in evidence_cols:
        df.loc[df["stage2_confirmed_flag"] == 0, c] = None

    # Create an output frame with a stable schema
    out_cols = [
        "patient_id",
        "stage1_dt",
        "stage2_confirmed_flag",
        "stage2_date_final",
        "stage2_tier_best",
        "stage2_rule_best",
        "stage2_delta_days_from_stage1",
        "best_note_type",
        "best_note_id",
        "best_note_dos",
        "best_note_op_date",
        "snippet",
    ]

    # If expander_bucket exists, carry it through (useful later)
    if "expander_bucket" in df.columns:
        out_cols.insert(3, "expander_bucket")
    if "mrn_from_staging" in df.columns:
        out_cols.insert(2, "mrn_from_staging")
    if "mrn_from_notes" in df.columns and "mrn_from_notes" not in out_cols:
        out_cols.insert(3, "mrn_from_notes")

    out = df[out_cols].copy()

    # Summary stats
    n_total = len(out)
    n_confirmed = int(out["stage2_confirmed_flag"].sum())
    pct_confirmed = (100.0 * n_confirmed / n_total) if n_total else 0.0

    tier_counts = out.loc[out["stage2_confirmed_flag"] == 1, "stage2_tier_best"].value_counts(dropna=False)
    timing_bins = None
    if n_confirmed > 0 and "stage2_delta_days_from_stage1" in out.columns:
        d = pd.to_numeric(out.loc[out["stage2_confirmed_flag"] == 1, "stage2_delta_days_from_stage1"], errors="coerce").dropna()
        if not d.empty:
            timing_bins = d.apply(bin_label).value_counts()

    # Write outputs
    out.to_csv(OUT_FINAL, index=False, encoding="utf-8")

    lines = []
    lines.append("=== Stage 2 Finalization (A/B only) ===")
    lines.append("Input: {}".format(IN_PATIENT_LEVEL))
    lines.append("Output: {}".format(OUT_FINAL))
    lines.append("")
    lines.append("Total patients in file: {}".format(n_total))
    lines.append("Stage2 confirmed (Tier A/B + after-index + date present): {} ({:.1f}%)".format(n_confirmed, pct_confirmed))
    lines.append("")

    lines.append("Confirmed tier distribution:")
    if tier_counts.empty:
        lines.append("  (none)")
    else:
        for k, v in tier_counts.items():
            lines.append("  {}: {}".format(str(k), int(v)))

    if timing_bins is not None:
        lines.append("")
        lines.append("Timing bins among confirmed (delta days from Stage1):")
        for k, v in timing_bins.items():
            lines.append("  {}: {}".format(k, int(v)))

    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_FINAL))
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
