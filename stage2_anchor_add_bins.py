# stage2_anchor_add_bins.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Keep Stage2-anchored rows with NO upper time limit, include same-day (>= Stage2 date),
#   and add timing bins so you can optionally filter later (e.g., exclude delta==0).
#
# Inputs:
#   1) stage2_final_ab_patient_level.csv   (AB-confirmed Stage2 patients with a Stage2 date)
#   2) stage2_complication_anchor_rows.csv (your Stage2-anchored rows output; already post Stage2)
#
# Outputs:
#   1) stage2_anchor_rows_with_bins.csv
#   2) stage2_anchor_rows_excluding_delta0.csv   (optional convenience file)
#   3) stage2_anchor_bins_summary.txt
#
# Notes:
#   - Python 3.6 safe
#   - Reads CSVs with latin1(errors=replace) to avoid decode crashes
#   - Does NOT enforce 1-year cutoff
#   - Same-day included by construction (DELTA_DAYS_FROM_STAGE2 >= 0)

from __future__ import print_function

import sys
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
AB_STAGE2_PATIENTS_CSV = "stage2_final_ab_patient_level.csv"
ANCHOR_ROWS_CSV = "stage2_complication_anchor_rows.csv"

OUT_WITH_BINS = "stage2_anchor_rows_with_bins.csv"
OUT_EXCLUDE_DELTA0 = "stage2_anchor_rows_excluding_delta0.csv"
OUT_SUMMARY = "stage2_anchor_bins_summary.txt"


# -------------------------
# Robust CSV read (Python 3.6 safe)
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


def find_first_existing_col(df, candidates):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def to_dt(series):
    return pd.to_datetime(series, errors="coerce")


def time_bin_days(x):
    if x is None or pd.isnull(x):
        return "NA"
    try:
        d = int(x)
    except Exception:
        return "NA"
    if d < 0:
        return "<0"
    if d == 0:
        return "0d"
    if d <= 30:
        return "1-30d"
    if d <= 90:
        return "31-90d"
    if d <= 180:
        return "91-180d"
    if d <= 365:
        return "181-365d"
    return ">365d"


def main():
    # -------------------------
    # Load AB-confirmed Stage2 patients (source of Stage2 date)
    # -------------------------
    ab = read_csv_safe(AB_STAGE2_PATIENTS_CSV)
    if "patient_id" not in ab.columns:
        raise RuntimeError("Missing column 'patient_id' in {}".format(AB_STAGE2_PATIENTS_CSV))

    # robust Stage2 date column detection
    stage2_candidates = [
        "stage2_event_dt_best",
        "stage2_dt_best",
        "stage2_date_ab",
        "stage2_date",
        "stage2_dt",
        "stage2_date_best",
        "stage2_dt_best_ab",
    ]
    stage2_col = find_first_existing_col(ab, stage2_candidates)
    if stage2_col is None:
        raise RuntimeError(
            "No Stage2 date column found in {}. Expected one of: {}".format(
                AB_STAGE2_PATIENTS_CSV, stage2_candidates
            )
        )

    ab["patient_id"] = ab["patient_id"].fillna("").astype(str)
    ab["STAGE2_DT"] = to_dt(ab[stage2_col])

    ab = ab[ab["patient_id"] != ""].copy()
    ab = ab[ab["STAGE2_DT"].notnull()].copy()

    if ab.empty:
        raise RuntimeError("No AB patients with non-null Stage2 date found in {}".format(AB_STAGE2_PATIENTS_CSV))

    stage2_map = dict(zip(ab["patient_id"].tolist(), ab["STAGE2_DT"].tolist()))
    ab_patients = set(stage2_map.keys())

    # -------------------------
    # Load Stage2-anchored rows
    # -------------------------
    ar = read_csv_safe(ANCHOR_ROWS_CSV)

    # Required: patient id + event date + delta days (if present)
    if "ENCRYPTED_PAT_ID" in ar.columns and "patient_id" not in ar.columns:
        ar = ar.rename(columns={"ENCRYPTED_PAT_ID": "patient_id"})

    if "patient_id" not in ar.columns:
        raise RuntimeError("Anchor rows file must include 'patient_id' or 'ENCRYPTED_PAT_ID': {}".format(ANCHOR_ROWS_CSV))

    # Event date column (depends on your anchor script)
    event_candidates = ["EVENT_DT", "event_dt", "EVENT_DATE", "NOTE_DATE_OF_SERVICE", "OPERATION_DATE"]
    event_col = find_first_existing_col(ar, event_candidates)
    if event_col is None:
        raise RuntimeError(
            "No EVENT date column found in {}. Expected one of: {}".format(ANCHOR_ROWS_CSV, event_candidates)
        )

    # Delta days column (optional; we can recompute if missing)
    delta_candidates = ["DELTA_DAYS_FROM_STAGE2", "delta_days_from_stage2", "delta_days"]
    delta_col = find_first_existing_col(ar, delta_candidates)

    ar["patient_id"] = ar["patient_id"].fillna("").astype(str)
    pre_rows = len(ar)

    # Keep only AB patients (since this is AB-only Stage2 anchoring)
    ar = ar[ar["patient_id"].isin(ab_patients)].copy()
    kept_prefilter = len(ar)

    # Attach Stage2 date
    ar["STAGE2_DT"] = ar["patient_id"].map(stage2_map)
    ar["EVENT_DT"] = to_dt(ar[event_col])

    # Compute delta if needed
    if delta_col is None:
        ar["DELTA_DAYS_FROM_STAGE2"] = (ar["EVENT_DT"] - ar["STAGE2_DT"]).dt.days
        delta_col = "DELTA_DAYS_FROM_STAGE2"
    else:
        # normalize into canonical name
        ar["DELTA_DAYS_FROM_STAGE2"] = pd.to_numeric(ar[delta_col], errors="coerce")

    # Ensure same-day included: keep delta >= 0
    ar = ar[ar["DELTA_DAYS_FROM_STAGE2"].notnull()].copy()
    ar = ar[ar["DELTA_DAYS_FROM_STAGE2"] >= 0].copy()
    kept_stage2 = len(ar)

    # Add bins
    ar["TIME_BIN_STAGE2"] = ar["DELTA_DAYS_FROM_STAGE2"].apply(time_bin_days)

    # Optional convenience exclusion: remove delta==0
    ar_no0 = ar[ar["DELTA_DAYS_FROM_STAGE2"] != 0].copy()

    # Write outputs
    ar.to_csv(OUT_WITH_BINS, index=False, encoding="utf-8")
    ar_no0.to_csv(OUT_EXCLUDE_DELTA0, index=False, encoding="utf-8")

    # Summary
    total_patients = len(ab_patients)
    patients_with_rows = int(ar["patient_id"].nunique()) if kept_stage2 else 0
    bin_counts = ar["TIME_BIN_STAGE2"].value_counts() if kept_stage2 else pd.Series(dtype=int)

    lines = []
    lines.append("=== Stage2 Anchored Rows (AB only) + Timing Bins ===")
    lines.append("Python: 3.6.8 compatible | Read encoding: latin1(errors=replace) | Write: utf-8")
    lines.append("")
    lines.append("AB Stage2 file: {}".format(AB_STAGE2_PATIENTS_CSV))
    lines.append("  Stage2 date column used: {}".format(stage2_col))
    lines.append("Anchor rows file: {}".format(ANCHOR_ROWS_CSV))
    lines.append("  Event date column used: {}".format(event_col))
    lines.append("")
    lines.append("AB Stage2 patients (with Stage2 date): {}".format(total_patients))
    lines.append("Anchor rows scanned (all patients): {}".format(pre_rows))
    lines.append("Anchor rows after AB patient pre-filter: {}".format(kept_prefilter))
    lines.append("Anchor rows kept (DELTA_DAYS_FROM_STAGE2 >= 0; no upper limit): {}".format(kept_stage2))
    lines.append("Patients with >=1 kept row: {}".format(patients_with_rows))
    lines.append("")
    lines.append("Timing bins across kept rows (TIME_BIN_STAGE2):")
    if not bin_counts.empty:
        # stable order
        order = ["0d", "1-30d", "31-90d", "91-180d", "181-365d", ">365d"]
        for k in order:
            if k in bin_counts.index:
                lines.append("  {:>8}: {}".format(k, int(bin_counts[k])))
        # any other bins
        for k, v in bin_counts.items():
            if k not in order:
                lines.append("  {:>8}: {}".format(str(k), int(v)))
    else:
        lines.append("  (no rows)")

    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_WITH_BINS))
    lines.append("  - {}".format(OUT_EXCLUDE_DELTA0))
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
