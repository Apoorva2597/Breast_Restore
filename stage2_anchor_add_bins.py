# stage2_anchor_add_bins.py
# Python 3.6.8+ (pandas required)
#
# Fix:
#   Your AB file doesn't have the exact Stage2 date column names we expected.
#   This version auto-detects the Stage2 date column by scanning headers for
#   stage2 + (date|dt) and excluding delta/derived columns. It also prints the
#   candidate columns it found so you can sanity-check quickly.
#
# Purpose:
#   Keep Stage2-anchored rows with NO upper time limit, include same-day (>= Stage2 date),
#   and add timing bins so you can optionally filter later (e.g., exclude delta==0).
#
# Inputs:
#   1) stage2_final_ab_patient_level.csv   (AB-confirmed Stage2 patients with a Stage2 date)
#   2) stage2_complication_anchor_rows.csv (Stage2-anchored rows output; already post Stage2)
#
# Outputs:
#   1) stage2_anchor_rows_with_bins.csv
#   2) stage2_anchor_rows_excluding_delta0.csv   (optional convenience file)
#   3) stage2_anchor_bins_summary.txt

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


def to_dt(series):
    return pd.to_datetime(series, errors="coerce")


def norm_colname(c):
    return str(c).strip().lower().replace(" ", "_")


def auto_detect_stage2_date_col(df):
    """
    Heuristic:
      - must contain 'stage2' or 'stage_2' or 'stage ii' (rare) in normalized name
      - and contain 'date' or 'dt' or 'datetime' or 'event'
      - exclude delta / days / diff / bin / tier / rule / after_index / flag columns
    Returns best column name or None.
    Also returns list of candidates (in priority order).
    """
    cols = list(df.columns)
    ncols = [(c, norm_colname(c)) for c in cols]

    bad_tokens = ["delta", "days", "diff", "bin", "tier", "rule", "after", "flag", "rank", "timing"]
    stage2_tokens = ["stage2", "stage_2", "stageii", "stage_ii", "stage-2", "stage2nd", "second_stage"]
    date_tokens = ["date", "dt", "datetime", "time", "event"]

    candidates = []
    for orig, n in ncols:
        if any(t in n for t in stage2_tokens):
            if any(t in n for t in date_tokens):
                if not any(bt in n for bt in bad_tokens):
                    candidates.append(orig)

    # Prefer more "date-like" columns first
    def score(col):
        n = norm_colname(col)
        s = 0
        # stronger signals
        if "event" in n:
            s += 5
        if "best" in n:
            s += 4
        if "recon" in n:
            s += 1
        if "op" in n or "surgery" in n:
            s += 1
        # prefer explicit "date" over generic "dt"
        if "date" in n:
            s += 3
        if "dt" in n:
            s += 2
        # avoid suspicious ones
        if "string" in n:
            s -= 2
        return -s  # sort ascending

    candidates_sorted = sorted(candidates, key=score)

    # If nothing found, do a fallback: any column containing stage2 and NOT bad tokens
    if not candidates_sorted:
        fallback = []
        for orig, n in ncols:
            if "stage2" in n or "stage_2" in n:
                if not any(bt in n for bt in bad_tokens):
                    fallback.append(orig)
        candidates_sorted = fallback

    best = candidates_sorted[0] if candidates_sorted else None
    return best, candidates_sorted


def auto_detect_event_dt_col(df):
    """
    For anchor rows, pick EVENT_DT if present; otherwise first reasonable date column.
    """
    cols = list(df.columns)
    norm = {c: norm_colname(c) for c in cols}

    preferred = ["event_dt", "event_date"]
    for p in preferred:
        for c, n in norm.items():
            if n == p:
                return c

    # common UM columns
    for p in ["note_date_of_service", "operation_date", "admit_date", "hosp_admsn_time"]:
        for c, n in norm.items():
            if n == p:
                return c

    # fallback: any column containing 'date' or 'dt' but not delta/days
    bad = ["delta", "days", "diff", "bin"]
    dateish = []
    for c, n in norm.items():
        if ("date" in n or "dt" in n or "time" in n) and (not any(b in n for b in bad)):
            dateish.append(c)
    return dateish[0] if dateish else None


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

    # patient id column (support either)
    if "patient_id" not in ab.columns:
        # allow ENCRYPTED_PAT_ID here too
        if "ENCRYPTED_PAT_ID" in ab.columns:
            ab = ab.rename(columns={"ENCRYPTED_PAT_ID": "patient_id"})
        else:
            raise RuntimeError(
                "Missing 'patient_id' (or 'ENCRYPTED_PAT_ID') in {}".format(AB_STAGE2_PATIENTS_CSV)
            )

    stage2_col, stage2_candidates = auto_detect_stage2_date_col(ab)
    if stage2_col is None:
        raise RuntimeError(
            "No Stage2 date column found in {}.\n"
            "I scanned headers for stage2+(date|dt|event) and excluded delta/tier/rule fields.\n"
            "Try opening the file and confirm the Stage2 date column name.\n"
            "Columns present: {}".format(AB_STAGE2_PATIENTS_CSV, list(ab.columns))
        )

    ab["patient_id"] = ab["patient_id"].fillna("").astype(str)
    ab["STAGE2_DT"] = to_dt(ab[stage2_col])

    ab = ab[ab["patient_id"] != ""].copy()
    ab = ab[ab["STAGE2_DT"].notnull()].copy()

    if ab.empty:
        raise RuntimeError(
            "Found Stage2 column '{}' but it parsed to all-null dates. "
            "Check formatting in {}.".format(stage2_col, AB_STAGE2_PATIENTS_CSV)
        )

    stage2_map = dict(zip(ab["patient_id"].tolist(), ab["STAGE2_DT"].tolist()))
    ab_patients = set(stage2_map.keys())

    # -------------------------
    # Load Stage2-anchored rows
    # -------------------------
    ar = read_csv_safe(ANCHOR_ROWS_CSV)

    if "patient_id" not in ar.columns:
        if "ENCRYPTED_PAT_ID" in ar.columns:
            ar = ar.rename(columns={"ENCRYPTED_PAT_ID": "patient_id"})
        else:
            raise RuntimeError(
                "Anchor rows file must include 'patient_id' or 'ENCRYPTED_PAT_ID': {}".format(ANCHOR_ROWS_CSV)
            )

    event_col = auto_detect_event_dt_col(ar)
    if event_col is None:
        raise RuntimeError(
            "No usable event date column found in {}. Columns present: {}".format(ANCHOR_ROWS_CSV, list(ar.columns))
        )

    # delta days column (optional)
    delta_col = None
    for c in ar.columns:
        n = norm_colname(c)
        if n in ["delta_days_from_stage2", "delta_days_stage2", "delta_days"]:
            delta_col = c
            break

    ar["patient_id"] = ar["patient_id"].fillna("").astype(str)
    pre_rows = len(ar)

    # Keep only AB patients
    ar = ar[ar["patient_id"].isin(ab_patients)].copy()
    kept_prefilter = len(ar)

    # Attach Stage2 date + parse event date
    ar["STAGE2_DT"] = ar["patient_id"].map(stage2_map)
    ar["EVENT_DT"] = to_dt(ar[event_col])

    # Compute delta if missing
    if delta_col is None:
        ar["DELTA_DAYS_FROM_STAGE2"] = (ar["EVENT_DT"] - ar["STAGE2_DT"]).dt.days
    else:
        ar["DELTA_DAYS_FROM_STAGE2"] = pd.to_numeric(ar[delta_col], errors="coerce")

    # Keep same-day and after (>= 0). NO upper limit.
    ar = ar[ar["DELTA_DAYS_FROM_STAGE2"].notnull()].copy()
    ar = ar[ar["DELTA_DAYS_FROM_STAGE2"] >= 0].copy()
    kept_stage2 = len(ar)

    # Add bins
    ar["TIME_BIN_STAGE2"] = ar["DELTA_DAYS_FROM_STAGE2"].apply(time_bin_days)

    # Convenience exclusion: remove delta==0
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
    lines.append("  Stage2 date column auto-detected: {}".format(stage2_col))
    if stage2_candidates:
        lines.append("  Other Stage2-like candidates found: {}".format(", ".join([str(x) for x in stage2_candidates[:10]])))
    lines.append("Anchor rows file: {}".format(ANCHOR_ROWS_CSV))
    lines.append("  Event date column auto-detected: {}".format(event_col))
    lines.append("")
    lines.append("AB Stage2 patients (with Stage2 date): {}".format(total_patients))
    lines.append("Anchor rows scanned (all patients): {}".format(pre_rows))
    lines.append("Anchor rows after AB patient pre-filter: {}".format(kept_prefilter))
    lines.append("Anchor rows kept (DELTA_DAYS_FROM_STAGE2 >= 0; no upper limit): {}".format(kept_stage2))
    lines.append("Patients with >=1 kept row: {}".format(patients_with_rows))
    lines.append("")
    lines.append("Timing bins across kept rows (TIME_BIN_STAGE2):")
    if not bin_counts.empty:
        order = ["0d", "1-30d", "31-90d", "91-180d", "181-365d", ">365d"]
        for k in order:
            if k in bin_counts.index:
                lines.append("  {:>8}: {}".format(k, int(bin_counts[k])))
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
