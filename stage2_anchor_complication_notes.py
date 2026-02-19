# stage2_anchor_complication_notes.py
# Python 3.6.8+ (pandas required)
#
# Goal:
#   For CONFIRMED Stage 2 patients (A/B only), collect ALL notes occurring
#   on/after the Stage 2 date (strict anchor; optional +buffer days).
#
# Inputs:
#   1) stage2_final_ab_patient_level.csv   (must include patient_id and SOME Stage2 date/dt column)
#   2) Notes files (Operation / Clinic / Inpatient)
#
# Outputs:
#   1) stage2_complication_anchor_rows.csv
#   2) stage2_complication_anchor_summary.txt
#
# Design choices:
#   - Canonical EVENT_DT:
#       operation_notes: OPERATION_DATE (preferred) else NOTE_DATE_OF_SERVICE
#       clinic_notes: NOTE_DATE_OF_SERVICE
#       inpatient_notes: NOTE_DATE_OF_SERVICE
#   - Anchor rule: EVENT_DT >= Stage2_DT (+BUFFER_DAYS)
#   - Robust CSV read: latin1(errors=replace), safe for 0xA0 / NBSP bytes
#
# NOTE:
#   This script does NOT classify complications. It only builds the Stage2-anchored
#   note set you will feed into complication extractors.

from __future__ import print_function

import re
import sys
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
STAGE2_AB_PATIENT_LEVEL_CSV = "stage2_final_ab_patient_level.csv"

NOTES_FILES = [
    ("operation_notes", "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"),
    ("clinic_notes",    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"),
    ("inpatient_notes", "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Notes.csv"),
]

OUT_ROWS_CSV = "stage2_complication_anchor_rows.csv"
OUT_SUMMARY_TXT = "stage2_complication_anchor_summary.txt"

CHUNKSIZE = 120000

# If you later decide you want a buffer, set e.g. BUFFER_DAYS = 1 for Stage2+1 day.
BUFFER_DAYS = 0

# Stage2 patient-level columns
COL_PAT_STAGE2 = "patient_id"  # in stage2_final_ab_patient_level.csv

# Notes columns
COL_PAT = "ENCRYPTED_PAT_ID"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_ID = "NOTE_ID"
COL_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"

# Other date columns we might keep (if present)
OTHER_DATE_COLS = ["ADMIT_DATE", "HOSP_ADMSN_TIME"]


# -------------------------
# Robust CSV helpers (Python 3.6 safe)
# -------------------------
def read_csv_safe_df(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def iter_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        for chunk in pd.read_csv(f, engine="python", **kwargs):
            yield chunk
    finally:
        try:
            f.close()
        except Exception:
            pass


def to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def clean_snippet(x, n=320):
    s = "" if x is None else str(x)
    try:
        s = s.replace(u"\xa0", " ")
    except Exception:
        pass
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:n] + "...") if len(s) > n else s


def canonical_event_dt(chunk, file_tag):
    """
    Return (event_dt_series, rule_used_string)
    """
    if file_tag == "operation_notes":
        op_dt = to_dt(chunk[COL_OP_DATE]) if COL_OP_DATE in chunk.columns else pd.Series([pd.NaT] * len(chunk))
        dos_dt = to_dt(chunk[COL_DOS]) if COL_DOS in chunk.columns else pd.Series([pd.NaT] * len(chunk))
        return op_dt.fillna(dos_dt), "OPERATION_DATE -> NOTE_DATE_OF_SERVICE"

    if COL_DOS in chunk.columns:
        return to_dt(chunk[COL_DOS]), "NOTE_DATE_OF_SERVICE"

    for c in [COL_OP_DATE] + OTHER_DATE_COLS:
        if c in chunk.columns:
            return to_dt(chunk[c]), "FALLBACK_" + c

    return pd.Series([pd.NaT] * len(chunk)), "NO_DATE_COLUMN_FOUND"


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


# -------------------------
# NEW: Stage2 date column auto-detection
# -------------------------
def detect_stage2_date_column(df):
    """
    Pick the best Stage2 date column automatically.

    Strategy:
      1) Consider any column whose name contains 'stage2' AND ('date' OR 'dt')
      2) Parse to datetime and choose the column with the most non-null parsed values.
      3) If ties, prefer names that include 'event' or 'best'.
    """
    cols = list(df.columns)
    candidates = []
    for c in cols:
        cl = str(c).lower()
        if "stage2" in cl and ("date" in cl or "dt" in cl):
            candidates.append(c)

    # If none matched, broaden slightly: any column with stage2 in it
    if not candidates:
        for c in cols:
            cl = str(c).lower()
            if "stage2" in cl:
                candidates.append(c)

    if not candidates:
        return None, {}

    stats = {}
    best_col = None
    best_nonnull = -1
    best_score = -1

    for c in candidates:
        parsed = to_dt(df[c])
        nonnull = int(parsed.notnull().sum())

        # name preference score
        cl = str(c).lower()
        score = 0
        if "event" in cl:
            score += 2
        if "best" in cl:
            score += 2
        if "ab" in cl:
            score += 1
        if "final" in cl:
            score += 1

        stats[str(c)] = {"parsed_nonnull": nonnull, "name_score": score}

        # choose by nonnull first, then score
        if nonnull > best_nonnull or (nonnull == best_nonnull and score > best_score):
            best_nonnull = nonnull
            best_score = score
            best_col = c

    # if best has 0 non-null, treat as failure (likely wrong column)
    if best_nonnull <= 0:
        return None, stats

    return best_col, stats


def main():
    # -------------------------
    # 1) Load confirmed Stage2 (A/B) patient-level file
    # -------------------------
    st2 = read_csv_safe_df(STAGE2_AB_PATIENT_LEVEL_CSV)

    if COL_PAT_STAGE2 not in st2.columns:
        raise RuntimeError("Missing '{}' in {}".format(COL_PAT_STAGE2, STAGE2_AB_PATIENT_LEVEL_CSV))

    st2[COL_PAT_STAGE2] = st2[COL_PAT_STAGE2].fillna("").astype(str)

    stage2_col, det_stats = detect_stage2_date_column(st2)
    if stage2_col is None:
        # Provide a helpful, deterministic error message without asking for headers again
        msg_lines = []
        msg_lines.append("No Stage2 date column found in {}.".format(STAGE2_AB_PATIENT_LEVEL_CSV))
        msg_lines.append("I looked for columns containing 'stage2' and 'date'/'dt' (and then any 'stage2' column).")
        msg_lines.append("Columns present ({}): {}".format(len(st2.columns), list(st2.columns)))
        if det_stats:
            msg_lines.append("Candidate parse stats: {}".format(det_stats))
        raise RuntimeError("\n".join(msg_lines))

    # Parse Stage2 column
    st2[stage2_col] = to_dt(st2[stage2_col])

    # Keep only patients with non-null Stage2 date
    st2_valid = st2[st2[stage2_col].notnull()].copy()
    stage2_map = dict(zip(st2_valid[COL_PAT_STAGE2].astype(str), st2_valid[stage2_col]))

    if not stage2_map:
        raise RuntimeError("Detected Stage2 column '{}' but it parsed to all-null datetimes.".format(stage2_col))

    stage2_ids = set(stage2_map.keys())

    print("Loaded Stage2 A/B patients with date present:", len(stage2_ids))
    print("Stage2 date column used:", stage2_col)
    print("Anchor buffer days:", BUFFER_DAYS)

    # -------------------------
    # 2) Scan notes files and keep rows on/after anchor date
    # -------------------------
    kept_frames = []
    summary_by_file = []
    total_rows_scanned = 0
    total_rows_stage2_patients = 0
    total_rows_kept = 0
    all_deltas = []

    for file_tag, path in NOTES_FILES:
        head = read_csv_safe_df(path, nrows=5)
        if COL_PAT not in head.columns:
            raise RuntimeError("Missing '{}' in {}".format(COL_PAT, path))

        usecols = [COL_PAT]
        for c in [COL_NOTE_TYPE, COL_NOTE_ID, COL_NOTE_TEXT, COL_DOS, COL_OP_DATE] + OTHER_DATE_COLS:
            if c in head.columns and c not in usecols:
                usecols.append(c)

        rows_scanned = 0
        rows_stage2 = 0
        rows_kept = 0
        event_rule_used = None

        for chunk in iter_csv_safe(path, usecols=usecols, chunksize=CHUNKSIZE):
            rows_scanned += len(chunk)
            total_rows_scanned += len(chunk)

            chunk[COL_PAT] = chunk[COL_PAT].fillna("").astype(str)
            chunk = chunk[chunk[COL_PAT].isin(stage2_ids)].copy()
            if chunk.empty:
                continue

            rows_stage2 += len(chunk)
            total_rows_stage2_patients += len(chunk)

            event_dt, rule = canonical_event_dt(chunk, file_tag)
            event_rule_used = rule
            chunk["EVENT_DT"] = event_dt

            chunk["STAGE2_DT_RAW"] = chunk[COL_PAT].map(stage2_map)
            chunk["STAGE2_ANCHOR_DT"] = chunk["STAGE2_DT_RAW"] + pd.to_timedelta(BUFFER_DAYS, unit="D")

            chunk = chunk[chunk["EVENT_DT"].notnull() & chunk["STAGE2_ANCHOR_DT"].notnull()].copy()
            if chunk.empty:
                continue

            chunk = chunk[chunk["EVENT_DT"] >= chunk["STAGE2_ANCHOR_DT"]].copy()
            if chunk.empty:
                continue

            chunk["DELTA_DAYS_FROM_STAGE2"] = (chunk["EVENT_DT"] - chunk["STAGE2_DT_RAW"]).dt.days
            rows_kept += len(chunk)
            total_rows_kept += len(chunk)

            all_deltas.extend(chunk["DELTA_DAYS_FROM_STAGE2"].dropna().astype(int).tolist())

            if COL_NOTE_TEXT in chunk.columns:
                chunk["NOTE_SNIPPET"] = chunk[COL_NOTE_TEXT].apply(lambda x: clean_snippet(x, n=320))
            else:
                chunk["NOTE_SNIPPET"] = ""

            out_cols = [
                "file_tag",
                COL_PAT,
                "STAGE2_DT_RAW",
                "STAGE2_ANCHOR_DT",
                "EVENT_DT",
                "DELTA_DAYS_FROM_STAGE2",
            ]
            for c in [COL_NOTE_TYPE, COL_NOTE_ID, COL_DOS, COL_OP_DATE] + OTHER_DATE_COLS:
                if c in chunk.columns:
                    out_cols.append(c)
            out_cols.append("NOTE_SNIPPET")

            chunk["file_tag"] = file_tag
            kept_frames.append(chunk[out_cols])

        summary_by_file.append({
            "file_tag": file_tag,
            "path": path,
            "rows_scanned": rows_scanned,
            "rows_stage2_patients": rows_stage2,
            "rows_kept_stage2_anchored": rows_kept,
            "event_dt_rule": event_rule_used if event_rule_used else "UNKNOWN",
        })

    out = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()

    if not out.empty:
        out = out.sort_values(by=[COL_PAT, "EVENT_DT", "file_tag"], ascending=[True, True, True])

    out.to_csv(OUT_ROWS_CSV, index=False, encoding="utf-8")

    patients_with_any = int(out[COL_PAT].nunique()) if (not out.empty and COL_PAT in out.columns) else 0

    bins_counts = {}
    for d in all_deltas:
        b = bin_label(d)
        if b is None:
            continue
        bins_counts[b] = bins_counts.get(b, 0) + 1

    lines = []
    lines.append("=== Stage 2 Complication Anchor (A/B only) ===")
    lines.append("Python: 3.6.8 compatible | Read encoding: latin1(errors=replace) | Outputs: utf-8")
    lines.append("")
    lines.append("Input Stage2 file: {}".format(STAGE2_AB_PATIENT_LEVEL_CSV))
    lines.append("Stage2 date column used: {}".format(stage2_col))
    lines.append("Stage2 patients with date present: {}".format(len(stage2_ids)))
    lines.append("Anchor rule: EVENT_DT >= (Stage2_DT + {}d buffer)".format(BUFFER_DAYS))
    lines.append("")
    lines.append("Row totals:")
    lines.append("  Total rows scanned (all files): {}".format(total_rows_scanned))
    lines.append("  Rows belonging to Stage2 patients (pre-filter): {}".format(total_rows_stage2_patients))
    lines.append("  Rows kept (Stage2-anchored): {}".format(total_rows_kept))
    lines.append("  Patients with >=1 kept row: {}".format(patients_with_any))
    lines.append("")
    lines.append("Per-file breakdown:")
    for s in summary_by_file:
        lines.append("  - {} | scanned={} | stage2_rows={} | kept={} | EVENT_DT rule={}".format(
            s["file_tag"],
            s["rows_scanned"],
            s["rows_stage2_patients"],
            s["rows_kept_stage2_anchored"],
            s["event_dt_rule"],
        ))

    if bins_counts:
        lines.append("")
        lines.append("Timing bins across kept rows (DELTA_DAYS_FROM_STAGE2):")
        for k in ["0-30d", "31-90d", "91-180d", "181-365d", ">365d"]:
            if k in bins_counts:
                lines.append("  {}: {}".format(k, bins_counts[k]))

    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_ROWS_CSV))
    lines.append("  - {}".format(OUT_SUMMARY_TXT))

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
