# qa_build_event_dt_all_notes.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Create a canonical EVENT_DT for each note row (for later windowing vs Stage2 date),
#   and produce QA summaries of which date fields exist and how often they're populated.
#
# Design:
#   - Robust read: latin1(errors=replace) to avoid UnicodeDecodeError (0xA0, etc.)
#   - Auto-detect common date columns (NOTE_DATE_OF_SERVICE, OPERATION_DATE, etc.)
#   - Canonical date rules:
#       * Operation notes: prefer OPERATION_DATE, then NOTE_DATE_OF_SERVICE
#       * Clinic/Inpatient notes: prefer NOTE_DATE_OF_SERVICE, then OPERATION_DATE
#     (If a file lacks these, we fall back to other date-like columns if found.)
#
# Outputs:
#   - qa_note_date_fields_summary.txt
#   - qa_note_date_fields_summary.csv
#   - qa_notes_with_event_dt_sample.csv   (small sample across files)
#
# Usage:
#   python qa_build_event_dt_all_notes.py

from __future__ import print_function

import os
import re
import sys
import pandas as pd

# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
NOTES_FILES = [
    ("operation_notes", "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"),
    ("clinic_notes",    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"),
    ("inpatient_notes", "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Notes.csv"),
]

OUT_SUMMARY_TXT = "qa_note_date_fields_summary.txt"
OUT_SUMMARY_CSV = "qa_note_date_fields_summary.csv"
OUT_SAMPLE_CSV = "qa_notes_with_event_dt_sample.csv"

CHUNKSIZE = 150000
SAMPLE_PER_FILE = 200  # keep small; just for sanity checks

# preferred core columns if present
COL_PAT = "ENCRYPTED_PAT_ID"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_ID = "NOTE_ID"
COL_NOTE_TEXT = "NOTE_TEXT"

# -------------------------
# Robust chunk reader
# -------------------------
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

def read_head(path, nrows=5):
    it = iter_csv_safe(path, nrows=nrows)
    for chunk in it:
        return chunk
    return None

def to_dt(series):
    # pandas handles many formats; errors=coerce turns bad values into NaT
    return pd.to_datetime(series, errors="coerce")

def norm_colname(c):
    return re.sub(r"\s+", " ", str(c).strip())

def looks_like_date_col(colname):
    c = str(colname).lower()
    # broad but safe heuristic; we only use these if core cols absent
    keys = ["date", "dt", "time", "timestamp", "dos", "service", "operation"]
    return any(k in c for k in keys)

def pick_existing_cols(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None

def build_event_dt(chunk, file_tag, date_cols):
    """
    date_cols: dict of detected columns for this file (strings or None)
    Returns (event_dt_series, used_rule_string)
    """
    # Extract candidate series if present
    # All are string-like; convert to datetime safely
    series_map = {}
    for k, col in date_cols.items():
        if col and col in chunk.columns:
            series_map[k] = to_dt(chunk[col])
        else:
            series_map[k] = None

    # Canonical rules (explicit, stable)
    if file_tag == "operation_notes":
        # prefer OPERATION_DATE, then NOTE_DATE_OF_SERVICE
        if series_map.get("op_date") is not None:
            event_dt = series_map["op_date"]
            rule = "OPERATION_DATE"
            if series_map.get("dos") is not None:
                event_dt = event_dt.fillna(series_map["dos"])
                rule = "OPERATION_DATE -> NOTE_DATE_OF_SERVICE"
            return event_dt, rule

        if series_map.get("dos") is not None:
            return series_map["dos"], "NOTE_DATE_OF_SERVICE"

    else:
        # clinic/inpatient: prefer NOTE_DATE_OF_SERVICE, then OPERATION_DATE
        if series_map.get("dos") is not None:
            event_dt = series_map["dos"]
            rule = "NOTE_DATE_OF_SERVICE"
            if series_map.get("op_date") is not None:
                event_dt = event_dt.fillna(series_map["op_date"])
                rule = "NOTE_DATE_OF_SERVICE -> OPERATION_DATE"
            return event_dt, rule

        if series_map.get("op_date") is not None:
            return series_map["op_date"], "OPERATION_DATE"

    # Fallback: other detected date-like columns in priority order
    for k in ["enc_date", "contact_date", "created_dt", "signed_dt", "other_date1", "other_date2"]:
        if series_map.get(k) is not None:
            return series_map[k], "FALLBACK:" + k

    # If nothing found
    return pd.to_datetime(pd.Series([None] * len(chunk))), "NO_DATE_COLUMNS_FOUND"

def main():
    summary_rows = []
    sample_rows = []

    lines = []
    lines.append("=== QA: Canonical EVENT_DT derivation across notes files ===")
    lines.append("Python 3.6.8 compatible | Read encoding: latin1(errors=replace) | Outputs: utf-8")
    lines.append("")

    for file_tag, path in NOTES_FILES:
        lines.append("File tag: {} | Path: {}".format(file_tag, path))

        head = read_head(path, nrows=5)
        if head is None:
            lines.append("  ERROR: Could not read file.")
            lines.append("")
            continue

        # Normalize columns
        cols = [norm_colname(c) for c in head.columns.tolist()]
        # Map normalized -> actual (to avoid surprises if whitespace differs)
        norm_to_actual = {norm_colname(c): c for c in head.columns.tolist()}
        df_cols = list(norm_to_actual.keys())

        # Detect core date columns first (exact matches)
        col_dos = pick_existing_cols(df_cols, ["NOTE_DATE_OF_SERVICE", "NOTE DOS", "DATE_OF_SERVICE", "DOS"])
        col_op  = pick_existing_cols(df_cols, ["OPERATION_DATE", "OP DATE", "DATE_OF_OPERATION", "SURGERY_DATE"])

        # Detect other likely date columns (best-effort)
        # Keep these conservative; only used if core cols absent
        date_like = [c for c in df_cols if looks_like_date_col(c)]
        # Try common alternates
        col_enc = pick_existing_cols(df_cols, ["ENCOUNTER_DATE", "CONTACT_DATE", "VISIT_DATE"])
        col_contact = pick_existing_cols(df_cols, ["CONTACT_DATE", "VISIT_DATE"])
        col_created = pick_existing_cols(df_cols, ["CREATED_DATE", "CREATED_DT", "CREATE_DATE", "CREATE_DTTM"])
        col_signed = pick_existing_cols(df_cols, ["SIGNED_DATE", "SIGNED_DT", "SIGN_DATE", "SIGN_DTTM"])

        # Create two "other" fallbacks from date_like that are not already chosen
        chosen = set([c for c in [col_dos, col_op, col_enc, col_contact, col_created, col_signed] if c])
        others = [c for c in date_like if c not in chosen]
        other1 = others[0] if len(others) > 0 else None
        other2 = others[1] if len(others) > 1 else None

        # Translate normalized -> actual names
        def actual(col_norm):
            return norm_to_actual.get(col_norm) if col_norm else None

        date_cols = {
            "dos": actual(col_dos),
            "op_date": actual(col_op),
            "enc_date": actual(col_enc),
            "contact_date": actual(col_contact),
            "created_dt": actual(col_created),
            "signed_dt": actual(col_signed),
            "other_date1": actual(other1),
            "other_date2": actual(other2),
        }

        # For QA: scan file in chunks and compute non-null rates for detected columns
        usecols = []
        for c in [COL_PAT, COL_NOTE_TYPE, COL_NOTE_ID, COL_NOTE_TEXT]:
            if c in df_cols:
                usecols.append(norm_to_actual[c])

        # Add date cols if present
        for k in ["dos", "op_date", "enc_date", "contact_date", "created_dt", "signed_dt", "other_date1", "other_date2"]:
            col = date_cols.get(k)
            if col and col not in usecols:
                usecols.append(col)

        total_rows = 0
        nonnull_counts = {k: 0 for k in date_cols.keys()}
        event_nonnull = 0
        used_rule_final = None

        # sample collector
        sample_kept = 0

        for chunk in iter_csv_safe(path, usecols=usecols, chunksize=CHUNKSIZE):
            total_rows += len(chunk)

            # Count non-null for each detected date col
            for k, col in date_cols.items():
                if col and col in chunk.columns:
                    nonnull_counts[k] += int(chunk[col].notnull().sum())

            # Build EVENT_DT for this chunk using canonical rules
            event_dt, used_rule = build_event_dt(chunk, file_tag, date_cols)
            used_rule_final = used_rule_final or used_rule
            event_nonnull += int(event_dt.notnull().sum())

            # Save sample rows (small)
            if sample_kept < SAMPLE_PER_FILE:
                n_take = min(SAMPLE_PER_FILE - sample_kept, len(chunk))
                sub = chunk.head(n_take).copy()
                sub["FILE_TAG"] = file_tag
                sub["EVENT_DT"] = event_dt.head(n_take)
                # Keep only a few columns
                keep = ["FILE_TAG"]
                for c in [COL_PAT, COL_NOTE_TYPE, COL_NOTE_ID]:
                    if c in sub.columns:
                        keep.append(c)
                # Keep the actual chosen date columns too
                for col in [date_cols.get("dos"), date_cols.get("op_date")]:
                    if col and col in sub.columns and col not in keep:
                        keep.append(col)
                keep.append("EVENT_DT")
                sample_rows.append(sub[keep])
                sample_kept += n_take

        # Summarize
        def pct(n, d):
            return (100.0 * float(n) / float(d)) if d else 0.0

        lines.append("  Rows scanned: {}".format(total_rows))
        lines.append("  Canonical EVENT_DT rule used: {}".format(used_rule_final))
        lines.append("  EVENT_DT non-null: {} ({:.1f}%)".format(event_nonnull, pct(event_nonnull, total_rows)))

        # Date column availability summary
        for k in ["dos", "op_date", "enc_date", "contact_date", "created_dt", "signed_dt", "other_date1", "other_date2"]:
            col = date_cols.get(k)
            if col:
                nn = nonnull_counts.get(k, 0)
                lines.append("    {:>14}: {:<35} non-null={} ({:.1f}%)".format(
                    k, col, nn, pct(nn, total_rows)
                ))

        lines.append("")

        row = {
            "file_tag": file_tag,
            "path": path,
            "rows_scanned": total_rows,
            "event_dt_rule": used_rule_final,
            "event_dt_nonnull": event_nonnull,
            "event_dt_nonnull_pct": round(pct(event_nonnull, total_rows), 2),
        }
        for k, col in date_cols.items():
            row["col_" + k] = col if col else ""
            row["nonnull_" + k] = int(nonnull_counts.get(k, 0))
            row["nonnull_pct_" + k] = round(pct(nonnull_counts.get(k, 0), total_rows), 2)
        summary_rows.append(row)

    # Write outputs
    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8")

    if sample_rows:
        sample_df = pd.concat(sample_rows, ignore_index=True)
        sample_df.to_csv(OUT_SAMPLE_CSV, index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(OUT_SAMPLE_CSV, index=False, encoding="utf-8")

    print("\n".join(lines))
    print("Wrote:")
    print("  - {}".format(OUT_SUMMARY_TXT))
    print("  - {}".format(OUT_SUMMARY_CSV))
    print("  - {}".format(OUT_SAMPLE_CSV))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
