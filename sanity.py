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
#     (If a file lacks these, fall back to other date-like columns if found.)
#
# Outputs:
#   - qa_note_date_fields_summary.txt
#   - qa_note_date_fields_summary.csv
#   - qa_notes_with_event_dt_sample.csv   (small sample across files)
#
# Usage:
#   python qa_build_event_dt_all_notes.py

from __future__ import print_function

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
# Robust readers (Python 3.6-safe)
# -------------------------
def read_csv_safe(path, **kwargs):
    """
    Always returns a DataFrame (non-chunked).
    Use for header reads or small reads.
    """
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass

def iter_csv_chunks_safe(path, **kwargs):
    """
    Always yields DataFrame chunks.
    Requires chunksize in kwargs.
    """
    if "chunksize" not in kwargs:
        raise RuntimeError("iter_csv_chunks_safe requires chunksize=...")

    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        reader = pd.read_csv(f, engine="python", **kwargs)
        for chunk in reader:
            yield chunk
    finally:
        try:
            f.close()
        except Exception:
            pass

def to_dt(series):
    return pd.to_datetime(series, errors="coerce")

def norm_colname(c):
    return re.sub(r"\s+", " ", str(c).strip())

def looks_like_date_col(colname):
    c = str(colname).lower()
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
    series_map = {}
    for k, col in date_cols.items():
        if col and col in chunk.columns:
            series_map[k] = to_dt(chunk[col])
        else:
            series_map[k] = None

    # Canonical rules
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

    # Fallback: other detected date-like columns
    for k in ["enc_date", "contact_date", "created_dt", "signed_dt", "other_date1", "other_date2"]:
        if series_map.get(k) is not None:
            return series_map[k], "FALLBACK:" + k

    # nothing found
    return pd.to_datetime(pd.Series([None] * len(chunk))), "NO_DATE_COLUMNS_FOUND"

def pct(n, d):
    return (100.0 * float(n) / float(d)) if d else 0.0

def main():
    summary_rows = []
    sample_rows = []

    lines = []
    lines.append("=== QA: Canonical EVENT_DT derivation across notes files ===")
    lines.append("Python 3.6.8 compatible | Read encoding: latin1(errors=replace) | Outputs: utf-8")
    lines.append("")

    for file_tag, path in NOTES_FILES:
        lines.append("File tag: {} | Path: {}".format(file_tag, path))

        head = read_csv_safe(path, nrows=5)
        if head is None or head.empty:
            lines.append("  ERROR: Could not read file or file is empty.")
            lines.append("")
            continue

        # Normalize columns
        norm_to_actual = {norm_colname(c): c for c in head.columns.tolist()}
        df_cols = list(norm_to_actual.keys())

        # Detect core date columns first
        col_dos = pick_existing_cols(df_cols, ["NOTE_DATE_OF_SERVICE", "NOTE DOS", "DATE_OF_SERVICE", "DOS"])
        col_op  = pick_existing_cols(df_cols, ["OPERATION_DATE", "OP DATE", "DATE_OF_OPERATION", "SURGERY_DATE"])

        # Detect other likely date columns
        date_like = [c for c in df_cols if looks_like_date_col(c)]
        col_enc = pick_existing_cols(df_cols, ["ENCOUNTER_DATE", "ENC_DATE"])
        col_contact = pick_existing_cols(df_cols, ["CONTACT_DATE", "VISIT_DATE"])
        col_created = pick_existing_cols(df_cols, ["CREATED_DATE", "CREATED_DT", "CREATE_DATE", "CREATE_DTTM"])
        col_signed = pick_existing_cols(df_cols, ["SIGNED_DATE", "SIGNED_DT", "SIGN_DATE", "SIGN_DTTM"])

        chosen = set([c for c in [col_dos, col_op, col_enc, col_contact, col_created, col_signed] if c])
        others = [c for c in date_like if c not in chosen]
        other1 = others[0] if len(others) > 0 else None
        other2 = others[1] if len(others) > 1 else None

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

        # Build usecols for faster scanning
        usecols = []
        for c in [COL_PAT, COL_NOTE_TYPE, COL_NOTE_ID, COL_NOTE_TEXT]:
            if c in df_cols:
                usecols.append(norm_to_actual[c])

        for k in ["dos", "op_date", "enc_date", "contact_date", "created_dt", "signed_dt", "other_date1", "other_date2"]:
            col = date_cols.get(k)
            if col and col not in usecols:
                usecols.append(col)

        total_rows = 0
        nonnull_counts = {k: 0 for k in date_cols.keys()}
        event_nonnull = 0
        used_rule_final = None
        sample_kept = 0

        for chunk in iter_csv_chunks_safe(path, usecols=usecols, chunksize=CHUNKSIZE):
            total_rows += len(chunk)

            for k, col in date_cols.items():
                if col and col in chunk.columns:
                    nonnull_counts[k] += int(chunk[col].notnull().sum())

            event_dt, used_rule = build_event_dt(chunk, file_tag, date_cols)
            if used_rule_final is None:
                used_rule_final = used_rule
            event_nonnull += int(event_dt.notnull().sum())

            if sample_kept < SAMPLE_PER_FILE:
                n_take = min(SAMPLE_PER_FILE - sample_kept, len(chunk))
                sub = chunk.head(n_take).copy()
                sub["FILE_TAG"] = file_tag
                sub["EVENT_DT"] = event_dt.head(n_take)

                keep = ["FILE_TAG"]
                for c in [COL_PAT, COL_NOTE_TYPE, COL_NOTE_ID]:
                    if c in sub.columns:
                        keep.append(c)

                for col in [date_cols.get("dos"), date_cols.get("op_date")]:
                    if col and col in sub.columns and col not in keep:
                        keep.append(col)

                keep.append("EVENT_DT")
                sample_rows.append(sub[keep])
                sample_kept += n_take

        lines.append("  Rows scanned: {}".format(total_rows))
        lines.append("  Canonical EVENT_DT rule used: {}".format(used_rule_final))
        lines.append("  EVENT_DT non-null: {} ({:.1f}%)".format(event_nonnull, pct(event_nonnull, total_rows)))

        for k in ["dos", "op_date", "enc_date", "contact_date", "created_dt", "signed_dt", "other_date1", "other_date2"]:
            col = date_cols.get(k)
            if col:
                nn = int(nonnull_counts.get(k, 0))
                lines.append("    {:>14}: {:<35} non-null={} ({:.1f}%)".format(k, col, nn, pct(nn, total_rows)))

        lines.append("")

        row = {
            "file_tag": file_tag,
            "path": path,
            "rows_scanned": total_rows,
            "event_dt_rule": used_rule_final if used_rule_final else "",
            "event_dt_nonnull": int(event_nonnull),
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
