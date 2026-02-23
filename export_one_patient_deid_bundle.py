#!/usr/bin/env python3
# export_one_patient_deid_bundle.py
# Python 3.6.8 compatible
#
# What this script does (single patient):
#  1) Bundles DE-ID note text from your DE-ID note CSVs (clinic v3, op v4, ip v5)
#     into: timeline.csv + ALL_NOTES_COMBINED.txt + per-note text files.
#  2) ALSO pulls encounter-level dates from your HPI11256 encounter CSVs
#     into: encounters_timeline.csv + stage2_anchor_summary.txt
#
# Default paths are hardcoded for YOUR environment so you don't have to type IDs.
#
# You can still override via CLI if you want, but you can also just run:
#   python export_one_patient_deid_bundle.py

from __future__ import print_function
import os
import re
import argparse
import random
import datetime
import pandas as pd


# ----------------------------
# HARD-CODED DEFAULTS (YOUR SETUP)
# ----------------------------
DEFAULT_PATIENT_ID = "63B0526207E98425D35E7EA737AB89AA"

# DE-ID note text CSVs (you said: v3 clinic, v4 op, v5 IP)
DEFAULT_DEID_NOTE_INPUTS = [
    "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv",
    "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv",
    "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v5.csv",
]

# Encounter CSVs (from /home/apokol/my_data_Breast/HPI-11526/HPI11256)
DEFAULT_ENCOUNTER_INPUTS = [
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Encounters.csv",
]

DEFAULT_OUT_DIR = "/home/apokol/Breast_Restore/PATIENT_BUNDLES"


# ----------------------------
# Helpers
# ----------------------------
def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def ensure_out_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def _open_text_for_read(path):
    # This avoids pandas.read_csv(errors=...) which is NOT supported in many pandas versions.
    # Instead, we open the file with errors='replace' and pass the handle to pandas.
    return open(path, "r", encoding="utf-8", errors="replace")

def read_csv_safely(path):
    # dtype=object keeps everything as strings (good for IDs)
    with _open_text_for_read(path) as f:
        return pd.read_csv(f, dtype=object, engine="python")

def normalize_note_type(x):
    t = _safe_str(x).strip()
    t = re.sub(r"\s+", " ", t)
    return t if t else "UNKNOWN_NOTE_TYPE"

def detect_col_exact(columns, target_upper):
    for c in columns:
        if _safe_str(c).strip().upper() == target_upper:
            return c
    return None

def detect_pid_col(columns):
    # Prefer exact match if present
    c = detect_col_exact(columns, "ENCRYPTED_PAT_ID")
    if c:
        return c
    # Otherwise heuristic
    for col in columns:
        lc = _safe_str(col).lower()
        if "encrypt" in lc and "pat" in lc and "id" in lc:
            return col
    for col in columns:
        lc = _safe_str(col).lower()
        if ("pat" in lc or "patient" in lc) and "id" in lc:
            return col
    return None

def detect_note_type_col(columns):
    c = detect_col_exact(columns, "NOTE_TYPE")
    if c:
        return c
    for col in columns:
        lc = _safe_str(col).lower()
        if "note" in lc and "type" in lc:
            return col
    return None

def detect_deid_text_col(columns):
    # Must be DEID text, not raw
    exacts = ["NOTE_TEXT_DEID", "NOTE_DEID", "TEXT_DEID", "NOTE_TEXT_DEIDENTIFIED"]
    for ex in exacts:
        c = detect_col_exact(columns, ex)
        if c:
            return c

    candidates = []
    for col in columns:
        uc = _safe_str(col).strip().upper()
        if "DEID" in uc or "DE-ID" in uc or "DE_IDENT" in uc:
            candidates.append(col)

    # Prefer ones containing NOTE and TEXT
    for col in candidates:
        uc = _safe_str(col).upper()
        if "NOTE" in uc and "TEXT" in uc:
            return col

    return candidates[0] if candidates else None

def detect_datetime_col(columns):
    """
    Best-effort date/time col detection for NOTE CSVs.
    Your DE-ID note exports may have dates wiped; that's OK.
    """
    date_like = []
    for c in columns:
        lc = _safe_str(c).lower()
        if any(k in lc for k in ["date", "datetime", "time", "created", "service"]):
            date_like.append(c)

    priority = [
        "service_date", "note_date", "note_datetime", "note_time",
        "encounter_date", "admit_date", "surgery_date", "created_date", "created"
    ]
    for key in priority:
        for c in date_like:
            if key in _safe_str(c).lower():
                return c

    return date_like[0] if date_like else None

def try_parse_datetime(series):
    """
    Attempt to parse datetimes; returns a datetime series or None if parsing fails badly.
    """
    try:
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        non_null = parsed.notnull().sum()
        if len(parsed) == 0:
            return None
        if non_null < max(3, int(0.05 * len(parsed))):
            return None
        return parsed
    except Exception:
        return None

def pick_patient_id(all_rows, min_notes):
    counts = {}
    for r in all_rows:
        pid = r["pid"]
        counts[pid] = counts.get(pid, 0) + 1
    eligible = [pid for pid, n in counts.items() if n >= min_notes]
    if not eligible:
        best = None
        bestn = -1
        for pid, n in counts.items():
            if n > bestn:
                bestn = n
                best = pid
        return best, bestn
    pid = random.choice(eligible)
    return pid, counts.get(pid, 0)

def _sanitize_filename(s, maxlen=60):
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", _safe_str(s)).strip("_")
    return (s[:maxlen] if s else "UNKNOWN")

def _best_event_datetime(row_dict, preferred_cols):
    """
    row_dict: dict of raw strings
    preferred_cols: ordered list of possible datetime column names present in the row_dict
    returns (best_raw, best_parsed_dt or None)
    """
    for c in preferred_cols:
        if c in row_dict:
            raw = _safe_str(row_dict.get(c, "")).strip()
            if raw:
                try:
                    dt = pd.to_datetime(raw, errors="coerce", infer_datetime_format=True)
                    if pd.notnull(dt):
                        return raw, dt.to_pydatetime()
                except Exception:
                    pass
    return "", None


# ----------------------------
# Note bundling
# ----------------------------
def write_note_bundle(patient_id, rows, out_dir):
    """
    rows: list of dicts
      keys: source_file, pid, note_type, note_text_deid, dt_raw, dt_parsed, row_idx
    """
    safe_pid = re.sub(r"[^A-Za-z0-9_\-]+", "_", patient_id)
    patient_dir = os.path.join(out_dir, safe_pid)
    ensure_out_dir(patient_dir)

    # Sort: parsed dt (if any) -> source -> row
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            (r["dt_parsed"] is None, r["dt_parsed"]) if r["dt_parsed"] is not None else (True, None),
            r["source_file"],
            r["row_idx"],
        )
    )

    # Timeline CSV
    timeline_path = os.path.join(patient_dir, "timeline.csv")
    timeline_df = pd.DataFrame([{
        "ENCRYPTED_PAT_ID": r["pid"],
        "NOTE_TYPE": r["note_type"],
        "NOTE_DATETIME_RAW": r["dt_raw"],
        "SOURCE_FILE": r["source_file"],
        "ROW_IDX": r["row_idx"],
        "NOTE_TEXT_DEID_LEN": len(_safe_str(r["note_text_deid"]))
    } for r in rows_sorted])
    timeline_df.to_csv(timeline_path, index=False, encoding="utf-8")

    # Combined file + per-note files
    combined_path = os.path.join(patient_dir, "ALL_NOTES_COMBINED.txt")
    with open(combined_path, "w", encoding="utf-8") as f_out:
        f_out.write("DE-ID PATIENT NOTE BUNDLE\n")
        f_out.write("=========================\n\n")
        f_out.write("ENCRYPTED_PAT_ID: {}\n".format(patient_id))
        f_out.write("TOTAL_NOTES: {}\n".format(len(rows_sorted)))
        f_out.write("\n---\n\n")

        for i, r in enumerate(rows_sorted, start=1):
            note_type = r["note_type"]
            dt = _safe_str(r["dt_raw"]).strip()
            src = r["source_file"]
            idx = r["row_idx"]
            text = _safe_str(r["note_text_deid"])

            note_fname = "note_{:04d}__{}.txt".format(i, _sanitize_filename(note_type))
            note_path = os.path.join(patient_dir, note_fname)

            with open(note_path, "w", encoding="utf-8") as nf:
                nf.write("ENCRYPTED_PAT_ID: {}\n".format(patient_id))
                nf.write("NOTE_NUMBER: {}\n".format(i))
                nf.write("NOTE_TYPE: {}\n".format(note_type))
                nf.write("NOTE_DATETIME_RAW: {}\n".format(dt))
                nf.write("SOURCE_FILE: {}\n".format(src))
                nf.write("ROW_IDX: {}\n".format(idx))
                nf.write("\n--- NOTE_TEXT_DEID ---\n\n")
                nf.write(text)

            f_out.write("NOTE {:04d}\n".format(i))
            f_out.write("---------\n")
            f_out.write("NOTE_TYPE: {}\n".format(note_type))
            f_out.write("NOTE_DATETIME_RAW: {}\n".format(dt))
            f_out.write("SOURCE_FILE: {}\n".format(src))
            f_out.write("ROW_IDX: {}\n".format(idx))
            f_out.write("\n--- NOTE_TEXT_DEID ---\n\n")
            f_out.write(text)
            f_out.write("\n\n" + ("=" * 80) + "\n\n")

    return patient_dir, timeline_path, combined_path


# ----------------------------
# Encounter timeline + Stage 2 anchor
# ----------------------------
def read_and_filter_encounters(encounter_paths, patient_id):
    """
    Returns:
      events: list of dicts with unified keys
      summaries: list of per-file summary tuples
    """
    events = []
    summaries = []

    for path in encounter_paths:
        if not os.path.exists(path):
            raise RuntimeError("Encounter input not found: {}".format(path))

        df = read_csv_safely(path)

        pid_col = detect_pid_col(df.columns)
        if pid_col is None:
            raise RuntimeError("Could not detect ENCRYPTED_PAT_ID in encounter file: {}".format(path))

        # Filter to patient
        df_pid = df[df[pid_col].astype(str).str.strip() == patient_id].copy()
        n = len(df_pid)

        # Common helpful cols (best-effort, not required)
        # We keep raw column names (as present) and map into a consistent output.
        def _col_if_exists(name_upper):
            c = detect_col_exact(df_pid.columns, name_upper)
            return c

        col_enc = _col_if_exists("PAT_ENC_CSN_ID") or _col_if_exists("ENCRYPTED_CSN") or _col_if_exists("ENCRYPTED_PAT_ENC_CSN_ID")
        col_type = _col_if_exists("ENCOUNTER_TYPE") or _col_if_exists("ENCOUNTER_TYPE_NAME")
        col_dept = _col_if_exists("DEPARTMENT") or _col_if_exists("OP_DEPARTMENT")
        col_reason = _col_if_exists("REASON_FOR_VISIT")
        col_proc = _col_if_exists("PROCEDURE")
        col_cpt = _col_if_exists("CPT_CODE")
        col_recon = _col_if_exists("RECONSTRUCTION_DATE")

        # Date preferences differ by file type; we detect by available columns.
        preferred_date_cols = []
        for cand in ["RECONSTRUCTION_DATE", "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME", "CHECKOUT_TIME",
                     "DISCHARGE_DATE_DT", "HOSP_DSCHRG_TIME"]:
            c = _col_if_exists(cand)
            if c:
                preferred_date_cols.append(c)

        # Build event rows
        for idx, row in df_pid.iterrows():
            row_dict = {c: row[c] for c in df_pid.columns}
            best_raw, best_parsed = _best_event_datetime(row_dict, preferred_date_cols)

            events.append({
                "ENCRYPTED_PAT_ID": patient_id,
                "SOURCE_FILE": os.path.basename(path),
                "ROW_IDX": int(idx) if _safe_str(idx).isdigit() else idx,
                "PAT_ENC_CSN_ID": _safe_str(row[col_enc]).strip() if col_enc else "",
                "ENCOUNTER_TYPE": _safe_str(row[col_type]).strip() if col_type else "",
                "DEPARTMENT": _safe_str(row[col_dept]).strip() if col_dept else "",
                "REASON_FOR_VISIT": _safe_str(row[col_reason]).strip() if col_reason else "",
                "PROCEDURE": _safe_str(row[col_proc]).strip() if col_proc else "",
                "CPT_CODE": _safe_str(row[col_cpt]).strip() if col_cpt else "",
                "RECONSTRUCTION_DATE": _safe_str(row[col_recon]).strip() if col_recon else "",
                "BEST_EVENT_DT_RAW": best_raw,
                "BEST_EVENT_DT_PARSED": best_parsed,
            })

        summaries.append((
            os.path.basename(path),
            pid_col,
            col_enc or "NONE",
            col_type or "NONE",
            col_dept or "NONE",
            col_recon or "NONE",
            ",".join(preferred_date_cols) if preferred_date_cols else "NONE",
            n
        ))

    return events, summaries

def write_encounter_outputs(patient_dir, encounter_events):
    """
    Writes:
      encounters_timeline.csv
      stage2_anchor_summary.txt
    """
    # Sort by best parsed dt if available, else push to bottom
    events_sorted = sorted(
        encounter_events,
        key=lambda e: (e["BEST_EVENT_DT_PARSED"] is None, e["BEST_EVENT_DT_PARSED"] or datetime.datetime.max)
    )

    enc_path = os.path.join(patient_dir, "encounters_timeline.csv")
    enc_df = pd.DataFrame([{
        "ENCRYPTED_PAT_ID": e["ENCRYPTED_PAT_ID"],
        "BEST_EVENT_DT_RAW": e["BEST_EVENT_DT_RAW"],
        "SOURCE_FILE": e["SOURCE_FILE"],
        "PAT_ENC_CSN_ID": e["PAT_ENC_CSN_ID"],
        "ENCOUNTER_TYPE": e["ENCOUNTER_TYPE"],
        "DEPARTMENT": e["DEPARTMENT"],
        "RECONSTRUCTION_DATE": e["RECONSTRUCTION_DATE"],
        "CPT_CODE": e["CPT_CODE"],
        "PROCEDURE": e["PROCEDURE"],
        "REASON_FOR_VISIT": e["REASON_FOR_VISIT"],
        "ROW_IDX": e["ROW_IDX"],
    } for e in events_sorted])
    enc_df.to_csv(enc_path, index=False, encoding="utf-8")

    # Anchor selection:
    # 1) If any RECONSTRUCTION_DATE parses -> pick earliest (most defensible)
    # 2) Else pick earliest BEST_EVENT_DT_PARSED
    recon_dates = []
    for e in events_sorted:
        rd = _safe_str(e.get("RECONSTRUCTION_DATE", "")).strip()
        if rd:
            try:
                dt = pd.to_datetime(rd, errors="coerce", infer_datetime_format=True)
                if pd.notnull(dt):
                    recon_dates.append(dt.to_pydatetime())
            except Exception:
                pass

    anchor = None
    anchor_source = ""
    if recon_dates:
        anchor = min(recon_dates)
        anchor_source = "RECONSTRUCTION_DATE (earliest parsed)"
    else:
        parsed = [e["BEST_EVENT_DT_PARSED"] for e in events_sorted if e["BEST_EVENT_DT_PARSED"] is not None]
        if parsed:
            anchor = min(parsed)
            anchor_source = "BEST_EVENT_DT_PARSED (earliest available; RECONSTRUCTION_DATE missing/unparseable)"

    summary_path = os.path.join(patient_dir, "stage2_anchor_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("STAGE 2 ANCHOR SUMMARY (ENCOUNTER-BASED)\n")
        f.write("=======================================\n\n")
        f.write("ENCRYPTED_PAT_ID: {}\n".format(events_sorted[0]["ENCRYPTED_PAT_ID"] if events_sorted else ""))
        f.write("TOTAL_ENCOUNTER_ROWS: {}\n".format(len(events_sorted)))
        f.write("\n")

        if anchor is None:
            f.write("ANCHOR_DATE: NOT FOUND\n")
            f.write("ANCHOR_SOURCE: NONE\n\n")
            f.write("NOTE: Your DE-ID note files likely have date wiping, so this script relies on encounter CSVs.\n")
            f.write("If RECONSTRUCTION_DATE exists but is still not parsing, we may need to inspect its format.\n")
        else:
            f.write("ANCHOR_DATE (parsed): {}\n".format(anchor.isoformat()))
            f.write("ANCHOR_SOURCE: {}\n".format(anchor_source))

            d30 = anchor + datetime.timedelta(days=30)
            d90 = anchor + datetime.timedelta(days=90)
            d365 = anchor + datetime.timedelta(days=365)

            f.write("\nWINDOWS (relative to ANCHOR_DATE)\n")
            f.write("--------------------------------\n")
            f.write("ANCHOR + 30 days : {}\n".format(d30.date().isoformat()))
            f.write("ANCHOR + 90 days : {}\n".format(d90.date().isoformat()))
            f.write("ANCHOR + 365 days: {}\n".format(d365.date().isoformat()))
            f.write("\nInterpretation:\n")
            f.write("  0–30d   = early\n")
            f.write("  31–90d  = intermediate\n")
            f.write("  91–365d = late\n")
            f.write("  >365d   = very late\n")

    return enc_path, summary_path, anchor


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    # NOTE: all args optional now, because you asked for hardcoded paths + no typing.
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory for patient bundles.")
    ap.add_argument("--patient_id", default=DEFAULT_PATIENT_ID, help="ENCRYPTED_PAT_ID to export.")
    ap.add_argument("--deid_inputs", nargs="+", default=DEFAULT_DEID_NOTE_INPUTS, help="DE-ID note CSV inputs.")
    ap.add_argument("--encounter_inputs", nargs="+", default=DEFAULT_ENCOUNTER_INPUTS, help="Encounter CSV inputs.")
    ap.add_argument("--pick_random", action="store_true", help="Pick a random patient with >= --min_notes (from DE-ID notes).")
    ap.add_argument("--min_notes", type=int, default=10, help="Used with --pick_random.")
    ap.add_argument("--max_rows_per_file", type=int, default=None, help="Optional: cap rows per DE-ID note file for fast tests.")
    args = ap.parse_args()

    ensure_out_dir(args.out_dir)

    # ----------------------------
    # 1) Load DE-ID notes across deid_inputs
    # ----------------------------
    all_rows = []
    file_summaries = []

    for path in args.deid_inputs:
        if not os.path.exists(path):
            raise RuntimeError("DE-ID note input not found: {}".format(path))

        df = read_csv_safely(path)

        pid_col = detect_pid_col(df.columns)
        note_type_col = detect_note_type_col(df.columns)
        deid_col = detect_deid_text_col(df.columns)
        dt_col = detect_datetime_col(df.columns)

        if pid_col is None:
            raise RuntimeError("Could not detect ENCRYPTED_PAT_ID column in: {}".format(path))
        if deid_col is None:
            raise RuntimeError(
                "Could not detect a DE-ID text column in: {}\n"
                "Expected something like NOTE_TEXT_DEID (must contain 'deid' in the header).".format(path)
            )

        # Guardrail: do NOT allow non-DEID text col
        if "deid" not in _safe_str(deid_col).lower():
            raise RuntimeError("Refusing to proceed: detected text col doesn't look de-identified: {}".format(deid_col))

        # Parse datetime if present (likely wiped)
        dt_parsed_series = None
        if dt_col is not None:
            dt_parsed_series = try_parse_datetime(df[dt_col].astype(str))

        n_rows = len(df)
        if args.max_rows_per_file is not None:
            n_rows = min(n_rows, args.max_rows_per_file)

        file_summaries.append((
            os.path.basename(path),
            pid_col,
            note_type_col or "NONE",
            deid_col,
            dt_col or "NONE",
            n_rows
        ))

        for i in range(n_rows):
            pid = _safe_str(df.iloc[i][pid_col]).strip()
            if not pid:
                continue

            note_type = normalize_note_type(df.iloc[i][note_type_col]) if note_type_col else "UNKNOWN_NOTE_TYPE"
            text_deid = _safe_str(df.iloc[i][deid_col])

            dt_raw = _safe_str(df.iloc[i][dt_col]) if dt_col is not None else ""
            dt_parsed = None
            if dt_parsed_series is not None:
                v = dt_parsed_series.iloc[i]
                if pd.notnull(v):
                    dt_parsed = v.to_pydatetime()

            all_rows.append({
                "source_file": os.path.basename(path),
                "pid": pid,
                "note_type": note_type,
                "note_text_deid": text_deid,
                "dt_raw": dt_raw,
                "dt_parsed": dt_parsed,
                "row_idx": i
            })

    if not all_rows:
        raise RuntimeError("No rows found across the provided DE-ID note inputs.")

    # Decide patient id
    patient_id = args.patient_id
    if args.pick_random:
        patient_id, n = pick_patient_id(all_rows, args.min_notes)
        print("Picked patient_id:", patient_id, "with notes:", n)

    # Filter note rows for that patient
    patient_rows = [r for r in all_rows if r["pid"] == patient_id]
    if not patient_rows:
        raise RuntimeError("No DE-ID note rows found for patient_id: {}".format(patient_id))

    # Write note bundle outputs
    patient_dir, timeline_path, combined_path = write_note_bundle(patient_id, patient_rows, args.out_dir)

    # ----------------------------
    # 2) Load encounters and write encounter timeline + anchor summary
    # ----------------------------
    encounter_events, encounter_summaries = read_and_filter_encounters(args.encounter_inputs, patient_id)

    if encounter_events:
        enc_timeline_path, anchor_summary_path, anchor_dt = write_encounter_outputs(patient_dir, encounter_events)
    else:
        enc_timeline_path, anchor_summary_path, anchor_dt = "", "", None

    # ----------------------------
    # Print summary
    # ----------------------------
    print("\nINPUT SUMMARY (DE-ID notes, per file):")
    for (fname, pid_col, type_col, deid_col, dt_col, n_rows) in file_summaries:
        print(" -", fname)
        print("    pid_col:", pid_col, "| note_type_col:", type_col, "| deid_text_col:", deid_col, "| datetime_col:", dt_col, "| rows_used:", n_rows)

    print("\nINPUT SUMMARY (Encounters, per file) [filtered to patient]:")
    for (fname, pid_col, enc_col, type_col, dept_col, recon_col, pref_dates, n_rows) in encounter_summaries:
        print(" -", fname)
        print("    pid_col:", pid_col, "| enc_id_col:", enc_col, "| type_col:", type_col, "| dept_col:", dept_col,
              "| recon_col:", recon_col, "| date_cols_considered:", pref_dates, "| rows_for_patient:", n_rows)

    print("\nEXPORTED PATIENT:")
    print("  ENCRYPTED_PAT_ID:", patient_id)
    print("  Notes exported  :", len(patient_rows))
    print("  Patient dir     :", patient_dir)
    print("  Notes timeline  :", timeline_path)
    print("  Combined notes  :", combined_path)
    if encounter_events:
        print("  Encounters file :", enc_timeline_path)
        print("  Anchor summary  :", anchor_summary_path)
        if anchor_dt is not None:
            print("  Anchor (parsed) :", anchor_dt.isoformat())
    else:
        print("  Encounters file : (none found for patient across encounter CSVs)")
    print("Done.")


if __name__ == "__main__":
    main()
