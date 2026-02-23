#!/usr/bin/env python3
# export_one_patient_deid_bundle.py
# Python 3.6.8 compatible
#
# Hardcoded for Breast_Restore + HPI-11526 paths (DE-ID notes + encounter tables).
# You can still override via CLI if you want, but you don't have to.

from __future__ import print_function
import os
import re
import argparse
import random
import pandas as pd


# ----------------------------
# HARD-CODED DEFAULTS (edit only if paths change)
# ----------------------------
DEFAULT_PATIENT_ID = "63B0526207E98425D35E7EA737AB89AA"
DEFAULT_OUT_DIR = "/home/apokol/Breast_Restore/PATIENT_BUNDLES"

# DE-ID note text files (you said: v3 clinic, v4 op, v5 IP)
DEFAULT_DEID_INPUTS = [
    "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv",
    "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv",
    "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v5.csv",
]

# Structured encounter files (has usable dates)
DEFAULT_ENCOUNTER_INPUTS = [
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Encounters.csv",
]


# ----------------------------
# Column detection helpers
# ----------------------------
def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def _norm_col(c):
    return re.sub(r"\s+", "_", _safe_str(c).strip().upper())

def detect_pid_col(columns):
    # Prefer exact match if present
    for c in columns:
        if _norm_col(c) == "ENCRYPTED_PAT_ID":
            return c
    # Otherwise heuristic
    for c in columns:
        lc = _safe_str(c).lower()
        if "encrypt" in lc and "id" in lc:
            return c
    for c in columns:
        lc = _safe_str(c).lower()
        if ("pat" in lc or "patient" in lc) and "id" in lc:
            return c
    return None

def detect_note_type_col(columns):
    for c in columns:
        if _norm_col(c) == "NOTE_TYPE":
            return c
    for c in columns:
        lc = _safe_str(c).lower()
        if "note" in lc and "type" in lc:
            return c
    return None

def detect_deid_text_col(columns):
    # Must be DEID text, not raw
    candidates = []
    for c in columns:
        uc = _safe_str(c).strip().upper()
        if uc in ("NOTE_TEXT_DEID", "NOTE_DEID", "TEXT_DEID", "NOTE_TEXT_DEIDENTIFIED"):
            return c
        if "DEID" in uc or "DE-ID" in uc or "DE_IDENT" in uc:
            candidates.append(c)
    # If multiple, prefer ones containing NOTE and TEXT
    for c in candidates:
        uc = _safe_str(c).upper()
        if "NOTE" in uc and "TEXT" in uc:
            return c
    return candidates[0] if candidates else None

def detect_datetime_col(columns):
    """
    Best-effort date/time col detection for NOTE tables.
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

def detect_encounter_date_cols(columns):
    """
    Encounter tables often have multiple relevant date fields.
    We return a prioritized list; the script will take the first parseable per row.
    """
    # Normalize and map back to original names
    norm_map = {}
    for c in columns:
        norm_map[_norm_col(c)] = c

    # Priority order (best guess for your files based on screenshots)
    priority_norm = [
        "RECONSTRUCTION_DATE",
        "OPERATION_DATE",
        "DISCHARGE_DATE_DT",
        "ADMIT_DATE",
        "HOSP_ADMSN_TIME",
        "HOSP_DISCHRG_TIME",
        "CHECKOUT_TIME",
        "ENCOUNTER_DATE",
        "VISIT_DATE",
        "DATE",
    ]

    out = []
    for p in priority_norm:
        if p in norm_map:
            out.append(norm_map[p])

    # Fallback: any column containing DATE/TIME
    if not out:
        for c in columns:
            n = _norm_col(c)
            if ("DATE" in n) or ("TIME" in n) or ("DT" in n):
                out.append(c)

    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


# ----------------------------
# IO helpers
# ----------------------------
def ensure_out_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def read_csv_robust(path):
    """
    Pandas on your environment threw:
      TypeError: read_csv() got an unexpected keyword argument 'errors'
    So we avoid that and do a simple encoding fallback.
    """
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        # Fallback often helps when there are weird characters
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")

def normalize_note_type(x):
    t = _safe_str(x).strip()
    t = re.sub(r"\s+", " ", t)
    return t if t else "UNKNOWN_NOTE_TYPE"

def try_parse_datetime(series):
    try:
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        non_null = parsed.notnull().sum()
        if non_null < max(3, int(0.05 * len(parsed))):
            return None
        return parsed
    except Exception:
        return None


# ----------------------------
# Bundle writers
# ----------------------------
def write_notes_bundle(patient_id, rows, out_dir):
    safe_pid = re.sub(r"[^A-Za-z0-9_\-]+", "_", patient_id)
    patient_dir = os.path.join(out_dir, safe_pid)
    ensure_out_dir(patient_dir)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            (r["dt_parsed"] is None, r["dt_parsed"]) if r["dt_parsed"] is not None else (True, None),
            r["source_file"],
            r["row_idx"],
        )
    )

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

    combined_path = os.path.join(patient_dir, "ALL_NOTES_COMBINED.txt")
    with open(combined_path, "w") as f_out:
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

            note_fname = "note_{:04d}__{}.txt".format(
                i,
                re.sub(r"[^A-Za-z0-9_\-]+", "_", note_type)[:60] or "UNKNOWN"
            )
            note_path = os.path.join(patient_dir, note_fname)
            with open(note_path, "w") as nf:
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

def write_encounters_timeline(patient_id, encounter_rows, patient_dir):
    out_path = os.path.join(patient_dir, "encounters_timeline.csv")
    df = pd.DataFrame(encounter_rows)
    # nice column order if present
    preferred = [
        "ENCRYPTED_PAT_ID", "BEST_EVENT_DT_RAW", "BEST_EVENT_DT_PARSED",
        "ANCHOR_SOURCE_COL", "SOURCE_FILE",
        "PAT_ENC_CSN_ID", "ENCRYPTED_CSN", "ENCOUNTER_TYPE",
        "DEPARTMENT", "OP_DEPARTMENT",
        "RECONSTRUCTION_DATE", "OPERATION_DATE", "ADMIT_DATE",
        "CPT_CODE", "PROCEDURE", "REASON_FOR_VISIT",
        "ROW_IDX"
    ]
    cols = []
    for c in preferred:
        if c in df.columns:
            cols.append(c)
    for c in df.columns:
        if c not in cols:
            cols.append(c)
    df = df[cols]
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def compute_anchor_from_encounters(encounter_rows):
    """
    Anchor = earliest parsed BEST_EVENT_DT_PARSED across encounter rows.
    Returns (anchor_dt, source_col, n_rows)
    """
    best = None
    best_source = None
    for r in encounter_rows:
        dtp = r.get("BEST_EVENT_DT_PARSED", None)
        if dtp is None:
            continue
        if best is None or dtp < best:
            best = dtp
            best_source = r.get("ANCHOR_SOURCE_COL", None)
    return best, best_source, len(encounter_rows)

def write_anchor_summary(patient_id, anchor_dt, anchor_source, n_rows, patient_dir):
    out_path = os.path.join(patient_dir, "stage2_anchor_summary.txt")
    with open(out_path, "w") as f:
        f.write("STAGE 2 ANCHOR SUMMARY (ENCOUNTER-BASED)\n")
        f.write("======================================\n\n")
        f.write("ENCRYPTED_PAT_ID: {}\n".format(patient_id))
        f.write("TOTAL_ENCOUNTER_ROWS: {}\n\n".format(n_rows))

        if anchor_dt is None:
            f.write("ANCHOR_DATE (parsed): NONE FOUND\n")
            f.write("ANCHOR_SOURCE: NONE\n\n")
            f.write("WARNING: No parseable encounter dates were found.\n")
            return out_path, None

        # Windows
        anchor_30 = anchor_dt + pd.Timedelta(days=30)
        anchor_90 = anchor_dt + pd.Timedelta(days=90)
        anchor_365 = anchor_dt + pd.Timedelta(days=365)

        f.write("ANCHOR_DATE (parsed): {}\n".format(anchor_dt.isoformat()))
        f.write("ANCHOR_SOURCE: {}\n\n".format(anchor_source if anchor_source else "UNKNOWN"))

        f.write("WINDOWS (relative to ANCHOR_DATE)\n")
        f.write("---------------------------------\n")
        f.write("ANCHOR + 30 days : {}\n".format(anchor_30.date().isoformat()))
        f.write("ANCHOR + 90 days : {}\n".format(anchor_90.date().isoformat()))
        f.write("ANCHOR + 365 days: {}\n\n".format(anchor_365.date().isoformat()))

        f.write("Interpretation:\n")
        f.write("  0–30d   = early\n")
        f.write("  31–90d  = intermediate\n")
        f.write("  91–365d = late\n")
        f.write("  >365d   = very late\n")

    return out_path, anchor_dt


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=None, help="DE-ID note CSV files (optional; uses defaults if omitted).")
    ap.add_argument("--encounters", nargs="+", default=None, help="Encounter CSV files (optional; uses defaults if omitted).")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory for patient bundles.")
    ap.add_argument("--patient_id", default=DEFAULT_PATIENT_ID, help="ENCRYPTED_PAT_ID to export.")
    ap.add_argument("--pick_random", action="store_true", help="Pick a random patient with >= --min_notes (from notes only).")
    ap.add_argument("--min_notes", type=int, default=10, help="Used with --pick_random.")
    ap.add_argument("--max_rows_per_file", type=int, default=None, help="Optional: cap rows per file for fast tests.")
    args = ap.parse_args()

    deid_inputs = args.inputs if args.inputs else list(DEFAULT_DEID_INPUTS)
    enc_inputs = args.encounters if args.encounters else list(DEFAULT_ENCOUNTER_INPUTS)

    ensure_out_dir(args.out_dir)

    # ---------
    # NOTES: Collect per-note rows for the patient
    # ---------
    all_note_rows = []
    file_summaries = []

    for path in deid_inputs:
        if not os.path.exists(path):
            raise RuntimeError("Input file not found: {}".format(path))

        df = read_csv_robust(path)

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

        if "deid" not in _safe_str(deid_col).lower():
            raise RuntimeError("Refusing to proceed: detected text col doesn't look de-identified: {}".format(deid_col))

        dt_parsed_series = None
        if dt_col is not None:
            dt_parsed_series = try_parse_datetime(df[dt_col].astype(str))

        n_rows = len(df)
        if args.max_rows_per_file is not None:
            n_rows = min(n_rows, args.max_rows_per_file)

        file_summaries.append((os.path.basename(path), pid_col, note_type_col or "NONE", deid_col, dt_col or "NONE", n_rows))

        for i in range(n_rows):
            pid = _safe_str(df.iloc[i][pid_col]).strip()
            if not pid:
                continue
            if args.pick_random:
                # for random selection we keep all
                pass
            else:
                # only keep the target patient
                if pid != args.patient_id:
                    continue

            note_type = normalize_note_type(df.iloc[i][note_type_col]) if note_type_col else "UNKNOWN_NOTE_TYPE"
            text_deid = _safe_str(df.iloc[i][deid_col])

            dt_raw = _safe_str(df.iloc[i][dt_col]) if dt_col is not None else ""
            dt_parsed = None
            if dt_parsed_series is not None:
                v = dt_parsed_series.iloc[i]
                if pd.notnull(v):
                    dt_parsed = v.to_pydatetime()

            all_note_rows.append({
                "source_file": os.path.basename(path),
                "pid": pid,
                "note_type": note_type,
                "note_text_deid": text_deid,
                "dt_raw": dt_raw,
                "dt_parsed": dt_parsed,
                "row_idx": i
            })

    if not all_note_rows:
        raise RuntimeError("No note rows found across the provided DE-ID inputs.")

    # Decide patient id (only if pick_random)
    patient_id = args.patient_id
    if args.pick_random:
        # compute counts
        counts = {}
        for r in all_note_rows:
            counts[r["pid"]] = counts.get(r["pid"], 0) + 1
        eligible = [pid for pid, n in counts.items() if n >= args.min_notes]
        if eligible:
            patient_id = random.choice(eligible)
        else:
            patient_id = max(counts.keys(), key=lambda k: counts[k])
        print("Picked patient_id:", patient_id, "with notes:", counts.get(patient_id, 0))

    patient_note_rows = [r for r in all_note_rows if r["pid"] == patient_id]
    if not patient_note_rows:
        raise RuntimeError("No note rows found for patient_id: {}".format(patient_id))

    # Write note bundle
    patient_dir, notes_timeline_path, combined_path = write_notes_bundle(patient_id, patient_note_rows, args.out_dir)

    # ---------
    # ENCOUNTERS: Build structured date timeline + anchor
    # ---------
    encounter_rows = []
    encounter_summaries = []

    for path in enc_inputs:
        if not os.path.exists(path):
            raise RuntimeError("Encounter file not found: {}".format(path))

        df = read_csv_robust(path)
        pid_col = detect_pid_col(df.columns)
        if pid_col is None:
            raise RuntimeError("Could not detect ENCRYPTED_PAT_ID in encounter file: {}".format(path))

        date_cols = detect_encounter_date_cols(df.columns)
        if not date_cols:
            # still write something minimal
            date_cols = []

        # pre-parse each candidate date col into datetime series
        parsed_cols = {}
        for dc in date_cols:
            parsed_cols[dc] = try_parse_datetime(df[dc].astype(str))

        # row cap
        n_rows = len(df)
        if args.max_rows_per_file is not None:
            n_rows = min(n_rows, args.max_rows_per_file)

        encounter_summaries.append((os.path.basename(path), pid_col, date_cols, n_rows))

        for i in range(n_rows):
            pid = _safe_str(df.iloc[i][pid_col]).strip()
            if not pid or pid != patient_id:
                continue

            best_dt = None
            best_raw = ""
            best_source = None

            for dc in date_cols:
                ser = parsed_cols.get(dc, None)
                if ser is None:
                    continue
                v = ser.iloc[i]
                if pd.notnull(v):
                    best_dt = v.to_pydatetime()
                    best_raw = _safe_str(df.iloc[i][dc])
                    best_source = dc
                    break

            row = {
                "ENCRYPTED_PAT_ID": pid,
                "BEST_EVENT_DT_RAW": best_raw,
                "BEST_EVENT_DT_PARSED": best_dt,
                "ANCHOR_SOURCE_COL": best_source if best_source else "",
                "SOURCE_FILE": os.path.basename(path),
                "ROW_IDX": i
            }

            # Add a few common fields if present (don’t break if absent)
            for want in [
                "PAT_ENC_CSN_ID", "ENCRYPTED_CSN", "ENCOUNTER_TYPE", "DEPARTMENT", "OP_DEPARTMENT",
                "RECONSTRUCTION_DATE", "OPERATION_DATE", "ADMIT_DATE", "DISCHARGE_DATE_DT",
                "CPT_CODE", "PROCEDURE", "REASON_FOR_VISIT"
            ]:
                # case-insensitive map
                for c in df.columns:
                    if _norm_col(c) == _norm_col(want):
                        row[want] = _safe_str(df.iloc[i][c])
                        break

            encounter_rows.append(row)

    encounters_path = ""
    anchor_summary_path = ""
    anchor_dt = None

    if encounter_rows:
        encounters_path = write_encounters_timeline(patient_id, encounter_rows, patient_dir)
        anchor_dt, anchor_source, n_enc = compute_anchor_from_encounters(encounter_rows)
        anchor_summary_path, anchor_dt = write_anchor_summary(patient_id, anchor_dt, anchor_source, n_enc, patient_dir)
    else:
        # still create an empty encounters file for consistency
        encounters_path = os.path.join(patient_dir, "encounters_timeline.csv")
        pd.DataFrame([]).to_csv(encounters_path, index=False, encoding="utf-8")
        anchor_summary_path = os.path.join(patient_dir, "stage2_anchor_summary.txt")
        with open(anchor_summary_path, "w") as f:
            f.write("No encounter rows found for this patient_id across provided encounter files.\n")

    # ---------
    # Print summary
    # ---------
    print("\nEXPORTED PATIENT:")
    print("  ENCRYPTED_PAT_ID:", patient_id)
    print("  Notes exported   :", len(patient_note_rows))
    print("  Patient dir      :", patient_dir)
    print("  Notes timeline   :", notes_timeline_path)
    print("  Combined notes   :", combined_path)
    print("  Encounters file  :", encounters_path)
    print("  Anchor summary   :", anchor_summary_path)
    print("  Anchor (parsed)  :", anchor_dt.isoformat() if anchor_dt else "NONE")
    print("Done.")

if __name__ == "__main__":
    main()
