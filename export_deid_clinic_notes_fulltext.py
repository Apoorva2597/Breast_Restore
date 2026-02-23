#!/usr/bin/env python3
# export_one_patient_deid_bundle.py
# Python 3.6.8 compatible
#
# PURPOSE
#   Export *all de-identified notes* for ONE ENCRYPTED_PAT_ID across:
#     - Clinic notes (v3)
#     - Operative notes (v4)
#     - Inpatient notes (v5)
#   into a single patient bundle folder:
#     - timeline.csv
#     - ALL_NOTES_COMBINED.txt
#     - one txt per note
#
# DEFAULT BEHAVIOR
#   - Uses the hardcoded INPUT_FILES and OUT_DIR below (no CLI required)
#   - Uses DEFAULT_PATIENT_ID below (so you don't have to type/copy/paste)
#
# OPTIONAL CLI OVERRIDES (still supported)
#   python export_one_patient_deid_bundle.py --patient_id <ID>
#   python export_one_patient_deid_bundle.py --pick_random --min_notes 20
#   python export_one_patient_deid_bundle.py --out_dir /some/path
#   python export_one_patient_deid_bundle.py --max_rows_per_file 5000

from __future__ import print_function
import os
import re
import argparse
import random

import pandas as pd


# ============================
# HARD-CODED PATHS (YOUR SET)
# ============================

BASE_DIR = "/home/apokol/Breast_Restore"

# v3 = clinic, v4 = op, v5 = IP (per your folder screenshot)
INPUT_FILES = [
    os.path.join(BASE_DIR, "DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv"),
    os.path.join(BASE_DIR, "DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv"),
    os.path.join(BASE_DIR, "DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v5.csv"),
]

OUT_DIR_DEFAULT = os.path.join(BASE_DIR, "PATIENT_BUNDLES")

# Put your patient id here once. You will not need to type it again.
DEFAULT_PATIENT_ID = "63B0526207E98425D35E7EA737AB89AA"


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


def detect_pid_col(columns):
    # Prefer exact match if present
    for c in columns:
        if c.strip().upper() == "ENCRYPTED_PAT_ID":
            return c
    # Otherwise heuristic
    for c in columns:
        lc = c.lower()
        if "encrypt" in lc and "id" in lc:
            return c
    for c in columns:
        lc = c.lower()
        if ("pat" in lc or "patient" in lc) and "id" in lc:
            return c
    return None


def detect_note_type_col(columns):
    for c in columns:
        if c.strip().upper() == "NOTE_TYPE":
            return c
    for c in columns:
        lc = c.lower()
        if "note" in lc and "type" in lc:
            return c
    return None


def detect_deid_text_col(columns):
    # Must be DEID text, not raw
    candidates = []
    for c in columns:
        uc = c.strip().upper()
        if uc in ("NOTE_TEXT_DEID", "NOTE_DEID", "TEXT_DEID", "NOTE_TEXT_DEIDENTIFIED"):
            return c
        if "DEID" in uc or "DE-ID" in uc or "DE_IDENT" in uc:
            candidates.append(c)

    # If multiple, prefer ones containing NOTE and TEXT
    for c in candidates:
        uc = c.upper()
        if "NOTE" in uc and "TEXT" in uc:
            return c
    return candidates[0] if candidates else None


def detect_datetime_col(columns):
    """
    Best-effort date/time col detection.
    We keep it broad: service_date, note_date, note_datetime, encounter_date, created, etc.
    """
    date_like = []
    for c in columns:
        lc = c.lower()
        if any(k in lc for k in ["date", "datetime", "time", "created", "service"]):
            date_like.append(c)

    priority = [
        "service_date", "note_date", "note_datetime", "note_time",
        "encounter_date", "admit_date", "surgery_date", "created_date", "created"
    ]
    for key in priority:
        for c in date_like:
            if key in c.lower():
                return c

    return date_like[0] if date_like else None


# ----------------------------
# Core logic
# ----------------------------
def read_deid_csv(path):
    """
    FIX FOR YOUR ERROR:
      Older pandas versions do NOT support read_csv(errors="replace").
      So we try utf-8; if decoding fails, fall back to latin-1.
    """
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin-1")


def normalize_note_type(x):
    t = _safe_str(x).strip()
    t = re.sub(r"\s+", " ", t)
    return t if t else "UNKNOWN_NOTE_TYPE"


def ensure_out_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def try_parse_datetime(series):
    """
    Attempt to parse datetimes; returns a datetime series or None if parsing fails badly.
    """
    try:
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        non_null = parsed.notnull().sum()
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


def write_bundle(patient_id, rows, out_dir):
    """
    rows: list of dicts
      keys: source_file, pid, note_type, note_text_deid, dt_raw, dt_parsed, row_idx
    """
    safe_pid = re.sub(r"[^A-Za-z0-9_\-]+", "_", patient_id)
    patient_dir = os.path.join(out_dir, safe_pid)
    ensure_out_dir(patient_dir)

    # Sort:
    # 1) parsed datetime if available
    # 2) source file name
    # 3) row index
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

    # One file per note + combined file
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

            # Per-note file
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

            # Combined section
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="Output directory for patient bundles.")
    ap.add_argument("--patient_id", default=DEFAULT_PATIENT_ID,
                    help="ENCRYPTED_PAT_ID to export (defaults to DEFAULT_PATIENT_ID in code).")
    ap.add_argument("--pick_random", action="store_true", help="Pick a random patient with >= --min_notes.")
    ap.add_argument("--min_notes", type=int, default=10, help="Used with --pick_random.")
    ap.add_argument("--max_rows_per_file", type=int, default=None, help="Optional: cap rows per file for fast tests.")
    args = ap.parse_args()

    ensure_out_dir(args.out_dir)

    # Collect all rows from all files (metadata + deid text only)
    all_rows = []
    file_summaries = []

    for path in INPUT_FILES:
        if not os.path.exists(path):
            raise RuntimeError("Input file not found: {}".format(path))

        df = read_deid_csv(path)

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

        # Guardrail: ensure we are NOT accidentally using raw note text
        if "deid" not in deid_col.lower():
            raise RuntimeError("Refusing to proceed: detected text col doesn't look de-identified: {}".format(deid_col))

        # Try parse datetime if present
        dt_parsed_series = None
        if dt_col is not None:
            dt_parsed_series = try_parse_datetime(df[dt_col].astype(str))

        n_rows = len(df)
        if args.max_rows_per_file is not None:
            n_rows = min(n_rows, args.max_rows_per_file)

        file_summaries.append((os.path.basename(path), pid_col, note_type_col or "NONE",
                               deid_col, dt_col or "NONE", n_rows))

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
        raise RuntimeError("No rows found across the provided inputs.")

    # Decide patient id
    patient_id = _safe_str(args.patient_id).strip()

    if (patient_id is None) or (patient_id == ""):
        if not args.pick_random:
            raise RuntimeError("Provide --patient_id OR use --pick_random.")
        patient_id, n = pick_patient_id(all_rows, args.min_notes)
        print("Picked patient_id:", patient_id, "with notes:", n)
    else:
        if args.pick_random:
            print("NOTE: --pick_random ignored because --patient_id is set (or defaulted).")

    # Filter rows for that patient
    patient_rows = [r for r in all_rows if r["pid"] == patient_id]
    if not patient_rows:
        raise RuntimeError("No rows found for patient_id: {}".format(patient_id))

    # Write outputs
    patient_dir, timeline_path, combined_path = write_bundle(patient_id, patient_rows, args.out_dir)

    print("\nINPUT SUMMARY (per file):")
    for (fname, pid_col, type_col, deid_col, dt_col, n_rows) in file_summaries:
        print(" -", fname)
        print("    pid_col:", pid_col,
              "| note_type_col:", type_col,
              "| deid_text_col:", deid_col,
              "| datetime_col:", dt_col,
              "| rows_used:", n_rows)

    print("\nEXPORTED PATIENT:")
    print("  ENCRYPTED_PAT_ID:", patient_id)
    print("  Notes exported  :", len(patient_rows))
    print("  Patient dir     :", patient_dir)
    print("  Timeline CSV    :", timeline_path)
    print("  Combined TXT    :", combined_path)
    print("Done.")


if __name__ == "__main__":
    main()
