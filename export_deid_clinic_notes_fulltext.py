#!/usr/bin/env python3
# export_one_patient_deid_bundle.py
# Python 3.6.8 compatible
#
# Usage examples (YOUR real files):
 # python export_one_patient_deid_bundle.py \
       --inputs \
         "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv" \
         "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv" \
         "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v5.csv" \
       --patient_id "63B0526207E98425D35E7EA737AB89AA" \
       --out_dir "/home/apokol/Breast_Restore/PATIENT_BUNDLES"
#
#   python export_one_patient_deid_bundle.py \
#       --inputs \
#         "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv" \
#         "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv" \
#         "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v5.csv" \
#       --pick_random --min_notes 20 \
#       --out_dir "/home/apokol/Breast_Restore/PATIENT_BUNDLES"
#
# Guardrail:
# - This script refuses to run unless each input looks like your DE-ID CTXWIPE files
#   AND the detected note text column is clearly DE-ID.

from __future__ import print_function
import os
import re
import argparse
import random
import datetime
import pandas as pd


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
    Best-effort date/time col detection. We keep it broad:
    - service_date, note_date, note_datetime, encounter_date, created, etc.
    """
    date_like = []
    for c in columns:
        lc = c.lower()
        if any(k in lc for k in ["date", "datetime", "time", "created", "service"]):
            date_like.append(c)

    # Prefer more specific names first
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
# DE-ID guardrails (file + content)
# ----------------------------
def looks_like_ctxwipe_deid_filename(path):
    """
    Your stated naming convention:
      v3 = clinic, v4 = op, v5 = ip
    We enforce "CTXWIPE" + "DEID" in filename to reduce risk of accidentally loading raw notes.
    """
    base = os.path.basename(path).upper()
    if "CTXWIPE" not in base:
        return False
    if not base.startswith("DEID_") and "DEID" not in base:
        return False
    # allow any CTXWIPE version (v2..vN), but your current set is v3/v4/v5.
    return True


def sample_looks_deidentified(text):
    """
    Content-level sanity check.
    We look for common de-id artifacts seen in your data: [NAME], [NAME_CTX_REDACTED], etc.
    This is intentionally conservative: if we cannot find any marker, we refuse.
    """
    t = _safe_str(text)
    if not t:
        return False
    # common tokens you showed in screenshots
    markers = ["[NAME", "CTX_REDACTED", "NAME_CTX_REDACTED"]
    up = t.upper()
    return any(m in up for m in markers)


# ----------------------------
# Core logic
# ----------------------------
def read_deid_csv(path):
    # Python 3.6 / pandas: "errors" here refers to decoding error handling for encoding.
    # Use engine="python" to be tolerant of odd quoting.
    df = pd.read_csv(path, dtype=object, engine="python", encoding="utf-8", errors="replace")
    return df


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
        # fallback: pick max
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
    # 1) parsed datetime (missing goes to end)
    # 2) source file name
    # 3) row index
    max_dt = datetime.datetime.max
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["dt_parsed"] if r["dt_parsed"] is not None else max_dt,
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
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more DE-ID CSV files (CTXWIPE).")
    ap.add_argument("--out_dir", required=True, help="Output directory for patient bundles.")
    ap.add_argument("--patient_id", default=None, help="ENCRYPTED_PAT_ID to export.")
    ap.add_argument("--pick_random", action="store_true", help="Pick a random patient with >= --min_notes.")
    ap.add_argument("--min_notes", type=int, default=10, help="Used with --pick_random.")
    ap.add_argument("--max_rows_per_file", type=int, default=None, help="Optional: cap rows per file for fast tests.")
    ap.add_argument(
        "--allow_non_ctxwipe_filenames",
        action="store_true",
        help="Override filename guardrail (NOT recommended)."
    )
    args = ap.parse_args()

    ensure_out_dir(args.out_dir)

    # Collect all rows from all files (metadata + deid text only)
    all_rows = []
    file_summaries = []

    for path in args.inputs:
        if not os.path.exists(path):
            raise RuntimeError("Input file not found: {}".format(path))

        # Guardrail 1: filename pattern (unless overridden)
        if (not args.allow_non_ctxwipe_filenames) and (not looks_like_ctxwipe_deid_filename(path)):
            raise RuntimeError(
                "Refusing to proceed: input does not look like your DE-ID CTXWIPE file.\n"
                "Got: {}\n"
                "Expected filename containing 'DEID' and 'CTXWIPE' (e.g., *_CTXWIPE_v3.csv).\n"
                "If you *really* want to override, pass --allow_non_ctxwipe_filenames.".format(path)
            )

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

        # Guardrail 2: ensure we are NOT accidentally using raw note text
        if "deid" not in deid_col.lower():
            raise RuntimeError(
                "Refusing to proceed: detected text col doesn't look de-identified: {} (file: {})".format(deid_col, path)
            )

        # Guardrail 3: content-level check (sample a few non-empty notes)
        # We check for de-id markers like [NAME...] to ensure this is your scrubbed set.
        found_marker = False
        # scan up to first 200 rows, looking for markers
        scan_n = min(len(df), 200)
        for j in range(scan_n):
            t = _safe_str(df.iloc[j][deid_col])
            if t and sample_looks_deidentified(t):
                found_marker = True
                break
        if not found_marker:
            raise RuntimeError(
                "Refusing to proceed: could not find DE-ID markers (e.g., [NAME], NAME_CTX_REDACTED) "
                "in the first {} rows of file:\n{}\n"
                "This suggests it may NOT be from your de-id CTXWIPE set (or the text column is wrong).".format(scan_n, path)
            )

        # Try parse datetime if present
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
    patient_id = args.patient_id
    if patient_id is None:
        if not args.pick_random:
            raise RuntimeError("Provide --patient_id OR use --pick_random.")
        patient_id, n = pick_patient_id(all_rows, args.min_notes)
        print("Picked patient_id:", patient_id, "with notes:", n)

    # Filter rows for that patient
    patient_rows = [r for r in all_rows if r["pid"] == patient_id]
    if not patient_rows:
        raise RuntimeError("No rows found for patient_id: {}".format(patient_id))

    # Write outputs
    patient_dir, timeline_path, combined_path = write_bundle(patient_id, patient_rows, args.out_dir)

    print("\nINPUT SUMMARY (per file):")
    for (fname, pid_col, type_col, deid_col, dt_col, n_rows) in file_summaries:
        print(" -", fname)
        print("    pid_col:", pid_col, "| note_type_col:", type_col, "| deid_text_col:", deid_col, "| datetime_col:", dt_col, "| rows_used:", n_rows)

    print("\nEXPORTED PATIENT:")
    print("  ENCRYPTED_PAT_ID:", patient_id)
    print("  Notes exported  :", len(patient_rows))
    print("  Patient dir     :", patient_dir)
    print("  Timeline CSV    :", timeline_path)
    print("  Combined TXT    :", combined_path)
    print("Done.")


if __name__ == "__main__":
    main()
