#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_WITH_AUDIT.py
Python 3.6.8 compatible

STAGING ONLY — produces:
  ./_outputs/patient_stage_summary.csv   (must contain HAS_STAGE2)
  ./_outputs/stage2_fn_raw_note_snippets.csv
"""

from __future__ import print_function
import os
import re
import pandas as pd

INPUT_NOTES = "_staging_inputs/HPI11526 Operation Notes.csv"
OUTPUT_SUMMARY = "_outputs/patient_stage_summary.csv"
OUTPUT_AUDIT = "_outputs/stage2_fn_raw_note_snippets.csv"

# ----------------------------
# Robust CSV reader (match validator style)
# ----------------------------

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))

# ----------------------------
# Regex Logic
# ----------------------------

EXCHANGE_STRICT = re.compile(
    r"""
    (
        (underwent|performed|taken\s+to\s+the\s+OR|returned\s+to\s+the\s+operating\s+room)
        .{0,120}?
        (exchange|removal|removed|replacement)
        .{0,120}?
        (tissue\s+expander|implant)
    )
    |
    (
        (exchange|removal|removed|replacement)
        .{0,80}?
        (tissue\s+expander|implant)
        .{0,80}?
        (for|with)
        .{0,80}?
        (permanent\s+)?(silicone|saline)?\s*(implant)
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

INTRAOP_SIGNALS = re.compile(
    r"""
    (estimated\s+blood\s+loss|EBL|
     specimen(s)?\s+sent|
     drains?\s+(placed|inserted)|
     anesthesia|
     incision\s+made|
     pocket\s+created|
     implant\s+placed|
     expander\s+removed|
     capsulotomy|capsulectomy|
     \bml\b\s+(removed|placed|instilled))
    """,
    re.IGNORECASE | re.VERBOSE,
)

NEGATIVE_CONTEXT = re.compile(
    r"""
    (scheduled\s+for|
     will\s+undergo|
     planning\s+to|
     considering|
     history\s+of|
     status\s+post|
     \bs/p\b|
     discussed|
     interested\s+in|
     plan:|
     follow[-\s]?up|
     here\s+for\s+follow)
    """,
    re.IGNORECASE | re.VERBOSE,
)

def is_true_exchange(text):
    if not EXCHANGE_STRICT.search(text):
        return False
    if NEGATIVE_CONTEXT.search(text):
        return False
    if not INTRAOP_SIGNALS.search(text):
        return False
    return True

def get_snippet(text, match_obj, window=200):
    start = max(match_obj.start() - window, 0)
    end = min(match_obj.end() + window, len(text))
    return text[start:end].replace("\n", " ").strip()

# ----------------------------
# Main
# ----------------------------

def main():

    root = os.path.abspath(".")
    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if not os.path.isfile(INPUT_NOTES):
        raise IOError("Operation Notes CSV not found: {}".format(INPUT_NOTES))

    df = read_csv_robust(INPUT_NOTES, dtype=str, low_memory=False)

    required_cols = ["ENCRYPTED_PAT_ID", "NOTE_ID", "NOTE_TEXT"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError("Missing required column: {}. Found: {}".format(c, list(df.columns)))

    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].fillna("").astype(str)

    stage2_patients = set()
    audit_rows = []

    for _, row in df.iterrows():

        pat_id = str(row["ENCRYPTED_PAT_ID"]).strip()
        note_id = str(row["NOTE_ID"])
        text = str(row["NOTE_TEXT"])

        if not pat_id:
            continue

        if is_true_exchange(text):

            stage2_patients.add(pat_id)

            for match in EXCHANGE_STRICT.finditer(text):
                snippet = get_snippet(text, match)
                audit_rows.append({
                    "ENCRYPTED_PAT_ID": pat_id,
                    "NOTE_ID": note_id,
                    "MATCH_TERM": "exchange_strict",
                    "SNIPPET": snippet,
                    "SOURCE_FILE": os.path.basename(INPUT_NOTES),
                })

    # ----------------------------
    # Build patient_stage_summary.csv
    # MUST contain: ENCRYPTED_PAT_ID + HAS_STAGE2
    # ----------------------------

    unique_patients = df["ENCRYPTED_PAT_ID"].dropna().unique()

    summary_rows = []
    for pid in unique_patients:
        summary_rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "HAS_STAGE2": 1 if pid in stage2_patients else 0
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUTPUT_AUDIT, index=False)

    print("Staging complete.")
    print("Patients:", len(unique_patients))
    print("Stage2 positive patients:", len(stage2_patients))

if __name__ == "__main__":
    main()
