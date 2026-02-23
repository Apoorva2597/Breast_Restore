#!/usr/bin/env python3
# export_deid_clinic_notes_fulltext.py
# Python 3.6.8 compatible
#
# Exports FULL clinic note text with PHI redacted.
# Keeps only:
#   - ENCRYPTED_PAT_ID
#   - NOTE_TYPE (if present)
#   - FULL REDACTED NOTE TEXT
#
# Removes:
#   - MRN
#   - patient_id
#   - note_date
#   - names
#   - phone numbers
#   - emails
#   - long numeric identifiers
#   - addresses
#   - provider names

from __future__ import print_function
import os
import re
import pandas as pd

INPUT_PATH  = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"
OUTPUT_PATH = "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes.csv"

MAX_ROWS = None  # set to a number (e.g., 500) if you want a smaller test export

def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def redact_phi(text):
    t = _safe_str(text)
    if t.strip() == "":
        return ""

    # Normalize whitespace
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Emails
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", t)

    # Phone numbers
    t = re.sub(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", t)

    # SSN
    t = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", t)

    # Long numeric IDs (6+ digits)
    t = re.sub(r"\b\d{6,}\b", "[ID]", t)

    # MRN / Account / Encounter patterns
    t = re.sub(r"(?i)\b(MRN|Medical Record Number)\s*[:#]?\s*[A-Za-z0-9\-]+\b", "MRN: [ID]", t)
    t = re.sub(r"(?i)\b(Account|Acct|Encounter|CSN)\s*[:#]?\s*[A-Za-z0-9\-]+\b", r"\1: [ID]", t)

    # Name lines
    t = re.sub(r"(?im)^(patient\s*name|name)\s*:\s*.*$", r"\1: [NAME]", t)

    # Provider names
    t = re.sub(r"\bDr\.?\s+[A-Z][a-zA-Z\-']+\b", "Dr [PROVIDER]", t)

    # Street addresses (heuristic)
    t = re.sub(
        r"\b\d{1,5}\s+[A-Za-z0-9\s]+\s+(Street|St|Avenue|Ave|Road|Rd|Blvd|Lane|Ln|Drive|Dr)\b",
        "[ADDRESS]",
        t,
        flags=re.IGNORECASE
    )

    return t

def auto_detect_columns(columns):
    text_col = None
    type_col = None
    enc_id_col = None

    for c in columns:
        lc = c.lower()
        if text_col is None and "text" in lc:
            text_col = c
        if type_col is None and "type" in lc:
            type_col = c
        if enc_id_col is None and "encrypt" in lc:
            enc_id_col = c

    return text_col, type_col, enc_id_col

def main():
    if not os.path.exists(INPUT_PATH):
        raise RuntimeError("Input file not found: {}".format(INPUT_PATH))

    print("Loading clinic notes...")
    df = pd.read_csv(INPUT_PATH, dtype=object, engine="python", encoding="latin1")

    text_col, type_col, enc_id_col = auto_detect_columns(df.columns)

    if text_col is None:
        raise RuntimeError("Could not detect note text column.")
    if enc_id_col is None:
        raise RuntimeError("Could not detect ENCRYPTED_PAT_ID column.")

    print("Detected columns:")
    print("  Text column:", text_col)
    print("  Type column:", type_col)
    print("  Encrypted ID column:", enc_id_col)

    rows = []
    count = 0

    for _, row in df.iterrows():
        if MAX_ROWS is not None and count >= MAX_ROWS:
            break

        full_text = redact_phi(row[text_col])

        rows.append({
            "ENCRYPTED_PAT_ID": row[enc_id_col],
            "NOTE_TYPE": row[type_col] if type_col else "",
            "NOTE_TEXT_DEID": full_text
        })

        count += 1

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Wrote:", OUTPUT_PATH)
    print("Rows exported:", len(out))
    print("Done.")

if __name__ == "__main__":
    main()
