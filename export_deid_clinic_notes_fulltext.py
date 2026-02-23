#!/usr/bin/env python3
# export_deid_clinic_notes_fulltext.py
# Python 3.6.8 compatible
#
# Export FULL clinic note text with aggressive PHI redaction.
# Output columns:
#   - ENCRYPTED_PAT_ID
#   - NOTE_TYPE (if present)
#   - NOTE_TEXT_DEID
#
# Removes:
#   - MRN / IDs
#   - dates are NOT removed (but note_date column is NOT exported)
#   - names (patient + staff/provider) via strong cue-based patterns
#   - phone numbers / emails / addresses
#
# ALSO: optionally drops signature blocks ("Electronically signed by...") onward.

from __future__ import print_function
import os
import re
import pandas as pd

INPUT_PATH  = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"
OUTPUT_PATH = "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes.csv"
QA_REPORT   = "/home/apokol/Breast_Restore/DEID_QA_name_hits_sample.txt"

MAX_ROWS = None            # e.g. 500 for quick test
DROP_SIGNATURE_BLOCK = True  # Highly recommended


def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def is_blank(x):
    if x is None:
        return True
    t = _safe_str(x).strip()
    return t == "" or t.lower() in ("nan", "none", "null", "na", "n/a")

def auto_detect_columns(columns):
    text_col = None
    type_col = None
    enc_id_col = None

    for c in columns:
        lc = c.lower()
        if enc_id_col is None and "encrypt" in lc:
            enc_id_col = c
        if text_col is None and "text" in lc:
            text_col = c
        if type_col is None and "type" in lc:
            type_col = c

    return text_col, type_col, enc_id_col


def redact_phi_strong(text):
    t = _safe_str(text)
    if t.strip() == "":
        return ""

    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # --- OPTIONAL: drop signature block onward (removes lots of names) ---
    if DROP_SIGNATURE_BLOCK:
        # Common markers that begin signature / authentication metadata
        sig_markers = [
            r"(?im)^\s*electronically\s+signed\s+by\b.*$",
            r"(?im)^\s*signed\s+by\b.*$",
            r"(?im)^\s*signature\s*:\b.*$",
            r"(?im)^\s*authenticated\s+by\b.*$",
        ]
        for pat in sig_markers:
            m = re.search(pat, t)
            if m:
                t = t[:m.start()].rstrip() + "\n[SIGNATURE_BLOCK_REDACTED]\n"
                break

    # Emails / phones
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", t)
    t = re.sub(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", t)

    # SSN
    t = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", t)

    # Long numeric IDs (6+ digits)
    t = re.sub(r"\b\d{6,}\b", "[ID]", t)

    # Address-ish patterns (heuristic)
    t = re.sub(
        r"\b\d{1,5}\s+[A-Za-z0-9\s]+\s+(Street|St|Avenue|Ave|Road|Rd|Blvd|Lane|Ln|Drive|Dr)\b",
        "[ADDRESS]",
        t,
        flags=re.IGNORECASE
    )

    # --- Header field redaction (very common in Epic text) ---
    header_fields = [
        "Patient", "Patient Name", "Name", "MRN", "DOB", "Date of Birth",
        "Sex", "Gender", "Address", "Phone", "Email",
        "PCP", "Primary Care Provider", "Author", "Attending", "Provider",
        "Ordering Provider", "Referring Provider",
        "Encounter", "Account", "CSN",
    ]
    # Example: "Patient: Jane Doe"
    for fld in header_fields:
        t = re.sub(r"(?im)^\s*%s\s*:\s*.*$" % re.escape(fld), "%s: [REDACTED]" % fld, t)

    # --- Strong provider/staff cue patterns ---
    # Lastname, Firstname (often provider lists)
    t = re.sub(r"\b([A-Z][a-zA-Z'\-]+),\s*([A-Z][a-zA-Z'\-]+)\b", "[NAME]", t)

    # Titles (Dr, MD, RN, PA, NP etc.) followed by likely name
    t = re.sub(r"\b(Dr|Doctor)\.?\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?\b", r"\1 [NAME]", t)
    t = re.sub(r"\b([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?)\s*,\s*(MD|DO|RN|PA|NP|MBBS)\b", "[NAME], \\2", t)

    # "Signed by/Author/Attending/Provider/PCP: First Last"
    cue_words = [
        "signed by", "electronically signed by", "author", "attending", "provider", "pcp",
        "referring provider", "ordering provider", "reviewed by", "discussed with",
        "seen by", "cc", "copied to"
    ]
    cue_pat = r"(?i)\b(" + "|".join([re.escape(x) for x in cue_words]) + r")\s*[:\-]\s*"
    # Replace cue + (First Last[ Middle]?) with cue + [NAME]
    t = re.sub(cue_pat + r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b", r"\1: [NAME]", t)

    # "Dear First Last," greetings
    t = re.sub(r"(?im)^\s*dear\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2}\s*,", "Dear [NAME],", t)

    # Family member labels sometimes include names: "Mother: Jane Doe"
    t = re.sub(r"(?im)^\s*(mother|father|husband|wife|daughter|son|sister|brother)\s*:\s*.*$",
               r"\1: [NAME]", t)

    return t


def find_remaining_name_like_hits(text, max_hits=20):
    """
    QA helper: find suspicious remaining "First Last" capitalized pairs.
    This will over-fire, but it's useful to sanity check.
    """
    t = _safe_str(text)
    hits = []
    # Two capitalized words in a row (not at sentence start only; heuristic)
    pat = re.compile(r"\b[A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+\b")
    for m in pat.finditer(t):
        hits.append(m.group(0))
        if len(hits) >= max_hits:
            break
    return hits


def main():
    if not os.path.exists(INPUT_PATH):
        raise RuntimeError("Input file not found: {}".format(INPUT_PATH))

    print("Loading clinic notes:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, dtype=object, engine="python", encoding="latin1")

    text_col, type_col, enc_id_col = auto_detect_columns(df.columns)

    if text_col is None:
        raise RuntimeError("Could not detect note text column (needs 'text' in header).")
    if enc_id_col is None:
        raise RuntimeError("Could not detect ENCRYPTED_PAT_ID column (needs 'encrypt' in header).")

    print("Detected columns:")
    print("  ENCRYPTED_PAT_ID:", enc_id_col)
    print("  NOTE_TYPE       :", type_col)
    print("  NOTE_TEXT       :", text_col)
    print("DROP_SIGNATURE_BLOCK:", DROP_SIGNATURE_BLOCK)

    rows = []
    qa_lines = []
    n = 0

    for i, row in df.iterrows():
        if MAX_ROWS is not None and n >= MAX_ROWS:
            break

        pid = row.get(enc_id_col, "")
        note_type = row.get(type_col, "") if type_col else ""

        raw_text = row.get(text_col, "")
        deid = redact_phi_strong(raw_text)

        rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "NOTE_TYPE": note_type,
            "NOTE_TEXT_DEID": deid
        })

        # QA: sample suspicious leftover name-like hits for first few notes
        if n < 25:
            hits = find_remaining_name_like_hits(deid, max_hits=10)
            if hits:
                qa_lines.append("ROW {} ENCRYPTED_PAT_ID={}\nHITS: {}\n".format(n, pid, hits))

        n += 1

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    with open(QA_REPORT, "w") as f:
        f.write("De-ID QA report (suspicious remaining Capitalized-Pair hits)\n")
        f.write("This is heuristic and may include false alarms.\n\n")
        if qa_lines:
            f.write("\n---\n".join(qa_lines))
        else:
            f.write("No suspicious hits found in first 25 notes.\n")

    print("Wrote:", OUTPUT_PATH)
    print("Wrote QA:", QA_REPORT)
    print("Rows exported:", len(out))
    print("Done.")


if __name__ == "__main__":
    main()
