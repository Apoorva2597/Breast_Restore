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
# Removes / redacts:
#   - MRN / long numeric IDs / account numbers
#   - phone numbers / emails / addresses
#   - patient & staff/provider names via cue-based patterns
#   - signature blocks (optional but recommended)
#
# Note: This is heuristic redaction (not a certified de-id system).
# It is designed to remove common direct identifiers in Epic-style notes.

from __future__ import print_function
import os
import re
import pandas as pd

INPUT_PATH  = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"
OUTPUT_PATH = "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes.csv"
QA_REPORT   = "/home/apokol/Breast_Restore/DEID_QA_name_hits_sample.txt"

MAX_ROWS = None               # set e.g. 500 for quick test
DROP_SIGNATURE_BLOCK = True   # highly recommended


def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


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

    # ------------------------------------------------------------
    # 0) Drop signature block onward (removes lots of trailing PHI)
    # ------------------------------------------------------------
    if DROP_SIGNATURE_BLOCK:
        sig_markers = [
            r"(?im)^\s*electronically\s+signed\s+by\b.*$",
            r"(?im)^\s*signed\s+by\b.*$",
            r"(?im)^\s*signature\s*:\b.*$",
            r"(?im)^\s*authenticated\s+by\b.*$",
            r"(?im)^\s*dictated\s+by\b.*$",
        ]
        for pat in sig_markers:
            m = re.search(pat, t)
            if m:
                t = t[:m.start()].rstrip() + "\n[SIGNATURE_BLOCK_REDACTED]\n"
                break

    # -----------------------
    # 1) Non-name identifiers
    # -----------------------
    # Emails
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", t)

    # Phones
    t = re.sub(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", t)

    # SSN (rare but possible)
    t = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", t)

    # Long numeric IDs (6+ digits) => MRN/account/etc.
    t = re.sub(r"\b\d{6,}\b", "[ID]", t)

    # Explicit MRN / Account / Encounter / CSN patterns
    t = re.sub(r"(?i)\b(MRN|Medical Record Number)\s*[:#]?\s*[A-Za-z0-9\-]+\b", "MRN: [ID]", t)
    t = re.sub(r"(?i)\b(Account|Acct|Encounter|CSN)\s*[:#]?\s*[A-Za-z0-9\-]+\b", r"\1: [ID]", t)

    # Addresses (heuristic)
    t = re.sub(
        r"\b\d{1,5}\s+[A-Za-z0-9\s]+\s+(Street|St|Avenue|Ave|Road|Rd|Blvd|Lane|Ln|Drive|Dr)\b",
        "[ADDRESS]",
        t,
        flags=re.IGNORECASE
    )

    # ------------------------------------------------------------
    # 2) Header-style fields (common Epic templates)
    #    Redact full lines like "Patient: Jane Doe"
    # ------------------------------------------------------------
    header_fields = [
        "Patient", "Patient Name", "Name", "MRN", "DOB", "Date of Birth",
        "Sex", "Gender", "Address", "Phone", "Email",
        "PCP", "Primary Care Provider", "Author", "Attending", "Provider",
        "Ordering Provider", "Referring Provider",
        "Encounter", "Account", "CSN",
    ]
    for fld in header_fields:
        t = re.sub(r"(?im)^\s*%s\s*:\s*.*$" % re.escape(fld), "%s: [REDACTED]" % fld, t)

    # Inline Patient: <name> patterns (not just full-line headers)
    t = re.sub(r"(?i)\bpatient\s*name\s*:\s*[A-Za-z ,.'\-]+\b", "Patient Name: [NAME]", t)
    t = re.sub(r"(?i)\bpatient\s*:\s*[A-Za-z ,.'\-]+\b", "Patient: [NAME]", t)

    # ------------------------------------------------------------
    # 3) Name patterns (cue-based to avoid over-redaction)
    # ------------------------------------------------------------

    # 3a) Lastname, Firstname (very common for providers)
    t = re.sub(r"\b([A-Z][a-zA-Z'\-]+),\s*([A-Z][a-zA-Z'\-]+)\b", "[NAME]", t)

    # 3b) Titles: Dr/Doctor + Name
    t = re.sub(r"\b(Dr|Doctor)\.?\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2}\b", r"\1 [NAME]", t)

    # 3c) Honorifics: Ms/Mr/Mrs/Miss + Name
    # NOTE: you said there is a space after "Ms" then the name (e.g., "Ms Jane Doe").
    # This pattern covers both "Ms Jane Doe" and "Ms. Jane Doe".
    t = re.sub(
        r"\b(Ms|Mr|Mrs|Miss)\.?\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2}\b",
        r"\1 [NAME]",
        t
    )

    # 3d) Credential formats: "Firstname Lastname, MD"
    t = re.sub(r"\b([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?)\s*,\s*(MD|DO|RN|PA|NP|MBBS)\b", "[NAME], \\2", t)

    # 3e) “Dear First Last,” greetings
    t = re.sub(r"(?im)^\s*dear\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2}\s*,", "Dear [NAME],", t)

    # 3f) Family member labels that often include names
    t = re.sub(
        r"(?im)^\s*(mother|father|husband|wife|daughter|son|sister|brother)\s*:\s*.*$",
        r"\1: [NAME]",
        t
    )

    # ------------------------------------------------------------
    # 4) Cue phrases inside prose (your “pleasure of seeing Jane Doe” case)
    # ------------------------------------------------------------
    # Strong, narrative cues that usually precede the patient's name.
    narrative_cues = [
        "we had the pleasure of seeing",
        "we had the pleasure to see",
        "it was a pleasure seeing",
        "it was a pleasure to see",
        "it was a pleasure meeting",
        "i had the pleasure of seeing",
        "i had the pleasure to see",
        "i saw",
        "we saw",
        "seen today is",
        "today we saw",
        "patient is",
        "this is",
        "met with",
        "spoke with",
        "spoke to",
        "discussed with",
    ]

    for cue in narrative_cues:
        t = re.sub(
            r"(?i)\b(" + re.escape(cue) + r")\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
            r"\1 [NAME]",
            t
        )

    # ------------------------------------------------------------
    # 5) Administrative cue fields (Author/Attending/PCP/etc.) inline
    # ------------------------------------------------------------
    cue_words = [
        "signed by", "electronically signed by", "author", "attending", "provider", "pcp",
        "referring provider", "ordering provider", "reviewed by", "copied to", "cc"
    ]
    cue_pat = r"(?i)\b(" + "|".join([re.escape(x) for x in cue_words]) + r")\s*[:\-]\s*"
    t = re.sub(
        cue_pat + r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
        r"\1: [NAME]",
        t
    )

    return t


def find_remaining_name_like_hits(text, max_hits=25):
    """
    QA helper: finds suspicious remaining "First Last" capitalized pairs.
    This will include false alarms (e.g., "Plastic Surgery"), but it's useful to spot leaks.
    """
    t = _safe_str(text)
    hits = []
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

    for _, row in df.iterrows():
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
        if n < 30:
            hits = find_remaining_name_like_hits(deid, max_hits=12)
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
            f.write("No suspicious hits found in first 30 notes.\n")

    print("Wrote:", OUTPUT_PATH)
    print("Wrote QA:", QA_REPORT)
    print("Rows exported:", len(out))
    print("Done.")


if __name__ == "__main__":
    main()
