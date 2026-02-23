#!/usr/bin/env python3
# export_deid_clinic_notes_fulltext_ctxwipe.py
# Python 3.6.8 compatible
#
# Export FULL clinic note text with strong PHI redaction and
# an additional "context wipe" around suspected names:
#   - remove N words before and after any suspected name-like span
#   - replace wiped region with [NAME_CTX_REDACTED]
#
# Output columns:
#   - ENCRYPTED_PAT_ID
#   - NOTE_TYPE (if present)
#   - NOTE_TEXT_DEID
#
# Writes QA report with wipe counts.

from __future__ import print_function
import os
import re
import pandas as pd

INPUT_PATH  = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"
OUTPUT_PATH = "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE.csv"
QA_REPORT   = "/home/apokol/Breast_Restore/DEID_QA_ctxwipe_report.txt"

MAX_ROWS = None
DROP_SIGNATURE_BLOCK = True

# Context wipe window: N words before and after suspected name span
CTX_WORDS_BEFORE = 3
CTX_WORDS_AFTER  = 3

# If True, we also wipe around "Firstname Lastname" pairs even without cues.
# More aggressive = safer, but may remove some non-PHI phrases like "Plastic Surgery".
AGGRESSIVE_PAIR_WIPE = False


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

def redact_nonname_phi(t):
    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Drop signature block onward
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

    # Emails
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", t)

    # Phones
    t = re.sub(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", t)

    # SSN
    t = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", t)

    # Long numeric IDs (6+ digits)
    t = re.sub(r"\b\d{6,}\b", "[ID]", t)

    # Explicit MRN/Account/Encounter/CSN patterns
    t = re.sub(r"(?i)\b(MRN|Medical Record Number)\s*[:#]?\s*[A-Za-z0-9\-]+\b", "MRN: [ID]", t)
    t = re.sub(r"(?i)\b(Account|Acct|Encounter|CSN)\s*[:#]?\s*[A-Za-z0-9\-]+\b", r"\1: [ID]", t)

    # Address-ish patterns
    t = re.sub(
        r"\b\d{1,5}\s+[A-Za-z0-9\s]+\s+(Street|St|Avenue|Ave|Road|Rd|Blvd|Lane|Ln|Drive|Dr)\b",
        "[ADDRESS]",
        t,
        flags=re.IGNORECASE
    )

    # Header fields (full-line)
    header_fields = [
        "Patient", "Patient Name", "Name", "MRN", "DOB", "Date of Birth",
        "Sex", "Gender", "Address", "Phone", "Email",
        "PCP", "Primary Care Provider", "Author", "Attending", "Provider",
        "Ordering Provider", "Referring Provider",
        "Encounter", "Account", "CSN",
    ]
    for fld in header_fields:
        t = re.sub(r"(?im)^\s*%s\s*:\s*.*$" % re.escape(fld), "%s: [REDACTED]" % fld, t)

    # Inline patient fields
    t = re.sub(r"(?i)\bpatient\s*name\s*:\s*[A-Za-z ,.'\-]+\b", "Patient Name: [REDACTED]", t)
    t = re.sub(r"(?i)\bpatient\s*:\s*[A-Za-z ,.'\-]+\b", "Patient: [REDACTED]", t)

    return t

def tokenize_with_spans(text):
    """
    Return list of (token, start_idx, end_idx).
    "Token" here is word-ish (letters/numbers/apostrophe/hyphen) or single punctuation.
    """
    tokens = []
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[^\s]", text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens

def is_capitalized_word(tok):
    # Heuristic: "Jane" or "O'Neil" or "McDonald"
    if not tok:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z'\-]*$", tok):
        return False
    return tok[0].isupper()

def find_name_token_windows(tokens):
    """
    Identify token index ranges [i, j] that likely represent names.
    We keep it conservative by focusing on cue-based triggers + honorifics.
    """
    ranges = []

    # Cue words (lowercase compare)
    cues = set([
        "author", "attending", "provider", "pcp", "signed", "by",
        "cc", "copied", "to", "reviewed", "with", "seen", "met", "spoke",
        "dear"
    ])

    # Honorifics/titles
    honorifics = set(["mr", "mrs", "ms", "miss", "dr", "doctor"])

    # Multi-word narrative cues handled at text-level with regex too,
    # but token-level catches variations.
    narrative_starts = set(["pleasure", "seeing", "meeting"])

    n = len(tokens)

    # 1) Honorific + 2-3 capitalized words
    for i in range(n):
        tok = tokens[i][0]
        tl = tok.lower().rstrip(".")
        if tl in honorifics:
            j = i + 1
            cap_count = 0
            while j < n and cap_count < 3:
                if is_capitalized_word(tokens[j][0]):
                    cap_count += 1
                    j += 1
                    continue
                break
            if cap_count >= 1:
                ranges.append((i, j - 1))

    # 2) "Lastname, Firstname"
    for i in range(n - 2):
        if is_capitalized_word(tokens[i][0]) and tokens[i + 1][0] == "," and is_capitalized_word(tokens[i + 2][0]):
            ranges.append((i, i + 2))

    # 3) Cue-based "cue : First Last"
    # look for cue then ":" or "-" then 2 capitalized words
    for i in range(n - 4):
        tok = tokens[i][0].lower().rstrip(".")
        if tok in cues and tokens[i + 1][0] in (":", "-"):
            if is_capitalized_word(tokens[i + 2][0]) and is_capitalized_word(tokens[i + 3][0]):
                # allow optional third word
                end = i + 3
                if i + 4 < n and is_capitalized_word(tokens[i + 4][0]):
                    end = i + 4
                ranges.append((i + 2, end))

    # 4) Optional aggressive: any "First Last" capitalized pair
    if AGGRESSIVE_PAIR_WIPE:
        for i in range(n - 1):
            if is_capitalized_word(tokens[i][0]) and is_capitalized_word(tokens[i + 1][0]):
                ranges.append((i, i + 1))

    return ranges

def merge_ranges(ranges):
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))
    merged = [list(ranges[0])]
    for a, b in ranges[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(x[0], x[1]) for x in merged]

def apply_context_wipes(text, tokens, name_ranges, before_words, after_words):
    """
    For each suspected name token range, wipe N words before/after (by token index)
    and replace wiped segment with [NAME_CTX_REDACTED].
    Returns (new_text, wipes_applied_count).
    """
    if not name_ranges:
        return text, 0

    wipes = []
    n = len(tokens)

    for (i, j) in name_ranges:
        wipe_i = max(0, i - before_words)
        wipe_j = min(n - 1, j + after_words)
        wipes.append((wipe_i, wipe_j))

    wipes = merge_ranges(wipes)

    # Convert token wipe indices to character spans
    spans = []
    for (wi, wj) in wipes:
        start = tokens[wi][1]
        end = tokens[wj][2]
        spans.append((start, end))

    # Apply from end to start to preserve indices
    out = text
    for (start, end) in sorted(spans, key=lambda x: x[0], reverse=True):
        out = out[:start] + "[NAME_CTX_REDACTED]" + out[end:]

    return out, len(spans)

def redact_phi_with_ctxwipe(text):
    t = _safe_str(text)
    if t.strip() == "":
        return "", 0

    # Step A: redact non-name PHI + some header-ish name fields
    t = redact_nonname_phi(t)

    # Step B: explicit narrative regex wipes for "pleasure of seeing Jane Doe"
    # This *directly* removes the name phrase (we still do ctxwipe too).
    narrative_regexes = [
        r"(?i)\bwe had the pleasure of seeing\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2}\b",
        r"(?i)\bit was a pleasure meeting\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2}\b",
        r"(?i)\bi saw\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2}\b",
        r"(?i)\bwe saw\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2}\b",
    ]
    for pat in narrative_regexes:
        t = re.sub(pat, lambda m: re.sub(r"\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2}\b", " [NAME]", m.group(0)), t)

    # Step C: context wipe around suspected names
    tokens = tokenize_with_spans(t)
    name_ranges = find_name_token_windows(tokens)
    name_ranges = merge_ranges(name_ranges)
    t2, wipes = apply_context_wipes(t, tokens, name_ranges, CTX_WORDS_BEFORE, CTX_WORDS_AFTER)

    return t2, wipes

def main():
    if not os.path.exists(INPUT_PATH):
        raise RuntimeError("Input file not found: {}".format(INPUT_PATH))

    print("Loading:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, dtype=object, engine="python", encoding="latin1")

    text_col, type_col, enc_id_col = auto_detect_columns(df.columns)
    if text_col is None:
        raise RuntimeError("Could not detect NOTE TEXT column (needs 'text' in header).")
    if enc_id_col is None:
        raise RuntimeError("Could not detect ENCRYPTED_PAT_ID column (needs 'encrypt' in header).")

    print("Detected columns:")
    print("  ENCRYPTED_PAT_ID:", enc_id_col)
    print("  NOTE_TYPE       :", type_col)
    print("  NOTE_TEXT       :", text_col)
    print("CTX wipe window:", CTX_WORDS_BEFORE, "before /", CTX_WORDS_AFTER, "after")
    print("AGGRESSIVE_PAIR_WIPE:", AGGRESSIVE_PAIR_WIPE)
    print("DROP_SIGNATURE_BLOCK:", DROP_SIGNATURE_BLOCK)

    out_rows = []
    qa_lines = []
    total_wipes = 0
    notes_with_wipes = 0

    n = 0
    for _, row in df.iterrows():
        if MAX_ROWS is not None and n >= MAX_ROWS:
            break

        pid = row.get(enc_id_col, "")
        note_type = row.get(type_col, "") if type_col else ""
        raw = row.get(text_col, "")

        deid, wipes = redact_phi_with_ctxwipe(raw)
        total_wipes += wipes
        if wipes > 0:
            notes_with_wipes += 1

        out_rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "NOTE_TYPE": note_type,
            "NOTE_TEXT_DEID": deid
        })

        if n < 30 and wipes > 0:
            qa_lines.append("ROW {} PID={} wipes={}\n".format(n, pid, wipes))

        n += 1

    out = pd.DataFrame(out_rows)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    with open(QA_REPORT, "w") as f:
        f.write("Context-wipe De-ID QA report\n")
        f.write("===========================\n\n")
        f.write("INPUT: {}\n".format(INPUT_PATH))
        f.write("OUTPUT: {}\n\n".format(OUTPUT_PATH))
        f.write("CTX_WORDS_BEFORE: {}\n".format(CTX_WORDS_BEFORE))
        f.write("CTX_WORDS_AFTER : {}\n".format(CTX_WORDS_AFTER))
        f.write("AGGRESSIVE_PAIR_WIPE: {}\n".format(AGGRESSIVE_PAIR_WIPE))
        f.write("DROP_SIGNATURE_BLOCK: {}\n\n".format(DROP_SIGNATURE_BLOCK))
        f.write("Rows exported: {}\n".format(len(out)))
        f.write("Notes with wipes: {}\n".format(notes_with_wipes))
        f.write("Total wipe segments applied: {}\n\n".format(total_wipes))
        if qa_lines:
            f.write("Sample (first 30 rows) wipe counts:\n")
            f.write("".join(qa_lines))

    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", QA_REPORT)
    print("Rows exported:", len(out))
    print("Notes with wipes:", notes_with_wipes)
    print("Total wipe segments:", total_wipes)
    print("Done.")

if __name__ == "__main__":
    main()
