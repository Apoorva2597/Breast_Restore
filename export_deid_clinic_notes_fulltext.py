#!/usr/bin/env python3
# export_deid_clinic_notes_fulltext_ctxwipe_v2.py
# Python 3.6.8 compatible

from __future__ import print_function
import os
import re
import pandas as pd

INPUT_PATH  = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"
OUTPUT_PATH = "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v2.csv"
QA_REPORT   = "/home/apokol/Breast_Restore/DEID_QA_ctxwipe_report_v2.txt"
QA_LEAKS    = "/home/apokol/Breast_Restore/DEID_QA_possible_name_leaks_v2.txt"

MAX_ROWS = None
DROP_SIGNATURE_BLOCK = True

# Wipe window around suspected names
CTX_WORDS_BEFORE = 3
CTX_WORDS_AFTER  = 3

# Turn ON: detect standalone "First Last" pairs anywhere
AGGRESSIVE_PAIR_WIPE = True

# Avoid wiping common clinical / org phrases that look like First Last
# Add more as you discover false positives
PAIR_ALLOWLIST = set([
    "Plastic Surgery",
    "Breast Clinic",
    "General Surgery",
    "Internal Medicine",
    "Family Medicine",
    "Radiation Oncology",
    "Medical Oncology",
    "Surgical Oncology",
    "Oncology Clinic",
    "University Hospital",
    "Michigan Medicine",
    "Ann Arbor",
    "United States",
    "Review Of",
    "History Of",
    "Physical Exam",
    "Assessment Plan",
    "Plan Of",
    "Follow Up",
])

# Name suffixes/credentials often attached to names
CRED_SUFFIX = set(["md","do","phd","rn","np","pa","pac","pa-c","crna","msw","lcsw","dnp","mba","mph"])


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

def is_blank(x):
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    t = str(x).strip()
    if t == "":
        return True
    if t.lower() in ("nan", "none", "null", "na", "n/a", ".", "-", "--"):
        return True
    return False

def redact_nonname_phi(t):
    t = t.replace("\r\n", "\n").replace("\r", "\n")

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

    # Addresses
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

    return t

def tokenize_with_spans(text):
    tokens = []
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[^\s]", text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens

def is_cap_word(tok):
    if not tok:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z'\-]*$", tok):
        return False
    return tok[0].isupper()

def looks_like_credential(tok):
    tl = tok.lower().strip().strip(".").replace("–","-")
    return tl in CRED_SUFFIX

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

def find_name_token_ranges(tokens):
    """
    Detect name-like token spans. Returns list of (i,j) token indices.
    """
    ranges = []
    n = len(tokens)

    honorifics = set(["mr","mrs","ms","miss","dr","doctor"])
    cues = set(["author","attending","provider","signed","by","pcp","referring","ordering"])

    # 1) Honorific + 1-3 cap words
    for i in range(n):
        tl = tokens[i][0].lower().rstrip(".")
        if tl in honorifics:
            j = i + 1
            cap = 0
            while j < n and cap < 3:
                if is_cap_word(tokens[j][0]):
                    cap += 1
                    j += 1
                else:
                    break
            if cap >= 1:
                ranges.append((i, j - 1))

    # 2) "Last, First"
    for i in range(n - 2):
        if is_cap_word(tokens[i][0]) and tokens[i+1][0] == "," and is_cap_word(tokens[i+2][0]):
            # maybe add middle name too
            end = i + 2
            if i + 3 < n and is_cap_word(tokens[i+3][0]):
                end = i + 3
            ranges.append((i, end))

    # 3) cue ":" First Last
    for i in range(n - 4):
        tl = tokens[i][0].lower().rstrip(".")
        if tl in cues and tokens[i+1][0] in (":", "-"):
            if is_cap_word(tokens[i+2][0]) and is_cap_word(tokens[i+3][0]):
                end = i + 3
                if i + 4 < n and is_cap_word(tokens[i+4][0]):
                    end = i + 4
                ranges.append((i+2, end))

    # 4) First M. Last (cap + cap initial + cap)
    for i in range(n - 4):
        if is_cap_word(tokens[i][0]) and re.match(r"^[A-Z]\.?$", tokens[i+1][0]) and is_cap_word(tokens[i+2][0]):
            ranges.append((i, i+2))

    # 5) Aggressive: any "Cap Cap" pair (standalone names)
    if AGGRESSIVE_PAIR_WIPE:
        for i in range(n - 1):
            a = tokens[i][0]
            b = tokens[i+1][0]
            if is_cap_word(a) and is_cap_word(b):
                phrase = a + " " + b
                if phrase in PAIR_ALLOWLIST:
                    continue

                # allow "Cap Cap, MD/DO/..."
                end = i + 1
                if i + 2 < n and tokens[i+2][0] in (",",):
                    if i + 3 < n and looks_like_credential(tokens[i+3][0]):
                        end = i + 3
                elif i + 2 < n and looks_like_credential(tokens[i+2][0]):
                    end = i + 2

                ranges.append((i, end))

    return merge_ranges(ranges)

def apply_context_wipes(text, tokens, name_ranges, before_words, after_words):
    if not name_ranges:
        return text, 0
    n = len(tokens)

    # Expand each detected name range by before/after words
    wipes = []
    for (i, j) in name_ranges:
        wi = max(0, i - before_words)
        wj = min(n - 1, j + after_words)
        wipes.append((wi, wj))
    wipes = merge_ranges(wipes)

    # Convert to char spans
    spans = []
    for (wi, wj) in wipes:
        start = tokens[wi][1]
        end = tokens[wj][2]
        spans.append((start, end))

    out = text
    for (start, end) in sorted(spans, key=lambda x: x[0], reverse=True):
        out = out[:start] + "[NAME_CTX_REDACTED]" + out[end:]
    return out, len(spans)

def deid_fulltext(text):
    t = _safe_str(text)
    if t.strip() == "":
        return "", 0

    # A) non-name PHI
    t = redact_nonname_phi(t)

    # B) explicit narrative patterns (direct name phrase)
    # replaces the name-ish tail with [NAME]
    narrative_regexes = [
        r"(?i)\bwe had the pleasure of seeing\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
        r"(?i)\bit was a pleasure meeting\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
        r"(?i)\bwe saw\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
        r"(?i)\bi saw\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
    ]
    for pat in narrative_regexes:
        t = re.sub(pat, lambda m: m.group(0).replace(m.group(1), "[NAME]"), t)

    # C) token-based name detection + context wipe
    tokens = tokenize_with_spans(t)
    name_ranges = find_name_token_ranges(tokens)
    t2, wipes = apply_context_wipes(t, tokens, name_ranges, CTX_WORDS_BEFORE, CTX_WORDS_AFTER)

    return t2, wipes

def find_possible_name_leaks(text, max_hits=50):
    """
    After de-id, scan for leftover 'Cap Cap' pairs (excluding allowlist).
    Returns list of strings.
    """
    hits = []
    toks = re.findall(r"\b[A-Z][a-zA-Z'\-]+\b", text)
    # very rough scan: look at adjacent CapWords in original word order by splitting
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    for i in range(len(words) - 1):
        a = words[i]
        b = words[i+1]
        if is_cap_word(a) and is_cap_word(b):
            phrase = a + " " + b
            if phrase in PAIR_ALLOWLIST:
                continue
            hits.append(phrase)
            if len(hits) >= max_hits:
                break
    return hits

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
    total_wipes = 0
    notes_with_wipes = 0

    leak_examples = {}
    n = 0

    for _, row in df.iterrows():
        if MAX_ROWS is not None and n >= MAX_ROWS:
            break

        pid = row.get(enc_id_col, "")
        note_type = row.get(type_col, "") if type_col else ""
        raw = row.get(text_col, "")

        deid, wipes = deid_fulltext(raw)
        total_wipes += wipes
        if wipes > 0:
            notes_with_wipes += 1

        # collect leak hits for QA (no context)
        leaks = find_possible_name_leaks(deid, max_hits=10)
        if leaks:
            leak_examples.setdefault(pid, [])
            for x in leaks:
                if x not in leak_examples[pid]:
                    leak_examples[pid].append(x)
                if len(leak_examples[pid]) >= 10:
                    break

        out_rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "NOTE_TYPE": note_type,
            "NOTE_TEXT_DEID": deid
        })

        n += 1

    out = pd.DataFrame(out_rows)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    with open(QA_REPORT, "w") as f:
        f.write("CTXWIPE v2 QA report\n")
        f.write("====================\n\n")
        f.write("INPUT: {}\n".format(INPUT_PATH))
        f.write("OUTPUT: {}\n\n".format(OUTPUT_PATH))
        f.write("CTX_WORDS_BEFORE: {}\n".format(CTX_WORDS_BEFORE))
        f.write("CTX_WORDS_AFTER : {}\n".format(CTX_WORDS_AFTER))
        f.write("AGGRESSIVE_PAIR_WIPE: {}\n".format(AGGRESSIVE_PAIR_WIPE))
        f.write("DROP_SIGNATURE_BLOCK: {}\n\n".format(DROP_SIGNATURE_BLOCK))
        f.write("Rows exported: {}\n".format(len(out)))
        f.write("Notes with wipes: {}\n".format(notes_with_wipes))
        f.write("Total wipe segments applied: {}\n".format(total_wipes))

    # Write leak scan report (ONLY the suspected leftover pairs, no note text)
    with open(QA_LEAKS, "w") as f:
        f.write("Possible leftover CapCap pairs after de-id (v2)\n")
        f.write("==============================================\n\n")
        f.write("These are NOT guaranteed to be names (some are clinical phrases).\n")
        f.write("Use this list to extend PAIR_ALLOWLIST or tighten rules.\n\n")
        shown = 0
        for pid in leak_examples:
            f.write("ENCRYPTED_PAT_ID: {}\n".format(pid))
            f.write("  pairs: {}\n\n".format(", ".join(leak_examples[pid][:10])))
            shown += 1
            if shown >= 200:
                break

    print("Wrote:", OUTPUT_PATH)
    print("Wrote:", QA_REPORT)
    print("Wrote:", QA_LEAKS)
    print("Rows exported:", len(out))
    print("Notes with wipes:", notes_with_wipes)
    print("Total wipe segments:", total_wipes)
    print("Done.")

if __name__ == "__main__":
    main()
