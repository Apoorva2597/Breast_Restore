#!/usr/bin/env python3
# export_deid_notes_fulltext_ctxwipe_v4.py
# Python 3.6.8 compatible
#
# v4 adds (critical for OP notes):
#  - ALWAYS redact honorific + name (Ms/Dr/Mr + 1-3 CapWords) => [NAME]
#  - OP note "time-out verified by name/DOB/reg number" & provider roster line removal
#  - Safer aggressive CapCap learning: blocks common clinical CapCap phrases (Tissue Expander, Breast Cancer, etc.)
#  - Optional --drop_dates to replace common date patterns with [DATE] for sharing
#  - Keeps full note text (no snippets)

from __future__ import print_function
import os
import re
import argparse
import pandas as pd

DEFAULT_INPUT  = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Notes.csv"
DEFAULT_OUTPUT = "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v5.csv"
DEFAULT_QA     = "/home/apokol/Breast_Restore/DEID_QA_ctxwipe_report_v5.txt"
DEFAULT_LEAKS  = "/home/apokol/Breast_Restore/DEID_QA_possible_name_leaks_v5.txt"
DEFAULT_NAMES  = "/home/apokol/Breast_Restore/DEID_learned_name_list_v5.txt"

DROP_SIGNATURE_BLOCK = True

# context wipe window around detected name spans
CTX_WORDS_BEFORE = 3
CTX_WORDS_AFTER  = 3

# detect standalone "First Last" everywhere
AGGRESSIVE_PAIR_WIPE = True

# ------------------------------------------------------------
# Allowlist phrases that should NOT be treated as person names
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Block common clinical CapCap pairs from being learned as names
# (OP notes are full of these)
# ------------------------------------------------------------
NON_NAME_SECOND_TOKENS = set([
    "expander", "implant", "implants", "surgery", "clinic", "anesthesia",
    "cancer", "carcinoma", "reconstruction", "mastectomy", "biopsy",
    "procedure", "procedures", "diagnosis", "findings", "indications",
    "pathology", "specimen", "specimens", "drain", "drains", "incision",
    "breast", "axilla", "node", "nodes", "therapy", "radiation", "chemo",
    "hospital", "medicine", "service", "services", "operation", "operative",
    "note", "notes", "report", "reports", "board", "team"
])

NON_NAME_FIRST_TOKENS = set([
    "tissue", "breast", "general", "regional", "local", "post", "pre",
    "operative", "operation", "operative", "estimated", "final", "routine",
    "right", "left", "bilateral"
])

CRED_SUFFIX = set(["md","do","phd","rn","np","pa","pac","pa-c","crna","msw","lcsw","dnp","mba","mph"])

HONORIFICS = ("mr", "mrs", "ms", "miss", "dr", "doctor")

def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def auto_detect_columns(columns):
    """
    Tries to detect:
      - encrypted patient id col: contains 'encrypt'
      - note text col: contains 'text'
      - note type col: contains 'type'
    """
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

def is_cap_word(tok):
    if not tok:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z'\-]*$", tok):
        return False
    return tok[0].isupper()

def looks_like_credential(tok):
    tl = tok.lower().strip().strip(".").replace("–","-")
    return tl in CRED_SUFFIX

def tokenize_with_spans(text):
    tokens = []
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[^\s]", text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens

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

def normalize_name_string(s):
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" ,.;:-")
    return s

def capcap_looks_like_clinical(a, b):
    """
    Blocks common OP-note CapCap pairs from being learned as names.
    """
    al = a.lower()
    bl = b.lower()
    if (al in NON_NAME_FIRST_TOKENS) or (bl in NON_NAME_SECOND_TOKENS):
        return True
    phrase = a + " " + b
    if phrase in PAIR_ALLOWLIST:
        return True
    return False

def extract_candidate_names_from_tokens(tokens):
    """
    Returns list of name strings extracted from token patterns:
      - Honorific + capwords
      - Last, First
      - First M. Last
      - Aggressive Cap Cap (+ optional credential), with clinical blocklist
    """
    names = []
    n = len(tokens)

    # Honorific + 1-3 cap words (store only the name words, not honorific)
    for i in range(n):
        tl = tokens[i][0].lower().rstrip(".")
        if tl in HONORIFICS:
            parts = []
            j = i + 1
            cap = 0
            while j < n and cap < 3:
                if is_cap_word(tokens[j][0]):
                    parts.append(tokens[j][0])
                    cap += 1
                    j += 1
                else:
                    break
            if cap >= 1:
                names.append(normalize_name_string(" ".join(parts)))

    # Last, First (optionally middle)
    for i in range(n - 2):
        if is_cap_word(tokens[i][0]) and tokens[i+1][0] == "," and is_cap_word(tokens[i+2][0]):
            parts = [tokens[i+2][0], tokens[i][0]]  # normalize to First Last
            if i + 3 < n and is_cap_word(tokens[i+3][0]):
                parts = [tokens[i+2][0], tokens[i+3][0], tokens[i][0]]
            names.append(normalize_name_string(" ".join(parts)))

    # First M. Last
    for i in range(n - 2):
        if is_cap_word(tokens[i][0]) and re.match(r"^[A-Z]\.?$", tokens[i+1][0]) and is_cap_word(tokens[i+2][0]):
            names.append(normalize_name_string(tokens[i][0] + " " + tokens[i+2][0]))

    # Aggressive Cap Cap (with clinical blocklist)
    if AGGRESSIVE_PAIR_WIPE:
        for i in range(n - 1):
            a = tokens[i][0]
            b = tokens[i+1][0]
            if is_cap_word(a) and is_cap_word(b):
                if capcap_looks_like_clinical(a, b):
                    continue
                phrase = a + " " + b
                names.append(normalize_name_string(phrase))

    # Filter obvious non-names
    out = []
    for nm in names:
        if not nm:
            continue
        if nm in PAIR_ALLOWLIST:
            continue
        if len(nm.split()) < 2:
            continue
        if re.match(r"^[A-Z\s\-]{4,}$", nm):
            continue
        out.append(nm)

    return out

def learn_name_dictionary(df, text_col, max_rows=None):
    seen = set()
    total = 0
    learned = 0

    for idx, raw in enumerate(df[text_col].values):
        if max_rows is not None and idx >= max_rows:
            break
        t = _safe_str(raw)
        if not t.strip():
            continue
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        toks = tokenize_with_spans(t)
        cands = extract_candidate_names_from_tokens(toks)
        total += 1
        for nm in cands:
            key = nm.lower()
            if key not in seen:
                seen.add(key)
                learned += 1

    names = sorted(list(seen), key=lambda x: (-len(x), x))
    return names, total, learned

def build_global_name_regex(learned_names_lower):
    patterns = []
    for nm in learned_names_lower:
        parts = nm.split()
        esc_parts = [re.escape(p) for p in parts]
        pat = r"\b" + r"\s+".join(esc_parts) + r"\b"
        patterns.append(pat)
        if len(patterns) >= 4000:
            break
    if not patterns:
        return None
    big = "(" + "|".join(patterns) + ")"
    return re.compile(big, flags=re.IGNORECASE)

def apply_global_name_redaction(text, name_re):
    if name_re is None:
        return text
    return name_re.sub("[NAME]", text)

def drop_common_opnote_verification_lines(t):
    """
    Remove the highest-risk OP note lines without harming clinical meaning.
    """
    patterns = [
        # time-out / verification lines
        r"(?im)^\s*(time\s*-?\s*out|timeout)\b.*$",
        r"(?im)^\s*verified\s+by\b.*$",
        r"(?im)^\s*verified\s+correct\s+patient\b.*$",
        r"(?im)^\s*patient\s+was\s+verified\s+by\b.*$",
        r"(?im)^\s*verified\s+by\s+name.*$",
        r"(?im)^\s*verified\s+by\s+name,\s*age.*$",
        r"(?im)^\s*verified\s+by\s+name,\s*date\s+of\s+birth.*$",
        r"(?im)^\s*verified\s+by\s+name,\s*dob.*$",
        r"(?im)^\s*verified\s+by\s+.*reg\s*number.*$",
        r"(?im)^\s*verified\s+by\s+.*medical\s*record.*$",

        # provider roster lines that often contain names
        r"(?im)^\s*(attending|surgeon|assistant|resident|fellow|anesthesiologist|circulator|scrub|author|provider)\s*:\s*.*$",
        r"(?im)^\s*(primary\s+surgeon|assistant\s+surgeon|dictated\s+by|transcribed\s+by)\s*:\s*.*$",
    ]
    for pat in patterns:
        t = re.sub(pat, "[LINE_REDACTED]", t)
    return t

def redact_nonname_phi(t, drop_dates=False):
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # OP note verification/provider roster lines
    t = drop_common_opnote_verification_lines(t)

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

    # Explicit MRN/Account/Encounter/CSN patterns
    t = re.sub(r"(?i)\b(MRN|Medical Record Number)\s*[:#]?\s*[A-Za-z0-9\-]+\b", "MRN: [ID]", t)
    t = re.sub(r"(?i)\b(Account|Acct|Encounter|CSN)\s*[:#]?\s*[A-Za-z0-9\-]+\b", r"\1: [ID]", t)

    # Long numeric IDs (6+ digits) AFTER the explicit patterns above
    t = re.sub(r"\b\d{6,}\b", "[ID]", t)

    # Addresses (simple)
    t = re.sub(
        r"\b\d{1,5}\s+[A-Za-z0-9\s]+\s+(Street|St|Avenue|Ave|Road|Rd|Blvd|Lane|Ln|Drive|Dr)\b",
        "[ADDRESS]",
        t,
        flags=re.IGNORECASE
    )

    # Optional: drop dates for sharing (you asked "lets not keep note dates")
    if drop_dates:
        # 09/05/2019, 9/5/19, 2019-09-05
        t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", t)
        t = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", "[DATE]", t)
        # "September 5, 2019"
        t = re.sub(r"\b(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},\s+\d{4}\b", "[DATE]", t, flags=re.IGNORECASE)

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

def collapse_leftover_title_patterns(t):
    """
    Hard rule: ANY honorific + (1-3) CapWords becomes [NAME].
    This is what prevents 'Ms. Linda Hammond' leaking.
    """
    # Ms. First Last
    t = re.sub(r"(?i)\b(Mr|Mrs|Ms|Miss|Dr|Doctor)\.?\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2})\b", "[NAME]", t)
    # Surgeon: Dr. First Last -> Surgeon: [NAME]
    t = re.sub(r"(?i)\b(Dr|Doctor)\.?\s+\[NAME\]\b", "[NAME]", t)
    # Ms. [NAME] / Ms. [NAME_CTX_REDACTED] etc.
    t = re.sub(r"(?i)\b(Mr|Mrs|Ms|Miss|Dr|Doctor)\.?\s+\[[A-Za-z_]+\]\b", "[NAME]", t)
    return t

def find_name_token_ranges(tokens):
    ranges = []
    n = len(tokens)

    # Honorific + 1-3 cap words (include honorific)
    for i in range(n):
        tl = tokens[i][0].lower().rstrip(".")
        if tl in HONORIFICS:
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

    # Last, First
    for i in range(n - 2):
        if is_cap_word(tokens[i][0]) and tokens[i+1][0] == "," and is_cap_word(tokens[i+2][0]):
            end = i + 2
            if i + 3 < n and is_cap_word(tokens[i+3][0]):
                end = i + 3
            ranges.append((i, end))

    # First M. Last
    for i in range(n - 2):
        if is_cap_word(tokens[i][0]) and re.match(r"^[A-Z]\.?$", tokens[i+1][0]) and is_cap_word(tokens[i+2][0]):
            ranges.append((i, i+2))

    # Aggressive: Cap Cap pair (+ optional credential), but block clinical pairs
    if AGGRESSIVE_PAIR_WIPE:
        for i in range(n - 1):
            a = tokens[i][0]
            b = tokens[i+1][0]
            if is_cap_word(a) and is_cap_word(b):
                if capcap_looks_like_clinical(a, b):
                    continue
                end = i + 1
                if i + 2 < n and tokens[i+2][0] == "," and i + 3 < n and looks_like_credential(tokens[i+3][0]):
                    end = i + 3
                elif i + 2 < n and looks_like_credential(tokens[i+2][0]):
                    end = i + 2
                ranges.append((i, end))

    return merge_ranges(ranges)

def apply_context_wipes(text, tokens, name_ranges, before_words, after_words):
    if not name_ranges:
        return text, 0
    n = len(tokens)

    wipes = []
    for (i, j) in name_ranges:
        wi = max(0, i - before_words)
        wj = min(n - 1, j + after_words)
        wipes.append((wi, wj))
    wipes = merge_ranges(wipes)

    spans = []
    for (wi, wj) in wipes:
        start = tokens[wi][1]
        end = tokens[wj][2]
        spans.append((start, end))

    out = text
    for (start, end) in sorted(spans, key=lambda x: x[0], reverse=True):
        out = out[:start] + "[NAME_CTX_REDACTED]" + out[end:]
    return out, len(spans)

def find_possible_name_leaks(text, max_hits=50):
    hits = []
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    for i in range(len(words) - 1):
        a = words[i]
        b = words[i+1]
        if is_cap_word(a) and is_cap_word(b):
            phrase = a + " " + b
            if phrase in PAIR_ALLOWLIST:
                continue
            if capcap_looks_like_clinical(a, b):
                continue
            hits.append(phrase)
            if len(hits) >= max_hits:
                break
    return hits

def deid_fulltext(raw_text, global_name_re, drop_dates=False):
    t = _safe_str(raw_text)
    if t.strip() == "":
        return "", 0

    # A) non-name PHI (+ OP note line drops)
    t = redact_nonname_phi(t, drop_dates=drop_dates)

    # B) narrative phrases
    narrative_regexes = [
        r"(?i)\bwe had the pleasure of seeing\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
        r"(?i)\bit was a pleasure meeting\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
        r"(?i)\bwe saw\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
        r"(?i)\bi saw\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})\b",
    ]
    for pat in narrative_regexes:
        t = re.sub(pat, lambda m: m.group(0).replace(m.group(1), "[NAME]"), t)

    # C) global learned-name redaction
    t = apply_global_name_redaction(t, global_name_re)

    # D) ALWAYS collapse honorific+name patterns (fixes Ms. Linda Hammond)
    t = collapse_leftover_title_patterns(t)

    # E) token-based detection + context wipe
    tokens = tokenize_with_spans(t)
    name_ranges = find_name_token_ranges(tokens)
    t2, wipes = apply_context_wipes(t, tokens, name_ranges, CTX_WORDS_BEFORE, CTX_WORDS_AFTER)

    # F) one more honorific collapse after wipes
    t2 = collapse_leftover_title_patterns(t2)

    return t2, wipes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--qa", default=DEFAULT_QA)
    ap.add_argument("--leaks", default=DEFAULT_LEAKS)
    ap.add_argument("--names_out", default=DEFAULT_NAMES)
    ap.add_argument("--max_rows", default=None, help="Optional: limit rows for quick test (e.g., 2000).")
    ap.add_argument("--drop_dates", action="store_true", help="Replace common dates with [DATE] for sharing.")
    args = ap.parse_args()

    max_rows = None
    if args.max_rows is not None:
        try:
            max_rows = int(args.max_rows)
        except Exception:
            raise RuntimeError("--max_rows must be an integer")

    if not os.path.exists(args.input):
        raise RuntimeError("Input file not found: {}".format(args.input))

    print("Loading:", args.input)
    df = pd.read_csv(args.input, dtype=object, engine="python", encoding="latin1")

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
    print("DROP_DATES:", bool(args.drop_dates))
    print("MAX_ROWS:", ("ALL" if max_rows is None else max_rows))

    # PASS 1: learn names
    learned_names, notes_scanned, learned_count = learn_name_dictionary(df, text_col, max_rows=max_rows)
    print("Pass1 learned name strings:", learned_count, "from notes scanned:", notes_scanned)

    with open(args.names_out, "w") as f:
        f.write("Learned name dictionary (lowercased)\n")
        f.write("===================================\n\n")
        for nm in learned_names[:5000]:
            f.write(nm + "\n")

    global_name_re = build_global_name_regex(learned_names)
    if global_name_re is None:
        print("WARNING: learned name regex is empty; proceeding without global-name redaction.")

    # PASS 2: de-id export
    out_rows = []
    total_wipes = 0
    notes_with_wipes = 0
    leak_examples = {}
    n_export = 0

    cols = list(df.columns)
    col_idx = {c: i for i, c in enumerate(cols)}
    values = df.itertuples(index=False, name=None)

    for i, row in enumerate(values):
        if max_rows is not None and i >= max_rows:
            break

        pid = row[col_idx[enc_id_col]]
        note_type = row[col_idx[type_col]] if type_col else ""
        raw = row[col_idx[text_col]]

        deid, wipes = deid_fulltext(raw, global_name_re, drop_dates=args.drop_dates)
        total_wipes += wipes
        if wipes > 0:
            notes_with_wipes += 1

        leaks = find_possible_name_leaks(deid, max_hits=10)
        if leaks:
            k = _safe_str(pid)
            leak_examples.setdefault(k, [])
            for x in leaks:
                if x not in leak_examples[k]:
                    leak_examples[k].append(x)
                if len(leak_examples[k]) >= 10:
                    break

        out_rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "NOTE_TYPE": note_type,
            "NOTE_TEXT_DEID": deid
        })
        n_export += 1

    out = pd.DataFrame(out_rows)
    out.to_csv(args.output, index=False, encoding="utf-8")

    with open(args.qa, "w") as f:
        f.write("CTXWIPE v4 QA report\n")
        f.write("====================\n\n")
        f.write("INPUT: {}\n".format(args.input))
        f.write("OUTPUT: {}\n".format(args.output))
        f.write("LEARNED_NAMES: {}\n\n".format(args.names_out))
        f.write("CTX_WORDS_BEFORE: {}\n".format(CTX_WORDS_BEFORE))
        f.write("CTX_WORDS_AFTER : {}\n".format(CTX_WORDS_AFTER))
        f.write("AGGRESSIVE_PAIR_WIPE: {}\n".format(AGGRESSIVE_PAIR_WIPE))
        f.write("DROP_SIGNATURE_BLOCK: {}\n".format(DROP_SIGNATURE_BLOCK))
        f.write("DROP_DATES: {}\n".format(bool(args.drop_dates)))
        f.write("MAX_ROWS: {}\n\n".format("ALL" if max_rows is None else max_rows))
        f.write("Rows exported: {}\n".format(len(out)))
        f.write("Pass1 learned strings: {}\n".format(learned_count))
        f.write("Notes with wipes: {}\n".format(notes_with_wipes))
        f.write("Total wipe segments applied: {}\n".format(total_wipes))

    with open(args.leaks, "w") as f:
        f.write("Possible leftover CapCap pairs after de-id (v4)\n")
        f.write("==============================================\n\n")
        f.write("These are NOT guaranteed to be names.\n")
        f.write("Use to extend PAIR_ALLOWLIST or tighten rules.\n\n")
        shown = 0
        for pid in leak_examples:
            f.write("ENCRYPTED_PAT_ID: {}\n".format(pid))
            f.write("  pairs: {}\n\n".format(", ".join(leak_examples[pid][:10])))
            shown += 1
            if shown >= 200:
                break

    print("Wrote:", args.output)
    print("Wrote:", args.qa)
    print("Wrote:", args.leaks)
    print("Wrote:", args.names_out)
    print("Rows exported:", len(out))
    print("Notes with wipes:", notes_with_wipes)
    print("Total wipe segments:", total_wipes)
    print("Done.")

if __name__ == "__main__":
    main()
