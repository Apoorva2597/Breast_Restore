#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_WITH_AUDIT.py  (Python 3.6.8 friendly)

PURPOSE
- Build Stage2 *staging* outputs only (NO validation).
- Rebuild patient_stage_summary.csv with the columns required by stage2_freeze_pack.py:
    STAGE2_DATE, STAGE2_NOTE_ID, STAGE2_NOTE_TYPE, STAGE2_MATCH_PATTERN, STAGE2_HITS
  (and also includes ENCRYPTED_PAT_ID, HAS_STAGE2 for convenience)

INPUTS (edit paths below if yours differ)
- ./_staging_inputs/HPI11526 Operation Notes.csv   (must include ENCRYPTED_PAT_ID and note text/snips)

OUTPUTS (names kept stable)
- ./_outputs/patient_stage_summary.csv
- ./_outputs/stage2_fn_raw_note_snippets.csv
- ./_outputs/stage2_fn_keyword_snippets_FINAL_FINAL.csv

NOTES
- Stage2 evidence is constrained toward intraoperative signals:
  EBL, drains, specimens, anesthesia context, explicit procedure language.
"""

from __future__ import print_function
import os
import re
import pandas as pd


# -------------------------
# IO helpers
# -------------------------

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))


def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df


def normalize_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit():
            return head
    return s


def safe_str(x):
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


# -------------------------
# Text assembly + snippet
# -------------------------

def build_note_text(row, text_col, snip_cols):
    parts = []
    if text_col:
        parts.append(safe_str(row.get(text_col, "")))
    for c in snip_cols:
        parts.append(safe_str(row.get(c, "")))
    txt = "\n".join([p for p in parts if p.strip()])
    return txt


def snippet_around(text, start, end, width=220):
    if not text:
        return ""
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    snip = text[lo:hi]
    snip = re.sub(r"\s+", " ", snip).strip()
    return snip


# -------------------------
# Stage2 logic (tight / intraop-biased)
# -------------------------

# Intraoperative / operative-context signals
INTRAOP_CONTEXT = re.compile(
    r"\b("
    r"operative\s+report|op\s+note|brief\s+op\s+note|procedure|procedures|"
    r"anesthesia|general\s+anesthesia|mac\s+anesthesia|lma|intubat|"
    r"estimated\s+blood\s+loss|ebl\b|"
    r"drain[s]?\b|jp\b|jackson[-\s]?pratt|blake\s+drain|"
    r"specimen[s]?\b|sent\s+to\s+pathology|pathology|"
    r"implant[s]?\b|"
    r"incision|closure|suture|"
    r"pre[-\s]?op|post[-\s]?op"
    r")\b",
    re.I
)

# Definitive stage2 procedure concepts (expander removal/exchange + implant)
# Keep patterns fairly strict to reduce clinic-note FP.
STAGE2_PATTERNS = [
    ("EXCHANGE_TIGHT",
     re.compile(
         r"\b("
         r"exchange\s+(?:of\s+)?(?:the\s+)?tissue\s+expander(?:s)?\s+(?:for|to)\s+(?:a\s+)?"
         r"(?:permanent\s+)?(?:silicone|saline)?\s*(?:breast\s+)?implant(?:s)?|"
         r"tissue\s+expander(?:s)?\s+exchange\s+(?:for|to)\s+(?:permanent\s+)?(?:breast\s+)?implant(?:s)?|"
         r"expander\s+exchange\s+(?:for|to)\s+(?:permanent\s+)?(?:breast\s+)?implant(?:s)?"
         r")\b",
         re.I
     )),
    ("REMOVE_PLUS_IMPLANT",
     re.compile(
         r"\b("
         r"remove(?:d|al)?\s+(?:of\s+)?(?:the\s+)?(?:tissue\s+)?expander(?:s)?\b.{0,80}\b"
         r"(?:place(?:d|ment)|insert(?:ed|ion)|exchange|implantation)\b.{0,60}\bimplant(?:s)?\b|"
         r"(?:tissue\s+)?expander(?:s)?\b.{0,60}\bremoved\b.{0,80}\bimplant(?:s)?\b.{0,40}\bplace(?:d|ment|)\b"
         r")\b",
         re.I | re.S
     )),
    ("IMPLANT_EXCHANGE_CONTEXT",
     re.compile(
         r"\b("
         r"implant\s+exchange\b|"
         r"exchange\s+(?:to|for)\s+(?:permanent\s+)?(?:breast\s+)?implant(?:s)?\b"
         r")\b",
         re.I
     )),
]

# If these appear without operative context, they are often clinic-followup mentions.
CLINIC_HEAVY_SIGNALS = re.compile(
    r"\b("
    r"return\s+to\s+clinic|follow[-\s]?up|post[-\s]?op\s+visit|"
    r"doing\s+well|incisions?\s+are\s+clean|jp\s+drains?\s+were\s+removed|"
    r"no\s+fever|no\s+chills|denies\s+fever|"
    r"plan\s+to\s+schedule|will\s+schedule|is\s+scheduled\s+to\s+have"
    r")\b",
    re.I
)


def find_stage2_hits(note_text):
    """
    Returns:
      - has_stage2 (0/1)
      - best_hit (dict) with keys: match_pattern, match_term, snippet
      - all_hits (list of dicts)
    """
    txt = note_text or ""
    if not txt.strip():
        return 0, None, []

    ctx_ok = True if INTRAOP_CONTEXT.search(txt) else False

    hits = []
    for name, pat in STAGE2_PATTERNS:
        for m in pat.finditer(txt):
            term = m.group(0)
            snip = snippet_around(txt, m.start(), m.end())
            hits.append({
                "match_pattern": name,
                "match_term": term,
                "snippet": snip,
                "start": m.start(),
                "end": m.end(),
                "ctx_ok": 1 if ctx_ok else 0
            })

    # Tighten: require intraop context for EXCHANGE_TIGHT and REMOVE_PLUS_IMPLANT
    # For IMPLANT_EXCHANGE_CONTEXT, require intraop context OR presence of hard operative tokens.
    filtered = []
    for h in hits:
        if h["match_pattern"] in ["EXCHANGE_TIGHT", "REMOVE_PLUS_IMPLANT"]:
            if h["ctx_ok"] == 1:
                filtered.append(h)
        else:
            if h["ctx_ok"] == 1:
                filtered.append(h)

    # If no filtered hits, no stage2
    if not filtered:
        return 0, None, []

    # Prefer hits with stronger patterns
    priority = {"REMOVE_PLUS_IMPLANT": 3, "EXCHANGE_TIGHT": 2, "IMPLANT_EXCHANGE_CONTEXT": 1}
    filtered.sort(key=lambda d: (priority.get(d["match_pattern"], 0), -(d["end"] - d["start"])), reverse=True)
    best = filtered[0]

    return 1, best, filtered


# -------------------------
# Main
# -------------------------

def main():
    root = os.path.abspath(".")
    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    INPUT_NOTES = os.path.join(root, "_staging_inputs", "HPI11526 Operation Notes.csv")

    if not os.path.isfile(INPUT_NOTES):
        raise IOError("Input not found: {}".format(INPUT_NOTES))

    df = normalize_cols(read_csv_robust(INPUT_NOTES, dtype=str, low_memory=False))

    # Required patient id
    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not enc_col:
        raise ValueError("Input must include ENCRYPTED_PAT_ID. Found: {}".format(list(df.columns)))
    if enc_col != "ENCRYPTED_PAT_ID":
        df = df.rename(columns={enc_col: "ENCRYPTED_PAT_ID"})
    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)

    # Note id / date / type (best-effort; freeze pack needs STAGE2_* columns, not necessarily these inputs)
    note_id_col = pick_first_existing(df, ["NOTE_ID", "NOTEID", "NOTEID_NUM", "NOTE_ID_NUM", "DOCUMENT_ID"])
    date_col = pick_first_existing(df, ["NOTE_DATE", "SERVICE_DATE", "DATE", "VISIT_DATE", "DOC_DATE"])
    note_type_col = pick_first_existing(df, ["NOTE_TYPE", "DOC_TYPE", "DOCUMENT_TYPE", "TYPE"])

    # Text columns: either a full note or multiple snips
    text_col = pick_first_existing(df, ["NOTE_TEXT", "TEXT", "NOTE", "NOTE_BODY", "BODY"])
    snip_cols = [c for c in df.columns if re.match(r"SNIP_\d+", c)]
    # Also allow old-style snippet columns
    snip_cols = snip_cols if snip_cols else [c for c in ["SNIP_01", "SNIP_02", "SNIP_03", "SNIP_04", "SNIP_05"] if c in df.columns]

    if not text_col and not snip_cols:
        raise ValueError("No note text found. Need NOTE_TEXT/TEXT or SNIP_01..SNIP_05. Found: {}".format(list(df.columns)))

    # Source file (optional)
    source_file_col = pick_first_existing(df, ["SOURCE_FILE", "FILE", "FILENAME", "SOURCE"])
    default_source = os.path.basename(INPUT_NOTES)

    # Build per-row notes
    staged_rows = []
    audit_rows = []
    raw_rows = []

    total_events = 0
    for i in range(len(df)):
        r = df.iloc[i]
        enc = normalize_id(r.get("ENCRYPTED_PAT_ID", ""))

        if not enc:
            continue

        note_id = safe_str(r.get(note_id_col, "")) if note_id_col else ""
        note_date = safe_str(r.get(date_col, "")) if date_col else ""
        note_type = safe_str(r.get(note_type_col, "")) if note_type_col else ""

        note_text = build_note_text(r, text_col, snip_cols)

        has_stage2, best, all_hits = find_stage2_hits(note_text)
        total_events += 1

        # If positive, capture the "best" row for patient-level collapse
        if has_stage2 == 1 and best:
            staged_rows.append({
                "ENCRYPTED_PAT_ID": enc,
                "STAGE2_DATE": note_date,
                "STAGE2_NOTE_ID": note_id,
                "STAGE2_NOTE_TYPE": note_type,
                "STAGE2_MATCH_PATTERN": best["match_pattern"],
                "STAGE2_HITS": best["match_term"],
                "HAS_STAGE2": 1
            })

            # Audit: one row per hit (keyword snippets)
            src = safe_str(r.get(source_file_col, default_source)) if source_file_col else default_source
            for h in all_hits:
                audit_rows.append({
                    "ENCRYPTED_PAT_ID": enc,
                    "NOTE_ID": note_id,
                    "MATCH_TERM": h["match_term"],
                    "SNIPPET": h["snippet"],
                    "SOURCE_FILE": src
                })
        else:
            # Raw note snippets bucket (helpful for FN review / why not detected)
            src = safe_str(r.get(source_file_col, default_source)) if source_file_col else default_source
            raw_rows.append({
                "ENCRYPTED_PAT_ID": enc,
                "NOTE_ID": note_id,
                "NOTE_TYPE": note_type,
                "NOTE_DATE": note_date,
                "SNIPPET": re.sub(r"\s+", " ", (note_text[:800] if note_text else "")).strip(),
                "SOURCE_FILE": src
            })

    # Collapse to one row per patient (max HAS_STAGE2; take earliest STAGE2_DATE where available)
    if staged_rows:
        pos = pd.DataFrame(staged_rows)
        # normalize empties
        for c in ["STAGE2_DATE", "STAGE2_NOTE_ID", "STAGE2_NOTE_TYPE", "STAGE2_MATCH_PATTERN", "STAGE2_HITS"]:
            pos[c] = pos[c].fillna("").map(lambda x: safe_str(x).strip())

        # Sort by date text (best-effort). If dates are ISO, this works; otherwise it still gives stable behavior.
        pos_sorted = pos.sort_values(by=["ENCRYPTED_PAT_ID", "STAGE2_DATE"], ascending=[True, True])
        # keep first positive per patient
        pos_first = pos_sorted.groupby("ENCRYPTED_PAT_ID", as_index=False).first()
        pos_first["HAS_STAGE2"] = 1
    else:
        pos_first = pd.DataFrame(columns=[
            "ENCRYPTED_PAT_ID", "STAGE2_DATE", "STAGE2_NOTE_ID", "STAGE2_NOTE_TYPE",
            "STAGE2_MATCH_PATTERN", "STAGE2_HITS", "HAS_STAGE2"
        ])

    # Ensure we output ALL patients seen (including negatives) with required columns present
    all_pats = df[["ENCRYPTED_PAT_ID"]].copy()
    all_pats["ENCRYPTED_PAT_ID"] = all_pats["ENCRYPTED_PAT_ID"].map(normalize_id)
    all_pats = all_pats[all_pats["ENCRYPTED_PAT_ID"] != ""].drop_duplicates()

    out = all_pats.merge(pos_first, on="ENCRYPTED_PAT_ID", how="left")

    # Fill negatives with empty Stage2 fields + HAS_STAGE2=0
    for c in ["STAGE2_DATE", "STAGE2_NOTE_ID", "STAGE2_NOTE_TYPE", "STAGE2_MATCH_PATTERN", "STAGE2_HITS"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("")
    out["HAS_STAGE2"] = out["HAS_STAGE2"].fillna(0).astype(int)

    # Write outputs (stable names)
    summary_path = os.path.join(out_dir, "patient_stage_summary.csv")
    raw_path = os.path.join(out_dir, "stage2_fn_raw_note_snippets.csv")
    audit_path = os.path.join(out_dir, "stage2_fn_keyword_snippets_FINAL_FINAL.csv")

    out.to_csv(summary_path, index=False)

    pd.DataFrame(raw_rows, columns=[
        "ENCRYPTED_PAT_ID", "NOTE_ID", "NOTE_TYPE", "NOTE_DATE", "SNIPPET", "SOURCE_FILE"
    ]).to_csv(raw_path, index=False)

    pd.DataFrame(audit_rows, columns=[
        "ENCRYPTED_PAT_ID", "NOTE_ID", "MATCH_TERM", "SNIPPET", "SOURCE_FILE"
    ]).to_csv(audit_path, index=False)

    print("Staging complete.")
    print("Patients:", int(out["ENCRYPTED_PAT_ID"].nunique()))
    print("Events:", int(total_events))
    print("")
    print("Wrote:")
    print("  ", summary_path)
    print("  ", raw_path)
    print("  ", audit_path)


if __name__ == "__main__":
    main()
