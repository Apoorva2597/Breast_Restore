#!/usr/bin/env python3
# build_stage12_WITH_AUDIT.py  (pandas compatible w/ older versions: no on_bad_lines, no encoding_errors)

import os
import re
import pandas as pd
from glob import glob

# ==============================
# CONFIG (ORIGINAL source files)
# ==============================

GOLD_PATH = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"

NOTES_V3_ORIG = "/home/apokol/Breast_Restore/_staging_inputs/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv"
NOTES_V4_ORIG = "/home/apokol/Breast_Restore/_staging_inputs/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv"

FALLBACK_GLOBS = [
    "/home/apokol/Breast_Restore/**/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv",
    "/home/apokol/Breast_Restore/**/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv",
]

OUTPUT_SUMMARY = "/home/apokol/Breast_Restore/_outputs/patient_stage_summary.csv"
OUTPUT_HITS    = "/home/apokol/Breast_Restore/_outputs/stage2_event_hits.csv"

# ==============================
# STAGE 2 DEFINITIONS
# ==============================

STAGE2_IMPLANT_PATTERNS = [
    r"exchange of tissue expanders? to (permanent )?implants?",
    r"exchange.*tissue expanders?.*implants?",
    r"expander.*exchange.*implants?",
    r"tissue expanders?.*removed.*implants?",
    r"\bimplant(s)? (are )?in\b",
    r"now with silicone implants?",
    r"\bs/p\b.*exchange.*(silicone|implant)",
]

NEGATION_PATTERNS = [
    r"\bno implants in log\b",
    r"\bimplant not placed\b",
]

CONTEXT_WINDOW = 250  # chars before/after match

IMPLANT_REGEX  = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in STAGE2_IMPLANT_PATTERNS]
NEGATION_REGEX = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in NEGATION_PATTERNS]

# ==============================
# HELPERS
# ==============================

def resolve_existing_path(primary_path, fallback_glob):
    if primary_path and os.path.exists(primary_path):
        return primary_path
    matches = glob(fallback_glob, recursive=True)
    return sorted(matches)[0] if matches else None

def read_notes_csv(path):
    """
    pandas<1.3 compatibility:
      - use error_bad_lines/warn_bad_lines instead of on_bad_lines
      - avoid encoding_errors
    """
    try:
        return pd.read_csv(
            path,
            dtype=str,
            engine="python",
            error_bad_lines=False,   # deprecated in newer pandas, but works in older
            warn_bad_lines=True      # deprecated in newer pandas, but works in older
        )
    except UnicodeDecodeError:
        # fallback if file contains non-utf8 bytes
        return pd.read_csv(
            path,
            dtype=str,
            engine="python",
            encoding="latin-1",
            error_bad_lines=False,
            warn_bad_lines=True
        )

def contains_negation(text):
    for rx in NEGATION_REGEX:
        if rx.search(text):
            return True
    return False

def find_stage2_hits(text):
    hits = []
    for rx in IMPLANT_REGEX:
        for m in rx.finditer(text):
            start = max(0, m.start() - CONTEXT_WINDOW)
            end   = min(len(text), m.end() + CONTEXT_WINDOW)
            snippet = text[start:end].replace("\n", " ")
            if not contains_negation(snippet):
                hits.append(snippet)
    return hits

def normalize_columns(df, want_map):
    """
    want_map: {CANONICAL: [possible_alts]}
    """
    cols = set(df.columns)
    rename = {}
    for canon, alts in want_map.items():
        if canon in cols:
            continue
        for a in alts:
            if a in cols:
                rename[a] = canon
                break
    return df.rename(columns=rename)

# ==============================
# RESOLVE INPUTS
# ==============================

notes_v3_path = resolve_existing_path(NOTES_V3_ORIG, FALLBACK_GLOBS[0])
notes_v4_path = resolve_existing_path(NOTES_V4_ORIG, FALLBACK_GLOBS[1])

if not notes_v3_path or not notes_v4_path:
    raise FileNotFoundError(
        "Could not find originals.\n"
        f"Checked:\n  {NOTES_V3_ORIG}\n  {NOTES_V4_ORIG}\n"
        f"Fallbacks:\n  {FALLBACK_GLOBS[0]}\n  {FALLBACK_GLOBS[1]}"
    )

# ==============================
# LOAD DATA
# ==============================

print("Loading gold...")
gold = pd.read_csv(GOLD_PATH, dtype=str)

print("Loading notes...")
notes_v3 = read_notes_csv(notes_v3_path)
notes_v4 = read_notes_csv(notes_v4_path)
notes = pd.concat([notes_v3, notes_v4], ignore_index=True)

notes = normalize_columns(notes, {
    "ENCRYPTED_PAT_ID": ["encrypted_pat_id", "PAT_ID", "PATID", "ENCRYPTED_ID"],
    "NOTE_TEXT_DEID":   ["note_text_deid", "NOTE_TEXT", "NOTE_TEXT_CLEAN", "TEXT"],
})

req = {"ENCRYPTED_PAT_ID", "NOTE_TEXT_DEID"}
missing = req.difference(set(notes.columns))
if missing:
    raise RuntimeError("Missing required columns in notes: %s" % sorted(missing))

notes["NOTE_TEXT_DEID"] = notes["NOTE_TEXT_DEID"].fillna("").astype(str)

# ==============================
# AGGREGATE NOTES BY PATIENT
# ==============================

print("Aggregating notes by patient...")
patient_text = (
    notes.groupby("ENCRYPTED_PAT_ID", dropna=False)["NOTE_TEXT_DEID"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)

# ==============================
# DETECT STAGE 2
# ==============================

print("Detecting Stage 2 events...")
summary_rows = []
hit_rows = []

for _, r in patient_text.iterrows():
    pid = r["ENCRYPTED_PAT_ID"]
    text = r["NOTE_TEXT_DEID"]

    hits = find_stage2_hits(text)
    has_stage2 = 1 if hits else 0

    summary_rows.append({
        "ENCRYPTED_PAT_ID": pid,
        "HAS_STAGE2": has_stage2,
        "HIT_COUNT": len(hits),
    })

    for h in hits:
        hit_rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "SNIPPET": h,
        })

summary_df = pd.DataFrame(summary_rows)
hits_df = pd.DataFrame(hit_rows)

# ==============================
# MERGE WITH GOLD + SAVE
# ==============================

gold = normalize_columns(gold, {
    "ENCRYPTED_PAT_ID": ["encrypted_pat_id", "PAT_ID", "PATID", "ENCRYPTED_ID"],
})

if "ENCRYPTED_PAT_ID" not in gold.columns:
    raise RuntimeError("Gold file missing ENCRYPTED_PAT_ID (or known alternate).")

print("Merging with gold...")
merged = gold.merge(summary_df, on="ENCRYPTED_PAT_ID", how="left")
merged["HAS_STAGE2"] = merged["HAS_STAGE2"].fillna(0).astype(int)
merged["HIT_COUNT"] = merged["HIT_COUNT"].fillna(0).astype(int)

print("Writing outputs...")
os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)
merged.to_csv(OUTPUT_SUMMARY, index=False)
hits_df.to_csv(OUTPUT_HITS, index=False)

print("Done.")
