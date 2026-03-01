#!/usr/bin/env python3
# build_stage12_WITH_AUDIT.py
# Uses MRN as the merge key

import os
import re
import pandas as pd
from glob import glob

# ==============================
# CONFIG
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

MERGE_KEY = "MRN"

# ==============================
# STAGE 2 PATTERNS
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

CONTEXT_WINDOW = 250

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

def read_csv_robust(path):
    try:
        df = pd.read_csv(path, dtype=str, engine="python",
                         error_bad_lines=False, warn_bad_lines=True)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, engine="python",
                         encoding="latin-1",
                         error_bad_lines=False, warn_bad_lines=True)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def normalize_key(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN"]
    for k in key_variants:
        if k in df.columns:
            df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError("MRN column not found in file.")
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df

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

# ==============================
# RESOLVE FILES
# ==============================

notes_v3_path = resolve_existing_path(NOTES_V3_ORIG, FALLBACK_GLOBS[0])
notes_v4_path = resolve_existing_path(NOTES_V4_ORIG, FALLBACK_GLOBS[1])

if not notes_v3_path or not notes_v4_path:
    raise FileNotFoundError("Original notes files not found.")

# ==============================
# LOAD DATA
# ==============================

print("Loading gold...")
gold = read_csv_robust(GOLD_PATH)
gold = normalize_key(gold)

print("Loading notes...")
notes_v3 = read_csv_robust(notes_v3_path)
notes_v4 = read_csv_robust(notes_v4_path)
notes = pd.concat([notes_v3, notes_v4], ignore_index=True)

notes = normalize_key(notes)

# Find note text column
note_text_col = None
for c in ["NOTE_TEXT_DEID", "NOTE_TEXT", "TEXT"]:
    if c in notes.columns:
        note_text_col = c
        break

if not note_text_col:
    raise RuntimeError("No note text column found.")

notes[note_text_col] = notes[note_text_col].fillna("").astype(str)

# ==============================
# AGGREGATE NOTES BY MRN
# ==============================

print("Aggregating notes by MRN...")
patient_text = (
    notes.groupby(MERGE_KEY)[note_text_col]
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
    mrn = r[MERGE_KEY]
    text = r[note_text_col]

    hits = find_stage2_hits(text)
    has_stage2 = 1 if hits else 0

    summary_rows.append({
        MERGE_KEY: mrn,
        "HAS_STAGE2": has_stage2,
        "HIT_COUNT": len(hits),
    })

    for h in hits:
        hit_rows.append({
            MERGE_KEY: mrn,
            "SNIPPET": h,
        })

summary_df = pd.DataFrame(summary_rows)
hits_df = pd.DataFrame(hit_rows)

# ==============================
# MERGE + SAVE
# ==============================

print("Merging with gold on MRN...")
merged = gold.merge(summary_df, on=MERGE_KEY, how="left")
merged["HAS_STAGE2"] = merged["HAS_STAGE2"].fillna(0).astype(int)
merged["HIT_COUNT"]  = merged["HIT_COUNT"].fillna(0).astype(int)

print("Writing outputs...")
os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)
merged.to_csv(OUTPUT_SUMMARY, index=False)
hits_df.to_csv(OUTPUT_HITS, index=False)

print("Done.")
