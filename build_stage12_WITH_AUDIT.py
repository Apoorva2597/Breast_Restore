#!/usr/bin/env python3

import os
import re
import pandas as pd
from glob import glob

# ==============================
# CONFIG (uses ORIGINAL source files)
# ==============================

GOLD_PATH = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"

# Your originals (from your logs/screenshots)
NOTES_V3_ORIG = "/home/apokol/Breast_Restore/_staging_inputs/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv"
NOTES_V4_ORIG = "/home/apokol/Breast_Restore/_staging_inputs/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv"

# Fallback: if paths differ, auto-find any matching originals under Breast_Restore
FALLBACK_GLOBS = [
    "/home/apokol/Breast_Restore/**/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv",
    "/home/apokol/Breast_Restore/**/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv",
]

OUTPUT_SUMMARY = "/home/apokol/Breast_Restore/_outputs/patient_stage_summary.csv"
OUTPUT_HITS = "/home/apokol/Breast_Restore/_outputs/stage2_event_hits.csv"

# ==============================
# STAGE 2 DEFINITIONS
# ==============================

STAGE2_IMPLANT_PATTERNS = [
    r"exchange of tissue expanders? to (permanent )?implants?",
    r"exchange.*tissue expanders?.*implants?",
    r"expander.*exchange.*implants?",
    r"tissue expanders?.*removed.*implants?",
    r"implant(s)? (are )?in\b",
    r"now with silicone implants?",
    r"s/p.*exchange.*(silicone|implant)",
]

NEGATION_PATTERNS = [
    r"\bno implants in log\b",
    r"\bno implant(s)?\b",
    r"\bimplant not placed\b",
]

CONTEXT_WINDOW = 250  # characters before/after match


# ==============================
# HELPERS
# ==============================

def compile_patterns(pattern_list):
    return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in pattern_list]

IMPLANT_REGEX = compile_patterns(STAGE2_IMPLANT_PATTERNS)
NEGATION_REGEX = compile_patterns(NEGATION_PATTERNS)

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
            end = min(len(text), m.end() + CONTEXT_WINDOW)
            snippet = text[start:end].replace("\n", " ")
            if not contains_negation(snippet):
                hits.append(snippet)
    return hits

def resolve_existing_path(p, fallback_globs):
    if p and os.path.exists(p):
        return p
    for g in fallback_globs:
        matches = glob(g, recursive=True)
        if matches:
            # pick first deterministic
            return sorted(matches)[0]
    return None

def read_notes_csv(path):
    # robust-ish read for messy CSVs
    return pd.read_csv(
        path,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
        encoding_errors="replace"
    )

# ==============================
# RESOLVE INPUTS
# ==============================

notes_v3_path = resolve_existing_path(NOTES_V3_ORIG, [FALLBACK_GLOBS[0]])
notes_v4_path = resolve_existing_path(NOTES_V4_ORIG, [FALLBACK_GLOBS[1]])

if not notes_v3_path or not notes_v4_path:
    raise FileNotFoundError(
        f"Could not find originals.\n"
        f"Checked:\n  {NOTES_V3_ORIG}\n  {NOTES_V4_ORIG}\n"
        f"And fallbacks:\n  {FALLBACK_GLOBS[0]}\n  {FALLBACK_GLOBS[1]}"
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

# Normalize/resolve expected columns
# Your snippets show ENCRYPTED_PAT_ID; handle common alternates just in case.
col_map = {}
if "ENCRYPTED_PAT_ID" not in notes.columns:
    for alt in ["encrypted_pat_id", "PAT_ID", "ENCRYPTED_ID", "PATID"]:
        if alt in notes.columns:
            col_map[alt] = "ENCRYPTED_PAT_ID"
            break

if "NOTE_TEXT_DEID" not in notes.columns:
    for alt in ["NOTE_TEXT", "NOTE_TEXT_CLEAN", "TEXT", "note_text_deid"]:
        if alt in notes.columns:
            col_map[alt] = "NOTE_TEXT_DEID"
            break

if col_map:
    notes = notes.rename(columns=col_map)

required = {"ENCRYPTED_PAT_ID", "NOTE_TEXT_DEID"}
missing = required.difference(set(notes.columns))
if missing:
    raise RuntimeError(f"Missing required columns in notes: {sorted(missing)}")

# ==============================
# AGGREGATE NOTES BY PATIENT
# ==============================

print("Aggregating notes by patient...")
notes["NOTE_TEXT_DEID"] = notes["NOTE_TEXT_DEID"].fillna("").astype(str)
notes_grouped = notes.groupby("ENCRYPTED_PAT_ID", dropna=False)["NOTE_TEXT_DEID"].apply(
    lambda x: " ".join(x)
).reset_index()

# ==============================
# DETECT STAGE 2
# ==============================

print("Detecting Stage 2 events...")
summary_rows = []
hit_rows = []

for _, r in notes_grouped.iterrows():
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

print("Merging with gold...")
if "ENCRYPTED_PAT_ID" not in gold.columns:
    # try common alternate in gold
    for alt in ["encrypted_pat_id", "PAT_ID", "PATID"]:
        if alt in gold.columns:
            gold = gold.rename(columns={alt: "ENCRYPTED_PAT_ID"})
            break

if "ENCRYPTED_PAT_ID" not in gold.columns:
    raise RuntimeError("gold file missing ENCRYPTED_PAT_ID (or known alternate).")

merged = gold.merge(summary_df, on="ENCRYPTED_PAT_ID", how="left")
merged["HAS_STAGE2"] = merged["HAS_STAGE2"].fillna(0).astype(int)
merged["HIT_COUNT"] = merged["HIT_COUNT"].fillna(0).astype(int)

print("Writing outputs...")
os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)
merged.to_csv(OUTPUT_SUMMARY, index=False)
hits_df.to_csv(OUTPUT_HITS, index=False)

print("Done.")
