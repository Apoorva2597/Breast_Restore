#!/usr/bin/env python3

import pandas as pd
import re
from pathlib import Path

# ==============================
# CONFIG
# ==============================

GOLD_PATH = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
NOTES_V3_PATH = "/home/apokol/Breast_Restore/_staging_inputs/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv"
NOTES_V4_PATH = "/home/apokol/Breast_Restore/_staging_inputs/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv"

OUTPUT_SUMMARY = "/home/apokol/Breast_Restore/_outputs/patient_stage_summary.csv"
OUTPUT_HITS = "/home/apokol/Breast_Restore/_outputs/stage2_event_hits.csv"

# ==============================
# STAGE 2 DEFINITIONS
# ==============================

STAGE2_IMPLANT_PATTERNS = [
    r"exchange.*tissue expander.*implant",
    r"expander.*exchange.*implant",
    r"exchange to (silicone|implant)",
    r"silicone implants? (are )?in",
    r"s/p.*exchange.*implant",
    r"tissue expander.*removed.*implant",
]

NEGATION_PATTERNS = [
    r"no implants in log",
    r"no implant",
    r"implant not placed",
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

# ==============================
# LOAD DATA
# ==============================

print("Loading gold...")
gold = pd.read_csv(GOLD_PATH)

print("Loading notes...")
notes_v3 = pd.read_csv(NOTES_V3_PATH)
notes_v4 = pd.read_csv(NOTES_V4_PATH)

notes = pd.concat([notes_v3, notes_v4], ignore_index=True)

# Ensure required columns exist
required_cols = {"ENCRYPTED_PAT_ID", "NOTE_TEXT_DEID"}
if not required_cols.issubset(notes.columns):
    raise RuntimeError("Missing required columns in notes file.")

# ==============================
# AGGREGATE NOTES BY PATIENT
# ==============================

print("Aggregating notes by patient...")
notes_grouped = notes.groupby("ENCRYPTED_PAT_ID")["NOTE_TEXT_DEID"].apply(
    lambda x: " ".join(x.astype(str))
).reset_index()

# ==============================
# DETECT STAGE 2
# ==============================

print("Detecting Stage 2 events...")
summary_rows = []
hit_rows = []

for _, row in notes_grouped.iterrows():
    pid = row["ENCRYPTED_PAT_ID"]
    text = row["NOTE_TEXT_DEID"]

    hits = find_stage2_hits(text)

    has_stage2 = 1 if len(hits) > 0 else 0

    summary_rows.append({
        "ENCRYPTED_PAT_ID": pid,
        "HAS_STAGE2": has_stage2,
        "HIT_COUNT": len(hits)
    })

    for h in hits:
        hit_rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "SNIPPET": h
        })

summary_df = pd.DataFrame(summary_rows)
hits_df = pd.DataFrame(hit_rows)

# ==============================
# MERGE WITH GOLD
# ==============================

print("Merging with gold...")
merged = gold.merge(summary_df, on="ENCRYPTED_PAT_ID", how="left")
merged["HAS_STAGE2"] = merged["HAS_STAGE2"].fillna(0).astype(int)

# ==============================
# SAVE OUTPUTS
# ==============================

print("Writing outputs...")
merged.to_csv(OUTPUT_SUMMARY, index=False)
hits_df.to_csv(OUTPUT_HITS, index=False)

print("Done.")
