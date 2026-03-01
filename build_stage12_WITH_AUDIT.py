#!/usr/bin/env python3
# build_stage12_WITH_AUDIT.py
# Uses ORIGINAL full-note source files: "HPI11526 * Notes.csv" (not DEID_*).
# Merge key: MRN
#
# Revision goals (TP-preserving, reduce FP):
# 1) Detect Stage2 per-NOTE (not on concatenated patient text), so we can require op/operative context.
# 2) Add "op-context gate": accept hits only if the note looks like an OP/operative note OR contains operative sections.
# 3) Add "history/plan gate": reject matches that are clearly historical ("s/p", "status post", "history of", "plan to", etc.)
# 4) Keep your paths/structure the same.

import os
import re
import pandas as pd
from glob import glob

# ==============================
# CONFIG
# ==============================

GOLD_PATH = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"

# Prefer explicit originals if you know them (add more if needed)
ORIG_NOTE_PATHS = [
    "/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Clinic Notes.csv",
    "/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Operation Notes.csv",
]

# Fallback search (broad) for any original "HPI11526 * Notes.csv" files
ORIG_NOTE_GLOBS = [
    "/home/apokol/Breast_Restore/**/HPI11526*Notes.csv",
    "/home/apokol/Breast_Restore/**/HPI11526*notes.csv",
]

OUTPUT_SUMMARY = "/home/apokol/Breast_Restore/_outputs/patient_stage_summary.csv"
OUTPUT_HITS    = "/home/apokol/Breast_Restore/_outputs/stage2_event_hits.csv"

MERGE_KEY = "MRN"

# ==============================
# STAGE 2 PATTERNS
# ==============================

# Keep your original patterns, but we will (a) score hits per note, and (b) gate by OP context.
STAGE2_IMPLANT_PATTERNS = [
    r"exchange of tissue expanders? to (permanent )?implants?",
    r"exchange.*tissue expanders?.*implants?",
    r"expander.*exchange.*implants?",
    r"tissue expanders?.*removed.*implants?",
    r"\bimplant exchange\b",
    # The next two are common FP sources in clinic notes; we keep them but only allow in OP context
    r"\bimplant(s)? (are )?in\b",
    r"now with silicone implants?",
    r"\bs/p\b.*exchange.*(silicone|implant)",
]

NEGATION_PATTERNS = [
    r"\bno implants in log\b",
    r"\bimplant not placed\b",
    r"\bno implant\b",
]

# New: "history/plan" cues that often drive FPs (especially in clinic notes / H&P)
HISTORY_PLAN_CUES = [
    r"\bhistory of\b",
    r"\bstatus post\b",
    r"\bs\/p\b",
    r"\bprior\b",
    r"\bprevious(ly)?\b",
    r"\bplan(s|ned)? to\b",
    r"\bwill (undergo|consider|proceed)\b",
    r"\brecommended\b",
    r"\bdiscussed\b",
    r"\bhere for (follow[- ]?up|post[- ]?op)\b",
]

# New: OP/operative note context signals
OP_NOTE_TYPE_CUES = [
    r"\bop note\b",
    r"\boperative\b",
    r"\boperation\b",
    r"\boperative report\b",
    r"\bbrief op\b",
    r"\bprocedure note\b",
    r"\bsurgery\b",
]

OP_SECTION_CUES = [
    r"\bprocedure\s*:",
    r"\boperation\s*:",
    r"\boperative\s+report\s*:",
    r"\bbrief\s+op\s*:",
    r"\bimplants?\s*:",
    r"\bhospital\s+course\s*:",
]

CONTEXT_WINDOW = 250
HISTORY_LEFT_WINDOW = 120  # look back this many chars before a match for history/plan cues

IMPLANT_REGEX  = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in STAGE2_IMPLANT_PATTERNS]
NEGATION_REGEX = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in NEGATION_PATTERNS]
HISTORY_REGEX  = [re.compile(p, re.IGNORECASE) for p in HISTORY_PLAN_CUES]
OPTYPE_REGEX   = [re.compile(p, re.IGNORECASE) for p in OP_NOTE_TYPE_CUES]
OPSEC_REGEX    = [re.compile(p, re.IGNORECASE) for p in OP_SECTION_CUES]

# ==============================
# HELPERS
# ==============================

def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Robust CSV read across older/newer pandas:
      - pandas>=1.3 supports on_bad_lines
      - older pandas uses error_bad_lines / warn_bad_lines
    """
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        # older pandas
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def normalize_mrn(df: pd.DataFrame) -> pd.DataFrame:
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MERGE_KEY:
                df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError(f"MRN column not found. Columns seen: {list(df.columns)[:40]}")
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df

def pick_note_text_col(df: pd.DataFrame) -> str:
    candidates = [
        "NOTE_TEXT", "NOTE_TEXT_FULL", "NOTE_TEXT_RAW", "NOTE", "TEXT",
        "NOTE_BODY", "NOTE_CONTENT", "FullText", "FULL_TEXT"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    text_like = [c for c in df.columns if "TEXT" in c.upper() or "NOTE" in c.upper()]
    if text_like:
        return text_like[0]
    raise RuntimeError(f"No obvious note text column found. Columns seen: {list(df.columns)[:40]}")

def pick_note_type_col(df: pd.DataFrame):
    candidates = ["NOTE_TYPE", "NOTE_TYPE_NAME", "TYPE", "NOTE_CATEGORY", "DOCUMENT_TYPE"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pick_note_date_col(df: pd.DataFrame):
    candidates = ["NOTE_DATETIME", "NOTE_DATE", "NOTE_DATE_RAW", "NOTE_DATETIME_RAW", "SERVICE_DATE", "DATE"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def contains_negation(snippet: str) -> bool:
    return any(rx.search(snippet) for rx in NEGATION_REGEX)

def has_op_context(note_type: str, source_file: str, text: str) -> bool:
    """
    Conservative gate to preserve TP:
      - If it's from "Operation Notes" file => OP context
      - OR note_type has operative cues
      - OR note has operative section headers
    """
    sf = (source_file or "").lower()
    nt = (note_type or "")
    if "operation" in sf or "operative" in sf or "op" in sf:
        # "Clinic Notes" will fail this, "Operation Notes" will pass
        if "clinic" not in sf:
            return True

    if any(rx.search(nt) for rx in OPTYPE_REGEX):
        return True

    if any(rx.search(text) for rx in OPSEC_REGEX):
        return True

    return False

def looks_like_history_or_plan(text: str, match_start: int) -> bool:
    """
    If the LEFT context (before the match) contains history/plan cues, treat as likely FP.
    """
    if not text:
        return False
    left = text[max(0, match_start - HISTORY_LEFT_WINDOW):match_start]
    return any(rx.search(left) for rx in HISTORY_REGEX)

def find_stage2_hits_in_note(text: str, note_type: str, source_file: str):
    """
    Returns list of snippets for Stage2 hits IN THIS NOTE that pass:
      - pattern match
      - no negation nearby
      - op-context gate
      - not clearly historical/plan mention (left-window cue)
    """
    hits = []
    if not text:
        return hits

    op_ok = has_op_context(note_type, source_file, text)
    if not op_ok:
        # key FP reducer: do not allow any Stage2 hits from non-op-context notes
        return hits

    for rx in IMPLANT_REGEX:
        for m in rx.finditer(text):
            # history/plan check
            if looks_like_history_or_plan(text, m.start()):
                continue

            start = max(0, m.start() - CONTEXT_WINDOW)
            end   = min(len(text), m.end() + CONTEXT_WINDOW)
            snippet = text[start:end].replace("\n", " ")
            if contains_negation(snippet):
                continue

            hits.append(snippet)

    return hits

def existing_files(paths, globs_list):
    found = []
    for p in paths:
        if p and os.path.exists(p):
            found.append(p)
    if found:
        return sorted(set(found))

    globbed = []
    for g in globs_list:
        globbed.extend(glob(g, recursive=True))
    return sorted(set(globbed))

# ==============================
# MAIN
# ==============================

print("Loading gold...")
gold = clean_cols(read_csv_robust(GOLD_PATH))
gold = normalize_mrn(gold)

note_files = existing_files(ORIG_NOTE_PATHS, ORIG_NOTE_GLOBS)
if not note_files:
    raise FileNotFoundError("No original HPI11526 * Notes.csv files found (checked explicit paths + globs).")

print("Loading ORIGINAL note files...")
note_dfs = []
for fp in note_files:
    df = clean_cols(read_csv_robust(fp))
    df = normalize_mrn(df)

    text_col = pick_note_text_col(df)
    type_col = pick_note_type_col(df)
    date_col = pick_note_date_col(df)

    df[text_col] = df[text_col].fillna("").astype(str)
    if type_col:
        df[type_col] = df[type_col].fillna("").astype(str)
    if date_col:
        df[date_col] = df[date_col].fillna("").astype(str)

    df["_SOURCE_FILE_"] = os.path.basename(fp)
    df["_NOTE_TYPE_"]   = df[type_col] if type_col else ""
    df["_NOTE_DATE_"]   = df[date_col] if date_col else ""

    note_dfs.append(
        df[[MERGE_KEY, text_col, "_SOURCE_FILE_", "_NOTE_TYPE_", "_NOTE_DATE_"]]
        .rename(columns={text_col: "NOTE_TEXT"})
    )

notes = pd.concat(note_dfs, ignore_index=True)

print("Detecting Stage 2 events (per-note, OP-context gated)...")
summary_rows = []
hit_rows = []

# Per-patient accumulators
hit_count_by_mrn = {}
has_stage2_by_mrn = {}

for _, r in notes.iterrows():
    mrn = r[MERGE_KEY]
    text = r["NOTE_TEXT"]
    source_file = r["_SOURCE_FILE_"]
    note_type = r["_NOTE_TYPE_"]
    note_date = r["_NOTE_DATE_"]

    hits = find_stage2_hits_in_note(text, note_type, source_file)

    if hits:
        has_stage2_by_mrn[mrn] = 1
        hit_count_by_mrn[mrn] = hit_count_by_mrn.get(mrn, 0) + len(hits)

        for h in hits:
            hit_rows.append({
                MERGE_KEY: mrn,
                "NOTE_DATE": note_date,
                "NOTE_TYPE": note_type,
                "SOURCE_FILE": source_file,
                "SNIPPET": h
            })
    else:
        # ensure patient appears even if no hits
        if mrn not in has_stage2_by_mrn:
            has_stage2_by_mrn[mrn] = 0
        if mrn not in hit_count_by_mrn:
            hit_count_by_mrn[mrn] = 0

# Build summary_df from per-note pass
for mrn, y in has_stage2_by_mrn.items():
    summary_rows.append({
        MERGE_KEY: mrn,
        "HAS_STAGE2": int(y),
        "HIT_COUNT": int(hit_count_by_mrn.get(mrn, 0))
    })

summary_df = pd.DataFrame(summary_rows)
hits_df = pd.DataFrame(hit_rows)

print("Merging with gold on MRN...")
merged = gold.merge(summary_df, on=MERGE_KEY, how="left")
merged["HAS_STAGE2"] = merged["HAS_STAGE2"].fillna(0).astype(int)
merged["HIT_COUNT"]  = merged["HIT_COUNT"].fillna(0).astype(int)

print("Writing outputs...")
os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)
merged.to_csv(OUTPUT_SUMMARY, index=False)
hits_df.to_csv(OUTPUT_HITS, index=False)

print("Done.")
