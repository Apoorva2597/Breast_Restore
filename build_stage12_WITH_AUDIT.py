#!/usr/bin/env python3
# build_stage12_WITH_AUDIT.py
# Uses ORIGINAL full-note source files: "HPI11526 * Notes.csv" (not DEID_*).
# Merge key: MRN

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

STAGE2_IMPLANT_PATTERNS = [
    r"exchange of tissue expanders? to (permanent )?implants?",
    r"exchange.*tissue expanders?.*implants?",
    r"expander.*exchange.*implants?",
    r"tissue expanders?.*removed.*implants?",
    r"\bimplant(s)? (are )?in\b",
    r"now with silicone implants?",
    r"\bs/p\b.*exchange.*(silicone|implant)",
    r"\bimplant exchange\b",
]

NEGATION_PATTERNS = [
    r"\bno implants in log\b",
    r"\bimplant not placed\b",
    r"\bno implant\b",
]

CONTEXT_WINDOW = 250

IMPLANT_REGEX  = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in STAGE2_IMPLANT_PATTERNS]
NEGATION_REGEX = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in NEGATION_PATTERNS]

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
    # fallback: pick the widest-looking text column
    text_like = [c for c in df.columns if "TEXT" in c.upper() or "NOTE" in c.upper()]
    if text_like:
        return text_like[0]
    raise RuntimeError(f"No obvious note text column found. Columns seen: {list(df.columns)[:40]}")

def contains_negation(snippet: str) -> bool:
    return any(rx.search(snippet) for rx in NEGATION_REGEX)

def find_stage2_hits(text: str):
    hits = []
    for rx in IMPLANT_REGEX:
        for m in rx.finditer(text):
            start = max(0, m.start() - CONTEXT_WINDOW)
            end   = min(len(text), m.end() + CONTEXT_WINDOW)
            snippet = text[start:end].replace("\n", " ")
            if not contains_negation(snippet):
                hits.append(snippet)
    return hits

def existing_files(paths, globs_list):
    found = []
    for p in paths:
        if p and os.path.exists(p):
            found.append(p)
    if found:
        return sorted(set(found))

    # fallback glob search
    globbed = []
    for g in globs_list:
        globbed.extend(glob(g, recursive=True))
    globbed = sorted(set(globbed))
    return globbed

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
    df[text_col] = df[text_col].fillna("").astype(str)
    df["_NOTE_TEXT_COL_"] = text_col
    df["_SOURCE_FILE_"] = os.path.basename(fp)
    note_dfs.append(df[[MERGE_KEY, text_col, "_SOURCE_FILE_"]].rename(columns={text_col: "NOTE_TEXT"}))

notes = pd.concat(note_dfs, ignore_index=True)

print("Aggregating notes by MRN...")
patient_text = (
    notes.groupby(MERGE_KEY)["NOTE_TEXT"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)

print("Detecting Stage 2 events...")
summary_rows = []
hit_rows = []

for _, r in patient_text.iterrows():
    mrn = r[MERGE_KEY]
    text = r["NOTE_TEXT"]

    hits = find_stage2_hits(text)
    has_stage2 = 1 if hits else 0

    summary_rows.append({MERGE_KEY: mrn, "HAS_STAGE2": has_stage2, "HIT_COUNT": len(hits)})
    for h in hits:
        hit_rows.append({MERGE_KEY: mrn, "SNIPPET": h})

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
