#!/usr/bin/env python3
# build_stage12_WITH_AUDIT.py
# Uses ORIGINAL full-note source files: "HPI11526 * Notes.csv" (not DEID_*).
# Merge key: MRN
#
# Revision (balanced): preserve TP/recall by allowing STRONG Stage2 phrases anywhere,
# while OP-gating only WEAK/ambiguous implant mentions to reduce FP.

import os
import re
import pandas as pd
from glob import glob

# ==============================
# CONFIG
# ==============================

GOLD_PATH = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"

ORIG_NOTE_PATHS = [
    "/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Clinic Notes.csv",
    "/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Operation Notes.csv",
]

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

# STRONG = usually true Stage2 when present (keep recall)
STRONG_STAGE2_PATTERNS = [
    r"exchange of tissue expanders? to (permanent )?implants?",
    r"exchange.*tissue expanders?.*implants?",
    r"expander.*exchange.*implants?",
    r"tissue expanders?.*removed.*implants?",
    r"\bimplant exchange\b",
    r"\b(expander|tissue expander).{0,40}\b(remov(ed|al)|exchang(e|ed))\b.{0,80}\bimplant",
]

# WEAK = easy FP in clinic/H&P (reduce FP without killing TP by OP-gating)
WEAK_STAGE2_PATTERNS = [
    r"\bimplant(s)? (are )?in\b",
    r"now with silicone implants?",
    r"\bs/p\b.*exchange.*(silicone|implant)",
]

NEGATION_PATTERNS = [
    r"\bno implants in log\b",
    r"\bimplant not placed\b",
    r"\bno implant\b",
]

# History/plan cues (we apply mainly to WEAK hits, or STRONG hits only if no procedure verbs nearby)
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
]

# OP/operative note context signals (used only to “unlock” WEAK hits)
OP_NOTE_TYPE_CUES = [
    r"\bop note\b",
    r"\boperative\b",
    r"\boperation\b",
    r"\boperative report\b",
    r"\bbrief op\b",
    r"\bprocedure note\b",
]

OP_SECTION_CUES = [
    r"\bprocedure\s*:",
    r"\boperation\s*:",
    r"\boperative\s+report\s*:",
    r"\bbrief\s+op\s*:",
    r"\bimplants?\s*:",
    r"\bhospital\s+course\s*:",
]

# procedure verbs that indicate an actual event (helps rescue recall)
PROCEDURE_VERB_CUES = [
    r"\bexchang(e|ed|ing)\b",
    r"\bremov(ed|al|ing)\b",
    r"\bplaced\b",
    r"\binsertion\b",
    r"\bimplantation\b",
]

CONTEXT_WINDOW = 250
LEFT_WINDOW = 120
AROUND_WINDOW = 180

STRONG_REGEX   = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in STRONG_STAGE2_PATTERNS]
WEAK_REGEX     = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in WEAK_STAGE2_PATTERNS]
NEGATION_REGEX = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in NEGATION_PATTERNS]
HISTORY_REGEX  = [re.compile(p, re.IGNORECASE) for p in HISTORY_PLAN_CUES]
OPTYPE_REGEX   = [re.compile(p, re.IGNORECASE) for p in OP_NOTE_TYPE_CUES]
OPSEC_REGEX    = [re.compile(p, re.IGNORECASE) for p in OP_SECTION_CUES]
PROCVERB_REGEX = [re.compile(p, re.IGNORECASE) for p in PROCEDURE_VERB_CUES]

# ==============================
# HELPERS
# ==============================

def read_csv_robust(path: str) -> pd.DataFrame:
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
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
    for c in ["NOTE_TYPE", "NOTE_TYPE_NAME", "TYPE", "NOTE_CATEGORY", "DOCUMENT_TYPE"]:
        if c in df.columns:
            return c
    return None

def pick_note_date_col(df: pd.DataFrame):
    for c in ["NOTE_DATETIME", "NOTE_DATE", "NOTE_DATE_RAW", "NOTE_DATETIME_RAW", "SERVICE_DATE", "DATE"]:
        if c in df.columns:
            return c
    return None

def contains_negation(snippet: str) -> bool:
    return any(rx.search(snippet) for rx in NEGATION_REGEX)

def left_has_history_cue(text: str, match_start: int) -> bool:
    left = text[max(0, match_start - LEFT_WINDOW):match_start]
    return any(rx.search(left) for rx in HISTORY_REGEX)

def around_has_procedure_verb(text: str, match_start: int, match_end: int) -> bool:
    around = text[max(0, match_start - AROUND_WINDOW):min(len(text), match_end + AROUND_WINDOW)]
    return any(rx.search(around) for rx in PROCVERB_REGEX)

def has_op_context(note_type: str, source_file: str, text: str) -> bool:
    sf = (source_file or "").lower()
    nt = (note_type or "")

    # operation file name is a strong signal
    if "operation" in sf or "operative" in sf:
        return True

    # note type cue
    if any(rx.search(nt) for rx in OPTYPE_REGEX):
        return True

    # section header cue
    if any(rx.search(text) for rx in OPSEC_REGEX):
        return True

    return False

def extract_hits(text: str, regex_list, label: str, note_type: str, source_file: str, gate_weak: bool):
    """
    gate_weak=True means:
      - require OP-context for WEAK patterns
      - apply history/plan filter strongly
    gate_weak=False means:
      - allow STRONG patterns without OP-context
      - history/plan filter only blocks if NO procedure verb nearby
    """
    hits = []
    if not text:
        return hits

    op_ok = has_op_context(note_type, source_file, text)

    for rx in regex_list:
        for m in rx.finditer(text):
            # build snippet
            start = max(0, m.start() - CONTEXT_WINDOW)
            end   = min(len(text), m.end() + CONTEXT_WINDOW)
            snippet = text[start:end].replace("\n", " ")

            # negation always blocks
            if contains_negation(snippet):
                continue

            # gating rules
            if gate_weak:
                if not op_ok:
                    continue
                # if it's WEAK and has history cues, drop it
                if left_has_history_cue(text, m.start()):
                    continue
            else:
                # STRONG: only drop on history cue if it looks like background (no procedure verbs nearby)
                if left_has_history_cue(text, m.start()) and not around_has_procedure_verb(text, m.start(), m.end()) and not op_ok:
                    continue

            hits.append((label, snippet))

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

print("Detecting Stage 2 events (balanced STRONG+WEAK logic)...")
summary_rows = []
hit_rows = []

hit_count_by_mrn = {}
has_stage2_by_mrn = {}

for _, r in notes.iterrows():
    mrn = r[MERGE_KEY]
    text = r["NOTE_TEXT"]
    source_file = r["_SOURCE_FILE_"]
    note_type = r["_NOTE_TYPE_"]
    note_date = r["_NOTE_DATE_"]

    strong_hits = extract_hits(text, STRONG_REGEX, "STRONG", note_type, source_file, gate_weak=False)
    weak_hits   = extract_hits(text, WEAK_REGEX,   "WEAK",   note_type, source_file, gate_weak=True)

    hits = strong_hits + weak_hits

    if hits:
        has_stage2_by_mrn[mrn] = 1
        hit_count_by_mrn[mrn] = hit_count_by_mrn.get(mrn, 0) + len(hits)

        for label, snippet in hits:
            hit_rows.append({
                MERGE_KEY: mrn,
                "NOTE_DATE": note_date,
                "NOTE_TYPE": note_type,
                "SOURCE_FILE": source_file,
                "HIT_STRENGTH": label,
                "SNIPPET": snippet
            })
    else:
        if mrn not in has_stage2_by_mrn:
            has_stage2_by_mrn[mrn] = 0
        if mrn not in hit_count_by_mrn:
            hit_count_by_mrn[mrn] = 0

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
