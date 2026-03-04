#!/usr/bin/env python3
# bart_stage2_zeroshot_score_notes_OFFLINE.py
#
# Zero-shot Stage2 scoring using local offline BART-MNLI model dir.
# Mirrors your build_stage12 script for:
#  - input paths/globs
#  - MRN key normalization
#  - note text col detection
# Outputs are isolated under _outputs_bart/ to avoid confusion.

import os
import re
import json
import time
import pandas as pd
from glob import glob
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ==============================
# CONFIG (hardcoded, no args)
# ==============================

BASE_DIR = "/home/apokol/Breast_Restore"

MODEL_DIR = os.path.join(BASE_DIR, "bart_large_mnli")  # offline dir with pytorch_model.bin

ORIG_NOTE_PATHS = [
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Clinic Notes.csv"),
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Operation Notes.csv"),
]

ORIG_NOTE_GLOBS = [
    os.path.join(BASE_DIR, "**", "HPI11526*Notes.csv"),
    os.path.join(BASE_DIR, "**", "HPI11526*notes.csv"),
]

OUT_DIR = os.path.join(BASE_DIR, "_outputs_bart")
OUT_NOTE_SCORES = os.path.join(OUT_DIR, "bart_stage2_note_scores.csv")
OUT_META = os.path.join(OUT_DIR, "bart_stage2_run_metadata.json")

MERGE_KEY = "MRN"

HYPOTHESIS = (
    "This clinical note describes {}."
)
POS_LABEL = "a Stage 2 breast reconstruction surgery where a tissue expander was exchanged for a permanent implant"
NEG_LABEL = "a clinical note that does not describe Stage 2 expander-to-implant exchange surgery"

# Runtime controls
BATCH_SIZE = 8          # CPU batching
MAX_CHARS = 3500        # truncate note text (speed)
THRESHOLD = 0.50        # you can change later if desired


# ==============================
# HELPERS (borrowed style from your build script)
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
        raise RuntimeError("MRN column not found. Columns seen: %s" % list(df.columns)[:40])
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
    text_like = [c for c in df.columns if ("TEXT" in c.upper()) or ("NOTE" in c.upper())]
    if text_like:
        return text_like[0]
    raise RuntimeError("No obvious note text column found. Columns seen: %s" % list(df.columns)[:40])

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

def truncate_text(s):
    if s is None:
        return ""
    s = str(s)
    if s.lower() == "nan":
        s = ""
    if MAX_CHARS > 0 and len(s) > MAX_CHARS:
        return s[:MAX_CHARS]
    return s


# ==============================
# MAIN
# ==============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_DIR):
        raise RuntimeError("MODEL_DIR not found: %s" % MODEL_DIR)

    # Load notes (same as build script style)
    note_files = existing_files(ORIG_NOTE_PATHS, ORIG_NOTE_GLOBS)
    if not note_files:
        raise FileNotFoundError("No original HPI11526 * Notes.csv files found (paths + globs).")

    print("Loading ORIGINAL note files for BART scoring...")
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
        df["_NOTE_TYPE_"] = df[type_col] if type_col else ""
        df["_NOTE_DATE_"] = df[date_col] if date_col else ""

        note_dfs.append(
            df[[MERGE_KEY, text_col, "_SOURCE_FILE_", "_NOTE_TYPE_", "_NOTE_DATE_"]]
            .rename(columns={text_col: "NOTE_TEXT"})
        )

    notes = pd.concat(note_dfs, ignore_index=True)
    notes["NOTE_TEXT"] = notes["NOTE_TEXT"].apply(truncate_text)

    print("Initializing offline zero-shot pipeline (CPU)...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    clf = pipeline("zero-shot-classification", model=mdl, tokenizer=tok, device=-1)

    candidate_labels = [NEG_LABEL, POS_LABEL]  # we'll map by label name
    bs = max(1, int(BATCH_SIZE))

    print("Scoring %d notes (batch_size=%d)..." % (len(notes), bs))
    t0 = time.time()

    out_rows = []
    n = len(notes)

    for start in tqdm(range(0, n, bs), desc="BART zero-shot", unit="batch"):
        end = min(n, start + bs)
        batch = notes.iloc[start:end]

        texts = batch["NOTE_TEXT"].tolist()

        results = clf(
            sequences=texts,
            candidate_labels=candidate_labels,
            hypothesis_template=HYPOTHESIS
        )

        # results aligns with batch order
        for i, res in enumerate(results):
            r = batch.iloc[i]
            labels = res.get("labels", [])
            scores = res.get("scores", [])

            score_map = {}
            for lab, sc in zip(labels, scores):
                score_map[str(lab).strip().lower()] = float(sc)

            stage2_score = score_map.get(POS_LABEL.lower(), None)
            pred_is_stage2 = 1 if (stage2_score is not None and stage2_score >= THRESHOLD) else 0

            out_rows.append({
                "MRN": str(r[MERGE_KEY]).strip(),
                "NOTE_DATE": str(r["_NOTE_DATE_"]),
                "NOTE_TYPE": str(r["_NOTE_TYPE_"]),
                "SOURCE_FILE": str(r["_SOURCE_FILE_"]),
                "bart_stage2_score": stage2_score if stage2_score is not None else "",
                "bart_threshold": THRESHOLD,
                "bart_pred_is_stage2": pred_is_stage2,
                "bart_pred_label": POS_LABEL if pred_is_stage2 else NEG_LABEL,
                "text_chars_used": len(texts[i]) if texts[i] is not None else 0,
            })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_NOTE_SCORES, index=False)

    meta = {
        "model_dir": MODEL_DIR,
        "note_files_used": note_files,
        "n_notes_scored": int(len(notes)),
        "batch_size": bs,
        "max_chars": MAX_CHARS,
        "candidate_labels": candidate_labels,
        "hypothesis": HYPOTHESIS,
        "threshold": THRESHOLD,
        "output_note_scores_csv": OUT_NOTE_SCORES,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:", OUT_NOTE_SCORES)
    print("Saved:", OUT_META)
    print("Done.")


if __name__ == "__main__":
    main()
