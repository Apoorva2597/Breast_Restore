#!/usr/bin/env python3
# bart_stage2_fast_verifier_resume.py
#
# FAST BART verifier:
#   - Reads rule hits only: /home/apokol/Breast_Restore/_outputs/stage2_event_hits.csv
#   - Scores SNIPPET (short) instead of full note text (fast)
#   - Resume-safe: appends to output CSV and skips already-processed rows
#   - Produces patient-level aggregation CSV
#
# Python 3.6.8 compatible (forces slow tokenizer).

import os
import csv
import json
import time
import hashlib
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ==============================
# CONFIG (hardcoded)
# ==============================

BASE_DIR = "/home/apokol/Breast_Restore"

IN_HITS = os.path.join(BASE_DIR, "_outputs", "stage2_event_hits.csv")

MODEL_DIR = os.path.join(BASE_DIR, "bart_large_mnli")  # offline model directory

OUT_DIR = os.path.join(BASE_DIR, "_outputs_bart")
OUT_HIT_SCORES = os.path.join(OUT_DIR, "bart_stage2_hit_verifier_scores.csv")
OUT_PATIENT = os.path.join(OUT_DIR, "bart_stage2_hit_verifier_patient_summary.csv")
OUT_META = os.path.join(OUT_DIR, "bart_stage2_hit_verifier_run_metadata.json")

# BART zero-shot settings
# IMPORTANT: hypothesis_template MUST contain {}.
HYPOTHESIS_TEMPLATE = "This clinical note describes {}."

POS_LABEL = "a Stage 2 breast reconstruction surgery where a tissue expander was exchanged for a permanent implant"
NEG_LABEL = "a note that does not describe Stage 2 expander-to-implant exchange surgery"

THRESHOLD = 0.50

# Speed controls
BATCH_SIZE = 16       # increase if CPU can handle; 16 usually ok for snippets
MAX_CHARS = 800       # snippets are already short; keep bounded for speed

# Columns expected in stage2_event_hits.csv from your rule script
COL_MRN = "MRN"
COL_SNIPPET = "SNIPPET"
COL_NOTE_DATE = "NOTE_DATE"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_SOURCE_FILE = "SOURCE_FILE"
COL_STRENGTH = "HIT_STRENGTH"

# ==============================
# Helpers
# ==============================

def _safe_str(x):
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s

def _truncate(s, n):
    s = _safe_str(s)
    if n > 0 and len(s) > n:
        return s[:n]
    return s

def make_row_id(mrn, note_date, source_file, strength, snippet):
    """
    Stable identifier for resume-skip.
    Uses a hash of snippet so we don't store huge ids.
    """
    base = "|".join([
        _safe_str(mrn).strip(),
        _safe_str(note_date).strip(),
        _safe_str(source_file).strip(),
        _safe_str(strength).strip(),
        hashlib.sha1(_safe_str(snippet).encode("utf-8", errors="ignore")).hexdigest()
    ])
    return base

def load_done_ids(path):
    done = set()
    if not os.path.exists(path):
        return done
    # read only row_id column from existing output
    try:
        for chunk in pd.read_csv(path, usecols=["row_id"], chunksize=50000, low_memory=False):
            for x in chunk["row_id"].astype(str).tolist():
                done.add(x)
    except Exception:
        # if file exists but is malformed, don't crash; just don't skip
        return set()
    return done

def append_rows_csv(path, rows, header_cols):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_cols)
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def build_patient_summary(hit_scores_csv, out_patient_csv):
    df = pd.read_csv(hit_scores_csv, low_memory=False)
    if COL_MRN not in df.columns:
        raise RuntimeError("MRN missing in hit scores CSV.")

    df["bart_stage2_score"] = pd.to_numeric(df["bart_stage2_score"], errors="coerce")
    df["bart_pred_is_stage2"] = pd.to_numeric(df["bart_pred_is_stage2"], errors="coerce").fillna(0).astype(int)

    g = df.groupby(COL_MRN, dropna=False)
    out = g.agg(
        bart_hit_stage2_score_max=("bart_stage2_score", "max"),
        bart_hit_stage2_score_mean=("bart_stage2_score", "mean"),
        bart_hits_scored=("bart_stage2_score", "count"),
        bart_any_hit_pred_stage2=("bart_pred_is_stage2", "max"),
        bart_any_strong_hit=("HIT_STRENGTH", lambda x: int(any(str(v).upper() == "STRONG" for v in x))),
        bart_any_weak_hit=("HIT_STRENGTH", lambda x: int(any(str(v).upper() == "WEAK" for v in x))),
    ).reset_index()

    out["bart_threshold"] = THRESHOLD
    os.makedirs(os.path.dirname(out_patient_csv), exist_ok=True)
    out.to_csv(out_patient_csv, index=False)
    return out_patient_csv

# ==============================
# Main
# ==============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(IN_HITS):
        raise RuntimeError("Missing input hits file: %s (run build_stage12_WITH_AUDIT.py first)" % IN_HITS)

    if not os.path.exists(MODEL_DIR):
        raise RuntimeError("Missing MODEL_DIR: %s" % MODEL_DIR)

    print("Loading stage2_event_hits.csv ...")
    hits = pd.read_csv(IN_HITS, low_memory=False)

    # Validate required columns
    required = [COL_MRN, COL_SNIPPET, COL_NOTE_DATE, COL_NOTE_TYPE, COL_SOURCE_FILE, COL_STRENGTH]
    missing = [c for c in required if c not in hits.columns]
    if missing:
        raise RuntimeError("Missing columns in %s: %s" % (IN_HITS, missing))

    # Build row_id
    print("Preparing resume keys...")
    hits["SNIPPET"] = hits["SNIPPET"].fillna("").astype(str).apply(lambda s: _truncate(s, MAX_CHARS))
    hits["row_id"] = hits.apply(
        lambda r: make_row_id(r[COL_MRN], r[COL_NOTE_DATE], r[COL_SOURCE_FILE], r[COL_STRENGTH], r[COL_SNIPPET]),
        axis=1
    )

    done_ids = load_done_ids(OUT_HIT_SCORES)
    if done_ids:
        print("Resume: found %d already-scored rows. Will skip them." % len(done_ids))

    to_score = hits[~hits["row_id"].isin(done_ids)].copy()
    print("Rows to score now:", len(to_score), "out of", len(hits))

    if len(to_score) == 0:
        print("Nothing new to score. Building patient summary from existing outputs...")
        build_patient_summary(OUT_HIT_SCORES, OUT_PATIENT)
        print("Saved:", OUT_PATIENT)
        return

    print("Loading offline BART MNLI (CPU)...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    clf = pipeline(
        "zero-shot-classification",
        model=mdl,
        tokenizer=tok,
        device=-1
    )

    candidate_labels = [NEG_LABEL, POS_LABEL]

    # Output columns
    out_cols = [
        "row_id",
        COL_MRN,
        COL_NOTE_DATE,
        COL_NOTE_TYPE,
        COL_SOURCE_FILE,
        COL_STRENGTH,
        "bart_stage2_score",
        "bart_threshold",
        "bart_pred_is_stage2",
        "bart_pred_label",
        "snippet_chars_used",
    ]

    # Batched scoring with incremental saves
    bs = max(1, int(BATCH_SIZE))
    t0 = time.time()

    print("Scoring with batch_size=%d on SNIPPET (fast)..." % bs)

    buffer_rows = []
    flush_every_batches = 25  # write every N batches to survive disconnects

    rows = to_score.to_dict(orient="records")
    total = len(rows)

    # iterate in batches
    for b_start in tqdm(range(0, total, bs), desc="BART verify", unit="batch"):
        b_end = min(total, b_start + bs)
        batch = rows[b_start:b_end]

        texts = [_safe_str(r[COL_SNIPPET]) for r in batch]

        results = clf(
            sequences=texts,
            candidate_labels=candidate_labels,
            hypothesis_template=HYPOTHESIS_TEMPLATE
        )

        for i, res in enumerate(results):
            r = batch[i]
            labels = res.get("labels", [])
            scores = res.get("scores", [])

            score_map = {}
            for lab, sc in zip(labels, scores):
                score_map[_safe_str(lab).strip().lower()] = float(sc)

            stage2_score = score_map.get(POS_LABEL.lower(), None)
            pred_is_stage2 = 1 if (stage2_score is not None and stage2_score >= THRESHOLD) else 0

            buffer_rows.append({
                "row_id": r["row_id"],
                COL_MRN: _safe_str(r[COL_MRN]).strip(),
                COL_NOTE_DATE: _safe_str(r[COL_NOTE_DATE]),
                COL_NOTE_TYPE: _safe_str(r[COL_NOTE_TYPE]),
                COL_SOURCE_FILE: _safe_str(r[COL_SOURCE_FILE]),
                COL_STRENGTH: _safe_str(r[COL_STRENGTH]),
                "bart_stage2_score": stage2_score if stage2_score is not None else "",
                "bart_threshold": THRESHOLD,
                "bart_pred_is_stage2": pred_is_stage2,
                "bart_pred_label": POS_LABEL if pred_is_stage2 else NEG_LABEL,
                "snippet_chars_used": len(texts[i]),
            })

        # flush periodically
        if ((b_start // bs) + 1) % flush_every_batches == 0 and buffer_rows:
            append_rows_csv(OUT_HIT_SCORES, buffer_rows, out_cols)
            buffer_rows = []

    # final flush
    if buffer_rows:
        append_rows_csv(OUT_HIT_SCORES, buffer_rows, out_cols)

    runtime = round(time.time() - t0, 2)

    meta = {
        "input_hits_csv": IN_HITS,
        "model_dir": MODEL_DIR,
        "hypothesis_template": HYPOTHESIS_TEMPLATE,
        "labels": candidate_labels,
        "threshold": THRESHOLD,
        "batch_size": bs,
        "max_chars": MAX_CHARS,
        "n_hits_total": int(len(hits)),
        "n_scored_this_run": int(len(to_score)),
        "output_hit_scores_csv": OUT_HIT_SCORES,
        "runtime_seconds": runtime,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved hit-level scores:", OUT_HIT_SCORES)
    print("Saved run metadata:", OUT_META)

    print("Building patient summary...")
    build_patient_summary(OUT_HIT_SCORES, OUT_PATIENT)
    print("Saved patient summary:", OUT_PATIENT)
    print("Done.")


if __name__ == "__main__":
    main()
