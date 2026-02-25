#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_fn_qa_snippets.py  (Python 3.6.8 compatible)

ROBUST VERSION:
- Does NOT assume any specific ENCRYPTED_PAT_ID column name
- Dynamically detects patient id column in OP notes
- Never hard-codes ENCRYPTED_PAT_ID
- Broad Stage2 discovery search
- Outputs NO MRN (safe to paste)
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
    raise IOError("Failed to read CSV: {}".format(path))


def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df


def normalize_id(x):
    return "" if x is None else str(x).strip()


def to01(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except:
        return 0


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


# -------------------------
# Broad Stage2 search
# -------------------------

STAGE2_PATTERNS = [
    r"\bimplant exchange\b",
    r"\bexpander exchange\b",
    r"\bexchange\b",
    r"\breplace(d|ment)?\b",
    r"\bremove(d|al)?\b.*\b(expander|tissue expander|expanders|te)\b",
    r"\b(explant(ed)?|take\s*out)\b.*\b(expander|implant)\b",
    r"\bexpander[- ]?to[- ]?implant\b",
    r"\bsecond stage\b",
    r"\bstage\s*2\b",
    r"\bpermanent implant\b",
    r"\bfinal implant\b",
    r"\b(expander|tissue expander|expanders|te)\b.*\b(implant|implants)\b",
    r"\b(implant|implants)\b.*\b(expander|tissue expander|expanders|te)\b",
    r"\b19342\b",
    r"\b11970\b",
    r"\b11971\b",
]

RX_STAGE2 = re.compile("|".join(["({})".format(p) for p in STAGE2_PATTERNS]), re.I)


def load_text(path):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except:
            continue
    return ""


def extract_snippets(text, rx, max_snips=8, ctx=350):
    if not text:
        return []

    snippets = []
    seen = set()

    for m in rx.finditer(text):
        start = max(0, m.start() - ctx)
        end = min(len(text), m.end() + ctx)
        snip = text[start:end]
        snip = re.sub(r"\s+", " ", snip).strip()

        key = snip.lower()
        if key in seen:
            continue
        seen.add(key)

        snippets.append(snip)
        if len(snippets) >= max_snips:
            break

    return snippets


# -------------------------
# Main
# -------------------------

def main():

    root = "/home/apokol/Breast_Restore"
    out_dir = os.path.join(root, "_outputs")

    merged_path = os.path.join(out_dir, "validation_merged.csv")
    op_notes_path = os.path.join(root, "_staging_inputs", "HPI11526 Operation Notes.csv")
    bundles_root = os.path.join(root, "PATIENT_BUNDLES")

    merged = normalize_cols(read_csv_robust(merged_path, dtype=str))
    op = normalize_cols(read_csv_robust(op_notes_path, dtype=str))

    # Identify columns in merged
    mrn_col = pick_first_existing(merged, ["MRN", "mrn"])
    gold_col = pick_first_existing(merged, ["GOLD_HAS_STAGE2", "Stage2_Applicable"])
    pred_col = pick_first_existing(merged, ["PRED_HAS_STAGE2", "HAS_STAGE2"])

    merged["MRN"] = merged[mrn_col].map(normalize_id)
    merged["GOLD_HAS_STAGE2"] = merged[gold_col].map(to01).astype(int)
    merged["PRED_HAS_STAGE2"] = merged[pred_col].map(to01).astype(int)

    fn = merged[(merged["GOLD_HAS_STAGE2"] == 1) &
                (merged["PRED_HAS_STAGE2"] == 0)].copy()

    # Identify MRN + patient id columns in op notes dynamically
    op_mrn_col = pick_first_existing(op, ["MRN", "mrn"])
    op_pid_col = pick_first_existing(op, [
        "ENCRYPTED_PAT_ID",
        "ENCRYPTED_PATID",
        "ENCRYPTED_PATIENT_ID",
        "patient_id",
        "PATIENT_ID"
    ])

    if not op_mrn_col or not op_pid_col:
        raise ValueError("Could not detect MRN and patient id columns in OP notes.")

    op["MRN"] = op[op_mrn_col].map(normalize_id)
    op["PATIENT_ID"] = op[op_pid_col].map(normalize_id)

    id_map = op[["MRN", "PATIENT_ID"]].drop_duplicates()

    fn = fn.merge(id_map, on="MRN", how="left")

    # Ensure PATIENT_ID exists safely
    if "PATIENT_ID" not in fn.columns:
        raise ValueError("Patient ID column missing after merge.")

    fn["PATIENT_ID"] = fn["PATIENT_ID"].fillna("").map(normalize_id)
    fn = fn[fn["PATIENT_ID"] != ""].copy()

    rows = []

    for _, r in fn.iterrows():
        pid = r["PATIENT_ID"]
        bundle_file = os.path.join(bundles_root, pid, "ALL_NOTES_COMBINED.txt")

        text = load_text(bundle_file) if os.path.isfile(bundle_file) else ""
        snippets = extract_snippets(text, RX_STAGE2)

        row = {
            "ENCRYPTED_PAT_ID": pid,
            "BUNDLE_FOUND": 1 if os.path.isfile(bundle_file) else 0
        }

        for i in range(8):
            row["SNIP_{:02d}".format(i+1)] = snippets[i] if i < len(snippets) else ""

        rows.append(row)

    qa_df = pd.DataFrame(rows)

    out_csv = os.path.join(out_dir, "stage2_fn_qa_snippets.csv")
    qa_df.to_csv(out_csv, index=False)

    print("Wrote:", out_csv)
    print("FN rows processed:", len(qa_df))


if __name__ == "__main__":
    main()
