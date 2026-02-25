#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_fn_qa_snippets_RAW_NOTES.py  (Python 3.6.8 compatible)

UPDATED:
- Uses RAW notes (Clinic + Inpatient + Operation Notes CSVs)
- Shorter snippets for faster manual review
- Broad Stage2 discovery search
- Outputs MRN + ENCRYPTED_PAT_ID + raw snippets
"""

from __future__ import print_function

import os
import re
import glob
import pandas as pd


# -------------------------
# Helpers
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
# Broad Stage2 discovery regex
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


def extract_snippets(text, rx, max_snips=5, ctx=150):
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
    staging_dir = os.path.join(root, "_staging_inputs")

    merged = normalize_cols(read_csv_robust(merged_path, dtype=str))

    mrn_col = pick_first_existing(merged, ["MRN", "mrn"])
    gold_col = pick_first_existing(merged, ["GOLD_HAS_STAGE2", "Stage2_Applicable"])
    pred_col = pick_first_existing(merged, ["PRED_HAS_STAGE2", "HAS_STAGE2"])

    merged["MRN"] = merged[mrn_col].map(normalize_id)
    merged["GOLD_HAS_STAGE2"] = merged[gold_col].map(to01).astype(int)
    merged["PRED_HAS_STAGE2"] = merged[pred_col].map(to01).astype(int)

    fn = merged[(merged["GOLD_HAS_STAGE2"] == 1) &
                (merged["PRED_HAS_STAGE2"] == 0)].copy()

    note_files = glob.glob(os.path.join(staging_dir, "*Notes*.csv"))
    notes_list = []

    for f in note_files:
        df = normalize_cols(read_csv_robust(f, dtype=str))
        mrn_note_col = pick_first_existing(df, ["MRN", "mrn"])
        text_col = pick_first_existing(df, ["NOTE_TEXT", "Note_Text", "note_text"])

        if mrn_note_col and text_col:
            df["MRN"] = df[mrn_note_col].map(normalize_id)
            df["NOTE_TEXT"] = df[text_col].fillna("").map(str)
            notes_list.append(df[["MRN", "NOTE_TEXT"]])

    if not notes_list:
        raise ValueError("No usable note files found.")

    notes = pd.concat(notes_list, ignore_index=True)

    rows = []

    for _, r in fn.iterrows():
        mrn = r["MRN"]

        patient_notes = notes[notes["MRN"] == mrn]

        snippets_all = []

        for _, nrow in patient_notes.iterrows():
            snips = extract_snippets(nrow["NOTE_TEXT"], RX_STAGE2)
            snippets_all.extend(snips)
            if len(snippets_all) >= 5:
                break

        row = {
            "MRN": mrn,
            "ENCRYPTED_PAT_ID": r.get("ENCRYPTED_PAT_ID", "")
        }

        for i in range(5):
            row["SNIP_{:02d}".format(i+1)] = snippets_all[i] if i < len(snippets_all) else ""

        rows.append(row)

    qa_df = pd.DataFrame(rows)

    out_csv = os.path.join(out_dir, "stage2_fn_raw_note_snippets.csv")
    qa_df.to_csv(out_csv, index=False)

    print("Wrote:", out_csv)
    print("FN rows processed:", len(qa_df))


if __name__ == "__main__":
    main()
