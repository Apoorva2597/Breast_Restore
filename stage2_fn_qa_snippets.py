#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_fn_qa_snippets.py  (Python 3.6.8 compatible)

UPDATED VERSION:
- Much broader Stage 2 keyword search
- Designed for discovery (FN exploration), not precision
- Extracts broad contextual snippets from ALL_NOTES_COMBINED.txt
- Outputs NO MRN (safe to paste)

Input assumptions:
- validation_merged.csv exists
- CROSSWALK__MRN_to_patient_id__vNEW.csv exists
- PATIENT_BUNDLES/<ENCRYPTED_PAT_ID>/ALL_NOTES_COMBINED.txt exists
"""

from __future__ import print_function

import os
import re
import glob
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
# BROAD Stage 2 discovery regex
# -------------------------

STAGE2_PATTERNS = [

    # direct exchange language
    r"\bimplant exchange\b",
    r"\bexpander exchange\b",
    r"\bexchange\b",
    r"\breplace(d|ment)?\b",
    r"\bre-implant\b",

    # expander removal
    r"\bremove(d|al)?\b.*\b(expander|tissue expander|expanders|te)\b",
    r"\b(explant(ed)?|take\s*out)\b.*\b(expander|implant)\b",

    # expander -> implant phrasing
    r"\bexpander[- ]?to[- ]?implant\b",
    r"\bconversion\b.*\bimplant\b",

    # second stage language
    r"\bsecond stage\b",
    r"\bstage\s*2\b",

    # permanent implant phrasing
    r"\bpermanent implant\b",
    r"\bfinal implant\b",

    # implant after expander mention anywhere in note
    r"\b(expander|tissue expander|expanders|te)\b.*\b(implant|implants)\b",
    r"\b(implant|implants)\b.*\b(expander|tissue expander|expanders|te)\b",

    # CPT hints sometimes written in notes
    r"\b19342\b",
    r"\b11970\b",
    r"\b11971\b",

    # narrative phrasing
    r"\bpatient returned for exchange\b",
    r"\bexchange of tissue expander\b",
    r"\bremoval of tissue expander\b",
    r"\bplacement of permanent implant\b",
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
# FN identification
# -------------------------

def main():

    root = "/home/apokol/Breast_Restore"
    out_dir = os.path.join(root, "_outputs")

    merged_path = os.path.join(out_dir, "validation_merged.csv")
    crosswalk_path = os.path.join(root, "CROSSWALK", "CROSSWALK__MRN_to_patient_id__vNEW.csv")
    bundles_root = os.path.join(root, "PATIENT_BUNDLES")

    merged = normalize_cols(read_csv_robust(merged_path, dtype=str))
    cw = normalize_cols(read_csv_robust(crosswalk_path, dtype=str))

    mrn_col = pick_first_existing(merged, ["MRN", "mrn"])
    gold_col = pick_first_existing(merged, ["GOLD_HAS_STAGE2", "Stage2_Applicable"])
    pred_col = pick_first_existing(merged, ["PRED_HAS_STAGE2", "HAS_STAGE2"])

    merged["MRN"] = merged[mrn_col].map(normalize_id)
    merged["GOLD_HAS_STAGE2"] = merged[gold_col].map(to01).astype(int)
    merged["PRED_HAS_STAGE2"] = merged[pred_col].map(to01).astype(int)

    fn = merged[(merged["GOLD_HAS_STAGE2"] == 1) &
                (merged["PRED_HAS_STAGE2"] == 0)].copy()

    cw_mrn = pick_first_existing(cw, ["MRN", "mrn"])
    cw_pid = pick_first_existing(cw, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "patient_id"])

    cw["MRN"] = cw[cw_mrn].map(normalize_id)
    cw["ENCRYPTED_PAT_ID"] = cw[cw_pid].map(normalize_id)

    cw = cw[["MRN", "ENCRYPTED_PAT_ID"]].drop_duplicates()

    fn = fn.merge(cw, on="MRN", how="left")
    fn = fn[fn["ENCRYPTED_PAT_ID"] != ""].copy()

    rows = []

    for _, r in fn.iterrows():
        pid = r["ENCRYPTED_PAT_ID"]
        bundle_file = os.path.join(bundles_root, pid, "ALL_NOTES_COMBINED.txt")

        text = load_text(bundle_file) if os.path.isfile(bundle_file) else ""
        snippets = extract_snippets(text, RX_STAGE2, max_snips=8, ctx=350)

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
