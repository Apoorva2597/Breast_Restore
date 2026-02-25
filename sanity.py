#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_reop_mismatches.py  (Python 3.6.8 compatible)

Run from:  ~/Breast_Restore
Reads:
  ./_outputs/validation_merged.csv
Writes:
  ./_outputs/reop_FN_sample.csv
  ./_outputs/reop_FP_sample.csv

Also prints a compact terminal view.

Goal:
- Reoperation FNs (gold=1 pred=0): include evidence fields (likely blank) + hint columns (STAGE2_DATE, window, sources/dates)
- Reoperation FPs (gold=0 pred=1): include reop_evidence_snippet (+ pattern/source/date) + hint columns
"""

from __future__ import print_function
import os
import pandas as pd


def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))


def to01(x):
    if x is None:
        return 0
    s = str(x).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def pick_cols(df, preferred):
    out = []
    for c in preferred:
        if c in df.columns and c not in out:
            out.append(c)
    return out


def main():
    root = os.path.abspath(".")
    in_path = os.path.join(root, "_outputs", "validation_merged.csv")
    out_fn = os.path.join(root, "_outputs", "reop_FN_sample.csv")
    out_fp = os.path.join(root, "_outputs", "reop_FP_sample.csv")

    if not os.path.isfile(in_path):
        raise IOError("Missing input: {}".format(in_path))

    df = read_csv_robust(in_path, dtype=str, low_memory=False)

    # Columns we expect (but handle alternate naming)
    gold_col = "GOLD_Stage2_Reoperation" if "GOLD_Stage2_Reoperation" in df.columns else None
    pred_col = "Stage2_Reoperation_pred" if "Stage2_Reoperation_pred" in df.columns else None

    if not gold_col:
        raise ValueError("Could not find GOLD_Stage2_Reoperation in columns.")
    if not pred_col:
        raise ValueError("Could not find Stage2_Reoperation_pred in columns.")

    df["_gold"] = df[gold_col].map(to01)
    df["_pred"] = df[pred_col].map(to01)

    fns = df[(df["_gold"] == 1) & (df["_pred"] == 0)].copy()
    fps = df[(df["_gold"] == 0) & (df["_pred"] == 1)].copy()

    # Build a useful column set (keep if exists)
    hint_cols = pick_cols(df, [
        "MRN",
        "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID",
        "PatientID",
        "STAGE2_DATE", "WINDOW_START", "WINDOW_END",
        gold_col, pred_col,
        # Evidence fields (your validator tends to carry these through)
        "reop_evidence_date", "reop_evidence_source", "reop_evidence_note_id",
        "reop_evidence_pattern", "reop_evidence_snippet",
        # Sometimes scripts used *_note_id naming variants
        "reop_evidence_noteid", "reop_evidence_note",
        # Any extra “signal” columns you might have
        "Stage2_MajorComp_pred", "Stage2_Rehospitalization_pred", "Stage2_Failure_pred", "Stage2_Revision_pred",
        "GOLD_Stage2_MajorComp", "GOLD_Stage2_Rehospitalization", "GOLD_Stage2_Failure", "GOLD_Stage2_Revision",
    ])

    # Save samples (all rows; you can open and filter, or limit later)
    fns_out = fns[hint_cols] if len(hint_cols) else fns
    fps_out = fps[hint_cols] if len(hint_cols) else fps

    fns_out.to_csv(out_fn, index=False)
    fps_out.to_csv(out_fp, index=False)

    # Terminal summary + top rows
    print("")
    print("Loaded:", in_path)
    print("Rows:", len(df))
    print("Reoperation FNs (gold=1 pred=0):", len(fns_out))
    print("Reoperation FPs (gold=0 pred=1):", len(fps_out))
    print("")
    print("Wrote:")
    print(" ", out_fn)
    print(" ", out_fp)
    print("")

    # Compact terminal preview (first 10)
    pd.set_option("display.max_colwidth", 140)
    pd.set_option("display.width", 200)

    show_cols_fn = pick_cols(df, [
        "MRN", "ENCRYPTED_PAT_ID", "STAGE2_DATE",
        gold_col, pred_col,
        "reop_evidence_date", "reop_evidence_source", "reop_evidence_pattern",
        "reop_evidence_snippet",
    ])
    show_cols_fp = show_cols_fn

    if len(fns_out) > 0:
        print("=== FN preview (first 10) ===")
        print(fns_out[show_cols_fn].head(10).to_string(index=False))
        print("")
    else:
        print("No Reoperation FNs found.\n")

    if len(fps_out) > 0:
        print("=== FP preview (first 10) ===")
        print(fps_out[show_cols_fp].head(10).to_string(index=False))
        print("")
    else:
        print("No Reoperation FPs found.\n")


if __name__ == "__main__":
    main()
