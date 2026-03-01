#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_fn_snippets.py
Python 3.6.8 compatible

Pull FN note snippets using:
- validation_mismatches_STAGE2_ANCHOR_FIXED.csv
- De-identified note bundle directory

OUTPUT:
- ./_outputs/fn_note_snippets_for_rule_refinement.csv
"""

from __future__ import print_function
import os
import pandas as pd

ROOT = os.path.abspath(".")
OUT_DIR = os.path.join(ROOT, "_outputs")

VALIDATION_FILE = os.path.join(OUT_DIR, "validation_mismatches_STAGE2_ANCHOR_FIXED.csv")

# <-- UPDATE if your de-id bundle root differs
DEID_BUNDLE_ROOT = os.path.join(ROOT, "_deidentified_note_bundles")

OUTPUT_FILE = os.path.join(OUT_DIR, "fn_note_snippets_for_rule_refinement.csv")


def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV: {}".format(path))


def main():

    if not os.path.isfile(VALIDATION_FILE):
        raise IOError("Validation mismatches file not found.")

    mism = read_csv_robust(VALIDATION_FILE, dtype=str, low_memory=False)

    # FN = GOLD_HAS_STAGE2 = 1 AND PRED_HAS_STAGE2 = 0
    fn = mism[
        (mism["GOLD_HAS_STAGE2"].astype(str) == "1") &
        (mism["PRED_HAS_STAGE2"].astype(str) == "0")
    ].copy()

    if len(fn) == 0:
        print("No FN cases found.")
        return

    print("FN cases found:", len(fn))

    snippets = []

    for _, row in fn.iterrows():

        mrn = str(row.get("MRN", "")).strip()
        if not mrn:
            continue

        # Expect bundle per patient like: _deidentified_note_bundles/<MRN>.csv
        bundle_path = os.path.join(DEID_BUNDLE_ROOT, "{}.csv".format(mrn))

        if not os.path.isfile(bundle_path):
            continue

        notes = read_csv_robust(bundle_path, dtype=str, low_memory=False)

        text_col = None
        for c in ["NOTE_TEXT", "TEXT", "CONTENT"]:
            if c in notes.columns:
                text_col = c
                break

        if not text_col:
            continue

        for _, nrow in notes.iterrows():
            text = str(nrow.get(text_col, ""))
            if not text.strip():
                continue

            snippets.append({
                "MRN": mrn,
                "SNIPPET": text[:600]
            })

    out_df = pd.DataFrame(snippets)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print("Wrote:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
