#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_encounters.py (Python 3.6.8)

Run from: ~/Breast_Restore
Goal: Print column inventory + samples for encounter files to decide which columns to use.
"""

from __future__ import print_function

import os
import glob
import pandas as pd

DATA_DIR = os.path.expanduser("~/my_data_Breast/HPI-11526/HPI11256")

CANDIDATES = [
    "HPI11526 Clinic Encounters.csv",
    "HPI11526 Inpatient Encounters.csv",
    "HPI11526 Operation Encounters.csv",
]

LIKELY_TEXT_COLS = [
    "PROCEDURE", "PROCEDURES", "PROC", "PROC_NAME", "PROC_DESC",
    "REASON", "REASON_FOR_VISIT", "CHIEF_COMPLAINT", "VISIT_REASON",
    "PRIMARY_REASON", "ENCOUNTER_REASON",
    "DIAGNOSIS", "DIAGNOSES", "DX", "DX_NAME", "DX_DESC",
    "CPT", "CPT_CODE", "ICD", "ICD_CODE"
]

LIKELY_DATE_COLS = [
    "DATE", "ENC_DATE", "ENCOUNTER_DATE", "ADMIT_DATE", "DISCH_DATE",
    "ARRIVAL_DATE", "SERVICE_DATE", "CONTACT_DATE"
]

LIKELY_ID_COLS = [
    "MRN", "ENCRYPTED_PAT_ID", "PAT_ID", "PATIENT_ID", "PatientID",
    "ENCRYPTED_CSN", "CSN", "PAT_ENC_CSN_ID"
]

def pick_files():
    files = []
    for name in CANDIDATES:
        p = os.path.join(DATA_DIR, name)
        if os.path.isfile(p):
            files.append(p)
    if files:
        return files
    # fallback: anything with "Encounters" in name
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*Encounters*.csv")))
    return files

def find_cols(df, candidates):
    cols = []
    upper_map = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in upper_map:
            cols.append(upper_map[cand.upper()])
    return cols

def show_samples(df, col, n=8):
    s = df[col].dropna().astype(str)
    s = s[s.str.strip() != ""]
    if s.empty:
        return []
    return list(s.head(n).values)

def main():
    files = pick_files()
    if not files:
        print("No encounter CSVs found in:", DATA_DIR)
        return

    print("Found encounter files:")
    for f in files:
        print(" -", f)

    for path in files:
        print("\n" + "="*80)
        print("FILE:", path)
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8", errors="replace")
        except TypeError:
            # older pandas sometimes doesn't accept errors=
            df = pd.read_csv(path, dtype=str, low_memory=False, encoding="latin-1")

        print("Rows:", len(df), "Cols:", len(df.columns))
        print("Columns:")
        print(list(df.columns))

        id_cols = find_cols(df, LIKELY_ID_COLS)
        date_cols = find_cols(df, LIKELY_DATE_COLS)
        text_cols = []
        for c in df.columns:
            if c.upper() in [x.upper() for x in LIKELY_TEXT_COLS]:
                text_cols.append(c)

        print("\nLikely ID cols:", id_cols)
        print("Likely date cols:", date_cols)
        print("Likely text cols:", text_cols)

        # show samples for up to 3 text columns
        for c in text_cols[:3]:
            samples = show_samples(df, c, n=8)
            print("\nSample values for:", c)
            for v in samples:
                print("  -", v[:160])

if __name__ == "__main__":
    main()
