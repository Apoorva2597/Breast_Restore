#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_fn_id_diagnostic_check.py

Purpose:
- Diagnose ID confusion between MRN, PatientID, ENCRYPTED_PAT_ID
- Report overlap between mismatches file and notes files
- NO PHI written to output (no MRN in final CSV)

Outputs:
  ./_outputs/stage2_id_diagnostic_report_FINAL_FINAL.csv
"""

from __future__ import print_function
import os
import glob
import re
import pandas as pd

MISMATCH_PATH = "./_outputs/validation_mismatches_STAGE2_ANCHOR_FINAL_FINAL.csv"
NOTES_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"
OUT_PATH = "./_outputs/stage2_id_diagnostic_report_FINAL_FINAL.csv"

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV: {}".format(path))

def normalize_cols(df):
    df.columns = [str(c).strip().replace(u"\xa0"," ") for c in df.columns]
    return df

def normalize_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    if re.match(r"^\d+\.0$", s):
        s = s.split(".")[0]
    if re.match(r"^\d+\.\d+$", s):
        try:
            f = float(s)
            if abs(f - int(f)) < 1e-9:
                s = str(int(f))
        except:
            pass
    return s

def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def list_csvs(root):
    return sorted([p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True) if os.path.isfile(p)])

def main():
    if not os.path.isfile(MISMATCH_PATH):
        raise IOError("Missing mismatches file.")
    if not os.path.isdir(NOTES_DIR):
        raise IOError("Missing notes dir.")

    mism = normalize_cols(read_csv_robust(MISMATCH_PATH, dtype=str, low_memory=False))

    # Identify ID columns in mismatches
    mrn_col = pick_first_existing(mism, ["MRN","mrn"])
    enc_col = pick_first_existing(mism, ["ENCRYPTED_PAT_ID","ENCRYPTED_PATID","ENCRYPTED_PATIENT_ID"])
    pid_col = pick_first_existing(mism, ["PatientID","PATIENT_ID","PATID"])

    mism_ids = {
        "MRN": set(mism[mrn_col].map(normalize_id)) if mrn_col else set(),
        "ENCRYPTED_PAT_ID": set(mism[enc_col].map(normalize_id)) if enc_col else set(),
        "PatientID": set(mism[pid_col].map(normalize_id)) if pid_col else set()
    }

    # Remove blanks
    for k in mism_ids:
        mism_ids[k] = set([x for x in mism_ids[k] if x != ""])

    # Collect IDs from notes
    note_csvs = list_csvs(NOTES_DIR)
    notes_ids = {
        "MRN": set(),
        "ENCRYPTED_PAT_ID": set(),
        "PatientID": set()
    }

    for p in note_csvs:
        try:
            df = normalize_cols(read_csv_robust(p, nrows=100, dtype=str, low_memory=False))
        except:
            continue

        mrn_c = pick_first_existing(df, ["MRN","mrn"])
        enc_c = pick_first_existing(df, ["ENCRYPTED_PAT_ID","ENCRYPTED_PATID","ENCRYPTED_PATIENT_ID"])
        pid_c = pick_first_existing(df, ["PatientID","PATIENT_ID","PATID"])

        if mrn_c:
            notes_ids["MRN"].update(set(df[mrn_c].map(normalize_id)))
        if enc_c:
            notes_ids["ENCRYPTED_PAT_ID"].update(set(df[enc_c].map(normalize_id)))
        if pid_c:
            notes_ids["PatientID"].update(set(df[pid_c].map(normalize_id)))

    # Remove blanks
    for k in notes_ids:
        notes_ids[k] = set([x for x in notes_ids[k] if x != ""])

    # Compute overlaps
    rows = []
    for mism_key in mism_ids:
        for note_key in notes_ids:
            overlap = mism_ids[mism_key].intersection(notes_ids[note_key])
            rows.append({
                "MISMATCH_ID_TYPE": mism_key,
                "NOTES_ID_TYPE": note_key,
                "MISMATCH_UNIQUE_COUNT": len(mism_ids[mism_key]),
                "NOTES_UNIQUE_COUNT": len(notes_ids[note_key]),
                "OVERLAP_COUNT": len(overlap)
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print("Wrote:", OUT_PATH)

if __name__ == "__main__":
    main()
