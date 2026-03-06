#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qa_age_remaining_mismatches.py

Find age mismatches that remain even after allowing:
gold_age == pred_age OR pred_age-1 OR pred_age+1

Outputs:
1) _outputs/qa_age_remaining_mismatches_summary.csv
2) _outputs/qa_age_remaining_mismatches_notes.csv

Python 3.6.8 compatible
"""

import os
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = os.path.join(BASE_DIR, "_outputs", "master_abstraction_rule_FINAL_NO_GOLD.csv")
GOLD_FILE = os.path.join(BASE_DIR, "gold_cleaned_for_cedar.csv")
NOTE_FILES = [
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Clinic Notes.csv"),
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Inpatient Notes.csv"),
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Operation Notes.csv"),
]
CLINIC_ENC_FILE = os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Clinic Encounters.csv")

OUT_SUMMARY = os.path.join(BASE_DIR, "_outputs", "qa_age_remaining_mismatches_summary.csv")
OUT_NOTES = os.path.join(BASE_DIR, "_outputs", "qa_age_remaining_mismatches_notes.csv")

MRN = "MRN"


def safe_read_csv(path):
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=str, encoding="latin1")


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MRN:
                df = df.rename(columns={k: MRN})
            break
    if MRN not in df.columns:
        raise RuntimeError("MRN column not found. Columns seen: {0}".format(list(df.columns)[:40]))
    df[MRN] = df[MRN].astype(str).str.strip()
    return df


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ["", "nan", "none", "null", "na"]:
        return ""
    return s


def reconstruct_notes():
    all_rows = []

    for fp in NOTE_FILES:
        if not os.path.isfile(fp):
            continue

        df = clean_cols(safe_read_csv(fp))
        df = normalize_mrn(df)

        note_text_col = None
        for c in ["NOTE_TEXT", "NOTE TEXT", "NOTE_TEXT_FULL", "TEXT", "NOTE"]:
            if c in df.columns:
                note_text_col = c
                break

        note_id_col = None
        for c in ["NOTE_ID", "NOTE ID"]:
            if c in df.columns:
                note_id_col = c
                break

        line_col = "LINE" if "LINE" in df.columns else None

        note_type_col = None
        for c in ["NOTE_TYPE", "NOTE TYPE"]:
            if c in df.columns:
                note_type_col = c
                break

        date_col = None
        for c in ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE", "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"]:
            if c in df.columns:
                date_col = c
                break

        if note_text_col is None or note_id_col is None:
            continue

        keep = [MRN, note_id_col, note_text_col]
        if line_col:
            keep.append(line_col)
        if note_type_col:
            keep.append(note_type_col)
        if date_col:
            keep.append(date_col)

        tmp = df[keep].copy()
        tmp["_SOURCE_FILE_"] = os.path.basename(fp)

        tmp = tmp.rename(columns={
            note_id_col: "NOTE_ID",
            note_text_col: "NOTE_TEXT"
        })

        if line_col is None:
            tmp["LINE"] = ""
        elif line_col != "LINE":
            tmp = tmp.rename(columns={line_col: "LINE"})

        if note_type_col is None:
            tmp["NOTE_TYPE"] = ""
        elif note_type_col != "NOTE_TYPE":
            tmp = tmp.rename(columns={note_type_col: "NOTE_TYPE"})

        if date_col is None:
            tmp["NOTE_DATE"] = ""
        else:
            if date_col != "NOTE_DATE":
                tmp = tmp.rename(columns={date_col: "NOTE_DATE"})

        all_rows.append(tmp)

    if not all_rows:
        return pd.DataFrame(columns=[MRN, "NOTE_ID", "NOTE_TYPE", "NOTE_DATE", "SOURCE_FILE", "NOTE_TEXT"])

    notes_raw = pd.concat(all_rows, ignore_index=True)

    def line_to_num(x):
        try:
            return int(float(str(x).strip()))
        except Exception:
            return 999999999

    reconstructed = []
    grouped = notes_raw.groupby([MRN, "NOTE_ID"], dropna=False)

    for (mrn, nid), g in grouped:
        mrn = clean_cell(mrn)
        nid = clean_cell(nid)
        if not mrn or not nid:
            continue

        gg = g.copy()
        gg["_LINE_NUM_"] = gg["LINE"].apply(line_to_num)
        gg = gg.sort_values("_LINE_NUM_")

        full_text = "\n".join(gg["NOTE_TEXT"].fillna("").astype(str).tolist()).strip()
        if not full_text:
            continue

        note_type = clean_cell(gg["NOTE_TYPE"].iloc[0]) if "NOTE_TYPE" in gg.columns else ""
        note_date = clean_cell(gg["NOTE_DATE"].iloc[0]) if "NOTE_DATE" in gg.columns else ""
        source_file = clean_cell(gg["_SOURCE_FILE_"].iloc[0])

        reconstructed.append({
            MRN: mrn,
            "NOTE_ID": nid,
            "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date,
            "SOURCE_FILE": source_file,
            "NOTE_TEXT": full_text
        })

    return pd.DataFrame(reconstructed)


def main():
    print("Loading master and gold...")
    master = clean_cols(safe_read_csv(MASTER_FILE))
    gold = clean_cols(safe_read_csv(GOLD_FILE))

    master = normalize_mrn(master)
    gold = normalize_mrn(gold)

    master = master.drop_duplicates(subset=[MRN]).copy()
    gold = gold.drop_duplicates(subset=[MRN]).copy()

    merged = pd.merge(
        master[[MRN, "Age"]],
        gold[[MRN, "Age"]],
        on=MRN,
        how="inner",
        suffixes=("_pred", "_gold")
    )

    pred = pd.to_numeric(merged["Age_pred"], errors="coerce")
    gold_age = pd.to_numeric(merged["Age_gold"], errors="coerce")

    mask = gold_age.notna()

    mismatch = merged[
        mask & ~((gold_age == pred) | (gold_age == pred - 1) | (gold_age == pred + 1))
    ].copy()

    print("Remaining age mismatches after floor/round rule:", len(mismatch))

    mismatch_mrns = mismatch[MRN].astype(str).str.strip().tolist()

    print("Loading clinic encounters...")
    clinic = clean_cols(safe_read_csv(CLINIC_ENC_FILE))
    clinic = normalize_mrn(clinic)

    keep_cols = [MRN]
    for c in ["AGE_AT_ENCOUNTER", "ADMIT_DATE", "RECONSTRUCTION_DATE", "CPT_CODE", "PROCEDURE", "REASON_FOR_VISIT"]:
        if c in clinic.columns:
            keep_cols.append(c)

    clinic_small = clinic[keep_cols].copy()
    clinic_for_mismatch = clinic_small[clinic_small[MRN].isin(mismatch_mrns)].copy()

    summary = pd.merge(
        mismatch,
        clinic_for_mismatch,
        on=MRN,
        how="left"
    )

    print("Reconstructing notes...")
    notes_df = reconstruct_notes()

    if len(notes_df) > 0 and len(mismatch) > 0:
        qa_notes = pd.merge(
            mismatch[[MRN, "Age_pred", "Age_gold"]],
            notes_df,
            on=MRN,
            how="left"
        )
        qa_notes["NOTE_SNIPPET_PREVIEW"] = qa_notes["NOTE_TEXT"].fillna("").astype(str).str.slice(0, 500)
    else:
        qa_notes = pd.DataFrame(columns=[
            MRN, "Age_pred", "Age_gold",
            "NOTE_ID", "NOTE_TYPE", "NOTE_DATE", "SOURCE_FILE", "NOTE_TEXT", "NOTE_SNIPPET_PREVIEW"
        ])

    os.makedirs(os.path.join(BASE_DIR, "_outputs"), exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    qa_notes.to_csv(OUT_NOTES, index=False)

    print("Wrote:", OUT_SUMMARY)
    print("Wrote:", OUT_NOTES)


if __name__ == "__main__":
    main()
