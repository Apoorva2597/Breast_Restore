#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qa_age_mismatches.py

Builds a QA file for patients whose predicted Age does not match gold Age.
Outputs:
1) _outputs/qa_age_mismatches_summary.csv
2) _outputs/qa_age_mismatches_notes.csv

Python 3.6.8 compatible
"""

import os
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = os.path.join(BASE_DIR, "_outputs", "master_abstraction_rule_FINAL_NO_GOLD.csv")
GOLD_FILE = os.path.join(BASE_DIR, "gold_cleaned_for_cedar.csv")
EVID_FILE = os.path.join(BASE_DIR, "_outputs", "rule_hit_evidence_FINAL_NO_GOLD.csv")

NOTE_FILES = [
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Clinic Notes.csv"),
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Inpatient Notes.csv"),
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Operation Notes.csv"),
]

OUT_SUMMARY = os.path.join(BASE_DIR, "_outputs", "qa_age_mismatches_summary.csv")
OUT_NOTES = os.path.join(BASE_DIR, "_outputs", "qa_age_mismatches_notes.csv")

MRN = "MRN"


def safe_read_csv(path):
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=str, encoding="latin1")


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ["", "nan", "none", "null", "na"]:
        return ""
    return s


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
    print("Loading files...")
    master = clean_cols(safe_read_csv(MASTER_FILE))
    gold = clean_cols(safe_read_csv(GOLD_FILE))
    evid = clean_cols(safe_read_csv(EVID_FILE))

    master = normalize_mrn(master)
    gold = normalize_mrn(gold)
    evid = normalize_mrn(evid)

    master = master.drop_duplicates(subset=[MRN]).copy()
    gold = gold.drop_duplicates(subset=[MRN]).copy()

    merged = pd.merge(
        master[[MRN, "Age"]],
        gold[[MRN, "Age"]],
        on=MRN,
        how="inner",
        suffixes=("_pred", "_gold")
    )

    merged["Age_pred"] = merged["Age_pred"].astype(str).str.strip()
    merged["Age_gold"] = merged["Age_gold"].astype(str).str.strip()

    mism = merged[
        (merged["Age_gold"] != "") &
        (merged["Age_gold"].str.lower() != "nan") &
        (merged["Age_pred"] != merged["Age_gold"])
    ].copy()

    print("Age mismatches:", len(mism))

    age_evid = evid[evid["FIELD"].astype(str).str.strip() == "Age"].copy()

    # Pull structured age evidence fields out of EVIDENCE text
    def extract_piece(text, key):
        text = clean_cell(text)
        if not text:
            return ""
        marker = key + "="
        if marker not in text:
            return ""
        rhs = text.split(marker, 1)[1]
        return rhs.split("|", 1)[0].strip()

    if len(age_evid) > 0:
        age_evid["AGE_AT_ENCOUNTER_USED"] = age_evid["EVIDENCE"].apply(lambda x: extract_piece(x, "AGE_AT_ENCOUNTER"))
        age_evid["STRUCT_DATE_USED"] = age_evid["EVIDENCE"].apply(lambda x: extract_piece(x, "STRUCT_DATE"))
        age_evid["TARGET_DATE_USED"] = age_evid["EVIDENCE"].apply(lambda x: extract_piece(x, "TARGET_DATE"))
        age_evid["AGE_FLOOR"] = age_evid["EVIDENCE"].apply(lambda x: extract_piece(x, "AGE_FLOOR"))
        age_evid["AGE_ROUND"] = age_evid["EVIDENCE"].apply(lambda x: extract_piece(x, "AGE_ROUND"))
        age_evid["FINAL_USED"] = age_evid["EVIDENCE"].apply(lambda x: extract_piece(x, "FINAL_USED"))

    age_evid = age_evid.sort_values([MRN, "NOTE_DATE"]).drop_duplicates(subset=[MRN], keep="last")

    summary = pd.merge(
        mism,
        age_evid[[
            MRN, "NOTE_TYPE", "NOTE_DATE", "AGE_AT_ENCOUNTER_USED",
            "STRUCT_DATE_USED", "TARGET_DATE_USED",
            "AGE_FLOOR", "AGE_ROUND", "FINAL_USED", "EVIDENCE"
        ]],
        on=MRN,
        how="left"
    )

    summary = summary.rename(columns={
        "NOTE_TYPE": "AGE_EVID_NOTE_TYPE",
        "NOTE_DATE": "AGE_EVID_NOTE_DATE",
        "EVIDENCE": "AGE_EVIDENCE_TEXT"
    })

    print("Reconstructing notes...")
    notes_df = reconstruct_notes()

    if len(notes_df) > 0 and len(mism) > 0:
        qa_notes = pd.merge(
            mism[[MRN, "Age_pred", "Age_gold"]],
            notes_df,
            on=MRN,
            how="left"
        )

        # helpful short snippet preview
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
