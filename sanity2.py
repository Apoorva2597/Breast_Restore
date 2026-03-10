#!/usr/bin/env python3
# qa_no_evidence_full_note_grep.py
#
# For smoking mismatches categorized as "no_evidence_row",
# search ALL reconstructed full notes (entire note text, not sections)
# using a tighter phrase-based smoking/tobacco pattern list.
#
# Outputs:
#   _outputs/qa_no_evidence_full_note_grep_hits.csv
#   _outputs/qa_no_evidence_full_note_grep_summary.csv
#
# Python 3.6.8 compatible

import os
import re
from glob import glob
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"
MERGE_KEY = "MRN"

QA_FILE = "_outputs/qa_smoking_mismatches_categorized.csv"
OUT_HITS = "_outputs/qa_no_evidence_full_note_grep_hits.csv"
OUT_SUMMARY = "_outputs/qa_no_evidence_full_note_grep_summary.csv"

NOTE_GLOBS = [
    "{0}/**/HPI11526*Clinic Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation notes.csv".format(BASE_DIR),
]

# Tighter phrase-based smoking pattern list.
# Intentionally avoids loose stems like "quit", "smok", "tob", "pack"
# by themselves because they produced too many false positives.
SMOKING_RX = re.compile(
    r"("
    r"\bcurrent smoker\b|"
    r"\bformer smoker\b|"
    r"\bnever smoker\b|"
    r"\bcurrent every day smoker\b|"
    r"\bcurrent some day smoker\b|"
    r"\bevery day smoker\b|"
    r"\bsome day smoker\b|"
    r"\bnonsmoker\b|"
    r"\bnon[- ]smoker\b|"
    r"\bsmoking status\b|"
    r"\btobacco use\b|"
    r"\bsmokeless tobacco\b|"
    r"\bquit smoking\b|"
    r"\bstopped smoking\b|"
    r"\bquit date\b|"
    r"\byears since quitting\b|"
    r"\blast attempt to quit\b|"
    r"\bpack years?\b|"
    r"\bpacks?/day\b|"
    r"\bcigarettes?\b|"
    r"\bdenies smoking\b|"
    r"\bdenies tobacco\b|"
    r"\bdenies tobacco use\b|"
    r"\bdenies use of tobacco products\b|"
    r"\bdenies use of tobacco\b|"
    r"\bdenies tobacco or alcohol use\b|"
    r"\bdenies use of tobacco,\s*alcohol\s*or\s*recreational drug use\b|"
    r"\bdenies use of tobacco,\s*alcohol\s*or\s*illicit drug use\b|"
    r"\bdoes not smoke\b|"
    r"\bdoes not smoke or use nicotine\b|"
    r"\bdoes not drink alcohol or smoke\b|"
    r"\bdoes not drink alcohol or use tobacco\b|"
    r"\bno smoking\b|"
    r"\bno tobacco use\b|"
    r"\bnever used tobacco\b|"
    r"\bno history of tobacco\b|"
    r"\bno history of tobacco use\b|"
    r"\bdoes not smoke or use nicotine\b|"
    r"\bsmokes?\s+(?:a\s+)?(?:couple|few)\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b|"
    r"\bsmokes?\s+\d+(?:\.\d+)?\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b|"
    r"\bsmokes?\s+\d+(?:\.\d+)?\s*packs?\s*/?\s*(?:day|week)\b|"
    r"\bsmokes?\s+every\s+once\s+in\s+a\s+while\b|"
    r"\bstill smoking\b|"
    r"\bcontinues to smoke\b|"
    r"\bcurrently smoking\b|"
    r"\bcurrently smokes\b"
    r")",
    re.IGNORECASE
)


def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                error_bad_lines=False,
                warn_bad_lines=True
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                on_bad_lines="skip"
            )
        except TypeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )

def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s

def normalize_mrn(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MERGE_KEY:
                df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError("MRN column not found. Columns seen: {0}".format(list(df.columns)[:40]))
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df

def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError("Required column missing. Tried={0}. Seen={1}".format(
            options, list(df.columns)[:60]
        ))
    return None

def to_int_safe(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def window_around(text, start, end, width=140):
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].replace("\n", " ").replace("\r", " ").strip()

def load_and_reconstruct_notes():
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))

    if not note_files:
        raise FileNotFoundError("No HPI11526 * Notes.csv files found via NOTE_GLOBS.")

    all_notes_rows = []

    for fp in note_files:
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)

        note_text_col = pick_col(df, ["NOTE_TEXT", "NOTE TEXT", "NOTE_TEXT_FULL", "TEXT", "NOTE"])
        note_id_col = pick_col(df, ["NOTE_ID", "NOTE ID"])
        line_col = pick_col(df, ["LINE"], required=False)
        note_type_col = pick_col(df, ["NOTE_TYPE", "NOTE TYPE"], required=False)
        date_col = pick_col(
            df,
            ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE", "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"],
            required=False
        )

        df[note_text_col] = df[note_text_col].fillna("").astype(str)
        df[note_id_col] = df[note_id_col].fillna("").astype(str)
        if line_col:
            df[line_col] = df[line_col].fillna("").astype(str)
        if note_type_col:
            df[note_type_col] = df[note_type_col].fillna("").astype(str)
        if date_col:
            df[date_col] = df[date_col].fillna("").astype(str)

        df["_SOURCE_FILE_"] = os.path.basename(fp)

        keep_cols = [MERGE_KEY, note_id_col, note_text_col, "_SOURCE_FILE_"]
        if line_col:
            keep_cols.append(line_col)
        if note_type_col:
            keep_cols.append(note_type_col)
        if date_col:
            keep_cols.append(date_col)

        tmp = df[keep_cols].copy()
        tmp = tmp.rename(columns={
            note_id_col: "NOTE_ID",
            note_text_col: "NOTE_TEXT",
        })

        if line_col and line_col != "LINE":
            tmp = tmp.rename(columns={line_col: "LINE"})
        if note_type_col and note_type_col != "NOTE_TYPE":
            tmp = tmp.rename(columns={note_type_col: "NOTE_TYPE"})
        if date_col and date_col != "NOTE_DATE_OF_SERVICE":
            tmp = tmp.rename(columns={date_col: "NOTE_DATE_OF_SERVICE"})

        if "LINE" not in tmp.columns:
            tmp["LINE"] = ""
        if "NOTE_TYPE" not in tmp.columns:
            tmp["NOTE_TYPE"] = ""
        if "NOTE_DATE_OF_SERVICE" not in tmp.columns:
            tmp["NOTE_DATE_OF_SERVICE"] = ""

        all_notes_rows.append(tmp)

    notes_raw = pd.concat(all_notes_rows, ignore_index=True)

    def join_note(group):
        tmp = group.copy()
        tmp["_LINE_NUM_"] = tmp["LINE"].apply(to_int_safe)
        tmp = tmp.sort_values(by=["_LINE_NUM_"], na_position="last")
        return "\n".join(tmp["NOTE_TEXT"].tolist()).strip()

    reconstructed = []
    grouped = notes_raw.groupby([MERGE_KEY, "NOTE_ID"], dropna=False)

    for (mrn, nid), g in grouped:
        mrn = str(mrn).strip()
        nid = str(nid).strip()
        if not nid:
            continue

        full_text = join_note(g)
        if not full_text:
            continue

        if g["NOTE_TYPE"].astype(str).str.strip().any():
            note_type = g["NOTE_TYPE"].astype(str).iloc[0]
        else:
            note_type = g["_SOURCE_FILE_"].astype(str).iloc[0]

        if g["NOTE_DATE_OF_SERVICE"].astype(str).str.strip().any():
            note_date = g["NOTE_DATE_OF_SERVICE"].astype(str).iloc[0]
        else:
            note_date = ""

        reconstructed.append({
            MERGE_KEY: mrn,
            "NOTE_ID": nid,
            "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date,
            "SOURCE_FILE": g["_SOURCE_FILE_"].astype(str).iloc[0],
            "NOTE_TEXT": full_text
        })

    return pd.DataFrame(reconstructed)


def main():
    print("Loading no-evidence QA file...")
    qa = read_csv_robust(QA_FILE)
    qa = clean_cols(qa)
    qa = normalize_mrn(qa)

    if "Mismatch_Category" not in qa.columns:
        raise RuntimeError("Mismatch_Category column not found in QA file.")

    target = qa[qa["Mismatch_Category"].astype(str).str.strip() == "no_evidence_row"].copy()
    target_mrns = sorted(set(target[MERGE_KEY].astype(str).str.strip().tolist()))

    print("No-evidence MRNs found: {0}".format(len(target_mrns)))

    print("Loading and reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    notes_df = notes_df[notes_df[MERGE_KEY].astype(str).str.strip().isin(target_mrns)].copy()
    print("Notes belonging to no-evidence MRNs: {0}".format(len(notes_df)))

    hit_rows = []
    summary_rows = []

    for mrn in target_mrns:
        patient_notes = notes_df[notes_df[MERGE_KEY].astype(str).str.strip() == mrn].copy()

        total_notes = len(patient_notes)
        total_hits = 0

        if total_notes == 0:
            summary_rows.append({
                "MRN": mrn,
                "Total_Notes": 0,
                "Notes_With_Grep_Hits": 0,
                "Total_Grep_Hits": 0,
                "Result": "NO_NOTES_FOUND"
            })
            continue

        notes_with_hits = 0

        for _, row in patient_notes.iterrows():
            text = clean_cell(row.get("NOTE_TEXT", ""))
            if not text:
                continue

            matches = list(SMOKING_RX.finditer(text))
            if not matches:
                continue

            notes_with_hits += 1
            total_hits += len(matches)

            for m in matches:
                hit_rows.append({
                    "MRN": mrn,
                    "NOTE_ID": clean_cell(row.get("NOTE_ID", "")),
                    "NOTE_DATE": clean_cell(row.get("NOTE_DATE", "")),
                    "NOTE_TYPE": clean_cell(row.get("NOTE_TYPE", "")),
                    "SOURCE_FILE": clean_cell(row.get("SOURCE_FILE", "")),
                    "MATCH_TEXT": m.group(0),
                    "MATCH_START": m.start(),
                    "CONTEXT": window_around(text, m.start(), m.end(), 160)
                })

        result = "GREP_HITS_FOUND" if total_hits > 0 else "NO_GREP_HITS_IN_FULL_NOTES"

        summary_rows.append({
            "MRN": mrn,
            "Total_Notes": total_notes,
            "Notes_With_Grep_Hits": notes_with_hits,
            "Total_Grep_Hits": total_hits,
            "Result": result
        })

    hits_df = pd.DataFrame(hit_rows)
    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(os.path.dirname(OUT_HITS), exist_ok=True)
    hits_df.to_csv(OUT_HITS, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved hit-level results to: {0}".format(OUT_HITS))
    print("Saved summary results to: {0}".format(OUT_SUMMARY))

    if len(summary_df) > 0:
        print("\nSummary counts:")
        print(summary_df["Result"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
