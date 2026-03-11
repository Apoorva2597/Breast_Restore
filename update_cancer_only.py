#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_master_rule_CANCER_RECON_PATCH.py

Patches the existing master abstraction CSV with updated:
- Indication_Left
- Indication_Right
- LymphNode

Main updates:
1. Uses note-priority aggregation
2. ALND > SLNB
3. Side-local indication parsing
4. Contralateral prophylactic handling
5. Writes back into original master CSV path for validation
6. Appends evidence rows for QA

Python 3.6.8 compatible
"""

import os
import sys
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------
# Paths - adjust if needed
# ---------------------------------------------------

MASTER_PATH = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
EVID_PATH = "_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv"

# Original HPI note files - adjust as needed
NOTE_FILES = [
    "Clinic_Notes.csv",
    "Inpatient_Notes.csv",
    "Operation_Notes.csv"
]

MRN_COL_CANDIDATES = ["MRN", "PAT_MRN_ID", "mrn"]
NOTE_ID_CANDIDATES = ["NOTE_ID", "note_id", "DOCUMENT_ID", "doc_id"]
LINE_NUM_CANDIDATES = ["LINE", "LINE_NO", "line_num", "SEQ_NUM"]
TEXT_CANDIDATES = ["NOTE_TEXT", "TEXT", "line_text", "NOTE_LINE"]
DATE_CANDIDATES = ["NOTE_DATE", "SERVICE_DATE", "CONTACT_DATE", "ENC_DATE"]
TYPE_CANDIDATES = ["NOTE_TYPE", "NOTE_NAME", "DOCUMENT_TYPE", "TYPE"]

# ---------------------------------------------------
# Import extractor
# ---------------------------------------------------

try:
    from extractors.extract_cancer_recon import extract_cancer_recon_from_notes
except Exception:
    # fallback if run from same directory
    from extract_cancer_recon import extract_cancer_recon_from_notes


# ---------------------------------------------------
# CSV helpers
# ---------------------------------------------------

def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8", dtype=str)
    except Exception:
        return pd.read_csv(path, encoding="latin1", dtype=str)


def pick_col(df, candidates):
    cols = set(df.columns.tolist())
    for c in candidates:
        if c in cols:
            return c
    return None


def norm(x):
    if x is None:
        return ""
    return str(x).strip()


# ---------------------------------------------------
# Note loading / reconstruction
# ---------------------------------------------------

def load_and_reconstruct_notes(note_files):
    """
    Returns:
      dict[mrn] = list of note dicts
    """
    by_mrn = defaultdict(list)

    for path in note_files:
        if not os.path.exists(path):
            continue

        df = safe_read_csv(path)
        if df is None or df.empty:
            continue

        mrn_col = pick_col(df, MRN_COL_CANDIDATES)
        note_id_col = pick_col(df, NOTE_ID_CANDIDATES)
        line_col = pick_col(df, LINE_NUM_CANDIDATES)
        text_col = pick_col(df, TEXT_CANDIDATES)
        date_col = pick_col(df, DATE_CANDIDATES)
        type_col = pick_col(df, TYPE_CANDIDATES)

        if mrn_col is None or text_col is None:
            continue

        # If no NOTE_ID, each row becomes its own "note"
        if note_id_col is None:
            df["_TMP_NOTE_ID"] = range(len(df))
            note_id_col = "_TMP_NOTE_ID"

        if line_col is None:
            df["_TMP_LINE_NO"] = 0
            line_col = "_TMP_LINE_NO"

        use_cols = [mrn_col, note_id_col, line_col, text_col]
        if date_col is not None:
            use_cols.append(date_col)
        if type_col is not None:
            use_cols.append(type_col)

        work = df[use_cols].copy()
        work[mrn_col] = work[mrn_col].astype(str).str.strip()
        work[note_id_col] = work[note_id_col].astype(str).str.strip()
        work[line_col] = pd.to_numeric(work[line_col], errors="coerce").fillna(0)

        grouped = work.sort_values([mrn_col, note_id_col, line_col]).groupby([mrn_col, note_id_col], dropna=False)

        for (mrn, note_id), g in grouped:
            lines = g[text_col].fillna("").astype(str).tolist()
            note_text = "\n".join(lines).strip()

            note_date = ""
            if date_col is not None and date_col in g.columns:
                vals = g[date_col].dropna().astype(str).tolist()
                if vals:
                    note_date = vals[0]

            note_type = ""
            if type_col is not None and type_col in g.columns:
                vals = g[type_col].dropna().astype(str).tolist()
                if vals:
                    note_type = vals[0]

            note_rec = {
                "mrn": norm(mrn),
                "note_id": norm(note_id),
                "note_date": norm(note_date),
                "note_type": norm(note_type),
                "section": "",
                "text": note_text
            }

            by_mrn[norm(mrn)].append(note_rec)

    return by_mrn


# ---------------------------------------------------
# Evidence helpers
# ---------------------------------------------------

def make_evidence_rows(mrn, extracted):
    rows = []

    for var in ["Indication_Left", "Indication_Right", "LymphNode"]:
        evid_key = var + "_Evidence"
        rows.append({
            "MRN": mrn,
            "Variable": var,
            "PredictedValue": extracted.get(var, ""),
            "EvidenceText": extracted.get(evid_key, "")
        })

    return rows


# ---------------------------------------------------
# Main patch logic
# ---------------------------------------------------

def ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def main():
    if not os.path.exists(MASTER_PATH):
        print("ERROR: Master file not found: {0}".format(MASTER_PATH))
        sys.exit(1)

    master = safe_read_csv(MASTER_PATH)
    if master is None or master.empty:
        print("ERROR: Master file is empty or unreadable.")
        sys.exit(1)

    if "MRN" not in master.columns:
        print("ERROR: MRN column not found in master.")
        sys.exit(1)

    master["MRN"] = master["MRN"].astype(str).str.strip()

    master = ensure_columns(master, [
        "Indication_Left",
        "Indication_Right",
        "LymphNode"
    ])

    notes_by_mrn = load_and_reconstruct_notes(NOTE_FILES)

    evidence_rows = []

    total = len(master)
    done = 0

    for idx in master.index:
        mrn = norm(master.at[idx, "MRN"])
        notes = notes_by_mrn.get(mrn, [])

        if not notes:
            done += 1
            continue

        extracted = extract_cancer_recon_from_notes(notes)

        master.at[idx, "Indication_Left"] = extracted.get("Indication_Left", master.at[idx, "Indication_Left"])
        master.at[idx, "Indication_Right"] = extracted.get("Indication_Right", master.at[idx, "Indication_Right"])
        master.at[idx, "LymphNode"] = extracted.get("LymphNode", master.at[idx, "LymphNode"])

        evidence_rows.extend(make_evidence_rows(mrn, extracted))

        done += 1
        if done % 100 == 0:
            print("Processed {0}/{1}".format(done, total))

    # Save patched master in original validation path
    os.makedirs(os.path.dirname(MASTER_PATH), exist_ok=True)
    master.to_csv(MASTER_PATH, index=False)

    # Append or write evidence
    evid_df = pd.DataFrame(evidence_rows)

    if os.path.exists(EVID_PATH):
        old = safe_read_csv(EVID_PATH)
        if old is not None and not old.empty:
            evid_df = pd.concat([old, evid_df], ignore_index=True, sort=False)

    os.makedirs(os.path.dirname(EVID_PATH), exist_ok=True)
    evid_df.to_csv(EVID_PATH, index=False)

    print("\nDONE.")
    print("- Patched existing master: {0}".format(MASTER_PATH))
    print("- Appended evidence: {0}".format(EVID_PATH))
    print("\nRun:")
    print(" python build_master_rule_CANCER_RECON_PATCH.py")


if __name__ == "__main__":
    main()
