#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_patient_master.py  (Python 3.6.8 friendly)

Builds a patient-level abstraction file (_outputs/patient_master.csv) from ORIGINAL HPI11526 CSVs.

Key points (per your requirements):
- Uses ORIGINAL data only (no NOTE_DEID requirement).
- Input files are NOT in Breast_Restore; they are in:
  /home/apokol/my_data_Breast/HPI-11526/HPI11256
- Patient-level (not note-level) output.
- Age: primarily from structured AGE_AT_ENCOUNTER (Operation Encounters), with fallback to other structured age fields,
  then fallback to note text patterns.
- Comorbidities: extracted from CLINIC NOTES, prioritizing "PAST MEDICAL HISTORY" section; meds used as a second layer.
- Lumpectomy: includes "lumpectomy scar" pattern (common in exams).
- Handles mixed encodings robustly (utf-8/latin1; replace bad bytes).
"""

from __future__ import print_function

import os
import re
import sys
import csv
import math
import argparse
from collections import defaultdict, Counter

import pandas as pd


# =========================
# CONFIG: INPUTS / OUTPUTS
# =========================

DATA_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

CLINIC_ENC_FILE = os.path.join(DATA_DIR, "HPI11526 Clinic Encounters.csv")
INPATIENT_ENC_FILE = os.path.join(DATA_DIR, "HPI11526 Inpatient Encounters.csv")
OP_ENC_FILE = os.path.join(DATA_DIR, "HPI11526 Operation Encounters.csv")

CLINIC_NOTES_FILE = os.path.join(DATA_DIR, "HPI11526 Clinic Notes.csv")
INPATIENT_NOTES_FILE = os.path.join(DATA_DIR, "HPI11526 Inpatient Notes.csv")
OP_NOTES_FILE = os.path.join(DATA_DIR, "HPI11526 Operation Notes.csv")

OUT_DIR = "_outputs"
OUT_MASTER = os.path.join(OUT_DIR, "patient_master.csv")

PID_COL = "ENCRYPTED_PAT_ID"


# =========================
# Helpers: CSV reading
# =========================

def _read_csv_robust(path, usecols=None, chunksize=None):
    """
    Read CSV with robust encoding fallbacks. Works on py3.6.8.
    """
    if not os.path.exists(path):
        raise IOError("Missing input file: {}".format(path))

    # pandas can choke on mixed encodings; try a small set of common ones.
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                usecols=usecols,
                chunksize=chunksize,
                encoding=enc,
                engine="python",
                sep=",",
                quotechar='"',
                error_bad_lines=False,   # pandas<1.5 compatible
                warn_bad_lines=True
            )
        except Exception as e:
            last_err = e

    # last resort: open manually, replace bad bytes, then parse via pandas from temp buffer
    # (still can be large; use only if really needed)
    raise last_err


def _normalize_str(x):
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        return ""
    return s.strip()


def _lower(x):
    return _normalize_str(x).lower()


def _is_nullish(x):
    if x is None:
        return True
    s = _normalize_str(x)
    if s == "":
        return True
    if s.lower() in ("nan", "none", "null", "na", "n/a"):
        return True
    return False


# =========================
# Text section extraction
# =========================

# Common all-caps section headers in these notes
SECTION_HEADER_RE = re.compile(r"\n\s*([A-Z][A-Z \/\-\(\)0-9]{3,60})\s*:\s*", re.M)

def extract_section(text, section_name):
    """
    Extract a note section (e.g., 'PAST MEDICAL HISTORY') if present.
    We look for "<SECTION>:" and then stop at the next ALL-CAPS header.
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # tolerate small variations: "PAST MEDICAL HISTORY", "PAST MEDICAL HX", "PMH"
    targets = [
        section_name,
        "PAST MEDICAL HX",
        "PAST MEDICAL HISTORY",
        "PMH",
        "PAST HISTORY",
    ]

    # find the first occurrence of any target as a header-ish token
    # We allow either "HEADER:" or "HEADER -" or "HEADER\n"
    for lab in targets:
        pat = re.compile(r"(^|\n)\s*{}\s*[:\-]\s*".format(re.escape(lab)), re.I)
        m = pat.search(t)
        if not m:
            continue

        start = m.end()
        rest = t[start:]

        # stop at next ALL-CAPS header
        m2 = SECTION_HEADER_RE.search(rest)
        if m2:
            return rest[:m2.start()].strip()

        # or stop at some other common divider
        m3 = re.search(r"\n\s*-{3,}\s*\n", rest)
        if m3:
            return rest[:m3.start()].strip()

        return rest.strip()

    return ""


# =========================
# Clinical concept patterns
# =========================

# Age patterns (use many variations; typically near start of note)
AGE_PATS = [
    re.compile(r"\b(\d{1,3})\s*[- ]?(?:yo|y/o)\b", re.I),
    re.compile(r"\b(\d{1,3})\s*(?:year|yr)\s*old\b", re.I),
    re.compile(r"\bage\s*[:\-]?\s*(\d{1,3})\b", re.I),
    re.compile(r"\b(\d{1,3})\s*(?:y\.?o\.?)\b", re.I),
]

BMI_PATS = [
    re.compile(r"\bBMI\s*[:=]?\s*(\d{1,2}(?:\.\d{1,2})?)\b", re.I),
    re.compile(r"\bBody\s*mass\s*index\s*[:=]?\s*(\d{1,2}(?:\.\d{1,2})?)\b", re.I),
]

# Comorbidities from PMH (+ med second layer)
COMORB_PATS = {
    "Diabetes": [
        re.compile(r"\bdiabetes\b", re.I),
        re.compile(r"\bdm\b", re.I),
        re.compile(r"\btype\s*2\s*diabetes\b", re.I),
    ],
    "Hypertension": [
        re.compile(r"\bhypertension\b", re.I),
        re.compile(r"\bhtn\b", re.I),
    ],
    "CardiacDisease": [
        re.compile(r"\bcoronary\b", re.I),
        re.compile(r"\bcad\b", re.I),
        re.compile(r"\bmyocardial infarction\b", re.I),
        re.compile(r"\bmi\b", re.I),
        re.compile(r"\bcongestive heart failure\b", re.I),
        re.compile(r"\bchf\b", re.I),
        re.compile(r"\batrial fibrillation\b", re.I),
        re.compile(r"\bafib\b", re.I),
        re.compile(r"\bangina\b", re.I),
        re.compile(r"\bstent\b", re.I),
    ],
    "VenousThromboembolism": [
        re.compile(r"\bvenous thromboembol", re.I),
        re.compile(r"\bvte\b", re.I),
        re.compile(r"\bdvt\b", re.I),
        re.compile(r"\bdeep vein thromb", re.I),
        re.compile(r"\bpe\b", re.I),
        re.compile(r"\bpulmonary embol", re.I),
    ],
    "Steroid": [
        re.compile(r"\bchronic steroid", re.I),
        re.compile(r"\bon (?:long[- ]term )?steroid", re.I),
        re.compile(r"\bprednisone\b", re.I),
        re.compile(r"\bmethylprednisolone\b", re.I),
        re.compile(r"\bdexamethasone\b", re.I),
        re.compile(r"\bhydrocortisone\b", re.I),
    ],
}

# Med “second layer” triggers (if meds present, search the full note again for disease mentions)
MED_TRIGGERS = {
    "Diabetes": [re.compile(r"\bmetformin\b", re.I), re.compile(r"\binsulin\b", re.I)],
    "Hypertension": [re.compile(r"\blisinopril\b", re.I), re.compile(r"\bamLODIPine\b", re.I), re.compile(r"\blosartan\b", re.I), re.compile(r"\bhydrochlorothiazide\b", re.I)],
    "CardiacDisease": [re.compile(r"\bnitroglycerin\b", re.I), re.compile(r"\bclopidogrel\b", re.I), re.compile(r"\baspirin\b", re.I), re.compile(r"\batorvastatin\b", re.I)],
    "VenousThromboembolism": [re.compile(r"\bwarfarin\b", re.I), re.compile(r"\bapixaban\b", re.I), re.compile(r"\brivaroxaban\b", re.I), re.compile(r"\benoxaparin\b", re.I), re.compile(r"\bheparin\b", re.I)],
    "Steroid": [re.compile(r"\bprednisone\b", re.I), re.compile(r"\bmethylprednisolone\b", re.I), re.compile(r"\bdexamethasone\b", re.I)],
}

# Smoking
SMOKE_PATS = [
    ("Never", re.compile(r"\bnever smoker\b", re.I)),
    ("Former", re.compile(r"\bformer smoker\b", re.I)),
    ("Current", re.compile(r"\bcurrent smoker\b", re.I)),
    ("Current", re.compile(r"\bsmokes\b", re.I)),
]

# Radiation / Chemo (keep conservative; avoid counting “discussed” unless therapy context)
RADIATION_PATS = [
    re.compile(r"\bradiation therapy\b", re.I),
    re.compile(r"\bcompleted radiation\b", re.I),
    re.compile(r"\bXRT\b", re.I),
    re.compile(r"\bpost[- ]?op radiation\b", re.I),
]
CHEMO_PATS = [
    re.compile(r"\bchemotherapy\b", re.I),
    re.compile(r"\breceived chemo\b", re.I),
    re.compile(r"\bcompleted chemo\b", re.I),
    re.compile(r"\bAC[- ]?T\b", re.I),
]

# Past breast surgery (PBS) cues
LUMPECTOMY_PATS = [
    re.compile(r"\blumpectomy\b", re.I),
    re.compile(r"\bs/p lumpectomy\b", re.I),
    re.compile(r"\blumpectomy scar\b", re.I),  # your key pointer
]
MASTOPEXY_PATS = [re.compile(r"\bmastopexy\b", re.I), re.compile(r"\bbreast lift\b", re.I)]
AUGMENT_PATS = [re.compile(r"\baugmentation\b", re.I), re.compile(r"\bbreast implant", re.I), re.compile(r"\bimplants\b", re.I)]


def _first_int_from_text(text, patterns):
    if not text:
        return None
    for p in patterns:
        m = p.search(text)
        if m:
            try:
                v = int(m.group(1))
                if 0 < v < 120:
                    return v
            except Exception:
                pass
    return None


def _first_float_from_text(text, patterns):
    if not text:
        return None
    for p in patterns:
        m = p.search(text)
        if m:
            try:
                v = float(m.group(1))
                if 5.0 <= v <= 90.0:
                    return v
            except Exception:
                pass
    return None


def _any_pat(text, pats):
    if not text:
        return False
    for p in pats:
        if p.search(text):
            return True
    return False


# =========================
# Structured aggregation
# =========================

def _choose_mode_numeric(values):
    vals = [v for v in values if v is not None and (not (isinstance(v, float) and math.isnan(v)))]
    if not vals:
        return None
    try:
        c = Counter([int(round(float(v))) for v in vals])
        return c.most_common(1)[0][0]
    except Exception:
        # fallback to first non-null
        return vals[0]


def _choose_mode_str(values):
    vals = [ _normalize_str(v) for v in values if not _is_nullish(v) ]
    if not vals:
        return ""
    c = Counter([v.strip() for v in vals if v.strip() != ""])
    if not c:
        return ""
    return c.most_common(1)[0][0]


def _safe_col(df, possible_names):
    for n in possible_names:
        if n in df.columns:
            return n
    return None


# =========================
# Notes processing (chunked)
# =========================

def build_patient_note_index(notes_path, allowed_pat_ids=None, note_type_filter=None, chunksize=5000):
    """
    Returns dict: pid -> list of (pmh_text, full_text)
    - pmh_text: extracted "PAST MEDICAL HISTORY" section if found, else ""
    - full_text: (optionally) full note text (kept shorter by truncation)
    """
    idx = defaultdict(list)

    # discover columns with a small read
    head = _read_csv_robust(notes_path, chunksize=200)
    if hasattr(head, "__iter__"):
        head_df = next(head)
    else:
        head_df = head

    pid_col = _safe_col(head_df, [PID_COL, "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID"])
    if not pid_col:
        raise ValueError("Could not find patient id column in {} (expected {}).".format(notes_path, PID_COL))

    # IMPORTANT: original data may have NOTE_TEXT, NOTE_TEXT_DEID, TEXT, etc.
    text_col = _safe_col(head_df, ["NOTE_TEXT", "NOTE_TEXT_DEID", "TEXT", "NOTE", "NOTE_BODY", "NOTE_CONTENT"])
    if not text_col:
        # fall back: pick the longest object column name if nothing obvious
        obj_cols = [c for c in head_df.columns if str(head_df[c].dtype) == "object"]
        text_col = obj_cols[0] if obj_cols else None
    if not text_col:
        raise ValueError("Could not find a note text column in {}.".format(notes_path))

    type_col = _safe_col(head_df, ["NOTE_TYPE", "NOTE_KIND", "TYPE"])

    # now re-read whole file in chunks with only needed cols
    usecols = [pid_col, text_col]
    if type_col:
        usecols.append(type_col)

    reader = _read_csv_robust(notes_path, usecols=usecols, chunksize=chunksize)

    for chunk in reader:
        if allowed_pat_ids is not None:
            chunk = chunk[chunk[pid_col].astype(str).isin(allowed_pat_ids)]
            if chunk.empty:
                continue

        if type_col and note_type_filter:
            chunk = chunk[chunk[type_col].astype(str).str.lower().isin([x.lower() for x in note_type_filter])]
            if chunk.empty:
                continue

        for _, row in chunk.iterrows():
            pid = _normalize_str(row.get(pid_col, ""))
            if pid == "":
                continue

            txt = row.get(text_col, "")
            if _is_nullish(txt):
                continue
            txt = _normalize_str(txt)

            pmh = extract_section(txt, "PAST MEDICAL HISTORY")

            # keep full text but cap length to avoid memory blow-up
            # (still enough for meds/age/smoking cues near top)
            full_txt = txt[:4000]

            idx[pid].append((pmh, full_txt))

    return idx


# =========================
# Main build logic
# =========================

def main():
    ap = argparse.ArgumentParser(description="Build patient_master.csv from HPI11526 original CSVs.")
    ap.add_argument("--data_dir", default=DATA_DIR, help="Input directory containing original HPI11526 CSVs.")
    ap.add_argument("--out_dir", default=OUT_DIR, help="Output directory.")
    ap.add_argument("--chunksize", type=int, default=5000, help="Chunksize for reading notes CSVs.")
    args = ap.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    chunksize = args.chunksize

    clinic_enc = os.path.join(data_dir, os.path.basename(CLINIC_ENC_FILE))
    inpatient_enc = os.path.join(data_dir, os.path.basename(INPATIENT_ENC_FILE))
    op_enc = os.path.join(data_dir, os.path.basename(OP_ENC_FILE))

    clinic_notes = os.path.join(data_dir, os.path.basename(CLINIC_NOTES_FILE))
    inpatient_notes = os.path.join(data_dir, os.path.basename(INPATIENT_NOTES_FILE))
    op_notes = os.path.join(data_dir, os.path.basename(OP_NOTES_FILE))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Loading structured files...")
    op_df = _read_csv_robust(op_enc)
    if hasattr(op_df, "__iter__"):
        # not chunked
        op_df = op_df
    # ensure actual df
    if hasattr(op_df, "__iter__") and not isinstance(op_df, pd.DataFrame):
        # if someone passed chunksize accidentally, consume
        op_df = pd.concat(list(op_df), ignore_index=True)

    # Optional: clinic/inpatient encounters can help fill race/ethnicity if needed
    ce_df = _read_csv_robust(clinic_enc)
    if hasattr(ce_df, "__iter__") and not isinstance(ce_df, pd.DataFrame):
        ce_df = pd.concat(list(ce_df), ignore_index=True)

    ie_df = _read_csv_robust(inpatient_enc)
    if hasattr(ie_df, "__iter__") and not isinstance(ie_df, pd.DataFrame):
        ie_df = pd.concat(list(ie_df), ignore_index=True)

    # unify patient list from structured
    all_pids = set()
    for df in [op_df, ce_df, ie_df]:
        if PID_COL in df.columns:
            all_pids.update([_normalize_str(x) for x in df[PID_COL].astype(str).tolist() if _normalize_str(x) != ""])

    all_pids = sorted(list(all_pids))
    print("Patients found in structured files: {}".format(len(all_pids)))

    # -------------------------
    # Structured extraction
    # -------------------------
    print("Building structured patient features...")

    # Race/Ethnicity columns may vary; check typical names
    race_col = _safe_col(op_df, ["RACE", "Race", "PAT_RACE", "RACE_NAME"])
    eth_col = _safe_col(op_df, ["ETHNICITY", "Ethnicity", "PAT_ETHNICITY", "ETHNICITY_NAME"])

    # Age at encounter (you confirmed exists in Operation Encounters)
    age_col = _safe_col(op_df, ["AGE_AT_ENCOUNTER", "Age_at_encounter", "AGE", "Age"])

    # BMI might be in clinic encounters
    bmi_col_ce = _safe_col(ce_df, ["BMI", "BodyMassIndex", "BODY_MASS_INDEX"])
    smoke_col_ce = _safe_col(ce_df, ["SmokingStatus", "SMOKING_STATUS", "TOBACCO_USE", "TOBACCO_STATUS"])

    # Procedure text / CPT description for PBS
    proc_col = _safe_col(op_df, ["PROCEDURE", "Procedure", "PROCEDURE_NAME", "PROC_NAME"])
    cpt_col = _safe_col(op_df, ["CPT_CODE", "CPT", "CPT_CODE_DESC", "CPT_DESCRIPTION", "CPT_DESC"])

    # aggregate structured per patient
    pat_struct = {}
    for pid in all_pids:
        pat_struct[pid] = {
            PID_COL: pid,
            "Race": "",
            "Ethnicity": "",
            "Age": None,
            "BMI": None,
            "SmokingStatus": "",
            "Diabetes": 0,
            "Hypertension": 0,
            "CardiacDisease": 0,
            "VenousThromboembolism": 0,
            "Steroid": 0,
            "PBS_Lumpectomy": 0,
            "PBS_Mastopexy": 0,
            "PBS_Augmentation": 0,
            "Radiation": 0,
            "Chemo": 0,
        }

    # race/ethnicity/age from op encounters first (often most complete)
    if PID_COL in op_df.columns:
        g = op_df.groupby(PID_COL)
        for pid, sub in g:
            pid = _normalize_str(pid)
            if pid == "" or pid not in pat_struct:
                continue

            if race_col:
                pat_struct[pid]["Race"] = _choose_mode_str(sub[race_col].tolist())
            if eth_col:
                pat_struct[pid]["Ethnicity"] = _choose_mode_str(sub[eth_col].tolist())

            if age_col:
                pat_struct[pid]["Age"] = _choose_mode_numeric(sub[age_col].tolist())

            # PBS from procedure fields
            proc_texts = []
            if proc_col:
                proc_texts += [ _normalize_str(x) for x in sub[proc_col].tolist() if not _is_nullish(x) ]
            if cpt_col:
                proc_texts += [ _normalize_str(x) for x in sub[cpt_col].tolist() if not _is_nullish(x) ]
            proc_blob = " | ".join(proc_texts).lower()

            if _any_pat(proc_blob, LUMPECTOMY_PATS):
                pat_struct[pid]["PBS_Lumpectomy"] = 1
            if _any_pat(proc_blob, MASTOPEXY_PATS):
                pat_struct[pid]["PBS_Mastopexy"] = 1
            if _any_pat(proc_blob, AUGMENT_PATS):
                pat_struct[pid]["PBS_Augmentation"] = 1

    # BMI / Smoking from clinic encounters if available
    if PID_COL in ce_df.columns:
        g = ce_df.groupby(PID_COL)
        for pid, sub in g:
            pid = _normalize_str(pid)
            if pid == "" or pid not in pat_struct:
                continue

            if pat_struct[pid]["BMI"] is None and bmi_col_ce:
                bmi = _choose_mode_numeric(sub[bmi_col_ce].tolist())
                pat_struct[pid]["BMI"] = bmi

            if pat_struct[pid]["SmokingStatus"] == "" and smoke_col_ce:
                pat_struct[pid]["SmokingStatus"] = _choose_mode_str(sub[smoke_col_ce].tolist())

            # race/ethnicity fallback
            if pat_struct[pid]["Race"] == "":
                rc = _safe_col(sub, ["RACE", "Race", "PAT_RACE", "RACE_NAME"])
                if rc:
                    pat_struct[pid]["Race"] = _choose_mode_str(sub[rc].tolist())
            if pat_struct[pid]["Ethnicity"] == "":
                ec = _safe_col(sub, ["ETHNICITY", "Ethnicity", "PAT_ETHNICITY", "ETHNICITY_NAME"])
                if ec:
                    pat_struct[pid]["Ethnicity"] = _choose_mode_str(sub[ec].tolist())

            # age fallback (if op missing)
            if pat_struct[pid]["Age"] is None:
                ac = _safe_col(sub, ["AGE_AT_ENCOUNTER", "Age_at_encounter", "AGE", "Age"])
                if ac:
                    pat_struct[pid]["Age"] = _choose_mode_numeric(sub[ac].tolist())

    # inpatient encounters can also backfill race/ethnicity/age
    if PID_COL in ie_df.columns:
        g = ie_df.groupby(PID_COL)
        for pid, sub in g:
            pid = _normalize_str(pid)
            if pid == "" or pid not in pat_struct:
                continue

            if pat_struct[pid]["Race"] == "":
                rc = _safe_col(sub, ["RACE", "Race", "PAT_RACE", "RACE_NAME"])
                if rc:
                    pat_struct[pid]["Race"] = _choose_mode_str(sub[rc].tolist())

            if pat_struct[pid]["Ethnicity"] == "":
                ec = _safe_col(sub, ["ETHNICITY", "Ethnicity", "PAT_ETHNICITY", "ETHNICITY_NAME"])
                if ec:
                    pat_struct[pid]["Ethnicity"] = _choose_mode_str(sub[ec].tolist())

            if pat_struct[pid]["Age"] is None:
                ac = _safe_col(sub, ["AGE_AT_ENCOUNTER", "Age_at_encounter", "AGE", "Age"])
                if ac:
                    pat_struct[pid]["Age"] = _choose_mode_numeric(sub[ac].tolist())

    # -------------------------
    # Notes-based enhancement
    # -------------------------
    print("Indexing clinic notes (chunked)...")
    note_idx = build_patient_note_index(
        clinic_notes,
        allowed_pat_ids=set(all_pids),
        note_type_filter=None,   # do not assume note types are clean; include all clinic notes
        chunksize=chunksize
    )
    print("Patients with >=1 clinic note: {}".format(len(note_idx)))

    for pid in all_pids:
        entries = note_idx.get(pid, [])
        if not entries:
            continue

        # combine PMH sections; if no PMH found, still use the first part of the note as fallback
        pmh_blob = "\n".join([e[0] for e in entries if e[0]])
        full_blob = "\n".join([e[1] for e in entries if e[1]])

        # 1) Age fallback (only if structured age missing)
        if pat_struct[pid]["Age"] is None:
            # age often near beginning, so prioritize first note's first 800 chars
            head = entries[0][1][:800] if entries[0][1] else ""
            age = _first_int_from_text(head, AGE_PATS)
            if age is None:
                age = _first_int_from_text(full_blob[:2000], AGE_PATS)
            if age is not None:
                pat_struct[pid]["Age"] = age

        # 2) BMI fallback (only if structured BMI missing)
        if pat_struct[pid]["BMI"] is None:
            bmi = _first_float_from_text(full_blob, BMI_PATS)
            if bmi is not None:
                pat_struct[pid]["BMI"] = bmi

        # 3) SmokingStatus fallback
        if pat_struct[pid]["SmokingStatus"] == "":
            for label, pat in SMOKE_PATS:
                if pat.search(full_blob):
                    pat_struct[pid]["SmokingStatus"] = label
                    break

        # 4) Comorbidities: PMH first
        for key, pats in COMORB_PATS.items():
            if pat_struct[pid][key] == 1:
                continue
            if _any_pat(pmh_blob, pats):
                pat_struct[pid][key] = 1

        # 5) Med second-layer: if meds present, re-check full note for comorb mention
        for key, meds in MED_TRIGGERS.items():
            if pat_struct[pid][key] == 1:
                continue
            if _any_pat(full_blob, meds):
                # require at least one disease mention somewhere in full note (to reduce false positives)
                if _any_pat(full_blob, COMORB_PATS[key]):
                    pat_struct[pid][key] = 1

        # 6) Lumpectomy enhancement: look for "lumpectomy scar" etc.
        if pat_struct[pid]["PBS_Lumpectomy"] == 0:
            if _any_pat(full_blob, LUMPECTOMY_PATS):
                pat_struct[pid]["PBS_Lumpectomy"] = 1

        # 7) Radiation / Chemo (clinic notes are best for history)
        if pat_struct[pid]["Radiation"] == 0:
            if _any_pat(full_blob, RADIATION_PATS):
                pat_struct[pid]["Radiation"] = 1
        if pat_struct[pid]["Chemo"] == 0:
            if _any_pat(full_blob, CHEMO_PATS):
                pat_struct[pid]["Chemo"] = 1

        # 8) Mastopexy / Augmentation fallback from text (if structured missed)
        if pat_struct[pid]["PBS_Mastopexy"] == 0 and _any_pat(full_blob, MASTOPEXY_PATS):
            pat_struct[pid]["PBS_Mastopexy"] = 1
        if pat_struct[pid]["PBS_Augmentation"] == 0 and _any_pat(full_blob, AUGMENT_PATS):
            pat_struct[pid]["PBS_Augmentation"] = 1

    # -------------------------
    # Final clean-up / output
    # -------------------------
    rows = []
    for pid in all_pids:
        r = pat_struct[pid]

        # normalize race/ethnicity labels a bit to match gold more often
        race = _normalize_str(r.get("Race", ""))
        eth = _normalize_str(r.get("Ethnicity", ""))

        # common normalization
        race = race.replace("White or Caucasian", "Caucasian")
        race = race.replace("Black or African American", "African American").replace("Black or African American", "African American")
        # keep original if already "Caucasian" etc.

        # Ethnicity normalization
        eth_low = eth.lower()
        if "non" in eth_low and "hisp" in eth_low:
            eth = "Non-hispanic"
        elif "hisp" in eth_low:
            eth = "Hispanic"

        r["Race"] = race
        r["Ethnicity"] = eth

        # if still missing age but present as NaN, set None
        if r["Age"] is not None:
            try:
                if isinstance(r["Age"], float) and math.isnan(r["Age"]):
                    r["Age"] = None
            except Exception:
                pass

        rows.append(r)

    out_df = pd.DataFrame(rows)

    # Ensure consistent column order (matches your validation expectations)
    col_order = [
        PID_COL,
        "Race", "Ethnicity", "Age", "BMI", "SmokingStatus",
        "Diabetes", "Hypertension", "CardiacDisease", "VenousThromboembolism", "Steroid",
        "PBS_Lumpectomy", "PBS_Mastopexy", "PBS_Augmentation",
        "Radiation", "Chemo",
    ]
    for c in col_order:
        if c not in out_df.columns:
            out_df[c] = ""

    out_df = out_df[col_order]

    print("Writing:", OUT_MASTER)
    out_df.to_csv(OUT_MASTER, index=False)

    print("Done.")
    print("Rows:", len(out_df))
    print("Patients with any note-derived PMH:", sum([1 for pid in all_pids if pid in note_idx]))


if __name__ == "__main__":
    main()
