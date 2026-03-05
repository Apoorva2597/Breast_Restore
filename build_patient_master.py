#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_patient_master.py (Python 3.6.8 compatible)

- No CLI arguments; hard-coded input/output paths.
- No mention of de-id columns.
- Robust CSV loading with encoding fallback (utf-8 -> cp1252 -> latin1)
- Extracts note-derived PMH, BMI, SmokingStatus (scaffold for comorbidities).
"""

from __future__ import print_function
import os
import re
import sys
import pandas as pd

# ---------------------------------------------------------------------
# HARD-CODED PATHS (edit here only)
# ---------------------------------------------------------------------
ENCOUNTERS_CSV = "/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Operation Encounters.csv"
CLINIC_NOTES_CSV = "/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Clinic Notes.csv"
OUT_CSV = "/home/apokol/Breast_Restore/_outputs/patient_master.csv"

# ---------------------------------------------------------------------
# Robust CSV loader (fixes UnicodeDecodeError)
# ---------------------------------------------------------------------
def read_csv_robust(path, dtype=str, low_memory=False):
    """
    Try common encodings for Epic/Cerner exports and text-heavy note files.
    """
    encodings_to_try = ["utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, dtype=dtype, low_memory=low_memory, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def norm(s):
    if s is None:
        return ""
    if isinstance(s, float):
        # NaN
        if s != s:
            return ""
    return str(s)

def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def collapse_ws(text):
    return re.sub(r"\s+", " ", text).strip()

def find_first_existing_col(df, candidates):
    cols = set([c.upper() for c in df.columns])
    for c in candidates:
        if c.upper() in cols:
            # Return actual column name matching case in df
            for real in df.columns:
                if real.upper() == c.upper():
                    return real
    return None

def extract_section(text, header_regex, max_chars=5000):
    """
    Extract text starting at header_regex until next ALLCAPS-ish header
    or until max_chars.
    Python 3.6-compatible (no re.Pattern typing).
    """
    if not text:
        return ""

    m = re.search(header_regex, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""

    start = m.start()
    chunk = text[start:start + max_chars]

    # Heuristic: stop at the next likely section header after the first line.
    # Examples: "PAST SURGICAL HISTORY:", "MEDICATIONS:", "ALLERGIES:", etc.
    # Allow headers with spaces and 2+ chars.
    stop = None
    header_pat = re.compile(r"\n[A-Z][A-Z \-/]{2,}:\s*", re.MULTILINE)
    for hm in header_pat.finditer(chunk):
        if hm.start() == 0:
            continue
        stop = hm.start()
        break

    if stop is None:
        return chunk
    return chunk[:stop]


# ---------------------------------------------------------------------
# Note extraction rules (based on your snippets)
# ---------------------------------------------------------------------
BMI_PATTERNS = [
    # "BMI 30.12 kg/m2" (degree symbol / weird chars may exist elsewhere)
    re.compile(r"\bBMI\s*[:=]?\s*([0-9]{1,2}\.?[0-9]{0,2})\s*(kg/m2|kg/m\^2|kg\/m2|kg\/m\^2)?\b", re.IGNORECASE),
    # Sometimes "Body mass index is 28.27"
    re.compile(r"\bbody\s+mass\s+index\s+(is\s+)?([0-9]{1,2}\.?[0-9]{0,2})\b", re.IGNORECASE),
]

SMOKING_PATTERNS = [
    # "<95> Smoking status: Former Smoker"
    re.compile(r"\bSmoking\s+status\s*[:=]\s*([A-Za-z \-]+)", re.IGNORECASE),
    # "Former smoker" style
    re.compile(r"\b(Current smoker|Former smoker|Never smoker)\b", re.IGNORECASE),
]

# PMH comorbidity keywords (expand as needed)
COMORBIDITY_MAP = {
    "Diabetes": [
        r"\bdiabetes\b", r"\bdm\b", r"\btype\s*2\s*diabetes\b", r"\bt2dm\b"
    ],
    "Hypertension": [
        r"\bhypertension\b", r"\bhtn\b", r"\bhigh blood pressure\b"
    ],
    "CardiacDisease": [
        r"\bcoronary artery disease\b", r"\bcad\b", r"\bchf\b", r"\bheart failure\b",
        r"\bmyocardial infarction\b", r"\bmi\b", r"\bafib\b", r"\batrial fibrillation\b"
    ],
    "VenousThromboembolism": [
        r"\bdeep vein thrombosis\b", r"\bdvt\b", r"\bpulmonary embolism\b", r"\bpe\b", r"\bvte\b"
    ],
    "Steroid": [
        r"\bsteroid\b", r"\bprednisone\b", r"\bmethylprednisolone\b", r"\bdexamethasone\b"
    ],
}

def extract_bmi(note_text):
    if not note_text:
        return None

    # Prioritize BMI near PHYSICAL EXAM / VITAL SIGNS
    # (your snippet shows "PHYSICAL EXAM: VITAL SIGNS: ... BMI 23.13 kg/m2")
    pe_chunk = extract_section(note_text, r"\bPHYSICAL\s+EXAM\s*:\s*", max_chars=2500)
    search_texts = [pe_chunk, note_text[:4000]]

    for t in search_texts:
        if not t:
            continue
        for pat in BMI_PATTERNS:
            m = pat.search(t)
            if m:
                # handle which group has number
                if pat.pattern.lower().find("body") >= 0:
                    val = safe_float(m.group(2))
                else:
                    val = safe_float(m.group(1))
                if val is not None and 10.0 <= val <= 80.0:
                    return val
    return None

def extract_smoking_status(note_text):
    if not note_text:
        return None

    # Look in SOCIAL HISTORY first; fallback to whole note head
    sh_chunk = extract_section(note_text, r"\bSOCIAL\s+HISTORY\s*:\s*", max_chars=2500)
    search_texts = [sh_chunk, note_text[:4000]]

    for t in search_texts:
        if not t:
            continue
        for pat in SMOKING_PATTERNS:
            m = pat.search(t)
            if not m:
                continue
            if m.lastindex and m.lastindex >= 1:
                cand = collapse_ws(m.group(1))
            else:
                cand = collapse_ws(m.group(0))

            cand_l = cand.lower()
            if "former" in cand_l:
                return "Former"
            if "never" in cand_l:
                return "Never"
            if "current" in cand_l:
                return "Current"

    return None

def extract_pmh_chunk(note_text):
    """
    Extract PMH while trying to avoid family-history bleed.
    You noted PMH sometimes contains family history => we clip at 'FAMILY HISTORY'
    or similar headers when present inside the PMH chunk.
    """
    pmh = extract_section(note_text, r"\bPAST\s+MEDICAL\s+HIST(ORY)?\s*:\s*", max_chars=6000)
    if not pmh:
        return ""

    # clip out family history if it appears inside
    fh = re.search(r"\bFAMILY\s+HIST(ORY)?\b", pmh, flags=re.IGNORECASE)
    if fh:
        pmh = pmh[:fh.start()]

    return pmh

def detect_comorbidities_from_pmh(note_text):
    """
    Returns dict of {var: 0/1} from PMH chunk only (your primary rule).
    """
    out = {}
    pmh = extract_pmh_chunk(note_text)
    pmh_l = pmh.lower()

    for var, pats in COMORBIDITY_MAP.items():
        found = 0
        for p in pats:
            if re.search(p, pmh_l, flags=re.IGNORECASE):
                found = 1
                break
        out[var] = found

    return out


# ---------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------
def build_patient_master(encounters_csv, clinic_notes_csv, out_csv):
    if not os.path.exists(encounters_csv):
        raise RuntimeError("Encounters CSV not found: %s" % encounters_csv)
    if not os.path.exists(clinic_notes_csv):
        raise RuntimeError("Clinic Notes CSV not found: %s" % clinic_notes_csv)

    print("Loading encounters: %s" % encounters_csv)
    enc = read_csv_robust(encounters_csv, dtype=str, low_memory=False)

    print("Loading clinic notes: %s" % clinic_notes_csv)
    notes = read_csv_robust(clinic_notes_csv, dtype=str, low_memory=False)

    # Expected core IDs
    enc_pid_col = find_first_existing_col(enc, ["ENCRYPTED_PAT_ID", "PAT_ID"])
    notes_pid_col = find_first_existing_col(notes, ["ENCRYPTED_PAT_ID", "PAT_ID"])

    if not enc_pid_col:
        raise RuntimeError("Could not find patient id column in encounters. Looked for ENCRYPTED_PAT_ID/PAT_ID")
    if not notes_pid_col:
        raise RuntimeError("Could not find patient id column in clinic notes. Looked for ENCRYPTED_PAT_ID/PAT_ID")

    # Optional columns
    age_col = find_first_existing_col(enc, ["AGE_AT_ENCOUNTER", "AGE"])
    race_col = find_first_existing_col(enc, ["RACE"])
    eth_col = find_first_existing_col(enc, ["ETHNICITY"])

    # Note text column (varies)
    note_text_col = find_first_existing_col(notes, ["NOTE_TEXT", "NOTE_TEXT_DEID", "TEXT", "NOTE_TEXT_CLEAN"])
    if not note_text_col:
        # last resort: try any column containing NOTE and TEXT
        for c in notes.columns:
            cu = c.upper()
            if "NOTE" in cu and "TEXT" in cu:
                note_text_col = c
                break
    if not note_text_col:
        raise RuntimeError("Could not find a note text column in clinic notes.")

    # -----------------------
    # Aggregate encounters to patient level (simple: first non-null)
    # -----------------------
    enc["__pid__"] = enc[enc_pid_col].map(norm)

    # pick one row per patient (you can change aggregation later)
    enc_first = enc.sort_values(by=[age_col] if age_col else [enc_pid_col]).groupby("__pid__", as_index=False).first()

    # Build base master
    master = pd.DataFrame()
    master["ENCRYPTED_PAT_ID"] = enc_first["__pid__"]

    if race_col:
        master["Race"] = enc_first[race_col].map(norm)
    else:
        master["Race"] = ""

    if eth_col:
        master["Ethnicity"] = enc_first[eth_col].map(norm)
    else:
        master["Ethnicity"] = ""

    if age_col:
        # Age should be numeric in encounters; keep as int where possible
        master["Age"] = enc_first[age_col].map(norm).apply(safe_int)
    else:
        master["Age"] = None

    # -----------------------
    # Aggregate clinic notes to patient level (concat a limited amount)
    # -----------------------
    notes["__pid__"] = notes[notes_pid_col].map(norm)
    notes["__text__"] = notes[note_text_col].map(norm)

    # Concatenate note text per patient (cap to avoid memory blowups)
    # NOTE: this is one reason scripts "take a while"—text concat is heavy.
    grouped = []
    for pid, g in notes.groupby("__pid__"):
        # join first N notes (or first N chars) to speed up
        texts = g["__text__"].tolist()
        joined = "\n\n".join(texts)
        joined = joined[:20000]  # cap per patient for performance
        grouped.append((pid, joined))

    note_agg = pd.DataFrame(grouped, columns=["ENCRYPTED_PAT_ID", "NOTE_TEXT_ALL"])

    master = master.merge(note_agg, on="ENCRYPTED_PAT_ID", how="left")
    master["NOTE_TEXT_ALL"] = master["NOTE_TEXT_ALL"].fillna("")

    # -----------------------
    # Extract note-derived fields
    # -----------------------
    bmis = []
    smokes = []
    pmh_any = []

    dm = []
    htn = []
    cardiac = []
    vte = []
    steroid = []

    for txt in master["NOTE_TEXT_ALL"].tolist():
        bmis.append(extract_bmi(txt))
        smokes.append(extract_smoking_status(txt))

        com = detect_comorbidities_from_pmh(txt)
        dm.append(com.get("Diabetes", 0))
        htn.append(com.get("Hypertension", 0))
        cardiac.append(com.get("CardiacDisease", 0))
        vte.append(com.get("VenousThromboembolism", 0))
        steroid.append(com.get("Steroid", 0))

        pmh_any.append(1 if extract_pmh_chunk(txt) else 0)

    master["BMI"] = bmis
    master["SmokingStatus"] = smokes

    master["Diabetes"] = dm
    master["Hypertension"] = htn
    master["CardiacDisease"] = cardiac
    master["VenousThromboembolism"] = vte
    master["Steroid"] = steroid

    master["HAS_NOTE_PMH"] = pmh_any

    # -----------------------
    # Write output
    # -----------------------
    out_dir = os.path.dirname(out_csv)
    if out_dir and (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    print("Writing: %s" % out_csv)
    master.to_csv(out_csv, index=False)
    print("Done.")
    print("Rows: %d" % len(master))
    print("Patients with any note-derived PMH: %d" % sum(master["HAS_NOTE_PMH"].tolist()))


def main():
    build_patient_master(ENCOUNTERS_CSV, CLINIC_NOTES_CSV, OUT_CSV)

if __name__ == "__main__":
    main()
