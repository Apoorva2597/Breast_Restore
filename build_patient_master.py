#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_patient_master.py (Python 3.6.8 compatible)

Fixes:
- UnicodeDecodeError handling for clinic notes
- No CLI args (hard-coded paths)
- No "de-id" column assumptions
- Correct note text column selection (by avg length)
- Normalize Race/Ethnicity to gold-style labels
- Age written as clean integer string (no 47.0)
- BMI extraction handles kg/m2 and kg/m² variants
- PMH extraction robust (block + table-ish)
"""

from __future__ import print_function
import os
import re
import pandas as pd


# ---------------------------------------------------------------------
# HARD-CODED PATHS
# ---------------------------------------------------------------------
ENCOUNTERS_CSV = "/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Operation Encounters.csv"
CLINIC_NOTES_CSV = "/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Clinic Notes.csv"
OUT_CSV = "/home/apokol/Breast_Restore/_outputs/patient_master.csv"

# ---------------------------------------------------------------------
# Robust CSV loader
# ---------------------------------------------------------------------
def read_csv_robust(path, dtype=str, low_memory=False):
    encodings_to_try = ["utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, dtype=dtype, low_memory=low_memory, encoding=enc)
        except Exception as e:
            last_err = e

    # Final fallback: do not crash on bad bytes
    try:
        return pd.read_csv(path, dtype=dtype, low_memory=low_memory, encoding="latin1")
    except Exception:
        raise last_err


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def norm(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and x != x:
            return ""
    except Exception:
        pass
    return str(x).strip()

def safe_int_str(x):
    s = norm(x)
    if not s:
        return ""
    try:
        i = int(float(s))
        return str(i)
    except Exception:
        return ""

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def collapse_ws(text):
    return re.sub(r"\s+", " ", text).strip()

def find_first_existing_col(df, candidates):
    cols_upper = set([c.upper() for c in df.columns])
    for c in candidates:
        cu = c.upper()
        if cu in cols_upper:
            for real in df.columns:
                if real.upper() == cu:
                    return real
    return None

def pick_note_text_col(notes_df):
    candidates = []
    for c in notes_df.columns:
        cu = c.upper()
        if ("NOTE" in cu and "TEXT" in cu) or cu in ("TEXT", "NOTE", "BODY"):
            candidates.append(c)

    if not candidates:
        return None

    best_col = None
    best_avg = -1.0
    for c in candidates:
        series = notes_df[c].fillna("").astype(str)
        samp = series.head(5000)
        lengths = samp.map(lambda s: len(s) if s else 0)
        avg_len = float(lengths.mean()) if len(lengths) else 0.0
        if avg_len > best_avg:
            best_avg = avg_len
            best_col = c

    return best_col


# ---------------------------------------------------------------------
# Normalization to match GOLD labels
# ---------------------------------------------------------------------
def normalize_race(val):
    v = norm(val).lower()
    if not v:
        return ""

    if "white" in v or "caucasian" in v:
        return "Caucasian"
    if "black" in v or "african" in v:
        return "Black"
    if "asian" in v:
        return "Asian"
    if "american indian" in v or "alaska" in v:
        return "American Indian/Alaska Native"
    if "native hawaiian" in v or "pacific" in v:
        return "Native Hawaiian/Pacific Islander"

    if "unknown" in v or "declined" in v or "refused" in v:
        return "Unknown"
    return "Other"

def normalize_ethnicity(val):
    v = norm(val).lower()
    if not v:
        return ""
    if "non" in v and "hisp" in v:
        return "Non-hispanic"
    if "hisp" in v or "latin" in v:
        return "Hispanic"
    if "unknown" in v or "declined" in v or "refused" in v:
        return "Unknown"
    return collapse_ws(norm(val))


# ---------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------
def extract_block(text, start_regex, max_chars=8000):
    if not text:
        return ""

    m = re.search(start_regex, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""

    start = m.start()
    chunk = text[start:start + max_chars]

    header_pat = re.compile(r"\n[A-Z][A-Z \-/]{2,}:\s*", re.MULTILINE)
    stop = None
    for hm in header_pat.finditer(chunk):
        if hm.start() == 0:
            continue
        stop = hm.start()
        break

    return chunk if stop is None else chunk[:stop]

def extract_pmh_chunks(note_text):
    if not note_text:
        return []

    chunks = []

    chunk_a = extract_block(note_text, r"\bPAST\s+MEDICAL\s+HIST(?:ORY)?\s*:?\s*")
    if chunk_a:
        chunks.append(chunk_a)

    m = re.search(r"\bPast\s+Medical\s+History\b", note_text, flags=re.IGNORECASE)
    if m:
        window = note_text[m.start():m.start() + 4000]
        chunks.append(window)

    cleaned = []
    for c in chunks:
        fh = re.search(r"\bFAMILY\s+HIST(?:ORY)?\b", c, flags=re.IGNORECASE)
        if fh:
            c = c[:fh.start()]
        cleaned.append(c)

    out = []
    seen = set()
    for c in cleaned:
        key = c[:200].lower()
        if key not in seen and len(c.strip()) > 20:
            out.append(c)
            seen.add(key)

    return out


# ---------------------------------------------------------------------
# BMI / Smoking extraction
# ---------------------------------------------------------------------
BMI_PATTERNS = [
    re.compile(r"\bBMI\s*[:=]?\s*([0-9]{1,2}\.?[0-9]{0,2})\s*(kg\s*/\s*m2|kg\s*/\s*m\^2|kg\s*/\s*m²|kg\s*/\s*m\u00b2|kg/m2|kg/m\^2|kg/m²)?\b", re.IGNORECASE),
    re.compile(r"\bbody\s+mass\s+index\s+(is\s+)?([0-9]{1,2}\.?[0-9]{0,2})\b", re.IGNORECASE),
]

def extract_bmi(note_text):
    if not note_text:
        return ""

    head = note_text[:10000]
    pe = extract_block(note_text, r"\bPHYSICAL\s+EXAM\s*:?\s*", max_chars=5000)
    texts = [pe, head] if pe else [head]

    for t in texts:
        if not t:
            continue
        for pat in BMI_PATTERNS:
            m = pat.search(t)
            if not m:
                continue
            if "body" in pat.pattern.lower():
                val = safe_float(m.group(2))
            else:
                val = safe_float(m.group(1))
            if val is not None and 10.0 <= val <= 80.0:
                return ("%.2f" % val).rstrip("0").rstrip(".")
    return ""

def extract_smoking_status(note_text):
    if not note_text:
        return ""

    sh = extract_block(note_text, r"\bSOCIAL\s+HISTORY\s*:?\s*", max_chars=5000)
    head = note_text[:12000]
    texts = [sh, head] if sh else [head]

    for t in texts:
        if not t:
            continue

        m = re.search(r"\bSmoking\s+status\s*[:=]\s*([A-Za-z \-]+)", t, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip().lower()
            if "former" in cand:
                return "Former"
            if "never" in cand:
                return "Never"
            if "current" in cand:
                return "Current"

        if re.search(r"\bnever\s+smok", t, flags=re.IGNORECASE):
            return "Never"
        if re.search(r"\bformer\s+smok|\bquit\s+smok", t, flags=re.IGNORECASE):
            return "Former"
        if re.search(r"\bcurrent\s+smok|\bsmokes\b", t, flags=re.IGNORECASE):
            return "Current"

    return ""


# ---------------------------------------------------------------------
# Comorbidities detection (ONLY inside PMH chunks)
# ---------------------------------------------------------------------
COMORBIDITY_MAP = {
    "Diabetes": [r"\bdiabetes\b", r"\btype\s*2\s*diabetes\b", r"\bt2dm\b", r"\bdm\b"],
    "Hypertension": [r"\bhypertension\b", r"\bhtn\b", r"\bhigh blood pressure\b"],
    "CardiacDisease": [
        r"\bcoronary artery disease\b", r"\bcad\b", r"\bchf\b", r"\bheart failure\b",
        r"\bmyocardial infarction\b", r"\bafib\b", r"\batrial fibrillation\b"
    ],
    "VenousThromboembolism": [r"\bdeep vein thrombosis\b", r"\bdvt\b", r"\bpulmonary embolism\b", r"\bpe\b", r"\bvte\b"],
    "Steroid": [r"\bprednisone\b", r"\bdexamethasone\b", r"\bmethylprednisolone\b", r"\bsteroid\b"],
}

def detect_comorbidities(note_text):
    out = {k: "0" for k in COMORBIDITY_MAP.keys()}
    chunks = extract_pmh_chunks(note_text)
    if not chunks:
        return out

    pmh_text = "\n\n".join(chunks).lower()

    for var, pats in COMORBIDITY_MAP.items():
        found = False
        for p in pats:
            if re.search(p, pmh_text, flags=re.IGNORECASE):
                found = True
                break
        out[var] = "1" if found else "0"

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

    enc_pid_col = find_first_existing_col(enc, ["ENCRYPTED_PAT_ID", "PAT_ID"])
    notes_pid_col = find_first_existing_col(notes, ["ENCRYPTED_PAT_ID", "PAT_ID"])
    if not enc_pid_col:
        raise RuntimeError("Encounters missing patient id (ENCRYPTED_PAT_ID/PAT_ID). Columns: %s" % list(enc.columns))
    if not notes_pid_col:
        raise RuntimeError("Clinic notes missing patient id (ENCRYPTED_PAT_ID/PAT_ID). Columns: %s" % list(notes.columns))

    age_col = find_first_existing_col(enc, ["AGE_AT_ENCOUNTER", "AGE"])
    race_col = find_first_existing_col(enc, ["RACE"])
    eth_col = find_first_existing_col(enc, ["ETHNICITY"])

    note_text_col = pick_note_text_col(notes)
    if not note_text_col:
        raise RuntimeError("Could not determine note text column from clinic notes. Columns: %s" % list(notes.columns))

    samp = notes[note_text_col].fillna("").astype(str).head(2000)
    avg_len = float(samp.map(len).mean()) if len(samp) else 0.0
    print("Selected note text column: %s (avg length first 2000 rows = %.1f)" % (note_text_col, avg_len))

    enc["__pid__"] = enc[enc_pid_col].map(norm)
    notes["__pid__"] = notes[notes_pid_col].map(norm)

    date_col = find_first_existing_col(enc, ["OPERATION_DATE", "DISCHARGE_DATE_DT", "ADMISSION_DATE", "ENC_DATE"])
    enc_sort = enc.sort_values(by=[date_col]) if date_col else enc
    enc_first = enc_sort.groupby("__pid__", as_index=False).first()

    master = pd.DataFrame()
    master["ENCRYPTED_PAT_ID"] = enc_first["__pid__"]

    master["Race"] = enc_first[race_col].map(normalize_race) if race_col else ""
    master["Ethnicity"] = enc_first[eth_col].map(normalize_ethnicity) if eth_col else ""
    master["Age"] = enc_first[age_col].map(safe_int_str) if age_col else ""

    notes["__text__"] = notes[note_text_col].fillna("").astype(str)

    grouped_rows = []
    for pid, g in notes.groupby("__pid__"):
        texts = g["__text__"].tolist()
        joined = "\n\n".join(texts[:8])
        joined = joined[:60000]
        grouped_rows.append((pid, joined))

    note_agg = pd.DataFrame(grouped_rows, columns=["ENCRYPTED_PAT_ID", "NOTE_TEXT_ALL"])
    master = master.merge(note_agg, on="ENCRYPTED_PAT_ID", how="left")
    master["NOTE_TEXT_ALL"] = master["NOTE_TEXT_ALL"].fillna("")

    bmi_list = []
    smoke_list = []
    has_pmh = []

    dm_list = []
    htn_list = []
    cardiac_list = []
    vte_list = []
    steroid_list = []

    for txt in master["NOTE_TEXT_ALL"].tolist():
        bmi_list.append(extract_bmi(txt))
        smoke_list.append(extract_smoking_status(txt))

        com = detect_comorbidities(txt)
        dm_list.append(com.get("Diabetes", "0"))
        htn_list.append(com.get("Hypertension", "0"))
        cardiac_list.append(com.get("CardiacDisease", "0"))
        vte_list.append(com.get("VenousThromboembolism", "0"))
        steroid_list.append(com.get("Steroid", "0"))

        pmh_chunks = extract_pmh_chunks(txt)
        has_pmh.append("1" if pmh_chunks else "0")

    master["BMI"] = bmi_list
    master["SmokingStatus"] = smoke_list

    master["Diabetes"] = dm_list
    master["Hypertension"] = htn_list
    master["CardiacDisease"] = cardiac_list
    master["VenousThromboembolism"] = vte_list
    master["Steroid"] = steroid_list
    master["HAS_NOTE_PMH"] = has_pmh

    def pct_nonempty(series):
        vals = series.map(norm)
        non = sum([1 for v in vals.tolist() if v != ""])
        return 100.0 * float(non) / float(len(vals)) if len(vals) else 0.0

    # FIX: escape the literal percent sign as %%
    print("Non-empty %%: Age=%.1f  BMI=%.1f  SmokingStatus=%.1f  PMH=%.1f" % (
        pct_nonempty(master["Age"]),
        pct_nonempty(master["BMI"]),
        pct_nonempty(master["SmokingStatus"]),
        pct_nonempty(master["HAS_NOTE_PMH"])
    ))

    out_dir = os.path.dirname(out_csv)
    if out_dir and (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    print("Writing: %s" % out_csv)
    master.to_csv(out_csv, index=False)
    print("Done.")
    print("Rows: %d" % len(master))
    print("Patients with any note-derived PMH: %d" % sum([1 for x in master["HAS_NOTE_PMH"].tolist() if x == "1"]))


def main():
    build_patient_master(ENCOUNTERS_CSV, CLINIC_NOTES_CSV, OUT_CSV)

if __name__ == "__main__":
    main()
