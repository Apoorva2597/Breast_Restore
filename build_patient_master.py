#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_patient_master.py  (Python 3.6.8 friendly)

Creates a PATIENT-LEVEL abstraction table (patient_master.csv) for the HPI11526 dataset.

Key updates (per your latest guidance):
- Comorbidities (Diabetes / HTN / Cardiac / VTE / Steroid) primarily from CLINIC NOTES:
    * section-aware extraction from "PAST MEDICAL HISTORY" (PMH) block
    * medication signals as a secondary confirmation layer
- Lumpectomy: include "lumpectomy scar" (commonly documented on exam) in CLINIC NOTES
- Uses ORIGINAL data only (no synthetic NOTE_DEID column assumptions):
    * auto-detects the note text column (NOTE_TEXT*, TEXT, BODY, etc.)
- Reads the real file paths under:
    /home/apokol/my_data_Breast/HPI-11526/HPI11256

Outputs:
  Breast_Restore/_outputs/patient_master.csv
  Breast_Restore/_outputs/build_audit_summary.csv

Run:
  (.venv) python build_patient_master.py
"""

from __future__ import print_function

import os
import re
import sys
import time
import pandas as pd
from collections import defaultdict, Counter

# -----------------------------
# INPUT PATHS (FULL PATHS)
# -----------------------------
DATA_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

CLINIC_ENC = os.path.join(DATA_DIR, "HPI11526 Clinic Encounters.csv")
INPAT_ENC  = os.path.join(DATA_DIR, "HPI11526 Inpatient Encounters.csv")
OP_ENC     = os.path.join(DATA_DIR, "HPI11526 Operation Encounters.csv")

CLINIC_NOTES = os.path.join(DATA_DIR, "HPI11526 Clinic Notes.csv")
INPAT_NOTES  = os.path.join(DATA_DIR, "HPI11526 Inpatient Notes.csv")
OP_NOTES     = os.path.join(DATA_DIR, "HPI11526 Operation Notes.csv")

# -----------------------------
# OUTPUTS (relative to CWD)
# -----------------------------
OUT_DIR = "_outputs"
MASTER_OUT = os.path.join(OUT_DIR, "patient_master.csv")
AUDIT_OUT  = os.path.join(OUT_DIR, "build_audit_summary.csv")

# -----------------------------
# ID COLUMN
# -----------------------------
PID_COL = "ENCRYPTED_PAT_ID"

# -----------------------------
# HELPERS: robust CSV read
# -----------------------------
def safe_read_csv(path, usecols=None, chunksize=None):
    """
    Tries a few encodings to avoid UnicodeDecodeError in older pandas/Python.
    """
    encodings = ["utf-8", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, usecols=usecols, chunksize=chunksize, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def norm_col(s):
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def pick_col(cols, candidates):
    """
    Find best match among columns given a list of candidate names (normalized).
    """
    norm_map = {norm_col(c): c for c in cols}
    for cand in candidates:
        c = norm_map.get(norm_col(cand))
        if c:
            return c
    # fallback: partial match
    for cand in candidates:
        nc = norm_col(cand)
        for k, v in norm_map.items():
            if nc and nc in k:
                return v
    return None


def ensure_outdir():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


# -----------------------------
# SECTION EXTRACTION: PMH block
# -----------------------------
PMH_HDR = re.compile(r"(^|\n)\s*(past medical history|pmh|medical history)\s*:?\s*(\n|$)",
                     re.IGNORECASE)
# Stop when next header appears (often ALL CAPS or Title Case section lines)
SECTION_STOP = re.compile(r"\n\s*[A-Z][A-Z \-/]{2,}\s*:?\s*\n")

def extract_pmh_block(text):
    """
    Returns the PMH block text if found, else None.
    """
    if not text:
        return None
    m = PMH_HDR.search(text)
    if not m:
        return None
    start = m.end()
    tail = text[start:]
    stop = SECTION_STOP.search(tail)
    if stop:
        return tail[:stop.start()]
    return tail[:2000]  # safety cap (keeps it fast)


# -----------------------------
# NEGATION HANDLING (lightweight)
# -----------------------------
NEG_WINDOW_CHARS = 40
NEG_RE = re.compile(r"\b(no|not|denies|deny|without|negative for|never had|rule out|r/o)\b",
                    re.IGNORECASE)

def has_positive(text, keyword_re):
    """
    Finds keyword occurrences not negated in a short preceding window.
    """
    if not text:
        return False
    for m in keyword_re.finditer(text):
        left = text[max(0, m.start() - NEG_WINDOW_CHARS):m.start()]
        if NEG_RE.search(left):
            continue
        return True
    return False


# -----------------------------
# Regex patterns (compiled)
# -----------------------------
# Comorbidities (PMH primary)
RX_DM = re.compile(r"\b(diabetes|dm\b|t2dm|type\s*2\s*dm|type\s*ii\s*dm|t1dm|type\s*1\s*dm)\b",
                   re.IGNORECASE)
RX_HTN = re.compile(r"\b(hypertension|htn|high blood pressure)\b", re.IGNORECASE)
RX_CARDIAC = re.compile(r"\b(coronary artery disease|cad\b|chf\b|heart failure|angina|"
                        r"myocardial infarction|\bmi\b|afib|atrial fibrillation)\b",
                        re.IGNORECASE)
RX_VTE = re.compile(r"\b(venous thromboembolism|vte\b|dvt\b|deep vein thrombosis|"
                    r"pulmonary embolism|\bpe\b)\b",
                    re.IGNORECASE)

# Meds (secondary)
RX_DM_MEDS = re.compile(r"\b(metformin|insulin|glipizide|glyburide|glimepiride|"
                        r"lantus|humalog|novolog|ozempic|semaglutide|"
                        r"jardiance|empagliflozin)\b",
                        re.IGNORECASE)

RX_HTN_MEDS = re.compile(r"\b(lisinopril|enalapril|benazepril|losartan|valsartan|"
                         r"amlodipine|diltiazem|metoprolol|carvedilol|"
                         r"hydrochlorothiazide|hctz\b)\b",
                         re.IGNORECASE)

RX_CARDIAC_MEDS = re.compile(r"\b(nitroglycerin|ntg\b|clopidogrel|plavix|"
                             r"warfarin|coumadin|"
                             r"atorvastatin|rosuvastatin|simvastatin)\b",
                             re.IGNORECASE)

RX_ANTICOAG = re.compile(r"\b(apixaban|eliquis|rivaroxaban|xarelto|dabigatran|"
                         r"pradaxa|warfarin|coumadin|enoxaparin|lovenox|heparin)\b",
                         re.IGNORECASE)

RX_STEROID = re.compile(r"\b(steroid|prednisone|prednisolone|methylprednisolone|"
                        r"dexamethasone|hydrocortisone)\b",
                        re.IGNORECASE)

# Smoking (clinic notes often have Tobacco/Smoking lines)
RX_SMOKE_LINE = re.compile(r"\b(tobacco|smok(ing|es)?)\b[^\n]{0,60}\b(current|former|never|none)\b",
                           re.IGNORECASE)

# Lumpectomy (include scar)
RX_LUMP = re.compile(r"\b(lumpectomy|lumpectomy scar|partial mastectomy)\b", re.IGNORECASE)
RX_REDUCT = re.compile(r"\b(breast reduction|reduction mammoplasty)\b", re.IGNORECASE)
RX_MASTOPEXY = re.compile(r"\b(mastopexy|breast lift)\b", re.IGNORECASE)
RX_AUG = re.compile(r"\b(augmentation|breast augmentation|implants?\b)\b", re.IGNORECASE)

# Chemo / Radiation (simple but negation-aware)
RX_RAD = re.compile(r"\b(radiation|xrt\b|radiotherapy)\b", re.IGNORECASE)

# Avoid endocrine-only meds being counted as chemo (tamoxifen etc.)
RX_CHEMO = re.compile(r"\b(chemotherapy|chemo\b|paclitaxel|taxol|docetaxel|taxotere|"
                      r"doxorubicin|adriamycin|cyclophosphamide|carboplatin|"
                      r"cisplatin|capecitabine|xeloda)\b",
                      re.IGNORECASE)

# Age/BMI from notes (fallback if not in encounters)
RX_AGE = re.compile(r"\b(\d{2})\s*[- ]?(yo|y/o|year old|years old)\b", re.IGNORECASE)
RX_BMI = re.compile(r"\bbmi\b[^\d]{0,10}(\d{2}\.\d|\d{2})\b", re.IGNORECASE)
RX_BMI2 = re.compile(r"\b(bmi of|bmi is)\s*(\d{2}\.\d|\d{2})\b", re.IGNORECASE)


def parse_smoking_status(text):
    if not text:
        return None
    m = RX_SMOKE_LINE.search(text)
    if not m:
        return None
    val = m.group(3).strip().lower()
    if val in ("none",):
        return "Never"
    if val == "never":
        return "Never"
    if val == "former":
        return "Former"
    if val == "current":
        return "Current"
    return None


def parse_age(text):
    if not text:
        return None
    m = RX_AGE.search(text)
    if not m:
        return None
    try:
        age = int(m.group(1))
        if 0 < age < 120:
            return age
    except Exception:
        return None
    return None


def parse_bmi(text):
    if not text:
        return None
    m = RX_BMI.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m2 = RX_BMI2.search(text)
    if m2:
        try:
            return float(m2.group(2))
        except Exception:
            pass
    return None


# -----------------------------
# Aggregation state per patient
# -----------------------------
def init_patient_state():
    return {
        "Race": None,
        "Ethnicity": None,
        "Age": None,
        "BMI": None,
        "SmokingStatus": None,
        "Diabetes": 0,
        "Hypertension": 0,
        "CardiacDisease": 0,
        "VenousThromboembolism": 0,
        "Steroid": 0,
        "PBS_Lumpectomy": 0,
        "PBS_Reduction": 0,
        "PBS_Mastopexy": 0,
        "PBS_Augmentation": 0,
        "Radiation": 0,
        "Chemo": 0,
        "_race_votes": Counter(),
        "_eth_votes": Counter(),
        "_smoke_votes": Counter(),
        "_age_votes": Counter(),
        "_bmi_votes": Counter(),
        "_pmh_hits": 0,
        "_med_hits": 0,
        "_notes_seen": 0
    }


def finalize_patient_state(st):
    # choose most common categorical values (if any votes)
    if st["_race_votes"]:
        st["Race"] = st["_race_votes"].most_common(1)[0][0]
    if st["_eth_votes"]:
        st["Ethnicity"] = st["_eth_votes"].most_common(1)[0][0]
    if st["_smoke_votes"]:
        st["SmokingStatus"] = st["_smoke_votes"].most_common(1)[0][0]
    if st["_age_votes"]:
        st["Age"] = st["_age_votes"].most_common(1)[0][0]
    if st["_bmi_votes"]:
        st["BMI"] = st["_bmi_votes"].most_common(1)[0][0]

    # cleanup internal fields
    for k in list(st.keys()):
        if k.startswith("_"):
            del st[k]
    return st


# -----------------------------
# Encounters: pick baseline demographics if available
# -----------------------------
def ingest_encounters(pstate, path):
    if not os.path.exists(path):
        print("WARNING: missing file:", path)
        return

    df = safe_read_csv(path)
    if hasattr(df, "__iter__") and not isinstance(df, pd.DataFrame):
        # chunks iterator not expected here
        df = pd.concat(list(df), ignore_index=True)

    cols = list(df.columns)
    pidc = pick_col(cols, [PID_COL, "ENCRYPTED_PAT_ID", "ENCRYPTEDPATID"])
    if not pidc:
        print("WARNING: could not find patient id in encounters:", path)
        return

    race_c = pick_col(cols, ["Race", "RACE"])
    eth_c  = pick_col(cols, ["Ethnicity", "ETHNICITY"])
    age_c  = pick_col(cols, ["Age", "AGE"])
    bmi_c  = pick_col(cols, ["BMI", "BodyMassIndex", "Body Mass Index"])

    # iterate rows (vectorizing here is fine, but keep it simple)
    for _, r in df.iterrows():
        pid = r.get(pidc)
        if pd.isna(pid):
            continue
        pid = str(pid).strip()
        st = pstate[pid]

        if race_c and not pd.isna(r.get(race_c)):
            val = str(r.get(race_c)).strip()
            if val:
                st["_race_votes"][val] += 1

        if eth_c and not pd.isna(r.get(eth_c)):
            val = str(r.get(eth_c)).strip()
            if val:
                st["_eth_votes"][val] += 1

        if age_c and not pd.isna(r.get(age_c)):
            try:
                agev = int(float(r.get(age_c)))
                if 0 < agev < 120:
                    st["_age_votes"][agev] += 1
            except Exception:
                pass

        if bmi_c and not pd.isna(r.get(bmi_c)):
            try:
                bmiv = float(r.get(bmi_c))
                if 10.0 <= bmiv <= 80.0:
                    st["_bmi_votes"][bmiv] += 1
            except Exception:
                pass


# -----------------------------
# Notes ingestion (chunked for speed)
# -----------------------------
def ingest_clinic_notes_for_flags(pstate, path, chunksize=2000):
    """
    CLINIC NOTES drive:
      - PMH comorbidities (primary)
      - meds confirmation (secondary)
      - smoking
      - lumpectomy scar + other PBS
      - fallback Age/BMI if needed
      - radiation/chemo cues (many notes include onc history)
    """
    if not os.path.exists(path):
        print("WARNING: missing file:", path)
        return

    # read just patient id + note text (auto-detected)
    reader = safe_read_csv(path, chunksize=chunksize)

    # get columns by peeking first chunk
    first = next(reader)
    cols = list(first.columns)

    pidc = pick_col(cols, [PID_COL, "ENCRYPTED_PAT_ID"])
    textc = pick_col(cols, ["NOTE_TEXT_DEID", "NOTE_TEXT", "NOTE_TEXT_CLEAN",
                            "NOTE_TEXT_RAW", "TEXT", "BODY", "NOTE"])

    if not pidc:
        raise RuntimeError("Could not find {} column in clinic notes.".format(PID_COL))
    if not textc:
        raise RuntimeError("Could not find note text column in clinic notes. "
                           "Tried NOTE_TEXT*, TEXT, BODY, NOTE.")

    def process_chunk(df):
        for _, r in df.iterrows():
            pid = r.get(pidc)
            if pd.isna(pid):
                continue
            pid = str(pid).strip()
            txt = r.get(textc)
            if pd.isna(txt):
                continue
            text = str(txt)

            st = pstate[pid]
            st["_notes_seen"] += 1

            # PMH block (primary source)
            pmh = extract_pmh_block(text)
            if pmh:
                st["_pmh_hits"] += 1
                if st["Diabetes"] == 0 and has_positive(pmh, RX_DM):
                    st["Diabetes"] = 1
                if st["Hypertension"] == 0 and has_positive(pmh, RX_HTN):
                    st["Hypertension"] = 1
                if st["CardiacDisease"] == 0 and has_positive(pmh, RX_CARDIAC):
                    st["CardiacDisease"] = 1
                if st["VenousThromboembolism"] == 0 and has_positive(pmh, RX_VTE):
                    st["VenousThromboembolism"] = 1
                if st["Steroid"] == 0 and has_positive(pmh, RX_STEROID):
                    st["Steroid"] = 1

            # Med confirmation layer (only if not found in PMH)
            # We do a light scan on full note for meds; still negation-aware via has_positive()
            if st["Diabetes"] == 0 and has_positive(text, RX_DM_MEDS):
                st["_med_hits"] += 1
                st["Diabetes"] = 1
            if st["Hypertension"] == 0 and has_positive(text, RX_HTN_MEDS):
                st["_med_hits"] += 1
                st["Hypertension"] = 1
            if st["CardiacDisease"] == 0 and has_positive(text, RX_CARDIAC_MEDS):
                st["_med_hits"] += 1
                st["CardiacDisease"] = 1
            if st["VenousThromboembolism"] == 0 and has_positive(text, RX_ANTICOAG):
                st["_med_hits"] += 1
                st["VenousThromboembolism"] = 1
            if st["Steroid"] == 0 and has_positive(text, RX_STEROID):
                # steroids appear frequently; allow but negation-aware
                st["_med_hits"] += 1
                st["Steroid"] = 1

            # Smoking
            smk = parse_smoking_status(text)
            if smk:
                st["_smoke_votes"][smk] += 1

            # PBS / lumpectomy scar logic
            if st["PBS_Lumpectomy"] == 0 and has_positive(text, RX_LUMP):
                st["PBS_Lumpectomy"] = 1
            if st["PBS_Reduction"] == 0 and has_positive(text, RX_REDUCT):
                st["PBS_Reduction"] = 1
            if st["PBS_Mastopexy"] == 0 and has_positive(text, RX_MASTOPEXY):
                st["PBS_Mastopexy"] = 1
            if st["PBS_Augmentation"] == 0 and has_positive(text, RX_AUG):
                st["PBS_Augmentation"] = 1

            # Radiation / Chemo
            if st["Radiation"] == 0 and has_positive(text, RX_RAD):
                st["Radiation"] = 1
            if st["Chemo"] == 0 and has_positive(text, RX_CHEMO):
                st["Chemo"] = 1

            # Fallback Age/BMI from notes if missing
            if not st["_age_votes"]:
                agev = parse_age(text)
                if agev is not None:
                    st["_age_votes"][agev] += 1
            if not st["_bmi_votes"]:
                bmiv = parse_bmi(text)
                if bmiv is not None and 10.0 <= bmiv <= 80.0:
                    st["_bmi_votes"][bmiv] += 1

    # process first chunk + remaining chunks
    process_chunk(first)
    for chunk in reader:
        process_chunk(chunk)


# -----------------------------
# Main
# -----------------------------
def main():
    t0 = time.time()
    ensure_outdir()

    pstate = defaultdict(init_patient_state)

    print("Loading encounters (demographics where available)...")
    for p in [CLINIC_ENC, INPAT_ENC, OP_ENC]:
        if os.path.exists(p):
            print("  -", p)
            ingest_encounters(pstate, p)

    print("Processing CLINIC notes for PMH comorbidities + PBS + smoking...")
    print("  -", CLINIC_NOTES)
    ingest_clinic_notes_for_flags(pstate, CLINIC_NOTES, chunksize=2000)

    # Build final dataframe
    rows = []
    audit = {
        "patients_total": 0,
        "patients_with_any_note": 0,
        "patients_with_pmh_block": 0,
        "patients_with_med_hit": 0
    }

    for pid, st in pstate.items():
        audit["patients_total"] += 1
        if st["_notes_seen"] > 0:
            audit["patients_with_any_note"] += 1
        if st["_pmh_hits"] > 0:
            audit["patients_with_pmh_block"] += 1
        if st["_med_hits"] > 0:
            audit["patients_with_med_hit"] += 1

        out = {"ENCRYPTED_PAT_ID": pid}
        out.update(finalize_patient_state(st))
        rows.append(out)

    df = pd.DataFrame(rows)

    # Stable column order
    col_order = [
        "ENCRYPTED_PAT_ID",
        "Race", "Ethnicity", "Age", "BMI", "SmokingStatus",
        "Diabetes", "Hypertension", "CardiacDisease", "VenousThromboembolism", "Steroid",
        "PBS_Lumpectomy", "PBS_Reduction", "PBS_Mastopexy", "PBS_Augmentation",
        "Radiation", "Chemo"
    ]
    for c in col_order:
        if c not in df.columns:
            df[c] = None
    df = df[col_order]

    df.to_csv(MASTER_OUT, index=False)
    pd.DataFrame([audit]).to_csv(AUDIT_OUT, index=False)

    dt = time.time() - t0
    print("\nWrote:", MASTER_OUT)
    print("Wrote:", AUDIT_OUT)
    print("Patients:", len(df))
    print("Done in %.1f seconds." % dt)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", str(e))
        sys.exit(1)
