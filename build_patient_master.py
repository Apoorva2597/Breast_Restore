#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_patient_master.py

Builds _outputs/patient_master.csv (patient-level abstraction) by combining:
  1) Structured patient file with ENCRYPTED_PAT_ID (Race/Ethnicity/Age/BMI/Smoking + surgery flags + etc.)
  2) Clinic notes CSV (free text) to extract:
        - Comorbidities from PAST MEDICAL HISTORY section (primary)
        - Medications as a trigger, then confirm by searching full note (secondary)
        - Lumpectomy evidence from clinic exam language (e.g., "lumpectomy scar")
        - Age fallback (if structured Age missing) from early lines of note text

IMPORTANT per user constraints:
  - Do NOT look for NOTE_TEXT_DEID (it does not exist in original data file).
  - Use broad age patterns.
  - Prefer structured Age column when present; use note only as fallback.

Outputs:
  - _outputs/patient_master.csv
  - _outputs/build_audit_summary.csv
  - _outputs/build_audit_patients.csv

Run:
  python build_patient_master.py
"""

import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd


# -----------------------------
# Paths / filenames (edit if needed)
# -----------------------------
STRUCTURED_FILE = "_outputs/patient_stage_summary.csv"   # your structured patient-level file (has ENCRYPTED_PAT_ID)
CLINIC_NOTES_FILE = "_staging_inputs/HPI11526 Clinic Notes.csv"  # your clinic notes CSV
OUT_MASTER = "_outputs/patient_master.csv"
OUT_AUDIT_SUMMARY = "_outputs/build_audit_summary.csv"
OUT_AUDIT_PATIENTS = "_outputs/build_audit_patients.csv"

PID = "ENCRYPTED_PAT_ID"


# -----------------------------
# What we output / validate (subset)
# -----------------------------
OUTPUT_COLUMNS = [
    PID,
    "Race",
    "Ethnicity",
    "Age",
    "BMI",
    "SmokingStatus",
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
    "PBS_Lumpectomy",
    "PBS_Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "Radiation",
    "Chemo",
]


# -----------------------------
# Helpers
# -----------------------------
def _safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def normalize_spaces(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s


def has_positive(text: str, pattern: re.Pattern) -> bool:
    """
    True if pattern matches and it's not negated in a short window.
    Simple negation guard (not full NLP).
    """
    if not text:
        return False
    for m in pattern.finditer(text):
        start = max(0, m.start() - 40)
        window = text[start:m.start()].lower()
        # crude negation triggers
        if re.search(r"\b(no|denies|deny|without|neg|negative for|h/o\s+no)\b", window):
            continue
        return True
    return False


def get_first_match_int(text: str, pattern: re.Pattern):
    m = pattern.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def majority_vote(counter: Counter):
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def normalize_race(val):
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in ("white", "caucasian", "w", "white or caucasian"):
        return "Caucasian"
    if "black" in v or "african" in v:
        return "African American"
    if "asian" in v:
        return "Asian"
    if "native" in v and "american" in v:
        return "Native American"
    if "pacific" in v or "hawai" in v:
        return "Pacific Islander"
    if "other" in v:
        return "Other"
    if "unknown" in v or "declined" in v or v == "nan":
        return None
    return str(val).strip()


def normalize_ethnicity(val):
    if val is None:
        return None
    v = str(val).strip().lower()
    if ("not" in v and "hisp" in v) or ("non" in v and "hisp" in v):
        return "Non-hispanic"
    if "hisp" in v or "latino" in v:
        return "Hispanic"
    if "unknown" in v or "declined" in v or v == "nan":
        return None
    return str(val).strip()


def read_csv_or_die(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"ERROR: Missing file: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -----------------------------
# Regex patterns
# -----------------------------
# PMH header (MEDICAL) + common section headers as stop markers
PMH_HDR = re.compile(
    r"(^|\n)\s*(past\s+medical\s+history|pmh|medical\s+history)\s*:?\s*(\n|$)",
    re.IGNORECASE,
)

SECTION_STOP = re.compile(
    r"(^|\n)\s*(history of present illness|hpi|assessment|plan|review of systems|ros|medications|allergies|family history|social history|surgical history|problem list)\s*:?\s*(\n|$)",
    re.IGNORECASE,
)

# Diseases
RX_DM = re.compile(r"\b(diabetes|dm\b|type\s*2\s*dm|type\s*ii\s*dm)\b", re.IGNORECASE)
RX_HTN = re.compile(r"\b(hypertension|htn\b|high blood pressure)\b", re.IGNORECASE)
RX_VTE = re.compile(r"\b(dvt|pe\b|pulmonary embol(ism)?|deep vein thrombosis|vte|thromboembol(ism)?)\b", re.IGNORECASE)

# Cardiac: keep broad but avoid common false positives like "no chest pain"
RX_CARDIAC = re.compile(
    r"\b(cad\b|coronary artery disease|mi\b|myocardial infarction|chf\b|heart failure|afib|a-fib|atrial fibrillation|angina|cardiomyopathy)\b",
    re.IGNORECASE,
)

# Steroids
RX_STEROID = re.compile(
    r"\b(steroid(s)?|prednisone|prednisolone|methylprednisolone|solu-medrol|dexamethasone|hydrocortisone)\b",
    re.IGNORECASE,
)

# Medication “trigger” lists (secondary layer)
RX_DM_MEDS = re.compile(r"\b(metformin|insulin|glipizide|glyburide|glimepiride|liraglutide|semaglutide|sitagliptin|empagliflozin|canagliflozin)\b", re.IGNORECASE)
RX_HTN_MEDS = re.compile(r"\b(lisinopril|losartan|valsartan|amlodipine|metoprolol|carvedilol|hydrochlorothiazide|chlorthalidone|diltiazem|verapamil)\b", re.IGNORECASE)
RX_CARDIAC_MEDS = re.compile(r"\b(nitroglycerin|atorvastatin|rosuvastatin|clopidogrel|prasugrel|ticagrelor)\b", re.IGNORECASE)
RX_ANTICOAG = re.compile(r"\b(warfarin|coumadin|heparin|enoxaparin|lovenox|apixaban|eliquis|rivaroxaban|xarelto|dabigatran|pradaxa)\b", re.IGNORECASE)

# Lumpectomy evidence from clinic exam language
RX_LUMPECTOMY = re.compile(
    r"\b(lumpectomy(\s+site|\s+scar)?|status\s+post\s+lumpectomy|s\/p\s+lumpectomy|lumpectomy\s+scar)\b",
    re.IGNORECASE,
)

# Age patterns (use many variants; search early note text first, then anywhere)
# Examples: "46 yo", "46 y.o.", "46-year-old", "Age: 46", "46-year old", "46 y/o"
RX_AGE = re.compile(
    r"\b(?:age\s*[:=]?\s*)?(\d{2})\s*"
    r"(?:-?\s*(?:yo|y\/o|y\.o\.|year\s*old|years\s*old|yr\s*old|yrs\s*old))\b",
    re.IGNORECASE,
)
RX_AGE2 = re.compile(r"\b(\d{2})\s*[- ]?\s*(?:year)\s*[- ]?\s*(?:old)\b", re.IGNORECASE)
RX_AGE3 = re.compile(r"\b(\d{2})\s*[- ]?\s*(?:yo)\b", re.IGNORECASE)


def extract_pmh_block(note_text: str):
    """
    Return PMH block string if found, else None.
    Extract from PMH header until next section stop.
    """
    if not note_text:
        return None

    txt = normalize_spaces(note_text)
    m = PMH_HDR.search(txt)
    if not m:
        return None

    start = m.end()
    tail = txt[start:]

    s = SECTION_STOP.search(tail)
    end = s.start() if s else len(tail)

    block = tail[:end].strip()
    if not block:
        return None
    return block


def extract_age_from_note(note_text: str):
    """
    Search the first ~30 lines first (age is often at the top), then entire note.
    """
    if not note_text:
        return None

    txt = normalize_spaces(note_text)
    lines = txt.split("\n")
    head = "\n".join(lines[:30])

    for pat in (RX_AGE, RX_AGE2, RX_AGE3):
        age = get_first_match_int(head, pat)
        if age is not None:
            return age

    # fallback: anywhere in note
    for pat in (RX_AGE, RX_AGE2, RX_AGE3):
        age = get_first_match_int(txt, pat)
        if age is not None:
            return age

    return None


def detect_text_column(df: pd.DataFrame):
    """
    Clinic notes file has NO DEID column. We try common raw text columns only.
    """
    candidates = [
        "NOTE_TEXT",
        "NOTE",
        "TEXT",
        "NOTE_BODY",
        "NOTE_CONTENT",
        "BODY",
        "FullText",
        "full_text",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: any column containing 'text' but NOT deid
    text_cols = [c for c in df.columns if "text" in c.lower() and "deid" not in c.lower()]
    if text_cols:
        return text_cols[0]
    return None


# -----------------------------
# Build pipeline
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
        "_age_votes": Counter(),
        "_bmi_votes": Counter(),
        "_smoke_votes": Counter(),
        "_pmh_seen": 0,
        "_med_hits": 0,
        "_clinic_notes_seen": 0,
        "_lumpectomy_hits": 0,
    }


def ingest_structured(structured: pd.DataFrame, patients: dict, audit: dict):
    """
    Take what we can from structured data FIRST.
    If Age exists here, keep it (note age becomes fallback only).
    """
    if PID not in structured.columns:
        raise ValueError(f"Structured file missing {PID}")

    # Identify likely column names
    col_race = "Race" if "Race" in structured.columns else None
    col_eth = "Ethnicity" if "Ethnicity" in structured.columns else None
    col_age = "Age" if "Age" in structured.columns else None
    col_bmi = "BMI" if "BMI" in structured.columns else None
    col_smoke = "SmokingStatus" if "SmokingStatus" in structured.columns else None

    # surgery / treatment flags if present
    for idx, row in structured.iterrows():
        pid = _safe_str(row.get(PID)).strip()
        if not pid:
            continue

        st = patients[pid]

        if col_race:
            v = normalize_race(row.get(col_race))
            if v:
                st["_race_votes"][v] += 1
        if col_eth:
            v = normalize_ethnicity(row.get(col_eth))
            if v:
                st["_eth_votes"][v] += 1

        # Age: prefer structured
        if col_age and not pd.isna(row.get(col_age)):
            try:
                a = int(float(row.get(col_age)))
                if 0 < a < 120:
                    st["_age_votes"][a] += 1
            except Exception:
                pass

        if col_bmi and not pd.isna(row.get(col_bmi)):
            try:
                b = float(row.get(col_bmi))
                if 5.0 < b < 100.0:
                    st["_bmi_votes"][round(b, 1)] += 1
            except Exception:
                pass

        if col_smoke and not pd.isna(row.get(col_smoke)):
            sv = str(row.get(col_smoke)).strip()
            if sv:
                st["_smoke_votes"][sv] += 1

        # Pass-through for existing flags if present in structured
        for flag in ["PBS_Lumpectomy", "PBS_Reduction", "PBS_Mastopexy", "PBS_Augmentation", "Radiation", "Chemo"]:
            if flag in structured.columns and not pd.isna(row.get(flag)):
                try:
                    st[flag] = max(int(st[flag]), int(row.get(flag)))
                except Exception:
                    pass

    audit["structured_rows"] = len(structured)


def ingest_clinic_notes(notes: pd.DataFrame, patients: dict, audit: dict):
    if PID not in notes.columns:
        raise ValueError(f"Clinic notes file missing {PID}")

    text_col = detect_text_column(notes)
    if not text_col:
        raise ValueError(
            "Could not find clinic note text column. "
            "Expected something like NOTE_TEXT / NOTE / TEXT / NOTE_BODY."
        )

    audit["clinic_text_col"] = text_col
    audit["clinic_rows"] = len(notes)

    for _, row in notes.iterrows():
        pid = _safe_str(row.get(PID)).strip()
        if not pid:
            continue

        text = _safe_str(row.get(text_col))
        if not text:
            continue

        st = patients[pid]
        st["_clinic_notes_seen"] += 1

        txt = normalize_spaces(text)

        # Lumpectomy from exam language
        if st["PBS_Lumpectomy"] == 0 and has_positive(txt, RX_LUMPECTOMY):
            st["PBS_Lumpectomy"] = 1
            st["_lumpectomy_hits"] += 1

        # Age fallback: only if no structured age vote exists
        if len(st["_age_votes"]) == 0:
            age = extract_age_from_note(txt)
            if age is not None and 0 < age < 120:
                st["_age_votes"][age] += 1

        # Primary: PMH block
        pmh = extract_pmh_block(txt)
        if pmh:
            st["_pmh_seen"] += 1

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

        # Secondary: meds as trigger, then confirm by searching full note
        dm_med = has_positive(txt, RX_DM_MEDS)
        htn_med = has_positive(txt, RX_HTN_MEDS)
        card_med = has_positive(txt, RX_CARDIAC_MEDS)
        vte_med = has_positive(txt, RX_ANTICOAG)
        steroid_med = has_positive(txt, RX_STEROID)

        if dm_med or htn_med or card_med or vte_med or steroid_med:
            st["_med_hits"] += 1

        if st["Diabetes"] == 0 and dm_med and has_positive(txt, RX_DM):
            st["Diabetes"] = 1
        if st["Hypertension"] == 0 and htn_med and has_positive(txt, RX_HTN):
            st["Hypertension"] = 1
        if st["CardiacDisease"] == 0 and card_med and has_positive(txt, RX_CARDIAC):
            st["CardiacDisease"] = 1
        if st["VenousThromboembolism"] == 0 and vte_med and has_positive(txt, RX_VTE):
            st["VenousThromboembolism"] = 1
        if st["Steroid"] == 0 and steroid_med and has_positive(txt, RX_STEROID):
            st["Steroid"] = 1


def finalize(patients: dict) -> pd.DataFrame:
    rows = []
    for pid, st in patients.items():
        out = {PID: pid}

        out["Race"] = majority_vote(st["_race_votes"])
        out["Ethnicity"] = majority_vote(st["_eth_votes"])
        out["Age"] = majority_vote(st["_age_votes"])
        out["BMI"] = majority_vote(st["_bmi_votes"])
        out["SmokingStatus"] = majority_vote(st["_smoke_votes"])

        # binary outputs + existing pass-through
        for col in ["Diabetes", "Hypertension", "CardiacDisease", "VenousThromboembolism", "Steroid",
                    "PBS_Lumpectomy", "PBS_Reduction", "PBS_Mastopexy", "PBS_Augmentation",
                    "Radiation", "Chemo"]:
            out[col] = int(st.get(col, 0))

        rows.append(out)

    df = pd.DataFrame(rows)
    # ensure output columns exist
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[OUTPUT_COLUMNS]


def build_audit(patients: dict, audit_meta: dict):
    """
    Write audit outputs to understand why fields are zero.
    """
    summary = {
        "structured_rows": audit_meta.get("structured_rows", 0),
        "clinic_rows": audit_meta.get("clinic_rows", 0),
        "clinic_text_col": audit_meta.get("clinic_text_col", ""),
        "patients_total": len(patients),
        "patients_with_any_clinic_note": sum(1 for _, st in patients.items() if st["_clinic_notes_seen"] > 0),
        "patients_with_pmh_block": sum(1 for _, st in patients.items() if st["_pmh_seen"] > 0),
        "patients_with_med_trigger": sum(1 for _, st in patients.items() if st["_med_hits"] > 0),
        "patients_with_lumpectomy_hit": sum(1 for _, st in patients.items() if st["_lumpectomy_hits"] > 0),
        "patients_diabetes_1": sum(1 for _, st in patients.items() if st["Diabetes"] == 1),
        "patients_htn_1": sum(1 for _, st in patients.items() if st["Hypertension"] == 1),
        "patients_cardiac_1": sum(1 for _, st in patients.items() if st["CardiacDisease"] == 1),
        "patients_vte_1": sum(1 for _, st in patients.items() if st["VenousThromboembolism"] == 1),
        "patients_steroid_1": sum(1 for _, st in patients.items() if st["Steroid"] == 1),
    }
    df_sum = pd.DataFrame([summary])

    df_pat = pd.DataFrame([
        {
            PID: pid,
            "clinic_notes_seen": st["_clinic_notes_seen"],
            "pmh_blocks_seen": st["_pmh_seen"],
            "med_triggers_seen": st["_med_hits"],
            "lumpectomy_hits": st["_lumpectomy_hits"],
            "Diabetes": st["Diabetes"],
            "Hypertension": st["Hypertension"],
            "CardiacDisease": st["CardiacDisease"],
            "VenousThromboembolism": st["VenousThromboembolism"],
            "Steroid": st["Steroid"],
            "Age_votes": len(st["_age_votes"]),
        }
        for pid, st in patients.items()
    ])

    ensure_dir(OUT_AUDIT_SUMMARY)
    df_sum.to_csv(OUT_AUDIT_SUMMARY, index=False)
    df_pat.to_csv(OUT_AUDIT_PATIENTS, index=False)


def main():
    print("Loading files...")
    structured = read_csv_or_die(STRUCTURED_FILE)
    clinic = read_csv_or_die(CLINIC_NOTES_FILE)

    print(f"Structured rows: {len(structured)}")
    print(f"Clinic note rows: {len(clinic)}")

    # init patients from structured IDs (and also allow notes-only patients)
    patients = defaultdict(init_patient_state)

    audit_meta = {}

    # Ingest structured first (Race/Ethnicity/Age/BMI/Smoking etc.)
    ingest_structured(structured, patients, audit_meta)

    # Ensure we also include patients appearing only in clinic notes
    for pid in clinic[PID].dropna().astype(str).str.strip().unique().tolist():
        _ = patients[pid]

    # Ingest clinic notes
    ingest_clinic_notes(clinic, patients, audit_meta)

    # Finalize
    master = finalize(patients)

    ensure_dir(OUT_MASTER)
    master.to_csv(OUT_MASTER, index=False)

    # Audit
    build_audit(patients, audit_meta)

    print("Build complete.")
    print(f"Wrote: {OUT_MASTER}")
    print(f"Wrote: {OUT_AUDIT_SUMMARY}")
    print(f"Wrote: {OUT_AUDIT_PATIENTS}")


if __name__ == "__main__":
    main()
