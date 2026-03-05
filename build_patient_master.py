#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_patient_master.py (NO CLI args)

Reads:
  1) Structured operation encounters CSV:
       - Must include: MRN, ENCRYPTED_PAT_ID
       - Uses (if present): AGE_AT_ENCOUNTER, RACE, ETHNICITY
  2) Clinic notes CSV:
       - Must include: ENCRYPTED_PAT_ID
       - Must include a note text column (auto-detected from common names)

Writes:
  _outputs/patient_master.csv

Run:
  python build_patient_master.py

Python 3.6 compatible.
"""

import os
import re
import pandas as pd
from typing import Optional, Any, List, Dict

# =========================
# CONFIG: EDIT PATHS HERE
# =========================
ENCOUNTERS_CSV = "/home/apokol/Breast_Restore/_staging_inputs/HP11526 Operation Encounters.csv"
CLINIC_NOTES_CSV = "/home/apokol/Breast_Restore/_staging_inputs/HP11526 Clinic Notes.csv"  
OUT_CSV = "_outputs/patient_master.csv"

# =========================
# Small utilities
# =========================

def _norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def _lower(x: Any) -> str:
    return _norm(x).lower()

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        return float(s)
    except Exception:
        return None

def safe_int(x: Any) -> Optional[int]:
    f = safe_float(x)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None

def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def most_frequent_nonnull(series: pd.Series) -> Optional[str]:
    s = series.dropna().astype(str).map(lambda z: z.strip()).replace("", pd.NA).dropna()
    if s.empty:
        return None
    return s.value_counts().idxmax()

def compile_re(pat: str, flags=re.IGNORECASE):
    return re.compile(pat, flags)

# =========================
# Section extraction
# =========================

SECTION_STOP_PAT = compile_re(
    r"\n\s*(PAST SURGICAL HISTORY|SURGICAL HISTORY|FAMILY HISTORY|SOCIAL HISTORY|"
    r"REVIEW OF SYSTEMS|ROS:|MEDICATIONS|ALLERGIES|PHYSICAL EXAM|VITAL SIGNS|ASSESSMENT|PLAN|IMPRESSION)\s*[:\n]"
)

PMH_HEADER = compile_re(r"(?:^|\n)\s*PAST\s+MEDICAL\s+HISTORY\s*[:\n]")

def extract_section(text: str, header_regex, max_chars: int = 4000) -> str:
    """header_regex is a compiled regex; left untyped for Python 3.6 compatibility."""
    if not text:
        return ""
    m = header_regex.search(text)
    if not m:
        return ""
    start = m.end()
    chunk = text[start:start + max_chars]
    stop = SECTION_STOP_PAT.search(chunk)
    if stop:
        chunk = chunk[:stop.start()]
    return chunk.strip()

def clean_pmh_block(pmh: str) -> str:
    """Drop obvious family-history lines to reduce false positives."""
    if not pmh:
        return ""
    lines = pmh.splitlines()
    kept = []
    for ln in lines:
        l = _lower(ln)
        if "family history" in l:
            continue
        if re.search(r"\b(mother|father|sister|brother|grandmother|grandfather|aunt|uncle)\b", l) and "history" in l:
            continue
        kept.append(ln)
    return "\n".join(kept).strip()

# =========================
# BMI extraction
# =========================

BMI_PATTERNS = [
    compile_re(r"\bBMI\b[^0-9]{0,15}(\d{1,2}\.\d{1,2})\b"),
    compile_re(r"\bBMI\b[^0-9]{0,15}(\d{1,2})\b"),
    compile_re(r"\bBody\s+mass\s+index\b[^0-9]{0,20}(\d{1,2}\.\d{1,2})\b"),
    compile_re(r"\bBody\s+mass\s+index\b[^0-9]{0,20}(\d{1,2})\b"),
    compile_re(r"\bkg\/m2\b[^0-9]{0,20}(\d{1,2}\.\d{1,2})\b"),
]

def extract_bmi(text: str) -> Optional[float]:
    if not text:
        return None

    blobs = []
    idx = _lower(text).find("physical exam")
    if idx != -1:
        blobs.append(text[idx: idx + 3000])  # vitals often immediately follow PHYSICAL EXAM
    blobs.append(text[:5000])
    blobs.append(text)

    for blob in blobs:
        for pat in BMI_PATTERNS:
            m = pat.search(blob)
            if m:
                val = safe_float(m.group(1))
                if val is not None and 10.0 <= val <= 80.0:
                    return float(val)
    return None

# =========================
# Smoking extraction
# =========================

def extract_smoking_status(text: str) -> Optional[str]:
    if not text:
        return None
    t = _lower(text)

    if "never smoker" in t or "never smoked" in t:
        return "Never"
    if "current smoker" in t:
        return "Current"
    if "former smoker" in t or "quit date" in t or "quit:" in t:
        return "Former"

    m = re.search(r"smoking\s+status\s*:\s*(never|former|current)", t)
    if m:
        return m.group(1).capitalize()

    if "denies tobacco" in t or "no tobacco" in t:
        if "quit" in t or "pack-year" in t:
            return "Former"
        return "Never"

    if "pack-year" in t or "packs/day" in t or "cigarettes" in t:
        if "quit" in t or "former" in t:
            return "Former"
        return "Current"

    return None

# =========================
# Comorbidity extraction from PMH (+ minimal backup)
# =========================

NEGATION_WINDOW = 40

def has_negated(term_pat, text: str) -> bool:
    for m in term_pat.finditer(text):
        left = text[max(0, m.start() - NEGATION_WINDOW):m.start()]
        if re.search(r"\b(no|denies|without|negative for)\b", left):
            return True
    return False

def positive_term(term_pat, text: str) -> bool:
    if not text or not term_pat.search(text):
        return False
    if has_negated(term_pat, text):
        for m in term_pat.finditer(text):
            left = text[max(0, m.start() - NEGATION_WINDOW):m.start()]
            if not re.search(r"\b(no|denies|without|negative for)\b", left):
                return True
        return False
    return True

PAT_DM = compile_re(r"\b(diabetes|dm\s*(i|ii|1|2)?|type\s*(i|ii|1|2)\s*diabetes)\b")
PAT_HTN = compile_re(r"\b(hypertension|htn)\b")
PAT_CAD = compile_re(r"\b(coronary\s+artery\s+disease|cad|chf|congestive\s+heart\s+failure|mi\b|myocardial\s+infarction)\b")

PAT_DVTPE = compile_re(r"\b(dvt|deep\s+vein\s+thrombosis|pulmonary\s+embol(ism|us)|\bpe\b)\b")
PAT_VTE_RISK = compile_re(r"\b(vte\s+risk|risk\s+assessment|risk\s+score|caprini|risk\s+level)\b")

PAT_STEROID = compile_re(r"\b(prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone|"
                         r"solu-medrol|medrol|steroid(s)?\b)\b")

PAT_DM_MEDS = compile_re(r"\b(metformin|insulin|lantus|levemir|humalog|novolog|glipizide|glyburide|"
                         r"januvia|sitagliptin|ozempic|semaglutide|trulicity|dulaglutide|"
                         r"empagliflozin|jardiance|dapagliflozin|farxiga)\b")

def extract_comorbidities(note_text: str) -> Dict[str, int]:
    out = {
        "Diabetes": 0,
        "Hypertension": 0,
        "CardiacDisease": 0,
        "VenousThromboembolism": 0,
        "Steroid": 0,
    }
    if not note_text:
        return out

    pmh_raw = extract_section(note_text, PMH_HEADER)
    pmh = clean_pmh_block(pmh_raw)

    full_low = _lower(note_text)
    pmh_low = _lower(pmh)

    dm_pos = positive_term(PAT_DM, pmh_low)
    dm_meds = bool(PAT_DM_MEDS.search(full_low))
    if dm_pos or (dm_meds and positive_term(PAT_DM, full_low)):
        out["Diabetes"] = 1

    if positive_term(PAT_HTN, pmh_low):
        out["Hypertension"] = 1

    if positive_term(PAT_CAD, pmh_low):
        out["CardiacDisease"] = 1

    # VTE: avoid risk-assessment sections producing false positives
    if positive_term(PAT_DVTPE, pmh_low):
        out["VenousThromboembolism"] = 1
    else:
        if PAT_DVTPE.search(full_low) and not PAT_VTE_RISK.search(full_low):
            if positive_term(PAT_DVTPE, full_low):
                out["VenousThromboembolism"] = 1

    if PAT_STEROID.search(full_low):
        out["Steroid"] = 1

    return out

# =========================
# Past breast surgery signals (lumpectomy scar in clinic notes)
# =========================

PAT_LUMP_SCAR = compile_re(r"\blumpectomy\s+scar\b")
PAT_LUMP = compile_re(r"\blumpectomy\b")
PAT_MASTOPEXY = compile_re(r"\bmastopexy\b")
PAT_AUGMENT = compile_re(r"\b(augmentation|breast\s+implants?)\b")

def extract_pbs_flags(note_text: str) -> Dict[str, int]:
    out = {"PBS_Lumpectomy": 0, "PBS_Mastopexy": 0, "PBS_Augmentation": 0}
    if not note_text:
        return out
    t = _lower(note_text)

    # prioritize "lumpectomy scar" signal
    if PAT_LUMP_SCAR.search(t) or re.search(r"\bhx\s+of\s+lumpectomy\b", t) or PAT_LUMP.search(t):
        out["PBS_Lumpectomy"] = 1
    if PAT_MASTOPEXY.search(t):
        out["PBS_Mastopexy"] = 1
    if PAT_AUGMENT.search(t):
        out["PBS_Augmentation"] = 1
    return out

# =========================
# Build
# =========================

def build_patient_master(encounters_csv: str, clinic_notes_csv: str, out_csv: str) -> None:
    print("Loading structured encounters:", encounters_csv)
    enc = pd.read_csv(encounters_csv, dtype=str, low_memory=False)

    for c in ["MRN", "ENCRYPTED_PAT_ID"]:
        if c not in enc.columns:
            raise ValueError("Encounters file missing required column: {}".format(c))

    age_col = pick_first_existing_col(enc, ["AGE_AT_ENCOUNTER", "AGE", "Age", "AGE_AT_VISIT"])
    race_col = pick_first_existing_col(enc, ["RACE", "Race", "RACE_NAME"])
    eth_col  = pick_first_existing_col(enc, ["ETHNICITY", "Ethnicity", "ETHNICITY_NAME"])

    enc["MRN"] = enc["MRN"].astype(str).str.strip()
    enc["ENCRYPTED_PAT_ID"] = enc["ENCRYPTED_PAT_ID"].astype(str).str.strip()

    grp = enc.groupby("ENCRYPTED_PAT_ID", dropna=False)

    base = pd.DataFrame({"ENCRYPTED_PAT_ID": list(grp.groups.keys())})
    base["MRN"] = grp["MRN"].apply(most_frequent_nonnull).values
    base["Race"] = grp[race_col].apply(most_frequent_nonnull).values if race_col else None
    base["Ethnicity"] = grp[eth_col].apply(most_frequent_nonnull).values if eth_col else None

    if age_col:
        def median_age(s):
            vals = [safe_int(x) for x in s.dropna().tolist()]
            vals = [v for v in vals if v is not None and 0 < v < 120]
            if not vals:
                return None
            vals = sorted(vals)
            mid = len(vals) // 2
            if len(vals) % 2 == 1:
                return vals[mid]
            return int(round((vals[mid - 1] + vals[mid]) / 2.0))
        base["Age"] = grp[age_col].apply(median_age).values
    else:
        base["Age"] = None

    print("Structured patients:", len(base))

    print("Loading clinic notes:", clinic_notes_csv)
    notes = pd.read_csv(clinic_notes_csv, dtype=str, low_memory=False)

    if "ENCRYPTED_PAT_ID" not in notes.columns:
        raise ValueError("Clinic notes file missing ENCRYPTED_PAT_ID column")

    # DE-ID REMOVED: only real text column names
    text_col = pick_first_existing_col(
        notes,
        ["NOTE_TEXT", "TEXT", "FULL_TEXT", "NOTE_BODY", "NOTE", "DOCUMENT_TEXT", "BODY", "CONTENT"]
    )
    if not text_col:
        raise ValueError(
            "Could not find note text column in clinic notes. "
            "Looked for NOTE_TEXT/TEXT/FULL_TEXT/NOTE/etc. Available columns: {}".format(list(notes.columns)[:80])
        )

    print("Using note text column:", text_col)

    notes["ENCRYPTED_PAT_ID"] = notes["ENCRYPTED_PAT_ID"].astype(str).str.strip()
    notes[text_col] = notes[text_col].fillna("").astype(str)

    notes_grp = notes.groupby("ENCRYPTED_PAT_ID", dropna=False)

    def agg_patient(pid: str) -> Dict[str, Any]:
        if pid not in notes_grp.groups:
            return {
                "BMI": None,
                "SmokingStatus": None,
                "Diabetes": 0, "Hypertension": 0, "CardiacDisease": 0, "VenousThromboembolism": 0, "Steroid": 0,
                "PBS_Lumpectomy": 0, "PBS_Mastopexy": 0, "PBS_Augmentation": 0,
            }

        dfp = notes_grp.get_group(pid)

        bmi_vals = []
        smoke_vals = []
        flags = {"Diabetes": 0, "Hypertension": 0, "CardiacDisease": 0, "VenousThromboembolism": 0, "Steroid": 0}
        pbs = {"PBS_Lumpectomy": 0, "PBS_Mastopexy": 0, "PBS_Augmentation": 0}

        for txt in dfp[text_col].tolist():
            if not txt:
                continue

            bmi = extract_bmi(txt)
            if bmi is not None:
                bmi_vals.append(bmi)

            sm = extract_smoking_status(txt)
            if sm:
                smoke_vals.append(sm)

            cm = extract_comorbidities(txt)
            for k in flags:
                if cm.get(k, 0) == 1:
                    flags[k] = 1

            pf = extract_pbs_flags(txt)
            for k in pbs:
                if pf.get(k, 0) == 1:
                    pbs[k] = 1

        bmi_final = None
        if bmi_vals:
            bmi_vals = sorted(bmi_vals)
            mid = len(bmi_vals) // 2
            bmi_final = bmi_vals[mid] if len(bmi_vals) % 2 == 1 else (bmi_vals[mid - 1] + bmi_vals[mid]) / 2.0

        smoke_final = None
        if smoke_vals:
            order = {"Current": 3, "Former": 2, "Never": 1}
            smoke_final = sorted(smoke_vals, key=lambda x: order.get(x, 0), reverse=True)[0]

        out = {"BMI": bmi_final, "SmokingStatus": smoke_final}
        out.update(flags)
        out.update(pbs)
        return out

    print("Deriving note-based features (PMH/BMI/Smoking/PBS)...")
    derived = pd.DataFrame([agg_patient(pid) for pid in base["ENCRYPTED_PAT_ID"].tolist()])

    master = pd.concat([base.reset_index(drop=True), derived.reset_index(drop=True)], axis=1)

    # Normalize a couple of common gold labels
    def norm_race(x):
        if not x:
            return x
        t = str(x).strip()
        t = t.replace("White or Caucasian", "Caucasian")
        t = t.replace("Black or African American", "Black")
        return t

    def norm_eth(x):
        if not x:
            return x
        t = str(x).strip()
        t = t.replace("Non-Hispanic", "Non-hispanic")
        t = t.replace("Hispanic or Latino", "Hispanic")
        return t

    master["Race"] = master["Race"].apply(norm_race)
    master["Ethnicity"] = master["Ethnicity"].apply(norm_eth)

    master["Age"] = master["Age"].apply(lambda x: safe_int(x) if x is not None else None)
    master["BMI"] = master["BMI"].apply(lambda x: float(x) if x is not None else None)

    bin_cols = ["Diabetes", "Hypertension", "CardiacDisease", "VenousThromboembolism", "Steroid",
                "PBS_Lumpectomy", "PBS_Mastopexy", "PBS_Augmentation"]
    for c in bin_cols:
        master[c] = master[c].fillna(0).astype(int)

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Rows:", len(master))
    print("Patients with BMI:", int(master["BMI"].notna().sum()))
    print("Patients with SmokingStatus:", int(master["SmokingStatus"].notna().sum()))
    print("Patients with any PMH flag:",
          int(((master["Diabetes"] + master["Hypertension"] + master["CardiacDisease"] +
                master["VenousThromboembolism"] + master["Steroid"]) > 0).sum()))

    print("Writing:", out_csv)
    master.to_csv(out_csv, index=False)
    print("Done.")

def main():
    # basic file existence checks
    for p in [ENCOUNTERS_CSV, CLINIC_NOTES_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError("File not found: {} (edit paths at top of script)".format(p))

    build_patient_master(ENCOUNTERS_CSV, CLINIC_NOTES_CSV, OUT_CSV)

if __name__ == "__main__":
    main()
