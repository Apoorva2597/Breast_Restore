#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a patient-level master table for validation.

Inputs:
  1) Operation encounters CSV (structured) with MRN + ENCRYPTED_PAT_ID + AGE_AT_ENCOUNTER + RACE + ETHNICITY
  2) Clinic notes CSV (text) with ENCRYPTED_PAT_ID and note text (column name may vary)

Outputs:
  _outputs/patient_master.csv

Python: 3.6+ compatible
"""

import os
import re
import sys
import argparse
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

# ----------------------------
# Helpers (Python 3.6-safe)
# ----------------------------

def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()

def _lower(s: Any) -> str:
    return _norm(s).lower()

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
    s = series.dropna().astype(str).map(lambda x: x.strip()).replace("", pd.NA).dropna()
    if s.empty:
        return None
    return s.value_counts().idxmax()

def compile_re(pat: str, flags=re.IGNORECASE):
    return re.compile(pat, flags)

# ----------------------------
# Note section extraction
# ----------------------------

SECTION_STOP_PAT = compile_re(
    r"\n\s*(PAST SURGICAL HISTORY|SURGICAL HISTORY|FAMILY HISTORY|SOCIAL HISTORY|"
    r"REVIEW OF SYSTEMS|ROS:|MEDICATIONS|ALLERGIES|PHYSICAL EXAM|VITAL SIGNS|ASSESSMENT|PLAN|IMPRESSION)\s*[:\n]"
)

def extract_section(text: str, header_regex: re.Pattern, max_chars: int = 4000) -> str:
    """
    Extract a section that starts with header_regex until the next SECTION_STOP_PAT.
    Returns "" if not found.
    """
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

PMH_HEADER = compile_re(r"(?:^|\n)\s*PAST\s+MEDICAL\s+HISTORY\s*[:\n]")
PSH_HEADER = compile_re(r"(?:^|\n)\s*PAST\s+SURGICAL\s+HISTORY\s*[:\n]")

# ----------------------------
# BMI extraction
# ----------------------------

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
    # Prefer BMI mentions near PHYSICAL EXAM / VITAL SIGNS if present
    # (but still fall back to full text)
    candidates = []

    # Try slicing around "PHYSICAL EXAM" if present
    idx = _lower(text).find("physical exam")
    if idx != -1:
        window = text[idx: idx + 3000]
        candidates.append(window)

    candidates.append(text[:5000])  # early note tends to have vitals
    candidates.append(text)

    for blob in candidates:
        for pat in BMI_PATTERNS:
            m = pat.search(blob)
            if m:
                val = safe_float(m.group(1))
                # sanity bounds
                if val is not None and 10.0 <= val <= 80.0:
                    return float(val)
    return None

# ----------------------------
# Smoking status extraction
# ----------------------------

def extract_smoking_status(text: str) -> Optional[str]:
    if not text:
        return None
    t = _lower(text)

    # Strong explicit phrases first
    if "never smoker" in t or "never smoked" in t:
        return "Never"
    if "current smoker" in t:
        return "Current"
    if "former smoker" in t or "quit date" in t or "quit:" in t:
        return "Former"

    # Structured line example: "Smoking status: Former Smoker"
    m = re.search(r"smoking\s+status\s*:\s*(never|former|current)", t)
    if m:
        return m.group(1).capitalize()

    # Heuristics
    if "denies tobacco" in t or "no tobacco" in t:
        # could still be former; keep as Never only if nothing indicates past use
        if "quit" in t or "pack-year" in t:
            return "Former"
        return "Never"

    if "pack-year" in t or "packs/day" in t or "cigarettes" in t:
        if "quit" in t or "former" in t:
            return "Former"
        return "Current"

    return None

# ----------------------------
# Comorbidity extraction from PMH (+ meds as backup)
# ----------------------------

NEGATION_WINDOW = 40

def has_negated(term_pat: re.Pattern, text: str) -> bool:
    """
    Detect 'no history of X' / 'denies X' close to term.
    """
    for m in term_pat.finditer(text):
        left = text[max(0, m.start() - NEGATION_WINDOW):m.start()]
        if re.search(r"\b(no|denies|without|negative for)\b", left):
            return True
    return False

def positive_term(term_pat: re.Pattern, text: str) -> bool:
    if not text:
        return False
    if not term_pat.search(text):
        return False
    # If ONLY negated occurrences, treat as negative
    if has_negated(term_pat, text):
        # still could be both; keep simple: require at least one non-negated hit
        for m in term_pat.finditer(text):
            left = text[max(0, m.start() - NEGATION_WINDOW):m.start()]
            if not re.search(r"\b(no|denies|without|negative for)\b", left):
                return True
        return False
    return True

# Term patterns
PAT_DM = compile_re(r"\b(diabetes|dm\s*(i|ii|1|2)?|type\s*(i|ii|1|2)\s*diabetes)\b")
PAT_HTN = compile_re(r"\b(hypertension|htn)\b")
PAT_CAD = compile_re(r"\b(coronary\s+artery\s+disease|cad|chf|congestive\s+heart\s+failure|mi\b|myocardial\s+infarction)\b")

PAT_DVTPE = compile_re(r"\b(dvt|deep\s+vein\s+thrombosis|pulmonary\s+embol(ism|us)|\bpe\b)\b")
PAT_VTE_RISK = compile_re(r"\b(vte\s+risk|risk\s+assessment|risk\s+score|caprini)\b")

# Steroid meds / terms
PAT_STEROID = compile_re(r"\b(prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone|"
                         r"solu-medrol|medrol|steroid(s)?\b)\b")

# Diabetes meds (backup signal)
PAT_DM_MEDS = compile_re(r"\b(metformin|insulin|lantus|levemir|humalog|novolog|glipizide|glyburide|"
                         r"januvia|sitagliptin|ozempic|semaglutide|trulicity|dulaglutide|"
                         r"empagliflozin|jardiance|dapagliflozin|farxiga)\b")

def clean_pmh_block(pmh: str) -> str:
    """
    Remove family-history-like lines from PMH block.
    """
    if not pmh:
        return ""
    lines = pmh.splitlines()
    kept = []
    for ln in lines:
        l = _lower(ln)
        # drop explicit family-history lines that sometimes appear inside PMH tables
        if "family history" in l:
            continue
        if re.search(r"\b(mother|father|sister|brother|grandmother|grandfather|aunt|uncle)\b", l) and "history" in l:
            # conservative: likely family history row
            continue
        kept.append(ln)
    return "\n".join(kept).strip()

def extract_comorbidities(note_text: str) -> Dict[str, int]:
    """
    Returns 0/1 flags: Diabetes, Hypertension, CardiacDisease, VenousThromboembolism, Steroid
    Uses PMH primarily; meds/full-note as backup.
    """
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

    # Diabetes
    dm_pos = positive_term(PAT_DM, pmh_low)
    dm_meds = bool(PAT_DM_MEDS.search(full_low))
    if dm_pos or (dm_meds and positive_term(PAT_DM, full_low)):
        out["Diabetes"] = 1

    # Hypertension
    if positive_term(PAT_HTN, pmh_low):
        out["Hypertension"] = 1

    # Cardiac disease
    if positive_term(PAT_CAD, pmh_low):
        out["CardiacDisease"] = 1

    # VTE: only count true history, not risk score language
    # If we see DVT/PE in PMH and it's not negated -> positive
    if positive_term(PAT_DVTPE, pmh_low):
        out["VenousThromboembolism"] = 1
    else:
        # Avoid triggering from VTE risk assessment blocks
        if PAT_DVTPE.search(full_low) and not PAT_VTE_RISK.search(full_low):
            if positive_term(PAT_DVTPE, full_low):
                out["VenousThromboembolism"] = 1

    # Steroids (prefer meds list / med strings)
    if PAT_STEROID.search(full_low):
        # If explicitly negated? uncommon; accept presence
        out["Steroid"] = 1

    return out

# ----------------------------
# Past breast surgery signals
# ----------------------------

PAT_LUMP_SCAR = compile_re(r"\blumpectomy\s+scar\b")
PAT_MASTOPEXY = compile_re(r"\bmastopexy\b")
PAT_AUGMENT = compile_re(r"\b(augmentation|breast\s+implants?)\b")

def extract_pbs_flags(note_text: str) -> Dict[str, int]:
    """
    Conservative flags for prior breast surgery types from clinic notes.
    """
    out = {
        "PBS_Lumpectomy": 0,
        "PBS_Mastopexy": 0,
        "PBS_Augmentation": 0,
    }
    if not note_text:
        return out

    t = _lower(note_text)

    # lumpectomy: your rule
    if PAT_LUMP_SCAR.search(t) or re.search(r"\bhx\s+of\s+lumpectomy\b", t):
        out["PBS_Lumpectomy"] = 1

    if PAT_MASTOPEXY.search(t):
        out["PBS_Mastopexy"] = 1

    if PAT_AUGMENT.search(t):
        out["PBS_Augmentation"] = 1

    return out

# ----------------------------
# Main build
# ----------------------------

def build_patient_master(encounters_csv: str,
                         clinic_notes_csv: str,
                         out_csv: str) -> None:

    print("Loading structured encounters:", encounters_csv)
    enc = pd.read_csv(encounters_csv, dtype=str, low_memory=False)

    # Required columns in encounters
    required_any = ["MRN", "ENCRYPTED_PAT_ID"]
    for c in required_any:
        if c not in enc.columns:
            raise ValueError("Encounters file missing required column: {}".format(c))

    # Age column (structured)
    age_col = pick_first_existing_col(enc, ["AGE_AT_ENCOUNTER", "AGE", "Age", "AGE_AT_VISIT"])
    if not age_col:
        print("WARNING: No structured age column found in encounters. Will rely on notes age (not ideal).")

    # Normalize key columns
    enc["MRN"] = enc["MRN"].astype(str).str.strip()
    enc["ENCRYPTED_PAT_ID"] = enc["ENCRYPTED_PAT_ID"].astype(str).str.strip()

    # Patient-level base table from encounters
    grp = enc.groupby("ENCRYPTED_PAT_ID", dropna=False)

    base = pd.DataFrame({
        "ENCRYPTED_PAT_ID": list(grp.groups.keys())
    })

    # MRN mapping (most frequent)
    base["MRN"] = grp["MRN"].apply(most_frequent_nonnull).values

    # Race / Ethnicity (most frequent)
    race_col = pick_first_existing_col(enc, ["RACE", "Race", "RACE_NAME"])
    eth_col  = pick_first_existing_col(enc, ["ETHNICITY", "Ethnicity", "ETHNICITY_NAME"])

    if race_col:
        base["Race"] = grp[race_col].apply(most_frequent_nonnull).values
    else:
        base["Race"] = None

    if eth_col:
        base["Ethnicity"] = grp[eth_col].apply(most_frequent_nonnull).values
    else:
        base["Ethnicity"] = None

    # Age: median of structured ages
    if age_col:
        def med_age(s):
            vals = [safe_int(x) for x in s.dropna().tolist()]
            vals = [v for v in vals if v is not None and 0 < v < 120]
            if not vals:
                return None
            vals_sorted = sorted(vals)
            mid = len(vals_sorted) // 2
            if len(vals_sorted) % 2 == 1:
                return vals_sorted[mid]
            return int(round((vals_sorted[mid-1] + vals_sorted[mid]) / 2.0))
        base["Age"] = grp[age_col].apply(med_age).values
    else:
        base["Age"] = None

    print("Structured patients:", len(base))

    # Load clinic notes
    print("Loading clinic notes:", clinic_notes_csv)
    notes = pd.read_csv(clinic_notes_csv, dtype=str, low_memory=False)

    if "ENCRYPTED_PAT_ID" not in notes.columns:
        raise ValueError("Clinic notes file missing ENCRYPTED_PAT_ID column")

    # Auto-detect note text column (DO NOT assume NOTE_TEXT_DEID)
    text_col = pick_first_existing_col(
        notes,
        [
            "NOTE_TEXT", "NOTE_TEXT_CLEAN", "TEXT", "FULL_TEXT", "NOTE_BODY",
            "NOTE_TEXT_DEID", "NOTE_TEXT_DEIDENTIFIED", "NOTE", "DOCUMENT_TEXT"
        ]
    )
    if not text_col:
        raise ValueError(
            "Could not find note text column in clinic notes. "
            "Looked for NOTE_TEXT/TEXT/FULL_TEXT/etc. Available columns: {}".format(list(notes.columns)[:50])
        )

    print("Using note text column:", text_col)

    notes["ENCRYPTED_PAT_ID"] = notes["ENCRYPTED_PAT_ID"].astype(str).str.strip()
    notes[text_col] = notes[text_col].fillna("").astype(str)

    # We only need clinic notes grouped by patient
    notes_grp = notes.groupby("ENCRYPTED_PAT_ID", dropna=False)

    # Aggregation for patient-level signals across multiple notes
    def agg_patient_notes(pid: str) -> Dict[str, Any]:
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

        # Iterate notes (keep lightweight)
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
                flags[k] = 1 if (flags[k] == 1 or cm.get(k, 0) == 1) else 0

            pf = extract_pbs_flags(txt)
            for k in pbs:
                pbs[k] = 1 if (pbs[k] == 1 or pf.get(k, 0) == 1) else 0

        # Patient BMI: choose median of extracted BMIs
        bmi_final = None
        if bmi_vals:
            bmi_vals_sorted = sorted(bmi_vals)
            mid = len(bmi_vals_sorted) // 2
            bmi_final = bmi_vals_sorted[mid] if len(bmi_vals_sorted) % 2 == 1 else (bmi_vals_sorted[mid-1] + bmi_vals_sorted[mid]) / 2.0

        # Smoking: if multiple, prefer Current > Former > Never (clinically sensible precedence)
        smoke_final = None
        if smoke_vals:
            order = {"Current": 3, "Former": 2, "Never": 1}
            smoke_final = sorted(smoke_vals, key=lambda x: order.get(x, 0), reverse=True)[0]

        out = {"BMI": bmi_final, "SmokingStatus": smoke_final}
        out.update(flags)
        out.update(pbs)
        return out

    print("Deriving note-based features (PMH/BMI/Smoking/PBS)...")
    derived_rows = []
    for pid in base["ENCRYPTED_PAT_ID"].tolist():
        derived_rows.append(agg_patient_notes(pid))

    derived = pd.DataFrame(derived_rows)

    # Merge into base
    master = pd.concat([base.reset_index(drop=True), derived.reset_index(drop=True)], axis=1)

    # Final cleanup / types
    # BMI numeric formatting (keep as float)
    # Age keep as int
    master["Age"] = master["Age"].apply(lambda x: safe_int(x) if x is not None else None)
    master["BMI"] = master["BMI"].apply(lambda x: float(x) if x is not None else None)

    # Optional: normalize Race/Ethnicity to match typical gold formatting
    def norm_race(x):
        if not x:
            return x
        t = str(x).strip()
        # common normalization examples
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

    # Fill missing binary flags with 0
    bin_cols = ["Diabetes", "Hypertension", "CardiacDisease", "VenousThromboembolism", "Steroid",
                "PBS_Lumpectomy", "PBS_Mastopexy", "PBS_Augmentation"]
    for c in bin_cols:
        master[c] = master[c].fillna(0).astype(int)

    # Basic reporting
    print("Rows:", len(master))
    print("Patients with BMI:", int(master["BMI"].notna().sum()))
    print("Patients with SmokingStatus:", int(master["SmokingStatus"].notna().sum()))
    print("Patients with any note-derived PMH flags:",
          int(((master["Diabetes"] + master["Hypertension"] + master["CardiacDisease"] +
                master["VenousThromboembolism"] + master["Steroid"]) > 0).sum()))

    # Write
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Writing:", out_csv)
    master.to_csv(out_csv, index=False)
    print("Done.")

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encounters", required=True, help="Structured operation encounters CSV")
    ap.add_argument("--clinic_notes", required=True, help="Clinic notes CSV (text)")
    ap.add_argument("--out", default="_outputs/patient_master.csv", help="Output patient master CSV")
    return ap.parse_args(argv)

def main():
    args = parse_args(sys.argv[1:])
    build_patient_master(args.encounters, args.clinic_notes, args.out)

if __name__ == "__main__":
    main()
