# build_master_rule_COMORBIDITY_PATCH.py
# Python 3.6.8 compatible
#
# Purpose:
#   Separate build script for comorbidity abstraction only.
#   It PATCHES the existing master file by adding/updating:
#   Obesity, Diabetes, Hypertension, CardiacDisease,
#   VenousThromboembolism, Steroid
#
# Important:
#   - Does NOT overwrite your old extractor files
#   - Does NOT create a master with only comorbidity columns
#   - Reads your existing master and writes a new patched output file
#
# Update the paths/globs below before running.

from __future__ import print_function

import os
import re
import sys
from glob import glob

import pandas as pd

from models import Candidate, SectionedNote

# =========================================================
# USER PATHS - EDIT THESE
# =========================================================

MASTER_IN = r"master.csv"
MASTER_OUT = r"master_with_comorbidity_patch.csv"

# Note chunk files / note exports
NOTE_GLOBS = [
    r"notes\*.csv",
    r"notes\**\*.csv",
]

MERGE_KEY = "MRN"

# =========================================================
# BASIC IO HELPERS
# =========================================================

def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", error_bad_lines=False, warn_bad_lines=True)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", error_bad_lines=False, warn_bad_lines=True)

def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def clean_cell(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def normalize_mrn(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "patient_mrn", "MedicalRecordNumber"]
    found = None
    for k in key_variants:
        if k in df.columns:
            found = k
            break
    if found is None:
        return df
    if found != MERGE_KEY:
        df[MERGE_KEY] = df[found].astype(str).str.strip()
    else:
        df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df

def first_existing(row, names):
    for n in names:
        if n in row and clean_cell(row.get(n)):
            return clean_cell(row.get(n))
    return ""

# =========================================================
# SECTIONING
# =========================================================

HEADER_RX = re.compile(r"^\s*([A-Z][A-Z0-9 /&\-\(\)]{2,80})\s*:\s*$")

def sectionize(text):
    text = clean_cell(text)
    if not text:
        return {"FULL": ""}

    lines = text.splitlines()
    sections = {}
    current = "FULL"
    bucket = []

    for line in lines:
        raw = line.rstrip("\n")
        m = HEADER_RX.match(raw.strip())
        if m:
            joined = "\n".join(bucket).strip()
            if joined:
                sections.setdefault(current, []).append(joined)
            current = m.group(1).strip().upper()
            bucket = []
        else:
            bucket.append(raw)

    joined = "\n".join(bucket).strip()
    if joined:
        sections.setdefault(current, []).append(joined)

    out = {}
    for k, v in sections.items():
        joined = "\n".join(v).strip()
        if joined:
            out[k] = joined
    return out if out else {"FULL": text}

def build_sectioned_note(note_text, note_type, note_id, note_date):
    return SectionedNote(
        sections=sectionize(note_text),
        note_type=note_type or "",
        note_id=note_id or "",
        note_date=note_date or ""
    )

def window_around(text, start, end, width):
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].strip()

# =========================================================
# AGGREGATION
# =========================================================

def cand_score(c):
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    nt = str(getattr(c, "note_type", "") or "").lower()
    op_bonus = 0.05 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    date_bonus = 0.01 if clean_cell(getattr(c, "note_date", "")) else 0.0
    return conf + op_bonus + date_bonus

def choose_best(existing, new):
    if existing is None:
        return new
    return new if cand_score(new) > cand_score(existing) else existing

def merge_boolean(existing, new):
    if existing is None:
        return new
    try:
        exv = bool(existing.value)
        nwv = bool(new.value)
    except Exception:
        return choose_best(existing, new)

    if nwv and not exv:
        return new
    if exv and not nwv:
        return existing
    return choose_best(existing, new)

# =========================================================
# FIELD MAP / TARGETS
# =========================================================

FIELD_MAP = {
    "Obesity": "Obesity",
    "Diabetes": "Diabetes",
    "DiabetesMellitus": "Diabetes",
    "Hypertension": "Hypertension",
    "CardiacDisease": "CardiacDisease",
    "VTE": "VenousThromboembolism",
    "VenousThromboembolism": "VenousThromboembolism",
    "SteroidUse": "Steroid",
    "Steroid": "Steroid",
}

BOOLEAN_FIELDS = {
    "Obesity",
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
}

TARGET_FIELDS = [
    "Obesity",
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
]

# =========================================================
# COMORBIDITY EXTRACTOR
# =========================================================

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "ALLERGIES",
    "REVIEW OF SYSTEMS",
}

PREFERRED_SECTIONS = {
    "PAST MEDICAL HISTORY",
    "PMH",
    "HISTORY AND PHYSICAL",
    "H&P",
    "ASSESSMENT",
    "ASSESSMENT AND PLAN",
    "MEDICAL HISTORY",
    "PROBLEM LIST",
    "ANESTHESIA",
    "ANESTHESIA H&P",
    "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS",
    "DIAGNOSIS",
    "IMPRESSION",
}

LOW_VALUE_SECTIONS = {
    "PAST SURGICAL HISTORY",
    "PSH",
    "SURGICAL HISTORY",
    "HISTORY",
    "GYNECOLOGIC HISTORY",
    "OB HISTORY",
}

NEGATION_RX = re.compile(
    r"\b(no|not|denies|denied|without|negative\s+for|free\s+of|absence\s+of)\b",
    re.IGNORECASE
)

FAMILY_RX = re.compile(
    r"\b(family history|mother|father|sister|brother|aunt|uncle|grandmother|grandfather)\b",
    re.IGNORECASE
)

HISTORICAL_ONLY_RX = re.compile(r"\b(history of|hx of|h/o)\b", re.IGNORECASE)

SYSTEMIC_STEROID_EXCLUDE_RX = re.compile(
    r"\b(inhaled|inhaler|intranasal|nasal|topical|cream|ointment|lotion|eye\s*drops?|otic|ear\s*drops?)\b",
    re.IGNORECASE
)

STEROID_NEG_CONTEXT_RX = re.compile(
    r"\b(no|not|denies|without)\b.{0,40}\b(steroid|prednisone|dexamethasone|medrol|methylprednisolone|hydrocortisone)\b",
    re.IGNORECASE
)

VTE_PROPHYLAXIS_RX = re.compile(
    r"\b(prophylaxis|ppx|dvt\s*ppx|vte\s*ppx|sequential\s+compression|compression\s+device|scd|scds|subcutaneous\s+heparin|heparin\s+prophylaxis|enoxaparin\s+prophylaxis)\b",
    re.IGNORECASE
)

CONCEPTS = {
    "Obesity": {
        "pos": [
            r"\bobesity\b",
            r"\bobese\b",
            r"\bmorbid obesity\b",
            r"\boverweight\b",
            r"\bbmi\s*(>|>=)?\s*30\b",
        ],
        "exclude": [],
        "base_conf": 0.80,
    },
    "Diabetes": {
        "pos": [
            r"\bdiabetes\b",
            r"\bdiabetes mellitus\b",
            r"\bdm\b",
            r"\bt1dm\b",
            r"\bt2dm\b",
            r"\btype\s*(i|ii|1|2)\s*diabetes\b",
            r"\binsulin[- ]dependent diabetes\b",
            r"\bnon[- ]insulin[- ]dependent diabetes\b",
            r"\biddm\b",
            r"\bniddm\b",
        ],
        "exclude": [
            r"\bprediabet(es|ic)\b",
            r"\bborderline\b.{0,20}\bdiabet",
            r"\bimpaired glucose tolerance\b",
            r"\bigt\b",
            r"\bgestational diabetes\b",
            r"\bdiabetes insipidus\b",
        ],
        "base_conf": 0.84,
    },
    "Hypertension": {
        "pos": [
            r"\bhypertension\b",
            r"\bhtn\b",
            r"\bhigh blood pressure\b",
        ],
        "exclude": [
            r"\bpulmonary hypertension\b",
            r"\bportal hypertension\b",
            r"\bgestational hypertension\b",
            r"\bpreeclampsia\b",
            r"\beclampsia\b",
            r"\bwhite coat hypertension\b",
        ],
        "base_conf": 0.84,
    },
    "CardiacDisease": {
        "pos": [
            r"\bcoronary artery disease\b",
            r"\bcad\b",
            r"\bcongestive heart failure\b",
            r"\bchf\b",
            r"\bheart failure\b",
            r"\bmyocardial infarction\b",
            r"\bprior mi\b",
            r"\bischemic heart disease\b",
            r"\bcardiomyopathy\b",
            r"\batrial fibrillation\b",
            r"\bafib\b",
            r"\ba[- ]fib\b",
        ],
        "exclude": [
            r"\bmitral valve prolapse\b",
            r"\bvalvular\b",
            r"\bheart murmur\b",
        ],
        "base_conf": 0.82,
    },
    "VenousThromboembolism": {
        "pos": [
            r"\bdeep vein thrombosis\b",
            r"\bdvt\b",
            r"\bpulmonary embol(ism)?\b",
            r"\bpe\b",
            r"\bvte\b",
        ],
        "exclude": [],
        "base_conf": 0.82,
    },
    "Steroid": {
        "pos": [
            r"\bprednisone\b",
            r"\bdexamethasone\b",
            r"\bmethylprednisolone\b",
            r"\bsolu[- ]medrol\b",
            r"\bmedrol\b",
            r"\bhydrocortisone\b",
            r"\bchronic steroid(s)?\b",
            r"\blong[- ]term steroid(s)?\b",
            r"\bsystemic steroid(s)?\b",
        ],
        "exclude": [],
        "base_conf": 0.78,
    },
}

DM_MED_STRONG = [
    r"\binsulin\b",
    r"\blantus\b",
    r"\bhumalog\b",
    r"\bnovolog\b",
    r"\blevemir\b",
    r"\bmetformin\b",
]

def _emit(field, value, status, evid, section, note, conf):
    return Candidate(
        field=field,
        value=value,
        status=status,
        evidence=evid,
        section=section,
        note_type=note.note_type,
        note_id=note.note_id,
        note_date=note.note_date,
        confidence=conf
    )

def _section_rank(section):
    s = clean_cell(section).upper()
    if s in PREFERRED_SECTIONS:
        return 0
    if s in LOW_VALUE_SECTIONS:
        return 2
    return 1

def _iter_sections(note):
    keys = list(note.sections.keys())
    keys.sort(key=_section_rank)
    for k in keys:
        ku = clean_cell(k).upper()
        if ku in SUPPRESS_SECTIONS:
            continue
        txt = clean_cell(note.sections.get(k, ""))
        if txt:
            yield ku, txt

def _has_any(patterns, text):
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

def _find_first(patterns, text):
    best = None
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            if best is None or m.start() < best.start():
                best = m
    return best

def _is_negated(evid):
    return bool(NEGATION_RX.search(evid))

def _family_context(evid):
    return bool(FAMILY_RX.search(evid))

def _status_from_context(evid):
    low = evid.lower()
    if _is_negated(low):
        return "denied"
    if HISTORICAL_ONLY_RX.search(low):
        return "history"
    return "history"

def _concept_confidence(section, base):
    rank = _section_rank(section)
    if rank == 0:
        return min(0.98, base + 0.05)
    if rank == 2:
        return max(0.55, base - 0.08)
    return base

def _extract_concept(field, note):
    cfg = CONCEPTS[field]
    cands = []

    for section, text in _iter_sections(note):
        m = _find_first(cfg["pos"], text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 220)
        low = evid.lower()

        if _family_context(low):
            continue

        if cfg.get("exclude") and _has_any(cfg["exclude"], low):
            continue

        if field == "VenousThromboembolism" and VTE_PROPHYLAXIS_RX.search(low):
            continue

        if field == "Steroid":
            if SYSTEMIC_STEROID_EXCLUDE_RX.search(low):
                continue
            if STEROID_NEG_CONTEXT_RX.search(low):
                continue

        status = _status_from_context(evid)
        value = False if status == "denied" else True
        conf = _concept_confidence(section, cfg.get("base_conf", 0.80))

        cands.append(_emit(
            field=field,
            value=value,
            status=status,
            evid=evid,
            section=section,
            note=note,
            conf=conf
        ))

        if value is True and _section_rank(section) == 0:
            break

    return cands

def _extract_diabetes_med_inference(note):
    cands = []

    for section, text in _iter_sections(note):
        m = _find_first(DM_MED_STRONG, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 220)
        low = evid.lower()

        if _family_context(low):
            continue
        if _has_any(CONCEPTS["Diabetes"]["exclude"], low):
            continue
        if _is_negated(low):
            continue

        conf = _concept_confidence(section, 0.76)
        cands.append(_emit(
            field="Diabetes",
            value=True,
            status="history",
            evid=evid,
            section=section,
            note=note,
            conf=conf
        ))

        if _section_rank(section) == 0:
            break

    return cands

def extract_comorbidities(note):
    cands = []
    cands.extend(_extract_concept("Obesity", note))
    cands.extend(_extract_concept("Diabetes", note))
    cands.extend(_extract_diabetes_med_inference(note))
    cands.extend(_extract_concept("Hypertension", note))
    cands.extend(_extract_concept("CardiacDisease", note))
    cands.extend(_extract_concept("VenousThromboembolism", note))
    cands.extend(_extract_concept("Steroid", note))
    return cands

# =========================================================
# NOTE LOADING
# =========================================================

def join_note(g):
    pieces = []
    if "NOTE_TEXT" in g.columns:
        for x in g["NOTE_TEXT"].tolist():
            val = clean_cell(x)
            if val:
                pieces.append(val)
    else:
        for col in ["LINE", "TEXT", "NOTE_LINE", "NOTE_TEXT_LINE"]:
            if col in g.columns:
                for x in g[col].tolist():
                    val = clean_cell(x)
                    if val:
                        pieces.append(val)

    return "\n".join(pieces).strip()

def load_notes():
    rows = []
    files = []
    for pat in NOTE_GLOBS:
        files.extend(glob(pat, recursive=True))

    for fp in sorted(set(files)):
        try:
            df = clean_cols(read_csv_robust(fp))
        except Exception as e:
            print("WARN: failed to read", fp, e)
            continue

        df = normalize_mrn(df)
        if MERGE_KEY not in df.columns:
            print("WARN: no MRN in", fp)
            continue

        note_id_col = None
        for c in ["NOTE_ID", "note_id", "NoteID", "DOCUMENT_ID", "DOC_ID"]:
            if c in df.columns:
                note_id_col = c
                break

        if note_id_col is None:
            df["NOTE_ID"] = [str(i + 1) for i in range(len(df))]
            note_id_col = "NOTE_ID"

        if "NOTE_TYPE" not in df.columns:
            if "NOTE_NAME" in df.columns:
                df["NOTE_TYPE"] = df["NOTE_NAME"]
            elif "DOCUMENT_TYPE" in df.columns:
                df["NOTE_TYPE"] = df["DOCUMENT_TYPE"]
            else:
                df["NOTE_TYPE"] = os.path.basename(fp)

        if "NOTE_DATE_OF_SERVICE" not in df.columns:
            if "NOTE_DATE" in df.columns:
                df["NOTE_DATE_OF_SERVICE"] = df["NOTE_DATE"]
            elif "DATE_OF_SERVICE" in df.columns:
                df["NOTE_DATE_OF_SERVICE"] = df["DATE_OF_SERVICE"]
            else:
                df["NOTE_DATE_OF_SERVICE"] = ""

        df["_SOURCE_FILE_"] = os.path.basename(fp)
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=[MERGE_KEY, "NOTE_ID", "NOTE_TYPE", "NOTE_DATE", "NOTE_TEXT"])

    raw = pd.concat(rows, ignore_index=True, sort=False)

    reconstructed = []
    grouped = raw.groupby([MERGE_KEY, note_id_col], dropna=False)
    for (mrn, nid), g in grouped:
        mrn = clean_cell(mrn)
        nid = clean_cell(nid)
        if not mrn or not nid:
            continue

        full_text = join_note(g)
        if not full_text:
            continue

        note_type = first_existing(g.iloc[0].to_dict(), ["NOTE_TYPE", "NOTE_NAME", "DOCUMENT_TYPE"])
        note_date = first_existing(g.iloc[0].to_dict(), ["NOTE_DATE_OF_SERVICE", "NOTE_DATE", "DATE_OF_SERVICE"])

        reconstructed.append({
            MERGE_KEY: mrn,
            "NOTE_ID": nid,
            "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date,
            "SOURCE_FILE": clean_cell(g["_SOURCE_FILE_"].iloc[0]),
            "NOTE_TEXT": full_text,
        })

    return pd.DataFrame(reconstructed)

# =========================================================
# CANDIDATE COLLECTION
# =========================================================

COMORBIDITY_PREFILTER = re.compile(
    r"\b("
    r"obesity|obese|overweight|bmi|"
    r"diabetes|dm|t1dm|t2dm|insulin|metformin|"
    r"hypertension|htn|high blood pressure|"
    r"cad|coronary artery disease|chf|heart failure|mi|afib|a-fib|cardiomyopathy|"
    r"dvt|deep vein thrombosis|pe|pulmonary embol|vte|"
    r"prednisone|dexamethasone|methylprednisolone|medrol|hydrocortisone|steroid"
    r")\b",
    re.IGNORECASE,
)

def collect_candidates(notes_df):
    by_mrn = {}

    for mrn, g in notes_df.groupby(MERGE_KEY):
        out = {}
        for _, row in g.iterrows():
            text = clean_cell(row.get("NOTE_TEXT"))
            if not text:
                continue
            if not COMORBIDITY_PREFILTER.search(text):
                continue

            note = build_sectioned_note(
                text,
                row.get("NOTE_TYPE", ""),
                row.get("NOTE_ID", ""),
                row.get("NOTE_DATE", "")
            )

            cands = extract_comorbidities(note)

            for c in cands:
                field = FIELD_MAP.get(c.field)
                if not field:
                    continue

                existing = out.get(field)
                if field in BOOLEAN_FIELDS:
                    out[field] = merge_boolean(existing, c)
                else:
                    out[field] = choose_best(existing, c)

        by_mrn[clean_cell(mrn)] = out

    return by_mrn

# =========================================================
# APPLY TO MASTER
# =========================================================

def ensure_target_columns(master):
    for col in TARGET_FIELDS:
        if col not in master.columns:
            master[col] = ""

def apply_to_master(master, cand_map):
    evidence_rows = []

    for idx in master.index:
        mrn = clean_cell(master.at[idx, MERGE_KEY])
        m = cand_map.get(mrn, {})

        for field in TARGET_FIELDS:
            c = m.get(field)
            if c is None:
                continue

            val = c.value
            if field in BOOLEAN_FIELDS:
                val = 1 if bool(val) else 0

            master.at[idx, field] = val

            evidence_rows.append({
                MERGE_KEY: mrn,
                "FIELD": field,
                "VALUE": val,
                "STATUS": clean_cell(getattr(c, "status", "")),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": clean_cell(getattr(c, "section", "")),
                "NOTE_ID": clean_cell(getattr(c, "note_id", "")),
                "NOTE_TYPE": clean_cell(getattr(c, "note_type", "")),
                "NOTE_DATE": clean_cell(getattr(c, "note_date", "")),
                "EVIDENCE": clean_cell(getattr(c, "evidence", "")),
            })

    evidence_df = pd.DataFrame(evidence_rows)
    return master, evidence_df

# =========================================================
# MAIN
# =========================================================

def main():
    if not os.path.exists(MASTER_IN):
        print("ERROR: MASTER_IN not found:", MASTER_IN)
        sys.exit(1)

    print("Loading master:", MASTER_IN)
    master = clean_cols(read_csv_robust(MASTER_IN))
    master = normalize_mrn(master)

    if MERGE_KEY not in master.columns:
        print("ERROR: master file has no merge key:", MERGE_KEY)
        sys.exit(1)

    ensure_target_columns(master)

    print("Loading notes...")
    notes_df = load_notes()
    print("Notes loaded:", len(notes_df))

    print("Collecting comorbidity candidates...")
    cand_map = collect_candidates(notes_df)
    print("MRNs with candidates:", len(cand_map))

    print("Applying patch to master...")
    patched_master, evidence_df = apply_to_master(master, cand_map)

    evidence_out = os.path.splitext(MASTER_OUT)[0] + "_evidence.csv"

    print("Writing patched master:", MASTER_OUT)
    patched_master.to_csv(MASTER_OUT, index=False)

    print("Writing evidence file:", evidence_out)
    evidence_df.to_csv(evidence_out, index=False)

    print("Done.")

if __name__ == "__main__":
    main()
