#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_comorbidity_only.py

Comorbidity-only updater for:
- Obesity
- Diabetes
- Hypertension
- CardiacDisease
- VenousThromboembolism
- Steroid

Strategy:
- Use the same BASE_DIR / NOTE_GLOBS / output style as update_pbs_only.py
- Reconstruct notes from HPI11526 note files
- Extract only the comorbidity fields above
- Update only these columns in the existing master
- Preserve all other master abstractions

Outputs:
1) /home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv
2) /home/apokol/Breast_Restore/_outputs/comorbidity_only_evidence.csv

Python 3.6.8 compatible.
"""

import os
import re
from glob import glob

import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_MASTER = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_EVID = "{0}/_outputs/comorbidity_only_evidence.csv".format(BASE_DIR)

MERGE_KEY = "MRN"

STRUCT_GLOBS = [
    "{0}/**/HPI11526*Clinic Encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation encounters.csv".format(BASE_DIR),
]

NOTE_GLOBS = [
    "{0}/**/HPI11526*Clinic Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation notes.csv".format(BASE_DIR),
]

from models import Candidate, SectionedNote  # noqa: E402

COMORBIDITY_FIELDS = [
    "Obesity",
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
]


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


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


HEADER_RX = re.compile(r"^\s*([A-Z][A-Z0-9 /&\-\(\)]{2,80})\s*:\s*$")


def sectionize(text):
    if not text:
        return {"FULL": ""}
    lines = text.splitlines()
    sections = {}
    current = "FULL"
    sections[current] = []

    for line in lines:
        m = HEADER_RX.match(line)
        if m:
            hdr = m.group(1).strip().upper()
            current = hdr
            if current not in sections:
                sections[current] = []
            continue
        sections[current].append(line)

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
    re.I
)

FAMILY_RX = re.compile(
    r"\b(family history|mother|father|sister|brother|aunt|uncle|grandmother|grandfather)\b",
    re.I
)

HISTORICAL_ONLY_RX = re.compile(r"\b(history of|hx of|h/o|s/p|status post)\b", re.I)

SYSTEMIC_STEROID_EXCLUDE_RX = re.compile(
    r"\b(inhaled|inhaler|intranasal|nasal|topical|cream|ointment|lotion|eye\s*drops?|otic|ear\s*drops?)\b",
    re.I
)

STEROID_NEG_CONTEXT_RX = re.compile(
    r"\b(no|not|denies|without)\b.{0,40}\b(steroid|prednisone|dexamethasone|medrol|methylprednisolone|hydrocortisone)\b",
    re.I
)

VTE_PROPHYLAXIS_RX = re.compile(
    r"\b(prophylaxis|ppx|dvt\s*ppx|vte\s*ppx|sequential\s+compression|compression\s+device|scd|scds|subcutaneous\s+heparin|heparin\s+prophylaxis|enoxaparin\s+prophylaxis)\b",
    re.I
)

CONCEPTS = {
    "Obesity": {
        "pos": [
            r"\bobesity\b",
            r"\bobese\b",
            r"\bmorbid obesity\b",
            r"\bbmi\s*(>|>=)?\s*30\b",
            r"\bbmi\s*(>|>=)?\s*3\d(?:\.\d+)?\b",
            r"\boverweight\b",
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
        if re.search(p, text, re.I):
            return True
    return False


def _find_first(patterns, text):
    best = None
    for p in patterns:
        m = re.search(p, text, re.I)
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


def candidate_score(c):
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    nt = str(getattr(c, "note_type", "") or "").lower()
    op_bonus = 0.05 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    date_bonus = 0.01 if clean_cell(getattr(c, "note_date", "")) else 0.0
    return conf + op_bonus + date_bonus


def choose_better(existing, new):
    if existing is None:
        return new
    return new if candidate_score(new) > candidate_score(existing) else existing


def choose_better_boolean(existing, new):
    if existing is None:
        return new

    ex_val = bool(getattr(existing, "value", False))
    nw_val = bool(getattr(new, "value", False))

    if nw_val and not ex_val:
        return new
    if ex_val and not nw_val:
        return existing
    return choose_better(existing, new)


COMORBIDITY_PREFILTER = re.compile(
    r"\b("
    r"obesity|obese|overweight|bmi|"
    r"diabetes|dm|t1dm|t2dm|insulin|metformin|"
    r"hypertension|htn|high blood pressure|"
    r"cad|coronary artery disease|chf|heart failure|mi|afib|a-fib|cardiomyopathy|"
    r"dvt|deep vein thrombosis|pe|pulmonary embol|vte|"
    r"prednisone|dexamethasone|methylprednisolone|medrol|hydrocortisone|steroid"
    r")\b",
    re.I
)


def main():
    print("Loading master...")
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)

    for c in COMORBIDITY_FIELDS:
        if c not in master.columns:
            master[c] = pd.NA

    print("Master rows: {0}".format(len(master)))

    print("Loading notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    for c in COMORBIDITY_FIELDS:
        master[c] = 0

    evidence_rows = []
    best_by_mrn = {}

    for _, row in notes_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        note_text = clean_cell(row.get("NOTE_TEXT", ""))
        if not note_text:
            continue

        if not COMORBIDITY_PREFILTER.search(note_text):
            continue

        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        snote = build_sectioned_note(
            note_text=row["NOTE_TEXT"],
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        try:
            cands = extract_comorbidities(snote)
        except Exception as e:
            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": row["NOTE_ID"],
                "NOTE_DATE": row["NOTE_DATE"],
                "NOTE_TYPE": row["NOTE_TYPE"],
                "FIELD": "EXTRACTOR_ERROR",
                "VALUE": "",
                "STATUS": "",
                "CONFIDENCE": "",
                "SECTION": "",
                "RULE_DECISION": "extractor_failed",
                "EVIDENCE": repr(e)
            })
            continue

        if not cands:
            continue

        if mrn not in best_by_mrn:
            best_by_mrn[mrn] = {}

        for c in cands:
            field = clean_cell(getattr(c, "field", ""))
            if field not in COMORBIDITY_FIELDS:
                continue

            evid = clean_cell(getattr(c, "evidence", ""))
            status = clean_cell(getattr(c, "status", ""))

            accept = False
            reason = ""

            if status == "denied":
                accept = False
                reason = "reject_denied"
            elif not evid:
                accept = False
                reason = "reject_no_evidence"
            else:
                accept = True
                reason = "accept_positive"

            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": getattr(c, "note_id", row["NOTE_ID"]),
                "NOTE_DATE": getattr(c, "note_date", row["NOTE_DATE"]),
                "NOTE_TYPE": getattr(c, "note_type", row["NOTE_TYPE"]),
                "FIELD": field,
                "VALUE": getattr(c, "value", True),
                "STATUS": status,
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "RULE_DECISION": reason,
                "EVIDENCE": evid
            })

            if not accept:
                continue

            existing = best_by_mrn[mrn].get(field)
            best_by_mrn[mrn][field] = choose_better_boolean(existing, c)

    print("Accepted comorbidity note-based predictions for MRNs: {0}".format(len(best_by_mrn)))

    for mrn, fields in best_by_mrn.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        for field in COMORBIDITY_FIELDS:
            cand = fields.get(field)
            if cand is None:
                continue
            master.loc[mask, field] = 1 if bool(getattr(cand, "value", False)) else 0

    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    master.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\nDONE.")
    print("- Updated master: {0}".format(OUTPUT_MASTER))
    print("- Comorbidity evidence: {0}".format(OUTPUT_EVID))
    print("\nRun:")
    print(" python update_comorbidity_only.py")


if __name__ == "__main__":
    main()
