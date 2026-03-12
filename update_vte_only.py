#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_vte_only.py

VTE-only updater for:
- VenousThromboembolism

Strategy:
- Uses Breast_Restore paths and note reconstruction style
- Updates only the VenousThromboembolism column in the existing master
- Preserves all other master abstractions
- Applies final VTE-specific precision cleanup:
    * reject procedure/consent complication-risk language
    * reject tamoxifen/raloxifene counseling language
    * reject prophylaxis / ppx / postop lovenox language
    * reject rule-out / duplex-to-rule-out language
    * reject symptom-warning education language
    * keep true history / prior DVT / prior PE / confirmed acute events

Outputs:
1) /home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv
2) /home/apokol/Breast_Restore/_outputs/vte_only_evidence.csv

Python 3.6.8 compatible.
"""

import os
import re
from glob import glob

import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_MASTER = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_EVID = "{0}/_outputs/vte_only_evidence.csv".format(BASE_DIR)

MERGE_KEY = "MRN"
TARGET_FIELD = "VenousThromboembolism"

NOTE_GLOBS = [
    "{0}/**/HPI11526*Clinic Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation notes.csv".format(BASE_DIR),
]

from models import Candidate, SectionedNote  # noqa: E402


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
    "ROS",
    "PERTINENT NEGATIVES",
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
    "PAST HISTORY",
    "DIAGNOSIS",
    "IMPRESSION",
    "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS",
}

LOW_VALUE_SECTIONS = {
    "PAST SURGICAL HISTORY",
    "PSH",
    "SURGICAL HISTORY",
    "HISTORY",
}

NEGATION_RX = re.compile(
    r"\b(no|not|denies|denied|without|negative\s+for|free\s+of|absence\s+of)\b",
    re.I
)

FAMILY_RX = re.compile(
    r"\b(family history|mother|father|sister|brother|aunt|uncle|grandmother|grandfather)\b",
    re.I
)

HISTORICAL_ONLY_RX = re.compile(
    r"\b(history of|hx of|h/o|s/p|status post|prior|previous|remote)\b",
    re.I
)

ROS_RX = re.compile(r"\breview of systems\b|\bros\b", re.I)
PERTINENT_NEG_RX = re.compile(r"\bpertinent negatives?\b", re.I)

VTE_POS = [
    r"\bdeep vein thrombosis\b",
    r"\bdvt\b",
    r"\bpulmonary embol(ism)?\b",
    r"\bpe\b",
    r"\bvte\b",
    r"\bhistory of dvt\b",
    r"\bhistory of pe\b",
    r"\bhistory of pulmonary embolism\b",
    r"\bhistory of deep vein thrombosis\b",
]

# Existing/strong excludes
VTE_PROPHYLAXIS_RX = re.compile(
    r"\b(prophylaxis|ppx|dvt\s*ppx|vte\s*ppx|sequential\s+compression|compression\s+device|scd|scds|subcutaneous\s+heparin|heparin\s+prophylaxis|enoxaparin\s+prophylaxis|postop lovenox|lovenox\s+\d+\s*(hr|hours?)\s*postop|continue ambulation|monitor for dvt)\b",
    re.I
)

VTE_RULEOUT_RX = re.compile(
    r"\b(rule out dvt|r/o dvt|to rule out dvt|ruled out dvt|ruled out pe|pe was ruled out|duplex.*rule out dvt|venous duplex.*rule out dvt|to rule out pe)\b",
    re.I
)

VTE_RISK_FORM_RX = re.compile(
    r"\b(vte risk assessment|risk assessment|risk score|caprini|risk factors score|venous thromboembolism risk assessment)\b",
    re.I
)

# New final patch excludes from QA
VTE_PROCEDURE_RISK_RX = re.compile(
    r"\b(risks?\s+associated\s+with\s+the\s+procedure|risks?\s+of\s+the\s+procedure|complications?\s+include|risk of deep vein thrombosis|risk of pulmonary embolism|risk of dvt|risk of pe)\b",
    re.I
)

VTE_MED_COUNSEL_RX = re.compile(
    r"\b(tamoxifen|raloxifene|aromatase inhibitor|aromatase inhibitors)\b.{0,120}\b(risk|blood clot|dvt|pe|pulmonary embolism|deep vein thrombosis)\b|\b(risk|blood clot|dvt|pe|pulmonary embolism|deep vein thrombosis)\b.{0,120}\b(tamoxifen|raloxifene|aromatase inhibitor|aromatase inhibitors)\b",
    re.I
)

VTE_WARNING_SIGNS_RX = re.compile(
    r"\b(symptoms?\s+of\s+dvt|symptoms?\s+of\s+pe|warning signs?.{0,40}(dvt|pe)|seek medical attention.{0,60}(dvt|pe)|call if she develops symptoms of concern)\b",
    re.I
)

VTE_GENERIC_COMPLICATIONS_RX = re.compile(
    r"\b(complications?\s+including\s+but\s+not\s+limited\s+to|potential complications?\s+include)\b.{0,160}\b(dvt|deep vein thrombosis|pe|pulmonary embolism)\b",
    re.I
)

VTE_TEMPLATE_LIST_RX = re.compile(
    r"\b(asthma|cad|copd|dvt|diabetes mellitus|mi|pulmonary embolism|sleep apnea|stroke)\b",
    re.I
)


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


def _looks_like_template_list(low):
    hits = len(re.findall(
        r"\b(asthma|cad|copd|dvt|diabetes mellitus|mi|pulmonary embolism|sleep apnea|stroke)\b",
        low,
        re.I
    ))
    return hits >= 3


def _is_bad_template_context(section, evid):
    low = clean_cell(evid).lower()
    sec = clean_cell(section).lower()

    if not low:
        return True

    if PERTINENT_NEG_RX.search(low):
        return True

    if ROS_RX.search(low) or ROS_RX.search(sec):
        return True

    if _looks_like_template_list(low) and ("pertinent negatives" in low or "patient active problem list" in low):
        return True

    return False


def _vte_extra_reject(low):
    if VTE_PROPHYLAXIS_RX.search(low):
        return True
    if VTE_RULEOUT_RX.search(low):
        return True
    if VTE_RISK_FORM_RX.search(low):
        return True
    if VTE_PROCEDURE_RISK_RX.search(low):
        return True
    if VTE_MED_COUNSEL_RX.search(low):
        return True
    if VTE_WARNING_SIGNS_RX.search(low):
        return True
    if VTE_GENERIC_COMPLICATIONS_RX.search(low):
        return True

    # common phrasing from QA
    if "tamoxifen has been shown" in low:
        return True
    if "raloxifene did increase" in low:
        return True
    if "chance of pulmonary embolism" in low:
        return True
    if "chance of deep vein thrombosis" in low:
        return True
    if "dvt prophylaxis" in low or "vte prophylaxis" in low:
        return True
    if "to lower risk of vte" in low:
        return True
    if "risk of vte" in low:
        return True

    return False


def extract_vte(note):
    cands = []

    for section, text in _iter_sections(note):
        m = _find_first(VTE_POS, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 320)
        low = evid.lower()

        if _family_context(low):
            continue

        if _is_bad_template_context(section, evid):
            continue

        if _vte_extra_reject(low):
            continue

        status = _status_from_context(evid)
        if status == "denied":
            continue

        cands.append(_emit(
            field=TARGET_FIELD,
            value=True,
            status=status,
            evid=evid,
            section=section,
            note=note,
            conf=0.84 if _section_rank(section) == 0 else 0.82
        ))

        if _section_rank(section) == 0:
            break

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


VTE_PREFILTER = re.compile(
    r"\b(dvt|deep vein thrombosis|pe|pulmonary embol|vte|lovenox|venous duplex|tamoxifen|raloxifene|blood clot)\b",
    re.I
)


def main():
    print("Loading master...")
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)

    if TARGET_FIELD not in master.columns:
        master[TARGET_FIELD] = pd.NA

    print("Master rows: {0}".format(len(master)))

    print("Loading notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    # Only rebuild VTE
    master[TARGET_FIELD] = 0

    evidence_rows = []
    best_by_mrn = {}

    for _, row in notes_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        note_text = clean_cell(row.get("NOTE_TEXT", ""))
        if not note_text:
            continue

        if not VTE_PREFILTER.search(note_text):
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
            cands = extract_vte(snote)
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

        for c in cands:
            evid = clean_cell(getattr(c, "evidence", ""))
            status = clean_cell(getattr(c, "status", ""))

            accept = False
            reason = ""

            if not evid:
                accept = False
                reason = "reject_no_evidence"
            elif status == "denied":
                accept = False
                reason = "reject_denied"
            elif _is_bad_template_context(getattr(c, "section", ""), evid):
                accept = False
                reason = "reject_template_context"
            elif _vte_extra_reject(evid.lower()):
                accept = False
                reason = "reject_vte_context"
            else:
                accept = True
                reason = "accept_positive"

            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": getattr(c, "note_id", row["NOTE_ID"]),
                "NOTE_DATE": getattr(c, "note_date", row["NOTE_DATE"]),
                "NOTE_TYPE": getattr(c, "note_type", row["NOTE_TYPE"]),
                "FIELD": TARGET_FIELD,
                "VALUE": getattr(c, "value", True),
                "STATUS": getattr(c, "status", ""),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "RULE_DECISION": reason,
                "EVIDENCE": evid
            })

            if not accept:
                continue

            existing = best_by_mrn.get(mrn)
            best_by_mrn[mrn] = choose_better(existing, c)

    print("Accepted VTE note-based predictions for MRNs: {0}".format(len(best_by_mrn)))

    for mrn, cand in best_by_mrn.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue
        master.loc[mask, TARGET_FIELD] = 1

    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    master.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\nDONE.")
    print("- Updated master: {0}".format(OUTPUT_MASTER))
    print("- VTE evidence: {0}".format(OUTPUT_EVID))
    print("\nRun:")
    print(" python update_vte_only.py")


if __name__ == "__main__":
    main()
