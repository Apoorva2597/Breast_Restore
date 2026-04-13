#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_pipeline.py
Python 3.6.8 compatible

PURPOSE:
    Single unified script replacing:
        build_patient_master.py
        update_bmi_smoking_only.py
        update_pbs_only.py
        update_cancer_only.py
        update_comorbidity_only.py
        update_vte_only.py

    Loads notes and structured encounters ONCE, runs all extractors in
    one pass, writes one clean master CSV.

    Does NOT touch the stage2 chain or complications patch.

RUN ORDER:
    1. python run_full_pipeline.py          <- this script
    2. (stage2 chain runs unchanged)
    3. python build_master_rule_COMPLICATIONS_PATCH.py
    4. python validate_abstraction.py

OUTPUTS:
    _outputs/master_abstraction_rule_FINAL_NO_GOLD.csv
    _outputs/pipeline_evidence.csv
"""

import os
import re
import math
from glob import glob
from datetime import datetime

import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "/home/apokol/Breast_Restore"

OUTPUT_MASTER = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_EVID   = "{0}/_outputs/pipeline_evidence.csv".format(BASE_DIR)

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

# ============================================================
# IMPORTS FROM REPO
# ============================================================
from models import SectionedNote, Candidate                       # noqa: E402
from extractors.age import extract_age                            # noqa: E402
from extractors.bmi import extract_bmi                            # noqa: E402
from extractors.smoking import extract_smoking                    # noqa: E402
from extractors.pbs import extract_pbs                            # noqa: E402
from extractors.mastectomy import extract_mastectomy              # noqa: E402
from extractors.cancer_treatment import extract_cancer_treatment  # noqa: E402
from extractors.breast_cancer_recon import extract_breast_cancer_recon  # noqa: E402

# ============================================================
# MASTER SCHEMA
# ============================================================
MASTER_COLUMNS = [
    "MRN", "ENCRYPTED_PAT_ID", "Last name", "DOB", "PatientID",
    "Race", "Ethnicity", "Age", "BMI", "CCI", "SmokingStatus",
    "Diabetes", "Obesity", "Hypertension", "CardiacDisease",
    "VenousThromboembolism", "Steroid",
    "PastBreastSurgery", "PBS_Lumpectomy", "PBS_Breast Reduction",
    "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other",
    "Mastectomy_Laterality", "Indication_Left", "Indication_Right",
    "LymphNode",
    "Radiation", "Radiation_Before", "Radiation_After",
    "Chemo", "Chemo_Before", "Chemo_After",
    "Recon_Laterality", "Recon_Type", "Recon_Classification", "Recon_Timing",
    "Stage1_MinorComp", "Stage1_Reoperation", "Stage1_Rehospitalization",
    "Stage1_MajorComp", "Stage1_Failure", "Stage1_Revision",
    "Stage2_MinorComp", "Stage2_Reoperation", "Stage2_Rehospitalization",
    "Stage2_MajorComp", "Stage2_Failure", "Stage2_Revision",
    "Stage2_Applicable",
]

# ============================================================
# SHARED UTILITIES
# ============================================================

def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df):
    for k in ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]:
        if k in df.columns:
            if k != MERGE_KEY:
                df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError("MRN column not found. Columns: {0}".format(list(df.columns)[:40]))
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df


def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError("Required column missing. Tried={0}. Seen={1}".format(
            options, list(df.columns)[:60]))
    return None


def to_int_safe(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def to_float_safe(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


def parse_date_safe(x):
    s = clean_cell(x)
    if not s:
        return None
    fmts = [
        "%Y-%m-%d", "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S",
        "%Y/%m/%d", "%d-%b-%Y", "%d-%b-%Y %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def days_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    return (dt1.date() - dt2.date()).days


def same_calendar_date(dt1, dt2):
    if dt1 is None or dt2 is None:
        return False
    return dt1.date() == dt2.date()


HEADER_RX = re.compile(r"^\s*([A-Z][A-Z0-9 /&\-]{2,60})\s*:\s*$")


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


# ============================================================
# STRUCTURED ENCOUNTER LOADING
# ============================================================

def load_structured_encounters():
    rows = []
    struct_files = []
    for g in STRUCT_GLOBS:
        struct_files.extend(glob(g, recursive=True))

    for fp in sorted(set(struct_files)):
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)
        source_name = os.path.basename(fp).lower()

        if "operation encounters" in source_name:
            encounter_source = "operation"
            priority = 1
        elif "clinic encounters" in source_name:
            encounter_source = "clinic"
            priority = 2
        elif "inpatient encounters" in source_name:
            encounter_source = "inpatient"
            priority = 3
        else:
            encounter_source = "other"
            priority = 9

        race_col   = pick_col(df, ["RACE", "Race"], required=False)
        eth_col    = pick_col(df, ["ETHNICITY", "Ethnicity"], required=False)
        age_col    = pick_col(df, ["AGE_AT_ENCOUNTER", "Age_at_encounter", "AGE"], required=False)
        admit_col  = pick_col(df, ["ADMIT_DATE", "Admit_Date"], required=False)
        recon_col  = pick_col(df, ["RECONSTRUCTION_DATE", "RECONSTRUCTION DATE"], required=False)
        cpt_col    = pick_col(df, ["CPT_CODE", "CPT CODE", "CPT"], required=False)
        proc_col   = pick_col(df, ["PROCEDURE", "Procedure"], required=False)
        reason_col = pick_col(df, ["REASON_FOR_VISIT", "REASON FOR VISIT"], required=False)
        date_col   = pick_col(df, ["OPERATION_DATE", "CHECKOUT_TIME", "DISCHARGE_DATE_DT"], required=False)

        out = pd.DataFrame()
        out[MERGE_KEY]                      = df[MERGE_KEY].astype(str).str.strip()
        out["STRUCT_SOURCE"]                = encounter_source
        out["STRUCT_PRIORITY"]              = priority
        out["STRUCT_DATE_RAW"]              = df[date_col].astype(str)   if date_col   else ""
        out["RACE_STRUCT"]                  = df[race_col].astype(str)   if race_col   else ""
        out["ETHNICITY_STRUCT"]             = df[eth_col].astype(str)    if eth_col    else ""
        out["AGE_AT_ENCOUNTER_STRUCT"]      = df[age_col].astype(str)    if age_col    else ""
        out["ADMIT_DATE_STRUCT"]            = df[admit_col].astype(str)  if admit_col  else ""
        out["RECONSTRUCTION_DATE_STRUCT"]   = df[recon_col].astype(str)  if recon_col  else ""
        out["CPT_CODE_STRUCT"]              = df[cpt_col].astype(str)    if cpt_col    else ""
        out["PROCEDURE_STRUCT"]             = df[proc_col].astype(str)   if proc_col   else ""
        out["REASON_FOR_VISIT_STRUCT"]      = df[reason_col].astype(str) if reason_col else ""
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            MERGE_KEY, "STRUCT_SOURCE", "STRUCT_PRIORITY", "STRUCT_DATE_RAW",
            "RACE_STRUCT", "ETHNICITY_STRUCT", "AGE_AT_ENCOUNTER_STRUCT",
            "ADMIT_DATE_STRUCT", "RECONSTRUCTION_DATE_STRUCT",
            "CPT_CODE_STRUCT", "PROCEDURE_STRUCT", "REASON_FOR_VISIT_STRUCT"
        ])

    return pd.concat(rows, ignore_index=True)


# ============================================================
# NOTE LOADING
# ============================================================

def load_and_reconstruct_notes():
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))

    if not note_files:
        raise FileNotFoundError("No HPI11526 Notes CSVs found.")

    all_rows = []

    for fp in note_files:
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)

        text_col = pick_col(df, ["NOTE_TEXT", "NOTE TEXT", "NOTE_TEXT_FULL", "TEXT", "NOTE"])
        id_col   = pick_col(df, ["NOTE_ID", "NOTE ID"])
        line_col = pick_col(df, ["LINE"], required=False)
        type_col = pick_col(df, ["NOTE_TYPE", "NOTE TYPE"], required=False)
        date_col = pick_col(df, ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE",
                                  "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"],
                             required=False)

        df[text_col] = df[text_col].fillna("").astype(str)
        df[id_col]   = df[id_col].fillna("").astype(str)
        if line_col: df[line_col] = df[line_col].fillna("").astype(str)
        if type_col: df[type_col] = df[type_col].fillna("").astype(str)
        if date_col: df[date_col] = df[date_col].fillna("").astype(str)

        df["_SOURCE_FILE_"] = os.path.basename(fp)

        keep = [MERGE_KEY, id_col, text_col, "_SOURCE_FILE_"]
        if line_col: keep.append(line_col)
        if type_col: keep.append(type_col)
        if date_col: keep.append(date_col)

        tmp = df[keep].copy().rename(columns={id_col: "NOTE_ID", text_col: "NOTE_TEXT"})

        if line_col and line_col != "LINE":
            tmp = tmp.rename(columns={line_col: "LINE"})
        if type_col and type_col != "NOTE_TYPE":
            tmp = tmp.rename(columns={type_col: "NOTE_TYPE"})
        if date_col and date_col != "NOTE_DATE_OF_SERVICE":
            tmp = tmp.rename(columns={date_col: "NOTE_DATE_OF_SERVICE"})

        for col in ["LINE", "NOTE_TYPE", "NOTE_DATE_OF_SERVICE"]:
            if col not in tmp.columns:
                tmp[col] = ""

        all_rows.append(tmp)

    notes_raw = pd.concat(all_rows, ignore_index=True)

    def join_note(group):
        tmp = group.copy()
        tmp["_LN_"] = tmp["LINE"].apply(to_int_safe)
        tmp = tmp.sort_values("_LN_", na_position="last")
        return "\n".join(tmp["NOTE_TEXT"].tolist()).strip()

    reconstructed = []
    for (mrn, nid), g in notes_raw.groupby([MERGE_KEY, "NOTE_ID"], dropna=False):
        mrn = str(mrn).strip()
        nid = str(nid).strip()
        if not nid:
            continue
        full_text = join_note(g)
        if not full_text:
            continue
        note_type = g["NOTE_TYPE"].astype(str).iloc[0] if g["NOTE_TYPE"].astype(str).str.strip().any() else g["_SOURCE_FILE_"].astype(str).iloc[0]
        note_date = g["NOTE_DATE_OF_SERVICE"].astype(str).iloc[0] if g["NOTE_DATE_OF_SERVICE"].astype(str).str.strip().any() else ""
        reconstructed.append({
            MERGE_KEY: mrn, "NOTE_ID": nid, "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date, "SOURCE_FILE": g["_SOURCE_FILE_"].astype(str).iloc[0],
            "NOTE_TEXT": full_text
        })

    return pd.DataFrame(reconstructed)


# ============================================================
# STRUCTURED ANCHOR: RECONSTRUCTION DATE
# ============================================================

PREFERRED_CPTS      = {"19357", "19340", "19342", "19361", "19364", "19367", "S2068"}
EXCLUDE_CPTS        = {"19325", "19330"}
FALLBACK_CPTS       = {"19350", "19380"}
RECON_KEYWORDS      = [
    "tissue expander", "breast recon", "latissimus", "diep", "tram",
    "flap", "free flap", "expander placmnt", "reconstruct", "reconstruction",
    "implant on same day of mastectomy",
    "insert or replcmnt breast implnt on sep day from mastectomy",
]


def _is_recon_row(row, has_pref):
    cpt  = clean_cell(row.get("CPT_CODE_STRUCT", "")).upper()
    proc = clean_cell(row.get("PROCEDURE_STRUCT", "")).lower()
    rvfv = clean_cell(row.get("REASON_FOR_VISIT_STRUCT", "")).lower()
    if cpt in EXCLUDE_CPTS:
        return False
    if cpt in PREFERRED_CPTS:
        return True
    if (not has_pref) and cpt in FALLBACK_CPTS:
        return True
    text = proc + " " + rvfv
    return any(kw in text for kw in RECON_KEYWORDS)


def build_recon_anchor_map(struct_df):
    """Returns mrn -> anchor dict with recon_date, admit_date, procedure, etc."""
    best = {}
    if len(struct_df) == 0:
        return best

    src_prio = {"clinic": 1, "operation": 2, "inpatient": 3}
    eligible = struct_df[struct_df["STRUCT_SOURCE"].isin(src_prio)].copy()

    has_pref = {}
    for mrn, g in eligible.groupby(MERGE_KEY):
        has_pref[mrn] = any(
            clean_cell(v).upper() in PREFERRED_CPTS
            for v in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist()
        )

    for _, row in eligible.iterrows():
        mrn    = clean_cell(row.get(MERGE_KEY, ""))
        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        if not mrn or source not in src_prio:
            continue

        admit_dt = parse_date_safe(row.get("ADMIT_DATE_STRUCT", ""))
        recon_dt = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))

        if admit_dt is None or recon_dt is None:
            continue
        if not _is_recon_row(row, has_pref.get(mrn, False)):
            continue

        score = (src_prio[source], recon_dt, admit_dt)
        cur = best.get(mrn)
        if cur is None or score < cur["score"]:
            best[mrn] = {
                "recon_date":       recon_dt.strftime("%Y-%m-%d"),
                "admit_date":       admit_dt.strftime("%Y-%m-%d"),
                "score":            score,
                "source":           source,
                "cpt_code":         clean_cell(row.get("CPT_CODE_STRUCT", "")),
                "procedure":        clean_cell(row.get("PROCEDURE_STRUCT", "")),
                "reason_for_visit": clean_cell(row.get("REASON_FOR_VISIT_STRUCT", "")),
            }

    # Backup: for MRNs with no primary anchor, use any recon-like row with a date
    for _, row in eligible.iterrows():
        mrn    = clean_cell(row.get(MERGE_KEY, ""))
        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        if not mrn or mrn in best or source not in src_prio:
            continue
        if not _is_recon_row(row, has_pref.get(mrn, False)):
            continue
        dt = (parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", "")) or
              parse_date_safe(row.get("ADMIT_DATE_STRUCT", "")) or
              parse_date_safe(row.get("STRUCT_DATE_RAW", "")))
        if dt is None:
            continue
        best[mrn] = {
            "recon_date":       dt.strftime("%Y-%m-%d"),
            "admit_date":       clean_cell(row.get("ADMIT_DATE_STRUCT", "")),
            "score":            (src_prio.get(source, 9), dt, dt),
            "source":           source,
            "cpt_code":         clean_cell(row.get("CPT_CODE_STRUCT", "")),
            "procedure":        clean_cell(row.get("PROCEDURE_STRUCT", "")),
            "reason_for_visit": clean_cell(row.get("REASON_FOR_VISIT_STRUCT", "")),
        }

    return best


# ============================================================
# RACE / ETHNICITY / AGE FROM STRUCTURED
# ============================================================

def _norm_race_token(x):
    s = clean_cell(x).lower()
    if not s:
        return ""
    if s in {"white or caucasian", "white", "caucasian"}:
        return "White"
    if s in {"black or african american", "black", "african american"}:
        return "Black or African American"
    if s in {"asian", "filipino", "other asian", "asian indian",
              "chinese", "japanese", "korean", "vietnamese"}:
        return "Asian"
    if s == "american indian or alaska native":
        return "American Indian or Alaska Native"
    if s in {"native hawaiian", "pacific islander",
              "native hawaiian or other pacific islander"}:
        return "Native Hawaiian or Other Pacific Islander"
    if s == "other":
        return "Other"
    if s in {"unknown", "patient refused", "declined", "refused",
              "unable to obtain", "choose not to disclose"}:
        return "Unknown / Declined / Not Reported"
    return clean_cell(x)


def build_race_map(struct_df):
    race_by_mrn = {}
    for _, row in struct_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        raw = clean_cell(row.get("RACE_STRUCT", ""))
        if not mrn or not raw:
            continue
        race_by_mrn.setdefault(mrn, []).append(raw)

    out = {}
    for mrn, vals in race_by_mrn.items():
        real = []
        saw_unk = False
        for v in vals:
            n = _norm_race_token(v)
            if not n:
                continue
            if n == "Unknown / Declined / Not Reported":
                saw_unk = True
                continue
            if n not in real:
                real.append(n)
        if not real:
            out[mrn] = "Unknown / Declined / Not Reported" if saw_unk else ""
        elif len(real) == 1:
            out[mrn] = real[0]
        else:
            out[mrn] = "Multiracial"
    return out


def build_ethnicity_map(struct_df):
    best = {}
    for _, row in struct_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        val = clean_cell(row.get("ETHNICITY_STRUCT", ""))
        pri = to_int_safe(row.get("STRUCT_PRIORITY", 9)) or 9
        if not mrn or not val:
            continue
        cur = best.get(mrn)
        if cur is None or pri < cur["score"]:
            best[mrn] = {"value": val, "score": pri}
    return {mrn: v["value"] for mrn, v in best.items()}


def build_age_map(struct_df, anchor_map):
    best = {}
    src_prio = {"clinic": 1, "operation": 2, "inpatient": 3}
    eligible = struct_df[struct_df["STRUCT_SOURCE"].isin(src_prio)].copy()

    has_pref = {}
    for mrn, g in eligible.groupby(MERGE_KEY):
        has_pref[mrn] = any(
            clean_cell(v).upper() in PREFERRED_CPTS
            for v in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist()
        )

    for _, row in eligible.iterrows():
        mrn    = clean_cell(row.get(MERGE_KEY, ""))
        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        if not mrn or source not in src_prio:
            continue

        age_raw  = clean_cell(row.get("AGE_AT_ENCOUNTER_STRUCT", ""))
        age_base = to_float_safe(age_raw)
        admit_dt = parse_date_safe(row.get("ADMIT_DATE_STRUCT", ""))
        recon_dt = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))
        cpt      = clean_cell(row.get("CPT_CODE_STRUCT", "")).upper()

        if age_base is None or admit_dt is None or recon_dt is None:
            continue
        if cpt in EXCLUDE_CPTS:
            continue
        if has_pref.get(mrn, False) and cpt in FALLBACK_CPTS:
            continue
        if not _is_recon_row(row, has_pref.get(mrn, False)):
            continue

        day_diff     = (recon_dt - admit_dt).days
        adj_age      = age_base + float(day_diff) / 365.25
        age_round    = int(math.floor(adj_age + 0.5))

        score = (src_prio[source], recon_dt, admit_dt)
        cur   = best.get(mrn)
        if cur is None or score < cur["score"]:
            best[mrn] = {"age": age_round, "score": score}

    return {mrn: v["age"] for mrn, v in best.items()}


# ============================================================
# CANDIDATE SCORING HELPERS
# ============================================================

def cand_score_basic(c):
    conf     = float(getattr(c, "confidence", 0.0) or 0.0)
    nt       = str(getattr(c, "note_type", "") or "").lower()
    op_bonus = 0.05 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    dt_bonus = 0.01 if clean_cell(getattr(c, "note_date", "")) else 0.0
    return conf + op_bonus + dt_bonus


def choose_best(existing, new):
    if existing is None:
        return new
    return new if cand_score_basic(new) > cand_score_basic(existing) else existing


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


# ============================================================
# BMI RANKING
# ============================================================

def _is_op_note(note_type):
    s = clean_cell(note_type).lower()
    return any(x in s for x in ["brief op", "op note", "operative", "operation", "oper report"])


def _is_clinic_note(note_type, source_file):
    s = clean_cell(note_type).lower() + " " + clean_cell(source_file).lower()
    return any(x in s for x in ["progress", "clinic", "office", "follow up",
                                  "follow-up", "pre-op", "preop", "consult", "h&p",
                                  "history and physical"])


def bmi_candidate_rank(c, recon_dt):
    note_dt   = parse_date_safe(getattr(c, "note_date", ""))
    note_type = clean_cell(getattr(c, "note_type", ""))
    dd        = days_between(note_dt, recon_dt)
    if dd is None:
        return (9, 9999, 9999, -cand_score_basic(c))
    abs_dd    = abs(dd)
    op        = _is_op_note(note_type)
    clinic    = _is_clinic_note(note_type, "")
    if dd == 0 and op:     return (0, 0, 0, -cand_score_basic(c))
    if dd == 0 and clinic: return (1, 0, 0, -cand_score_basic(c))
    if abs_dd <= 3 and op: return (2, abs_dd, 0 if dd <= 0 else 1, -cand_score_basic(c))
    if dd < 0 and abs_dd <= 45 and clinic:  return (3, abs_dd, 0, -cand_score_basic(c))
    if dd > 0 and abs_dd <= 14 and clinic:  return (4, abs_dd, 1, -cand_score_basic(c))
    if abs_dd <= 45: return (5, abs_dd, 0 if dd <= 0 else 1, -cand_score_basic(c))
    return (9, abs_dd, 0 if dd <= 0 else 1, -cand_score_basic(c))


def choose_best_bmi(existing, new, recon_dt):
    if existing is None:
        return new
    return new if bmi_candidate_rank(new, recon_dt) < bmi_candidate_rank(existing, recon_dt) else existing


def bmi_in_window(note_dt, recon_dt):
    dd = days_between(note_dt, recon_dt)
    if dd is None:
        return False
    return -45 <= dd <= 14


# ============================================================
# SMOKING RANKING
# ============================================================

def smoking_value_priority(val):
    v = clean_cell(val)
    if v == "Current": return 0
    if v == "Former":  return 1
    if v == "Never":   return 2
    return 9


def smoking_candidate_rank(c, recon_dt):
    note_dt = parse_date_safe(getattr(c, "note_date", ""))
    dd      = days_between(note_dt, recon_dt)
    if dd is None:
        return None
    sec     = clean_cell(getattr(c, "section", "")).lower()
    conf    = float(getattr(c, "confidence", 0.0) or 0.0)
    sec_rank = 0 if "social" in sec else 1
    return (abs(dd), sec_rank, -conf)


def choose_best_smoking(existing, new, recon_dt):
    if existing is None:
        return new
    ex_rank = smoking_candidate_rank(existing, recon_dt)
    nw_rank = smoking_candidate_rank(new, recon_dt)
    if ex_rank is None: return new
    if nw_rank is None: return existing
    if nw_rank < ex_rank: return new
    if nw_rank == ex_rank:
        ex_pri = smoking_value_priority(getattr(existing, "value", ""))
        nw_pri = smoking_value_priority(getattr(new, "value", ""))
        if nw_pri < ex_pri: return new
    return existing


def note_on_or_before(note_dt, recon_dt):
    dd = days_between(note_dt, recon_dt)
    return dd is not None and dd <= 0


def note_in_window(note_dt, recon_dt, before, after):
    dd = days_between(note_dt, recon_dt)
    return dd is not None and -before <= dd <= after


# ============================================================
# PBS ACCEPT/REJECT LOGIC (with laterality fix)
# ============================================================

LEFT_RX  = re.compile(r"\b(left|lt)\b|\bleft\s+breast\b|\bleft[- ]sided\b|\(left\)|\(lt\)", re.I)
RIGHT_RX = re.compile(r"\b(right|rt)\b|\bright\s+breast\b|\bright[- ]sided\b|\(right\)|\(rt\)", re.I)
BILAT_RX = re.compile(r"\b(bilateral|bilat|both\s+breasts?)\b", re.I)

HISTORY_CUE_RX = re.compile(
    r"\b(s/p|status\s+post|history\s+of|with\s+a\s+history\s+of|prior|previous|"
    r"remote|previously|underwent|treated\s+with)\b", re.I)

NEGATIVE_HISTORY_RX = re.compile(
    r"\b(no\s+prior\s+breast\s+surgery|no\s+history\s+of\s+breast\s+surgery|"
    r"denies\s+prior\s+breast\s+surgery|never\s+had\s+breast\s+surgery)\b", re.I)

CANCER_CONTEXT_RX = re.compile(
    r"\b(ductal\s+carcinoma|lobular\s+carcinoma|dcis|invasive\s+ductal|breast\s+cancer|"
    r"sentinel\s+lymph\s+node|slnb|alnd|radiation|chemo|xrt)\b", re.I)

YEAR_RX = re.compile(r"\b(?:19|20)\d{2}\b", re.I)

AUGMENT_NEG_RX = re.compile(
    r"\b(reconstruction|implant[- ]based\s+reconstruction|tissue\s+expander|expander|"
    r"implant\s+exchange|permanent\s+(?:silicone|saline)\s+breast\s+implants?|"
    r"breast\s+implant\s+reconstruction|post[- ]mastectomy|mastectomy)\b", re.I)

AUGMENT_POS_RX = re.compile(
    r"\b(cosmetic|augmentation|history\s+of|prior|previous|previously|s/p|"
    r"submuscular|saline|silicone|(?:19|20)\d{2})\b", re.I)

LUMPECTOMY_FP_RX = re.compile(
    r"\b(candidate\s+for\s+lumpectomy|lumpectomy\s+vs\.?\s+mastectomy|"
    r"discussion\s+of\s+lumpectomy|discussed\s+lumpectomy|"
    r"recommend(?:ed)?\s+lumpectomy|planned\s+lumpectomy|"
    r"scheduled\s+for\s+lumpectomy)\b", re.I)


def _norm_lat(x):
    s = clean_cell(x).lower()
    if not s: return ""
    if "bilat" in s or "bilateral" in s or "both" in s: return "bilateral"
    if "left" in s or s == "l": return "left"
    if "right" in s or s == "r": return "right"
    return ""


def _extract_lat(text):
    t = clean_cell(text)
    if not t: return ""
    hb = BILAT_RX.search(t) is not None
    hl = LEFT_RX.search(t) is not None
    hr = RIGHT_RX.search(t) is not None
    if hb or (hl and hr): return "bilateral"
    if hl: return "left"
    if hr: return "right"
    return ""


def _lat_relation(recon_lat, proc_lat, ctx):
    recon_lat = _norm_lat(recon_lat)
    proc_lat  = _norm_lat(proc_lat)
    ctx_low   = clean_cell(ctx).lower()
    if recon_lat == "bilateral": return "accept"
    if recon_lat in {"left", "right"}:
        if proc_lat == recon_lat: return "accept"
        if proc_lat == "bilateral": return "accept"
        if proc_lat in {"left", "right"} and proc_lat != recon_lat: return "reject_contralateral"
        if "contralateral" in ctx_low: return "reject_contralateral"
        return "unknown_unilateral"
    return "unknown_recon"


def _pbs_history_ok(field, ctx):
    c = clean_cell(ctx)
    if field == "PBS_Lumpectomy":
        return (HISTORY_CUE_RX.search(c) is not None or
                CANCER_CONTEXT_RX.search(c) is not None or
                YEAR_RX.search(c) is not None or
                bool(re.search(r"\bunderwent\b", c, re.I)))
    if field in {"PBS_Breast Reduction", "PBS_Mastopexy", "PBS_Other"}:
        return HISTORY_CUE_RX.search(c) is not None
    if field == "PBS_Augmentation":
        explicit = re.search(
            r"\b(breast\s+augmentation|augmentation\s+mammaplasty|cosmetic\s+augmentation|"
            r"breast\s+implants?\s+for\s+augmentation)\b", c, re.I) is not None
        if explicit: return True
        return AUGMENT_POS_RX.search(c) is not None and not AUGMENT_NEG_RX.search(c)
    return False


def pbs_accept(field, evid, day_diff, recon_lat, proc_lat, combined_ctx):
    """
    Returns (accept: bool, reason: str).
    Key fix: PBS_Breast Reduction / Mastopexy / Augmentation / Other
    are NOT gated by laterality — they are past cosmetic/surgical history.
    """
    neg  = NEGATIVE_HISTORY_RX.search(combined_ctx)
    hist = _pbs_history_ok(field, combined_ctx)

    if neg:
        return False, "reject_negative_history"
    if day_diff is None:
        return False, "reject_missing_date_diff"
    if field == "PBS_Lumpectomy" and LUMPECTOMY_FP_RX.search(combined_ctx):
        return False, "reject_lumpectomy_planning_context"

    # FIX: non-lumpectomy PBS fields — skip laterality check entirely
    if field != "PBS_Lumpectomy":
        if not hist:
            return False, "reject_no_history_context"
        return True, "accept_non_lumpectomy_history"

    # PBS_Lumpectomy: laterality-aware (original logic preserved)
    lat = _lat_relation(recon_lat, proc_lat, combined_ctx)

    if day_diff < 0:
        if lat == "accept":
            return True, "accept_pre_recon_historical" if hist else "accept_pre_recon_lumpectomy"
        if lat == "reject_contralateral":
            return False, "reject_contralateral"
        if lat == "unknown_unilateral":
            return False, "reject_unknown_laterality_unilateral"
        # unknown_recon
        if hist:
            return True, "accept_pre_recon_unknown_lat_history"
        return False, "reject_unknown_recon_laterality"
    else:
        if not hist:
            return False, "reject_post_recon_not_historical"
        if lat == "accept":
            return True, "accept_post_recon_historical"
        if lat == "reject_contralateral":
            return False, "reject_contralateral"
        if lat == "unknown_unilateral":
            return False, "reject_unknown_laterality_unilateral"
        # unknown_recon — accept if history present
        return True, "accept_post_recon_history_no_recon_lat"


def pbs_stage_rank(c, recon_dt):
    note_dt    = parse_date_safe(getattr(c, "note_date", ""))
    note_type  = clean_cell(getattr(c, "note_type", ""))
    source     = clean_cell(getattr(c, "_source_file", ""))
    post_hist  = getattr(c, "_accepted_post_hist", False)
    dd         = days_between(note_dt, recon_dt)
    if dd is None:
        return (9, 9999, 9)
    op     = _is_op_note(note_type) or "operation" in source.lower()
    clinic = _is_clinic_note(note_type, source)
    if dd < 0:
        if op:     return (0, abs(dd), 0)
        if clinic: return (1, abs(dd), 1)
        return (2, abs(dd), 2)
    if dd >= 0 and post_hist:
        if op:     return (3, abs(dd), 0)
        if clinic: return (4, abs(dd), 1)
        return (5, abs(dd), 2)
    return (9, abs(dd), 9)


def choose_best_pbs(existing, new, recon_dt):
    if existing is None:
        return new
    ex_r = pbs_stage_rank(existing, recon_dt)
    nw_r = pbs_stage_rank(new, recon_dt)
    if nw_r < ex_r: return new
    if ex_r < nw_r: return existing
    return new if cand_score_basic(new) > cand_score_basic(existing) else existing


# ============================================================
# CANCER/RECON HELPERS (from update_cancer_only)
# ============================================================

MASTECTOMY_RX = re.compile(
    r"\b(mastectomy|simple\s+mastectomy|total\s+mastectomy|"
    r"skin[- ]sparing\s+mastectomy|nipple[- ]sparing\s+mastectomy|\bMRM\b)\b", re.I)

CANCER_KEYWORD_RX = re.compile(
    r"\b(mastectomy|diep|tram|siea|gap|latissimus|flap|reconstruction|expander|implant|"
    r"radiation|xrt|pmrt|chemo|chemotherapy|taxol|herceptin|sentinel|axillary|alnd|slnb|"
    r"prophylactic|carcinoma|dcis|lcis|oncology|lymphatic\s+mapping)\b", re.I)


def _infer_lat(text):
    low = clean_cell(text).lower()
    if BILAT_RX.search(low): return "BILATERAL"
    hl = bool(LEFT_RX.search(low))
    hr = bool(RIGHT_RX.search(low))
    if hl and hr: return "BILATERAL"
    if hl: return "LEFT"
    if hr: return "RIGHT"
    return None


def _infer_recon_type(text):
    low = clean_cell(text).lower()
    found = []
    if "diep" in low: found.append("DIEP")
    if "tram" in low: found.append("TRAM")
    if "siea" in low: found.append("SIEA")
    if "latissimus" in low: found.append("latissimus dorsi")
    has_flap = len(found) > 0 or " flap" in low
    has_dti  = bool(re.search(r"\bdirect[- ]to[- ]implant\b", low))
    has_exp  = "tissue expander" in low or "expander" in low
    has_impl = "implant" in low

    if len(set(found)) >= 2: return "mixed flaps", "autologous"
    if "DIEP" in found: return "DIEP", "autologous"
    if "TRAM" in found: return "TRAM", "autologous"
    if "SIEA" in found: return "SIEA", "autologous"
    if "latissimus dorsi" in found: return "latissimus dorsi", "autologous"
    if has_flap: return "other", "autologous"
    if has_dti:  return "direct-to-implant", "implant"
    if has_exp or has_impl: return "expander/implant", "implant"
    return None, None


def build_recon_structured_map(struct_df):
    """Build structured recon type/laterality/timing from encounter data."""
    best = {}
    src_prio = {"operation": 1, "clinic": 2, "inpatient": 3}

    has_pref = {}
    for mrn, g in struct_df.groupby(MERGE_KEY):
        has_pref[mrn] = any(
            clean_cell(v).upper() in PREFERRED_CPTS
            for v in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist()
        )

    for _, row in struct_df.iterrows():
        mrn    = clean_cell(row.get(MERGE_KEY, ""))
        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        if not mrn or source not in src_prio:
            continue
        if not _is_recon_row(row, has_pref.get(mrn, False)):
            continue
        recon_dt = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))
        if recon_dt is None:
            continue
        proc = clean_cell(row.get("PROCEDURE_STRUCT", ""))
        lat  = _infer_lat(proc)
        rtype, rclass = _infer_recon_type(proc)
        score = (src_prio.get(source, 9), recon_dt)
        cur = best.get(mrn)
        if cur is None or score < cur["score"]:
            best[mrn] = {
                "recon_date": recon_dt.strftime("%Y-%m-%d"),
                "laterality": lat,
                "recon_type": rtype,
                "recon_class": rclass,
                "procedure": proc,
                "score": score,
            }
    return best


def build_mastectomy_events(struct_df):
    out = {}
    for _, row in struct_df.iterrows():
        mrn  = clean_cell(row.get(MERGE_KEY, ""))
        proc = clean_cell(row.get("PROCEDURE_STRUCT", ""))
        if not mrn or not proc or not MASTECTOMY_RX.search(proc):
            continue
        ev_dt = parse_date_safe(row.get("STRUCT_DATE_RAW", "")) or \
                parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))
        lat   = _infer_lat(proc)
        out.setdefault(mrn, []).append({"date": ev_dt, "laterality": lat, "procedure": proc})
    return out


def choose_best_mastectomy(events, recon_dt):
    if not events:
        return None
    best_same = None
    best_prior = None
    for ev in events:
        ev_dt = ev.get("date")
        if ev_dt is None:
            continue
        if recon_dt is not None and same_calendar_date(ev_dt, recon_dt):
            if best_same is None:
                best_same = ev
        elif recon_dt is None or ev_dt.date() < recon_dt.date():
            if best_prior is None or ev_dt > best_prior.get("date"):
                best_prior = ev
    return best_same if best_same is not None else best_prior


# ============================================================
# COMORBIDITY EXTRACTION (inline — avoids circular import issues)
# ============================================================

SUPPRESS_SEC = {"FAMILY HISTORY", "ALLERGIES", "REVIEW OF SYSTEMS", "ROS", "PERTINENT NEGATIVES"}
PREF_SEC     = {"PAST MEDICAL HISTORY", "PMH", "HISTORY AND PHYSICAL", "H&P", "ASSESSMENT",
                "ASSESSMENT AND PLAN", "MEDICAL HISTORY", "PROBLEM LIST", "PAST HISTORY",
                "DIAGNOSIS", "IMPRESSION", "PREOPERATIVE DIAGNOSIS", "POSTOPERATIVE DIAGNOSIS",
                "ANESTHESIA", "ANESTHESIA H&P"}
LOW_SEC      = {"PAST SURGICAL HISTORY", "PSH", "SURGICAL HISTORY", "HISTORY",
                "GYNECOLOGIC HISTORY", "OB HISTORY"}

NEG_RX      = re.compile(r"\b(no|not|denies|denied|without|negative\s+for|free\s+of|absence\s+of)\b", re.I)
FAMILY_RX   = re.compile(r"\b(family history|mother|father|sister|brother|aunt|uncle|grandmother|grandfather)\b", re.I)
HIST_RX     = re.compile(r"\b(history of|hx of|h/o|s/p|status post|prior|previous|remote)\b", re.I)
PERT_NEG_RX = re.compile(r"\bpertinent negatives?\b", re.I)
ROS_RX      = re.compile(r"\breview of systems\b|\bros\b", re.I)
NONE_TMPL_RX = re.compile(
    r"\b(diabetes|heart failure|cad|coronary artery disease|copd|asthma|renal failure|"
    r"liver disease|pulmonary|neuro|endo)\s*:\s*\(none\)\b", re.I)
VTE_PROPH_RX = re.compile(
    r"\b(prophylaxis|ppx|dvt\s*ppx|vte\s*ppx|sequential\s+compression|compression\s+device|"
    r"scd|scds|subcutaneous\s+heparin|heparin\s+prophylaxis|enoxaparin\s+prophylaxis|"
    r"postop lovenox|lovenox\s+\d+\s*(hr|hours?)\s*postop|continue ambulation|monitor for dvt)\b", re.I)
VTE_RISK_RX  = re.compile(
    r"\b(vte risk assessment|risk assessment|risk score|caprini|venous thromboembolism risk assessment|"
    r"risk of dvt|risk of pe|risk of pulmonary embolism|symptoms?\s+of\s+dvt|symptoms?\s+of\s+pe|"
    r"rule out dvt|r/o dvt|ruled out dvt|ruled out pe|tamoxifen has been shown|"
    r"chance of pulmonary embolism|chance of deep vein thrombosis|dvt prophylaxis|vte prophylaxis|"
    r"to lower risk of vte|risk of vte|risk of thrombosis)\b", re.I)
CARDIAC_RISK_RX = re.compile(
    r"\b(risk of cardiomyopathy|risk of cardiac dysfunction|risk of heart failure|"
    r"cardiac monitoring|echo every 3 months|baseline echo|cardiotoxicity|"
    r"anthracycline|trastuzumab|herceptin|doxorubicin|risk of chf)\b", re.I)
DM_INFO_RX   = re.compile(
    r"\b(may include diabetes|risk of developing diabetes|can include diabetes|"
    r"diabetes\s*:\s*\(none\))\b", re.I)
STEROID_SYS_RX = re.compile(
    r"\b(inhaled|inhaler|intranasal|nasal|topical|cream|ointment|lotion|"
    r"eye\s*drops?|otic|ear\s*drops?)\b", re.I)
STEROID_NEG_RX = re.compile(
    r"\b(no|not|denies|without)\b.{0,40}\b(steroid|prednisone|dexamethasone|medrol|"
    r"methylprednisolone|hydrocortisone)\b", re.I)
STEROID_SHORT_RX = re.compile(
    r"\b(x\s*[1-9]\d?\s*days?|for\s*[1-9]\d?\s*days?|one[- ]time|one time|single dose|"
    r"one dose|pulse|short course|burst)\b", re.I)
STEROID_CHEMO_RX = re.compile(
    r"\b(chemo|chemotherapy|infusion|premed|premedication|antiemetic|for nausea|"
    r"before chemo|after chemo|compazine|emend|zofran|adriamycin|taxotere|cytoxan)\b", re.I)
STEROID_PERIOP_RX = re.compile(
    r"\b(pre[- ]op|preop|post[- ]op|postop|during surgery|at surgery|intraop|perioperative)\b", re.I)
STEROID_CHRONIC_RX = re.compile(
    r"\b(chronic|long[- ]term|maintenance|daily|regular|currently taking|current medications?|"
    r"medication list|home medication|on prednisone|on dexamethasone|takes prednisone|"
    r"taking prednisone|chronic steroid therapy|steroid dependent)\b", re.I)
STEROID_DZ_RX = re.compile(
    r"\b(copd|asthma|rheumatologic|rheumatoid arthritis|lupus|sle|inflammatory bowel disease|"
    r"crohn'?s|ulcerative colitis|autoimmune|adrenal insufficiency)\b", re.I)

COMORB_CONCEPTS = {
    "Diabetes": {
        "pos": [r"\bdiabetes\b", r"\bdiabetes mellitus\b", r"\bt[12]dm\b", r"\biddm\b",
                r"\bniddm\b", r"\bdiabetic\b", r"\bdm2\b", r"\btype\s*[12]\s*diabetes\b"],
        "excl": [r"\bprediabet", r"\bgestational diabetes\b", r"\bdiabetes insipidus\b"],
        "base_conf": 0.84,
    },
    "Hypertension": {
        "pos": [r"\bhypertension\b", r"\bhtn\b", r"\bhigh blood pressure\b"],
        "excl": [r"\bpulmonary hypertension\b", r"\bportal hypertension\b",
                 r"\bgestational hypertension\b", r"\bwhite coat\b"],
        "base_conf": 0.84,
    },
    "CardiacDisease": {
        "pos": [r"\bcoronary artery disease\b", r"\bcad\b", r"\bcongestive heart failure\b",
                r"\bchf\b", r"\bheart failure\b", r"\bmyocardial infarction\b",
                r"\bischemic heart disease\b", r"\bcardiomyopathy\b",
                r"\batrial fibrillation\b", r"\bafib\b", r"\ba[- ]fib\b"],
        "excl": [r"\bmitral valve prolapse\b", r"\bheart murmur\b"],
        "base_conf": 0.82,
    },
    "VenousThromboembolism": {
        "pos": [r"\bdeep vein thrombosis\b", r"\bdvt\b", r"\bpulmonary embol",
                r"\bvte\b", r"\bhistory of dvt\b", r"\bhistory of pe\b"],
        "excl": [r"\brisk of dvt\b", r"\brisk of pe\b", r"\brisk of pulmonary embolism\b"],
        "base_conf": 0.82,
    },
    "Steroid": {
        "pos": [r"\bprednisone\b", r"\bdexamethasone\b", r"\bmethylprednisolone\b",
                r"\bsolu[- ]medrol\b", r"\bmedrol\b", r"\bhydrocortisone\b",
                r"\bprednisolone\b", r"\bcorticosteroid\b", r"\bsteroid\b",
                r"\bimmunosuppress(ant|ive)\b", r"\bmethotrexate\b", r"\bazathioprine\b",
                r"\bmycophenolate\b", r"\btacrolimus\b", r"\bcyclosporine\b"],
        "excl": [],
        "base_conf": 0.84,
    },
}

COMORB_PREFILTER = re.compile(
    r"\b(diabetes|diabetic|dm|insulin|metformin|hypertension|htn|high blood pressure|"
    r"cad|coronary artery disease|chf|heart failure|mi|atrial fibrillation|afib|cardiomyopathy|"
    r"dvt|deep vein thrombosis|pe|pulmonary embol|vte|lovenox|"
    r"prednisone|dexamethasone|methylprednisolone|medrol|hydrocortisone|steroid|"
    r"immunosuppress|methotrexate|azathioprine|mycophenolate|tacrolimus|cyclosporine)\b", re.I)


def _sec_rank(sec):
    s = clean_cell(sec).upper()
    if s in PREF_SEC: return 0
    if s in LOW_SEC:  return 2
    return 1


def _window(text, start, end, width=260):
    return text[max(0, start - width):min(len(text), end + width)].strip()


def _looks_like_template(low):
    return len(re.findall(
        r"\b(asthma|cad|copd|dvt|diabetes mellitus|mi|pulmonary embolism|sleep apnea|stroke)\b",
        low, re.I)) >= 3


def _bad_context(field, sec, evid):
    low = clean_cell(evid).lower()
    if not low: return True
    if PERT_NEG_RX.search(low): return True
    if NONE_TMPL_RX.search(low): return True
    if ROS_RX.search(low) or ROS_RX.search(sec.lower()): return True
    if _looks_like_template(low) and ("pertinent negatives" in low or "active problem list" in low):
        return True
    if field == "VenousThromboembolism":
        if VTE_PROPH_RX.search(low) or VTE_RISK_RX.search(low): return True
    if field == "CardiacDisease":
        if CARDIAC_RISK_RX.search(low): return True
    if field == "Diabetes":
        if DM_INFO_RX.search(low): return True
    return False


def _steroid_ok(low):
    if STEROID_SYS_RX.search(low): return False
    if STEROID_NEG_RX.search(low): return False
    if STEROID_SHORT_RX.search(low): return False
    if STEROID_CHEMO_RX.search(low): return False
    if STEROID_PERIOP_RX.search(low): return False
    return STEROID_CHRONIC_RX.search(low) is not None or STEROID_DZ_RX.search(low) is not None


def extract_comorbidities_inline(note):
    cands = []
    sections = list(note.sections.keys())
    sections.sort(key=_sec_rank)

    for sec in sections:
        sec_u = clean_cell(sec).upper()
        if sec_u in SUPPRESS_SEC:
            continue
        text = clean_cell(note.sections.get(sec, ""))
        if not text:
            continue

        for field, cfg in COMORB_CONCEPTS.items():
            m = None
            for p in cfg["pos"]:
                mm = re.search(p, text, re.I)
                if mm:
                    if m is None or mm.start() < m.start():
                        m = mm
            if not m:
                continue

            evid = _window(text, m.start(), m.end(), 260)
            low  = evid.lower()

            if FAMILY_RX.search(low): continue
            if _bad_context(field, sec_u, evid): continue
            if cfg["excl"] and any(re.search(e, low, re.I) for e in cfg["excl"]): continue
            if field == "Steroid" and not _steroid_ok(low): continue
            if field == "VenousThromboembolism" and VTE_PROPH_RX.search(low): continue

            status = "denied" if NEG_RX.search(low) else "history"
            if status == "denied": continue

            rank = _sec_rank(sec_u)
            conf = cfg["base_conf"]
            if rank == 0: conf = min(0.98, conf + 0.05)
            elif rank == 2: conf = max(0.55, conf - 0.08)

            cands.append(Candidate(
                field=field, value=True, status=status,
                evidence=evid, section=sec_u,
                note_type=note.note_type, note_id=note.note_id,
                note_date=note.note_date, confidence=conf
            ))

    return cands


# ============================================================
# MASTER SEEDING
# ============================================================

def seed_master(struct_df):
    mrns = set()
    for _, row in struct_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if mrn:
            mrns.add(mrn)

    if not mrns:
        # fallback: collect MRNs from note files
        for g in NOTE_GLOBS:
            for fp in glob(g, recursive=True):
                try:
                    df = clean_cols(read_csv_robust(fp))
                    df = normalize_mrn(df)
                    mrns.update(df[MERGE_KEY].dropna().astype(str).str.strip().tolist())
                except Exception:
                    pass

    mrns = sorted([m for m in mrns if m])
    master = pd.DataFrame({MERGE_KEY: mrns})
    master["ENCRYPTED_PAT_ID"] = master[MERGE_KEY]
    for c in MASTER_COLUMNS:
        if c not in master.columns:
            master[c] = pd.NA
    return master[MASTER_COLUMNS].copy()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("run_full_pipeline.py")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load structured encounters (once)
    # ----------------------------------------------------------
    print("\n[1/6] Loading structured encounters...")
    struct_df = load_structured_encounters()
    print("      Encounter rows: {0}".format(len(struct_df)))

    recon_anchor_map   = build_recon_anchor_map(struct_df)
    recon_struct_map   = build_recon_structured_map(struct_df)
    mastectomy_evt_map = build_mastectomy_events(struct_df)
    race_map           = build_race_map(struct_df)
    eth_map            = build_ethnicity_map(struct_df)
    age_map            = build_age_map(struct_df, recon_anchor_map)

    print("      Recon anchors: {0}".format(len(recon_anchor_map)))
    print("      Race entries:  {0}".format(len(race_map)))

    # ----------------------------------------------------------
    # 2. Seed master
    # ----------------------------------------------------------
    print("\n[2/6] Seeding master...")
    master = seed_master(struct_df)
    master = normalize_mrn(master)
    print("      MRNs: {0}".format(len(master)))

    # Fill structured demographics
    for mrn in master[MERGE_KEY].astype(str).str.strip().tolist():
        mask = master[MERGE_KEY].astype(str).str.strip() == mrn
        if race_map.get(mrn):
            master.loc[mask, "Race"] = race_map[mrn]
        if eth_map.get(mrn):
            master.loc[mask, "Ethnicity"] = eth_map[mrn]
        if age_map.get(mrn) is not None:
            master.loc[mask, "Age"] = age_map[mrn]

    # Fill structured recon fields
    for mrn, info in recon_struct_map.items():
        mask = master[MERGE_KEY].astype(str).str.strip() == mrn
        if not mask.any():
            continue
        if info.get("laterality"):
            master.loc[mask, "Recon_Laterality"] = info["laterality"]
        if info.get("recon_type"):
            master.loc[mask, "Recon_Type"] = info["recon_type"]
        if info.get("recon_class"):
            master.loc[mask, "Recon_Classification"] = info["recon_class"]

    # ----------------------------------------------------------
    # 3. Load notes (once)
    # ----------------------------------------------------------
    print("\n[3/6] Loading and reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("      Reconstructed notes: {0}".format(len(notes_df)))

    # ----------------------------------------------------------
    # 4. Run all extractors in one pass
    # ----------------------------------------------------------
    print("\n[4/6] Running extractors...")

    evidence_rows = []

    # Accumulators
    best_bmi       = {}   # mrn -> best BMI candidate
    best_smoking   = {}   # mrn -> best smoking candidate
    best_pbs       = {}   # mrn -> {field -> best candidate}
    best_comorb    = {}   # mrn -> {field -> best candidate}
    best_cancer    = {}   # mrn -> {field -> best candidate}
    therapy_dates  = {}   # mrn -> {"Radiation": [dt,...], "Chemo": [dt,...], "Mastectomy_Date": [dt,...]}
    lymphnode_cands = {}  # mrn -> [candidates]

    note_count = 0

    for _, row in notes_df.iterrows():
        mrn       = clean_cell(row.get(MERGE_KEY, ""))
        note_text = clean_cell(row.get("NOTE_TEXT", ""))
        note_dt   = parse_date_safe(row.get("NOTE_DATE", ""))
        if not mrn or not note_text:
            continue

        mask = master[MERGE_KEY].astype(str).str.strip() == mrn
        if not mask.any():
            continue

        anchor   = recon_anchor_map.get(mrn)
        recon_dt = parse_date_safe((anchor or {}).get("recon_date", ""))

        snote = build_sectioned_note(
            note_text=note_text,
            note_type=row.get("NOTE_TYPE", ""),
            note_id=row.get("NOTE_ID", ""),
            note_date=row.get("NOTE_DATE", "")
        )

        # ---------- BMI ----------
        if anchor is not None and recon_dt is not None and note_dt is not None:
            if bmi_in_window(note_dt, recon_dt):
                try:
                    for c in extract_bmi(snote):
                        best_bmi[mrn] = choose_best_bmi(best_bmi.get(mrn), c, recon_dt)
                        evidence_rows.append({
                            MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                            "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                            "FIELD": "BMI", "VALUE": getattr(c, "value", ""),
                            "STATUS": getattr(c, "status", ""),
                            "CONFIDENCE": getattr(c, "confidence", ""),
                            "SECTION": getattr(c, "section", ""), "EVIDENCE": getattr(c, "evidence", "")
                        })
                except Exception as e:
                    evidence_rows.append({MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                                          "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                                          "FIELD": "EXTRACTOR_ERROR", "VALUE": "", "STATUS": "",
                                          "CONFIDENCE": "", "SECTION": "", "EVIDENCE": "extract_bmi: " + repr(e)})

        # ---------- Smoking ----------
        if anchor is not None and recon_dt is not None and note_dt is not None:
            if note_on_or_before(note_dt, recon_dt):
                try:
                    for c in extract_smoking(snote):
                        val = clean_cell(getattr(c, "value", ""))
                        if val in {"Current", "Former", "Never"}:
                            best_smoking[mrn] = choose_best_smoking(best_smoking.get(mrn), c, recon_dt)
                            evidence_rows.append({
                                MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                                "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                                "FIELD": "SmokingStatus", "VALUE": val,
                                "STATUS": getattr(c, "status", ""),
                                "CONFIDENCE": getattr(c, "confidence", ""),
                                "SECTION": getattr(c, "section", ""), "EVIDENCE": getattr(c, "evidence", "")
                            })
                except Exception as e:
                    evidence_rows.append({MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                                          "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                                          "FIELD": "EXTRACTOR_ERROR", "VALUE": "", "STATUS": "",
                                          "CONFIDENCE": "", "SECTION": "", "EVIDENCE": "extract_smoking: " + repr(e)})

        # ---------- PBS ----------
        if anchor is not None and recon_dt is not None and note_dt is not None:
            try:
                recon_lat = ""
                if "Recon_Laterality" in master.columns:
                    recon_lat = clean_cell(master.loc[mask, "Recon_Laterality"].iloc[0])

                full_text = clean_cell(row.get("NOTE_TEXT", ""))
                for c in extract_pbs(snote):
                    field = clean_cell(getattr(c, "field", ""))
                    if field not in {"PBS_Lumpectomy", "PBS_Breast Reduction",
                                     "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other"}:
                        continue
                    evid = clean_cell(getattr(c, "evidence", ""))
                    if not evid:
                        continue
                    combined = evid + "\n" + full_text
                    proc_lat = _extract_lat(combined)
                    day_diff = days_between(note_dt, recon_dt)
                    accept, reason = pbs_accept(field, evid, day_diff, recon_lat, proc_lat, combined)

                    evidence_rows.append({
                        MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                        "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                        "FIELD": field, "VALUE": getattr(c, "value", ""),
                        "STATUS": reason, "CONFIDENCE": getattr(c, "confidence", ""),
                        "SECTION": getattr(c, "section", ""), "EVIDENCE": evid
                    })

                    if accept:
                        setattr(c, "_source_file", row.get("SOURCE_FILE", ""))
                        setattr(c, "_accepted_post_hist", bool(day_diff is not None and day_diff >= 0 and _pbs_history_ok(field, combined)))
                        best_pbs.setdefault(mrn, {})
                        best_pbs[mrn][field] = choose_best_pbs(best_pbs[mrn].get(field), c, recon_dt)
            except Exception as e:
                evidence_rows.append({MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                                       "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                                       "FIELD": "EXTRACTOR_ERROR", "VALUE": "", "STATUS": "",
                                       "CONFIDENCE": "", "SECTION": "", "EVIDENCE": "extract_pbs: " + repr(e)})

        # ---------- Comorbidities ----------
        if COMORB_PREFILTER.search(note_text):
            try:
                for c in extract_comorbidities_inline(snote):
                    field = clean_cell(getattr(c, "field", ""))
                    evid  = clean_cell(getattr(c, "evidence", ""))
                    if not evid: continue
                    status = clean_cell(getattr(c, "status", ""))
                    if status == "denied": continue
                    if _bad_context(field, getattr(c, "section", ""), evid): continue

                    evidence_rows.append({
                        MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                        "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                        "FIELD": field, "VALUE": getattr(c, "value", ""),
                        "STATUS": "accept_positive", "CONFIDENCE": getattr(c, "confidence", ""),
                        "SECTION": getattr(c, "section", ""), "EVIDENCE": evid
                    })

                    best_comorb.setdefault(mrn, {})
                    best_comorb[mrn][field] = merge_boolean(best_comorb[mrn].get(field), c)
            except Exception as e:
                evidence_rows.append({MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                                       "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                                       "FIELD": "EXTRACTOR_ERROR", "VALUE": "", "STATUS": "",
                                       "CONFIDENCE": "", "SECTION": "", "EVIDENCE": "extract_comorbidities: " + repr(e)})

        # ---------- Cancer / Recon / LymphNode ----------
        if CANCER_KEYWORD_RX.search(note_text):
            try:
                for c in extract_breast_cancer_recon(snote):
                    field = clean_cell(str(getattr(c, "field", "")))

                    evidence_rows.append({
                        MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                        "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                        "FIELD": field, "VALUE": getattr(c, "value", ""),
                        "STATUS": getattr(c, "status", ""), "CONFIDENCE": getattr(c, "confidence", ""),
                        "SECTION": getattr(c, "section", ""), "EVIDENCE": getattr(c, "evidence", "")
                    })

                    # Collect therapy dates for timing
                    if field == "Radiation":
                        dt = parse_date_safe(getattr(c, "note_date", row.get("NOTE_DATE", "")))
                        if dt: therapy_dates.setdefault(mrn, {"Radiation": [], "Chemo": [], "Mastectomy_Date": []})["Radiation"].append(dt)
                    elif field == "Chemo":
                        dt = parse_date_safe(getattr(c, "note_date", row.get("NOTE_DATE", "")))
                        if dt: therapy_dates.setdefault(mrn, {"Radiation": [], "Chemo": [], "Mastectomy_Date": []})["Chemo"].append(dt)
                    elif field == "Mastectomy_Date":
                        dt = parse_date_safe(getattr(c, "value", ""))
                        if dt: therapy_dates.setdefault(mrn, {"Radiation": [], "Chemo": [], "Mastectomy_Date": []})["Mastectomy_Date"].append(dt)
                    elif field == "LymphNode":
                        lymphnode_cands.setdefault(mrn, []).append(c)
                        continue
                    else:
                        best_cancer.setdefault(mrn, {})
                        existing = best_cancer[mrn].get(field)
                        if field in {"Radiation", "Chemo"}:
                            best_cancer[mrn][field] = merge_boolean(existing, c)
                        elif field in {"Indication_Left", "Indication_Right"}:
                            if existing is None or cand_score_basic(c) > cand_score_basic(existing):
                                best_cancer[mrn][field] = c
                        else:
                            best_cancer[mrn][field] = choose_best(existing, c)
            except Exception as e:
                evidence_rows.append({MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                                       "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                                       "FIELD": "EXTRACTOR_ERROR", "VALUE": "", "STATUS": "",
                                       "CONFIDENCE": "", "SECTION": "", "EVIDENCE": "extract_breast_cancer_recon: " + repr(e)})

        note_count += 1
        if note_count % 5000 == 0:
            print("      Processed {0} notes...".format(note_count))

    print("      Done. Notes processed: {0}".format(note_count))

    # ----------------------------------------------------------
    # 5. Write results to master
    # ----------------------------------------------------------
    print("\n[5/6] Writing results to master...")

    for mrn in master[MERGE_KEY].astype(str).str.strip().tolist():
        mask = master[MERGE_KEY].astype(str).str.strip() == mrn
        if not mask.any():
            continue

        anchor   = recon_anchor_map.get(mrn)
        recon_dt = parse_date_safe((anchor or {}).get("recon_date", ""))

        # BMI
        bmi_cand = best_bmi.get(mrn)
        if bmi_cand is not None:
            try:
                bmi_val = round(float(getattr(bmi_cand, "value", 0)), 1)
                master.loc[mask, "BMI"]     = bmi_val
                master.loc[mask, "Obesity"] = 1 if bmi_val >= 30.0 else 0
            except Exception:
                pass

        # Smoking
        smoke_cand = best_smoking.get(mrn)
        if smoke_cand is not None:
            val = clean_cell(getattr(smoke_cand, "value", ""))
            if val:
                master.loc[mask, "SmokingStatus"] = val

        # PBS
        pbs_fields = best_pbs.get(mrn, {})
        any_pbs = False
        for field in ["PBS_Lumpectomy", "PBS_Breast Reduction", "PBS_Mastopexy",
                       "PBS_Augmentation", "PBS_Other"]:
            cand = pbs_fields.get(field)
            if cand is not None:
                master.loc[mask, field] = 1
                any_pbs = True
        master.loc[mask, "PastBreastSurgery"] = 1 if any_pbs else 0

        # Comorbidities
        for field, cand in best_comorb.get(mrn, {}).items():
            if field in master.columns:
                master.loc[mask, field] = 1 if bool(getattr(cand, "value", False)) else 0

        # Cancer / Recon fields
        for field, cand in best_cancer.get(mrn, {}).items():
            if field == "Mastectomy_Date":
                continue
            val = getattr(cand, "value", pd.NA)
            if field in {"Radiation", "Chemo"}:
                try:
                    val = 1 if bool(val) else 0
                except Exception:
                    val = pd.NA
            if field in master.columns:
                master.loc[mask, field] = val

        # LymphNode — simple best score (full episode logic omitted for cleanliness;
        # can be enhanced if LymphNode accuracy needs improvement)
        ln_cands = lymphnode_cands.get(mrn, [])
        if ln_cands:
            best_ln = None
            for c in ln_cands:
                val = clean_cell(getattr(c, "value", ""))
                if val in {"ALND", "SLNB"}:
                    best_ln = choose_best(best_ln, c)
            if best_ln is not None:
                master.loc[mask, "LymphNode"] = getattr(best_ln, "value", "")
        # Default to "none" if still empty
        if clean_cell(master.loc[mask, "LymphNode"].iloc[0]) == "":
            master.loc[mask, "LymphNode"] = "none"

        # Recon timing + radiation/chemo before/after
        if recon_dt is not None:
            # Timing
            timing_val = clean_cell(master.loc[mask, "Recon_Timing"].iloc[0])
            if not timing_val:
                immediate = False
                delayed   = False
                for ev in mastectomy_evt_map.get(mrn, []):
                    ev_dt = ev.get("date")
                    if ev_dt is None: continue
                    if same_calendar_date(ev_dt, recon_dt): immediate = True; break
                    if ev_dt.date() < recon_dt.date(): delayed = True
                if not immediate:
                    for ev_dt in therapy_dates.get(mrn, {}).get("Mastectomy_Date", []):
                        if same_calendar_date(ev_dt, recon_dt): immediate = True; break
                        if ev_dt.date() < recon_dt.date(): delayed = True
                if immediate:   master.loc[mask, "Recon_Timing"] = "Immediate"
                elif delayed:   master.loc[mask, "Recon_Timing"] = "Delayed"

            # Mastectomy laterality
            mast_lat = clean_cell(master.loc[mask, "Mastectomy_Laterality"].iloc[0])
            if not mast_lat:
                best_mev = choose_best_mastectomy(mastectomy_evt_map.get(mrn, []), recon_dt)
                if best_mev and clean_cell(best_mev.get("laterality", "")):
                    master.loc[mask, "Mastectomy_Laterality"] = best_mev["laterality"]

            # Radiation/Chemo timing
            rad_dates   = therapy_dates.get(mrn, {}).get("Radiation", [])
            chemo_dates = therapy_dates.get(mrn, {}).get("Chemo", [])

            rad_before = rad_after = chemo_before = chemo_after = 0
            for dt in rad_dates:
                dd = days_between(dt, recon_dt)
                if dd is None: continue
                if dd < 0: rad_before = 1
                elif dd > 0: rad_after = 1
            for dt in chemo_dates:
                dd = days_between(dt, recon_dt)
                if dd is None: continue
                if dd < 0: chemo_before = 1
                elif dd > 0: chemo_after = 1

            master.loc[mask, "Radiation_Before"] = rad_before
            master.loc[mask, "Radiation_After"]  = rad_after
            master.loc[mask, "Chemo_Before"]      = chemo_before
            master.loc[mask, "Chemo_After"]       = chemo_after

            cur_rad = clean_cell(master.loc[mask, "Radiation"].iloc[0])
            if rad_before or rad_after:
                master.loc[mask, "Radiation"] = 1
            elif cur_rad not in {"1", "True", "true"}:
                master.loc[mask, "Radiation"] = 0

            cur_chemo = clean_cell(master.loc[mask, "Chemo"].iloc[0])
            if chemo_before or chemo_after:
                master.loc[mask, "Chemo"] = 1
            elif cur_chemo not in {"1", "True", "true"}:
                master.loc[mask, "Chemo"] = 0

    # Zero-out Stage outcome columns (filled by complications patch later)
    stage_cols = [
        "Stage1_MinorComp", "Stage1_Reoperation", "Stage1_Rehospitalization",
        "Stage1_MajorComp", "Stage1_Failure", "Stage1_Revision",
        "Stage2_MinorComp", "Stage2_Reoperation", "Stage2_Rehospitalization",
        "Stage2_MajorComp", "Stage2_Failure", "Stage2_Revision",
    ]
    for col in stage_cols:
        if col in master.columns:
            master[col] = 0

    # ----------------------------------------------------------
    # 6. Write outputs
    # ----------------------------------------------------------
    print("\n[6/6] Writing outputs...")
    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    master.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\n" + "=" * 60)
    print("DONE.")
    print("Master: {0}".format(OUTPUT_MASTER))
    print("Evidence: {0}".format(OUTPUT_EVID))
    print("\nNext steps:")
    print("  1. Run stage2 chain (unchanged)")
    print("  2. python build_master_rule_COMPLICATIONS_PATCH.py")
    print("  3. python validate_abstraction.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
