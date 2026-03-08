#!/usr/bin/env python3
# build_master_rule_FINAL_NO_GOLD.py
#
# RULE-BASED ONLY builder that DOES NOT use the gold sheet as a base.
# - Builds a clean master dataframe from STRUCTURED encounter files (or MRNs in notes as fallback)
# - Loads ORIGINAL HPI11526 note CSVs (Clinic/Inpatient/Operation Notes)
# - Reconstructs full note text by NOTE_ID + LINE ordering
# - Lightweight sectionizer
# - Enriches Race / Ethnicity / Age from structured encounters
# - Runs rule-based extractors from ./extractors
# - Writes:
#   1) /home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv
#   2) /home/apokol/Breast_Restore/_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv
#
# UPDATE:
# - BMI anchoring widened to peri-reconstruction notes.
# - BMI is now searched across ALL notes in the best available date window
#   around the structured reconstruction date:
#     (1) same day
#     (2) +/- 1 day
#     (3) +/- 3 days
# - Within that window, note types are prioritized as:
#     OP NOTE / BRIEF OP NOTE / operative / operation
#     then anesthesia / pre-op
#     then progress / clinic / H&P / physical exam style notes
# - This keeps BMI reconstruction-timed while no longer assuming the BMI
#   must be written inside the OP NOTE text itself.
#
# Python 3.6.8 compatible

import os
import re
import math
from glob import glob
from datetime import datetime

import pandas as pd

# -----------------------
# CONFIG (NO USER INPUTS)
# -----------------------
BASE_DIR = "/home/apokol/Breast_Restore"

STRUCT_GLOBS = [
    f"{BASE_DIR}/**/HPI11526*Clinic Encounters.csv",
    f"{BASE_DIR}/**/HPI11526*Inpatient Encounters.csv",
    f"{BASE_DIR}/**/HPI11526*Operation Encounters.csv",
    f"{BASE_DIR}/**/HPI11526*clinic encounters.csv",
    f"{BASE_DIR}/**/HPI11526*inpatient encounters.csv",
    f"{BASE_DIR}/**/HPI11526*operation encounters.csv",
]

NOTE_GLOBS = [
    f"{BASE_DIR}/**/HPI11526*Clinic Notes.csv",
    f"{BASE_DIR}/**/HPI11526*Inpatient Notes.csv",
    f"{BASE_DIR}/**/HPI11526*Operation Notes.csv",
    f"{BASE_DIR}/**/HPI11526*clinic notes.csv",
    f"{BASE_DIR}/**/HPI11526*inpatient notes.csv",
    f"{BASE_DIR}/**/HPI11526*operation notes.csv",
]

OUTPUT_MASTER = f"{BASE_DIR}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
OUTPUT_EVID = f"{BASE_DIR}/_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv"
MERGE_KEY = "MRN"

# -----------------------
# Imports from your repo
# -----------------------
from models import SectionedNote, Candidate  # noqa: E402
from extractors.age import extract_age  # noqa: E402
from extractors.bmi import extract_bmi  # noqa: E402
from extractors.smoking import extract_smoking  # noqa: E402
from extractors.comorbidities import extract_comorbidities  # noqa: E402
from extractors.pbs import extract_pbs  # noqa: E402
from extractors.mastectomy import extract_mastectomy  # noqa: E402
from extractors.cancer_treatment import extract_cancer_treatment  # noqa: E402

# -----------------------
# Robust CSV read
# -----------------------
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
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
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
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%Y/%m/%d",
        "%d-%b-%Y",
        "%d-%b-%Y %H:%M:%S",
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


def same_calendar_date(d1, d2):
    if d1 is None or d2 is None:
        return False
    try:
        return d1.date() == d2.date()
    except Exception:
        return False


def abs_day_diff(d1, d2):
    if d1 is None or d2 is None:
        return None
    try:
        return abs((d1.date() - d2.date()).days)
    except Exception:
        return None


def is_operation_note_type(note_type):
    nt = clean_cell(note_type).upper()
    return nt in {"OP NOTE", "BRIEF OP NOTE"} or ("OPERATIVE" in nt) or ("OP NOTE" in nt) or ("BRIEF OP NOTE" in nt)


def bmi_note_type_priority(note_type):
    nt = clean_cell(note_type).lower()

    if ("op note" in nt) or ("brief op note" in nt) or ("operative" in nt) or ("operation" in nt):
        return 1

    if ("anesthesia" in nt) or ("pre-op" in nt) or ("pre op" in nt) or ("preprocedure" in nt) or ("pre-procedure" in nt):
        return 2

    if ("progress" in nt) or ("clinic" in nt) or ("h&p" in nt) or ("history and physical" in nt) or ("physical exam" in nt):
        return 3

    return 9


# -----------------------
# Lightweight sectionizer
# -----------------------
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


# -----------------------
# Aggregation logic
# -----------------------
def cand_score(c):
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    nt = str(getattr(c, "note_type", "") or "").lower()
    op_bonus = 0.05 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    date_bonus = 0.01 if (getattr(c, "note_date", "") or "").strip() else 0.0
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


# -----------------------
# Field mapping to your FINAL columns
# -----------------------
FIELD_MAP = {
    "Age": "Age",
    "Age_DOS": "Age",
    "BMI": "BMI",
    "SmokingStatus": "SmokingStatus",
    "Diabetes": "Diabetes",
    "DiabetesMellitus": "Diabetes",
    "Hypertension": "Hypertension",
    "CardiacDisease": "CardiacDisease",
    "VTE": "VenousThromboembolism",
    "VenousThromboembolism": "VenousThromboembolism",
    "SteroidUse": "Steroid",
    "Steroid": "Steroid",
    "PastBreastSurgery": "PastBreastSurgery",
    "PBS_Lumpectomy": "PBS_Lumpectomy",
    "PBS_Breast Reduction": "PBS_Breast Reduction",
    "PBS_Mastopexy": "PBS_Mastopexy",
    "PBS_Augmentation": "PBS_Augmentation",
    "PBS_Other": "PBS_Other",
    "Mastectomy_Laterality": "Mastectomy_Laterality",
    "Radiation": "Radiation",
    "Chemo": "Chemo",
}

BOOLEAN_FIELDS = {
    "Diabetes", "Hypertension", "CardiacDisease",
    "VenousThromboembolism", "Steroid",
    "PastBreastSurgery", "PBS_Lumpectomy", "PBS_Breast Reduction",
    "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other",
    "Radiation", "Chemo"
}

MASTER_COLUMNS = [
    "MRN",
    "ENCRYPTED_PAT_ID",
    "Last name",
    "DOB",
    "PatientID",
    "Race",
    "Ethnicity",
    "Age",
    "BMI",
    "CCI",
    "SmokingStatus",
    "Diabetes",
    "Obesity",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
    "PastBreastSurgery",
    "PBS_Lumpectomy",
    "PBS_Breast Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "PBS_Other",
    "Mastectomy_Laterality",
    "Indication_Left",
    "Indication_Right",
    "LymphNode",
    "Radiation",
    "Radiation_Before",
    "Radiation_After",
    "Chemo",
    "Chemo_Before",
    "Chemo_After",
    "Recon_Laterality",
    "Recon_Type",
    "Recon_Classification",
    "Recon_Timing",
    "Stage1_MinorComp",
    "Stage1_Reoperation",
    "Stage1_Rehospitalization",
    "Stage1_MajorComp",
    "Stage1_Failure",
    "Stage1_Revision",
    "Stage2_MinorComp",
    "Stage2_Reoperation",
    "Stage2_Rehospitalization",
    "Stage2_MajorComp",
    "Stage2_Failure",
    "Stage2_Revision",
    "Stage2_Applicable",
]


def seed_master_from_structured():
    struct_files = []
    for g in STRUCT_GLOBS:
        struct_files.extend(glob(g, recursive=True))
    struct_files = sorted(set(struct_files))

    mrns = set()

    if struct_files:
        for fp in struct_files:
            df = clean_cols(read_csv_robust(fp))
            df = normalize_mrn(df)
            mrns.update(df[MERGE_KEY].dropna().astype(str).str.strip().tolist())
    else:
        note_files = []
        for g in NOTE_GLOBS:
            note_files.extend(glob(g, recursive=True))
        note_files = sorted(set(note_files))

        if not note_files:
            raise FileNotFoundError("No structured encounters OR notes found to seed MRNs.")

        for fp in note_files:
            df = clean_cols(read_csv_robust(fp))
            df = normalize_mrn(df)
            mrns.update(df[MERGE_KEY].dropna().astype(str).str.strip().tolist())

    mrns = sorted([m for m in mrns if m])
    master = pd.DataFrame({MERGE_KEY: mrns})
    master["ENCRYPTED_PAT_ID"] = master[MERGE_KEY]

    for c in MASTER_COLUMNS:
        if c not in master.columns:
            master[c] = pd.NA

    master = master[MASTER_COLUMNS]
    return master


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
            ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE", "OPERATION_DATE",
             "ADMIT_DATE", "HOSP_ADMSN_TIME"],
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


# -----------------------
# Structured enrichment: Race / Ethnicity / Age / BMI anchor date
# -----------------------
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

        race_col = pick_col(df, ["RACE", "Race"], required=False)
        eth_col = pick_col(df, ["ETHNICITY", "Ethnicity"], required=False)
        age_col = pick_col(df, ["AGE_AT_ENCOUNTER", "Age_at_encounter", "AGE"], required=False)
        admit_col = pick_col(df, ["ADMIT_DATE", "Admit_Date"], required=False)
        recon_col = pick_col(df, ["RECONSTRUCTION_DATE", "RECONSTRUCTION DATE"], required=False)
        cpt_col = pick_col(df, ["CPT_CODE", "CPT CODE", "CPT"], required=False)
        proc_col = pick_col(df, ["PROCEDURE", "Procedure"], required=False)
        reason_col = pick_col(df, ["REASON_FOR_VISIT", "REASON FOR VISIT"], required=False)
        date_col = pick_col(df, ["OPERATION_DATE", "CHECKOUT_TIME", "DISCHARGE_DATE_DT"], required=False)

        out = pd.DataFrame()
        out[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
        out["STRUCT_SOURCE"] = encounter_source
        out["STRUCT_PRIORITY"] = priority
        out["STRUCT_DATE_RAW"] = df[date_col].astype(str) if date_col else ""
        out["RACE_STRUCT"] = df[race_col].astype(str) if race_col else ""
        out["ETHNICITY_STRUCT"] = df[eth_col].astype(str) if eth_col else ""
        out["AGE_AT_ENCOUNTER_STRUCT"] = df[age_col].astype(str) if age_col else ""
        out["ADMIT_DATE_STRUCT"] = df[admit_col].astype(str) if admit_col else ""
        out["RECONSTRUCTION_DATE_STRUCT"] = df[recon_col].astype(str) if recon_col else ""
        out["CPT_CODE_STRUCT"] = df[cpt_col].astype(str) if cpt_col else ""
        out["PROCEDURE_STRUCT"] = df[proc_col].astype(str) if proc_col else ""
        out["REASON_FOR_VISIT_STRUCT"] = df[reason_col].astype(str) if reason_col else ""
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            MERGE_KEY, "STRUCT_SOURCE", "STRUCT_PRIORITY", "STRUCT_DATE_RAW",
            "RACE_STRUCT", "ETHNICITY_STRUCT", "AGE_AT_ENCOUNTER_STRUCT",
            "ADMIT_DATE_STRUCT", "RECONSTRUCTION_DATE_STRUCT",
            "CPT_CODE_STRUCT", "PROCEDURE_STRUCT", "REASON_FOR_VISIT_STRUCT"
        ])

    struct_df = pd.concat(rows, ignore_index=True)
    return struct_df


def normalize_race_token(x):
    s = clean_cell(x).lower()
    if not s:
        return ""

    if s in {"white or caucasian", "white", "caucasian"}:
        return "White"
    if s in {"black or african american", "black", "african american"}:
        return "Black or African American"
    if s in {"asian", "filipino", "other asian", "asian indian", "chinese", "japanese", "korean", "vietnamese"}:
        return "Asian"
    if s == "american indian or alaska native":
        return "American Indian or Alaska Native"
    if s in {"native hawaiian", "pacific islander", "native hawaiian or other pacific islander"}:
        return "Native Hawaiian or Other Pacific Islander"
    if s == "other":
        return "Other"
    if s in {"unknown", "patient refused", "declined", "refused", "unable to obtain", "choose not to disclose"}:
        return "Unknown / Declined / Not Reported"

    return clean_cell(x)


def normalize_ethnicity_value(x):
    s = clean_cell(x)
    if not s:
        return ""
    return s


def normalize_race_value_list(raw_values):
    real_races = []
    unknown_seen = False

    for raw in raw_values:
        norm = normalize_race_token(raw)
        if not norm:
            continue
        if norm == "Unknown / Declined / Not Reported":
            unknown_seen = True
            continue
        if norm not in real_races:
            real_races.append(norm)

    if len(real_races) == 0:
        if unknown_seen:
            return "Unknown / Declined / Not Reported"
        return ""

    if len(real_races) == 1:
        return real_races[0]

    return "Multiracial"


def choose_best_ethnicity(struct_df):
    eth_best = {}

    if len(struct_df) == 0:
        return eth_best

    for _, row in struct_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        eth_val = normalize_ethnicity_value(row.get("ETHNICITY_STRUCT", ""))
        if not eth_val:
            continue

        priority = to_int_safe(row.get("STRUCT_PRIORITY", 9))
        if priority is None:
            priority = 9

        cur = eth_best.get(mrn)
        score = priority

        if cur is None or score < cur["score"]:
            eth_best[mrn] = {
                "value": eth_val,
                "score": score,
                "source": row.get("STRUCT_SOURCE", ""),
                "date_raw": row.get("STRUCT_DATE_RAW", "")
            }

    return eth_best


def choose_race_us_categories(struct_df):
    race_by_mrn = {}

    if len(struct_df) == 0:
        return race_by_mrn

    for _, row in struct_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        raw_race = clean_cell(row.get("RACE_STRUCT", ""))
        if not raw_race:
            continue

        if mrn not in race_by_mrn:
            race_by_mrn[mrn] = []

        race_by_mrn[mrn].append(raw_race)

    out = {}
    for mrn, raw_values in race_by_mrn.items():
        out[mrn] = normalize_race_value_list(raw_values)

    return out


def _recon_anchor_helper(struct_df, require_age=False):
    out_best = {}

    if len(struct_df) == 0:
        return out_best

    source_priority = {
        "clinic": 1,
        "operation": 2,
        "inpatient": 3
    }

    preferred_cpts = set([
        "19357",
        "19340",
        "19342",
        "19361",
        "19364",
        "19367",
        "S2068"
    ])

    primary_exclude_cpts = set([
        "19325",
        "19330"
    ])

    fallback_allowed_cpts = set([
        "19350",
        "19380"
    ])

    eligible_sources = struct_df[struct_df["STRUCT_SOURCE"].isin(["clinic", "operation", "inpatient"])].copy()
    if len(eligible_sources) == 0:
        return out_best

    has_preferred_cpt = {}

    for mrn, g in eligible_sources.groupby(MERGE_KEY):
        found = False
        for val in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist():
            cpt = clean_cell(val).upper()
            if cpt in preferred_cpts:
                found = True
                break
        has_preferred_cpt[mrn] = found

    for _, row in eligible_sources.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        if source not in source_priority:
            continue

        age_raw = clean_cell(row.get("AGE_AT_ENCOUNTER_STRUCT", ""))
        age_base = to_float_safe(age_raw)

        admit_date = parse_date_safe(row.get("ADMIT_DATE_STRUCT", ""))
        recon_date = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))

        cpt_code = clean_cell(row.get("CPT_CODE_STRUCT", "")).upper()
        procedure = clean_cell(row.get("PROCEDURE_STRUCT", "")).lower()
        reason_for_visit = clean_cell(row.get("REASON_FOR_VISIT_STRUCT", "")).lower()

        if require_age and age_base is None:
            continue

        if admit_date is None or recon_date is None:
            continue

        if cpt_code in primary_exclude_cpts:
            continue

        if has_preferred_cpt.get(mrn, False) and cpt_code in fallback_allowed_cpts:
            continue

        is_anchor = False

        if cpt_code in preferred_cpts:
            is_anchor = True

        if (not has_preferred_cpt.get(mrn, False)) and (cpt_code in fallback_allowed_cpts):
            is_anchor = True

        if not is_anchor:
            if (
                ("tissue expander" in procedure) or
                ("breast recon" in procedure) or
                ("implant on same day of mastectomy" in procedure) or
                ("insert or replcmnt breast implnt on sep day from mastectomy" in procedure) or
                ("latissimus" in procedure) or
                ("diep" in procedure) or
                ("tram" in procedure) or
                ("flap" in procedure)
            ):
                is_anchor = True

        if not is_anchor:
            continue

        score = (
            source_priority[source],
            recon_date,
            admit_date
        )

        current_best = out_best.get(mrn)

        if current_best is None or score < current_best["score"]:
            payload = {
                "admit_date": admit_date.strftime("%Y-%m-%d"),
                "recon_date": recon_date.strftime("%Y-%m-%d"),
                "score": score,
                "source": source,
                "cpt_code": cpt_code,
                "procedure": clean_cell(row.get("PROCEDURE_STRUCT", "")),
                "reason_for_visit": clean_cell(row.get("REASON_FOR_VISIT_STRUCT", ""))
            }

            if require_age:
                day_diff = (recon_date - admit_date).days
                adjusted_age = float(age_base) + (float(day_diff) / 365.25)
                age_floor = int(math.floor(adjusted_age))
                age_round = int(math.floor(adjusted_age + 0.5))
                payload.update({
                    "age_at_encounter": age_raw,
                    "day_diff": day_diff,
                    "value_floor": age_floor,
                    "value_round": age_round
                })

            out_best[mrn] = payload

    return out_best


def choose_best_clinic_age_rows(struct_df):
    return _recon_anchor_helper(struct_df, require_age=True)


def choose_best_bmi_recon_rows(struct_df):
    return _recon_anchor_helper(struct_df, require_age=False)


def enrich_master_with_structured_demo(master, notes_df, evidence_rows):
    print("Loading structured encounters for Race / Ethnicity / Age...")
    struct_df = load_structured_encounters()
    print("Structured encounter rows: {0}".format(len(struct_df)))

    race_map = choose_race_us_categories(struct_df)
    eth_map = choose_best_ethnicity(struct_df)
    age_map = choose_best_clinic_age_rows(struct_df)

    print("Structured race values found for MRNs: {0}".format(len(race_map)))
    print("Structured ethnicity values found for MRNs: {0}".format(len(eth_map)))
    print("Structured encounter-based age rows found for MRNs: {0}".format(len(age_map)))

    for mrn in master[MERGE_KEY].astype(str).str.strip().tolist():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        race_val = race_map.get(mrn, "")
        if race_val:
            master.loc[mask, "Race"] = race_val
            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": "",
                "NOTE_DATE": "",
                "NOTE_TYPE": "STRUCTURED_MULTI_SOURCE",
                "FIELD": "Race",
                "VALUE": race_val,
                "STATUS": "structured_fill",
                "CONFIDENCE": "1.0",
                "SECTION": "STRUCTURED_ENCOUNTER",
                "EVIDENCE": "Race harmonized to US-style categories from structured encounter values"
            })

        eth_info = eth_map.get(mrn)
        if eth_info is not None and clean_cell(eth_info.get("value")):
            master.loc[mask, "Ethnicity"] = eth_info["value"]
            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": "",
                "NOTE_DATE": eth_info.get("date_raw", ""),
                "NOTE_TYPE": "STRUCTURED_{0}".format(eth_info.get("source", "")).upper(),
                "FIELD": "Ethnicity",
                "VALUE": eth_info.get("value", ""),
                "STATUS": "structured_fill",
                "CONFIDENCE": "1.0",
                "SECTION": "STRUCTURED_ENCOUNTER",
                "EVIDENCE": "Ethnicity from structured encounter"
            })

        age_info = age_map.get(mrn)
        if age_info is not None:
            age_floor = age_info.get("value_floor")
            age_round = age_info.get("value_round")

            final_age = age_round if age_round is not None else age_floor

            if final_age is not None:
                master.loc[mask, "Age"] = final_age

                ev = (
                    "Age from structured encounter row using AGE_AT_ENCOUNTER + ADMIT_DATE + RECONSTRUCTION_DATE with source priority and fallback CPT logic | "
                    "AGE_AT_ENCOUNTER={0} | ADMIT_DATE={1} | RECONSTRUCTION_DATE={2} | DAY_DIFF={3} | "
                    "AGE_FLOOR={4} | AGE_ROUND={5} | FINAL_USED={6} | SOURCE={7} | CPT_CODE={8} | PROCEDURE={9} | REASON_FOR_VISIT={10}"
                ).format(
                    age_info.get("age_at_encounter", ""),
                    age_info.get("admit_date", ""),
                    age_info.get("recon_date", ""),
                    age_info.get("day_diff", ""),
                    age_floor,
                    age_round,
                    final_age,
                    age_info.get("source", ""),
                    age_info.get("cpt_code", ""),
                    age_info.get("procedure", ""),
                    age_info.get("reason_for_visit", "")
                )

                evidence_rows.append({
                    MERGE_KEY: mrn,
                    "NOTE_ID": "",
                    "NOTE_DATE": age_info.get("admit_date", ""),
                    "NOTE_TYPE": "STRUCTURED_ENCOUNTER_RECON_ROW",
                    "FIELD": "Age",
                    "VALUE": final_age,
                    "STATUS": "structured_fill",
                    "CONFIDENCE": "1.0",
                    "SECTION": "STRUCTURED_ENCOUNTER",
                    "EVIDENCE": ev
                })

    return master, evidence_rows


def choose_bmi_candidate_note_ids_by_mrn(notes_df, bmi_anchor_map):
    """
    For each MRN with structured reconstruction date anchor, return ALL candidate
    note NOTE_IDs in the best available date window:
      same day > +/-1 day > +/-3 days

    Within that best window, keep notes with the best note-type priority.
    """
    out = {}

    if len(notes_df) == 0 or len(bmi_anchor_map) == 0:
        return out

    tmp = notes_df.copy()
    tmp["NOTE_DATE_PARSED"] = tmp["NOTE_DATE"].apply(parse_date_safe)
    tmp["BMI_NOTE_TYPE_PRIORITY"] = tmp["NOTE_TYPE"].apply(bmi_note_type_priority)

    # keep note types we are willing to consider for BMI
    tmp = tmp[tmp["BMI_NOTE_TYPE_PRIORITY"] < 9].copy()

    if len(tmp) == 0:
        return out

    for mrn, g in tmp.groupby(MERGE_KEY):
        mrn = str(mrn).strip()
        anchor = bmi_anchor_map.get(mrn)
        if anchor is None:
            continue

        recon_dt = parse_date_safe(anchor.get("recon_date", ""))
        if recon_dt is None:
            continue

        rows = []
        for _, row in g.iterrows():
            note_dt = row.get("NOTE_DATE_PARSED")
            dd = abs_day_diff(note_dt, recon_dt)
            if dd is None:
                continue
            if dd <= 3:
                rows.append({
                    "NOTE_ID": str(row["NOTE_ID"]).strip(),
                    "NOTE_DATE": row["NOTE_DATE"],
                    "NOTE_TYPE": row["NOTE_TYPE"],
                    "DAY_DIFF": dd,
                    "NOTE_TYPE_PRIORITY": row["BMI_NOTE_TYPE_PRIORITY"]
                })

        if not rows:
            continue

        same_day = [r for r in rows if r["DAY_DIFF"] == 0]
        within_1 = [r for r in rows if r["DAY_DIFF"] <= 1]
        within_3 = [r for r in rows if r["DAY_DIFF"] <= 3]

        chosen_rows = []
        match_tier = ""

        if same_day:
            chosen_rows = same_day
            match_tier = "same_day"
        elif within_1:
            chosen_rows = within_1
            match_tier = "within_1_day"
        elif within_3:
            chosen_rows = within_3
            match_tier = "within_3_days"

        if chosen_rows:
            best_pri = min([r["NOTE_TYPE_PRIORITY"] for r in chosen_rows])
            chosen_rows = [r for r in chosen_rows if r["NOTE_TYPE_PRIORITY"] == best_pri]

            out[mrn] = {
                "note_ids": set([r["NOTE_ID"] for r in chosen_rows]),
                "match_tier": match_tier,
                "rows": chosen_rows
            }

    return out


def main():
    print("Seeding clean master WITHOUT gold...")
    master = seed_master_from_structured()
    master = normalize_mrn(master)

    if "ENCRYPTED_PAT_ID" in master.columns:
        master["ENCRYPTED_PAT_ID"] = master["MRN"].astype(str).str.strip()

    print("Master seeded: {0} MRNs".format(len(master)))

    print("Loading & reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    evidence_rows = []

    master, evidence_rows = enrich_master_with_structured_demo(master, notes_df, evidence_rows)

    struct_df_for_bmi = load_structured_encounters()
    bmi_anchor_map = choose_best_bmi_recon_rows(struct_df_for_bmi)
    print("Structured reconstruction-date BMI anchor rows found for MRNs: {0}".format(len(bmi_anchor_map)))

    bmi_candidate_note_map = choose_bmi_candidate_note_ids_by_mrn(notes_df, bmi_anchor_map)
    print("BMI candidate peri-reconstruction notes found for MRNs: {0}".format(len(bmi_candidate_note_map)))

    print("Running rule-based extractors...")
    extractor_fns = [
        extract_age,
        extract_bmi,
        extract_smoking,
        extract_comorbidities,
        extract_pbs,
        extract_mastectomy,
        extract_cancer_treatment,
    ]

    best_by_mrn = {}

    for _, row in notes_df.iterrows():
        mrn = str(row[MERGE_KEY]).strip()

        snote = build_sectioned_note(
            note_text=row["NOTE_TEXT"],
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        all_cands = []

        for fn in extractor_fns:
            if fn == extract_bmi:
                target = bmi_candidate_note_map.get(mrn)
                if target is None:
                    continue
                if str(row["NOTE_ID"]).strip() not in target.get("note_ids", set()):
                    continue

            try:
                new_cands = fn(snote)
                all_cands.extend(new_cands)

                if fn == extract_bmi:
                    target = bmi_candidate_note_map.get(mrn)
                    if target is not None and str(row["NOTE_ID"]).strip() in target.get("note_ids", set()):
                        evidence_rows.append({
                            MERGE_KEY: mrn,
                            "NOTE_ID": row["NOTE_ID"],
                            "NOTE_DATE": row["NOTE_DATE"],
                            "NOTE_TYPE": row["NOTE_TYPE"],
                            "FIELD": "BMI_NOTE_SELECTION",
                            "VALUE": "",
                            "STATUS": "targeted_selection",
                            "CONFIDENCE": "",
                            "SECTION": "STRUCTURED_PLUS_NOTE_DATE_WINDOW",
                            "EVIDENCE": "BMI note eligible by reconstruction-date anchoring | RECON_DATE={0} | MATCH_TIER={1}".format(
                                bmi_anchor_map.get(mrn, {}).get("recon_date", ""),
                                target.get("match_tier", "")
                            )
                        })

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
                    "EVIDENCE": "{0} failed: {1}".format(fn.__name__, repr(e))
                })

        if not all_cands:
            continue

        if mrn not in best_by_mrn:
            best_by_mrn[mrn] = {}

        for c in all_cands:
            logical = FIELD_MAP.get(str(c.field))
            if not logical:
                continue

            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": getattr(c, "note_id", row["NOTE_ID"]),
                "NOTE_DATE": getattr(c, "note_date", row["NOTE_DATE"]),
                "NOTE_TYPE": getattr(c, "note_type", row["NOTE_TYPE"]),
                "FIELD": logical,
                "VALUE": getattr(c, "value", ""),
                "STATUS": getattr(c, "status", ""),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "EVIDENCE": getattr(c, "evidence", "")
            })

            existing = best_by_mrn[mrn].get(logical)
            if logical in BOOLEAN_FIELDS:
                best_by_mrn[mrn][logical] = merge_boolean(existing, c)
            else:
                best_by_mrn[mrn][logical] = choose_best(existing, c)

    print("Aggregated note-based predictions for {0} MRNs".format(len(best_by_mrn)))

    for mrn, fields in best_by_mrn.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        for logical, cand in fields.items():
            if logical in {"Race", "Ethnicity", "Age"}:
                continue

            val = getattr(cand, "value", pd.NA)

            if logical in BOOLEAN_FIELDS:
                try:
                    val = 1 if bool(val) else 0
                except Exception:
                    val = pd.NA

            if logical == "BMI" and not pd.isna(val):
                try:
                    bmi_val = round(float(val))
                    master.loc[mask, "BMI"] = bmi_val
                    master.loc[mask, "Obesity"] = 1 if bmi_val >= 30 else 0
                except Exception:
                    master.loc[mask, "BMI"] = pd.NA
                continue

            if logical in master.columns:
                master.loc[mask, logical] = val
            else:
                master[logical] = pd.NA
                master.loc[mask, logical] = val

    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    master.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\nDONE.")
    print("- Master (NO GOLD): {0}".format(OUTPUT_MASTER))
    print("- Evidence: {0}".format(OUTPUT_EVID))
    print("\nRun:")
    print(" python build_master_rule_FINAL_NO_GOLD.py")


if __name__ == "__main__":
    main()
