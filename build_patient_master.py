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

import os
import re
import math
from glob import glob
from datetime import datetime
from typing import Dict, List, Optional, Any

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
# Structured enrichment: Race / Ethnicity / Age
# -----------------------
def load_structured_encounters():
    rows = []
    for fp in sorted(set(sum([glob(g, recursive=True) for g in STRUCT_GLOBS], []))):
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
        date_col = pick_col(
            df,
            ["OPERATION_DATE", "ENCOUNTER_DATE", "ADMIT_DATE", "DISCHARGE_DATE_DT"],
            required=False
        )

        out = pd.DataFrame()
        out[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
        out["STRUCT_SOURCE"] = encounter_source
        out["STRUCT_PRIORITY"] = priority
        out["STRUCT_DATE_RAW"] = df[date_col].astype(str) if date_col else ""
        out["RACE_STRUCT"] = df[race_col].astype(str) if race_col else ""
        out["ETHNICITY_STRUCT"] = df[eth_col].astype(str) if eth_col else ""
        out["AGE_AT_ENCOUNTER_STRUCT"] = df[age_col].astype(str) if age_col else ""
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            MERGE_KEY, "STRUCT_SOURCE", "STRUCT_PRIORITY", "STRUCT_DATE_RAW",
            "RACE_STRUCT", "ETHNICITY_STRUCT", "AGE_AT_ENCOUNTER_STRUCT"
        ])

    struct_df = pd.concat(rows, ignore_index=True)
    return struct_df


def build_target_dates(notes_df, struct_df):
    target_dates = {}

    # 1) operation note date preferred
    if len(notes_df) > 0:
        for _, row in notes_df.iterrows():
            mrn = clean_cell(row.get(MERGE_KEY, ""))
            if not mrn:
                continue

            note_type = clean_cell(row.get("NOTE_TYPE", "")).lower()
            source_file = clean_cell(row.get("SOURCE_FILE", "")).lower()
            note_date = parse_date_safe(row.get("NOTE_DATE", ""))

            is_operation_note = (
                ("operation" in note_type) or
                ("operative" in note_type) or
                ("operation" in source_file) or
                ("operative" in source_file)
            )

            if is_operation_note and note_date is not None:
                prev = target_dates.get(mrn)
                if prev is None or note_date < prev:
                    target_dates[mrn] = note_date

    # 2) fallback to operation encounter date
    if len(struct_df) > 0:
        op_df = struct_df[struct_df["STRUCT_SOURCE"] == "operation"].copy()
        if len(op_df) > 0:
            for _, row in op_df.iterrows():
                mrn = clean_cell(row.get(MERGE_KEY, ""))
                if not mrn or mrn in target_dates:
                    continue
                dt = parse_date_safe(row.get("STRUCT_DATE_RAW", ""))
                if dt is not None:
                    target_dates[mrn] = dt

    # 3) fallback to any earliest note date
    if len(notes_df) > 0:
        for _, row in notes_df.iterrows():
            mrn = clean_cell(row.get(MERGE_KEY, ""))
            if not mrn or mrn in target_dates:
                continue
            note_date = parse_date_safe(row.get("NOTE_DATE", ""))
            if note_date is not None:
                prev = target_dates.get(mrn)
                if prev is None or note_date < prev:
                    target_dates[mrn] = note_date

    return target_dates


def normalize_race_value(x):
    s = clean_cell(x)
    if not s:
        return ""
    return s


def normalize_ethnicity_value(x):
    s = clean_cell(x)
    if not s:
        return ""
    return s


def round_half_up(x):
    try:
        return int(math.floor(float(x) + 0.5))
    except Exception:
        return None


def choose_structured_demo_values(struct_df, target_dates):
    best = {}

    if len(struct_df) == 0:
        return best

    for _, row in struct_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        race_val = normalize_race_value(row.get("RACE_STRUCT", ""))
        eth_val = normalize_ethnicity_value(row.get("ETHNICITY_STRUCT", ""))
        age_raw = clean_cell(row.get("AGE_AT_ENCOUNTER_STRUCT", ""))
        age_base = to_float_safe(age_raw)
        struct_date = parse_date_safe(row.get("STRUCT_DATE_RAW", ""))
        priority = to_int_safe(row.get("STRUCT_PRIORITY", 9))
        if priority is None:
            priority = 9

        if mrn not in best:
            best[mrn] = {
                "race": None,
                "ethnicity": None,
                "age": None
            }

        target_date = target_dates.get(mrn)

        # -------- Race --------
        if race_val:
            score = (priority, 999999999)
            if target_date is not None and struct_date is not None:
                score = (priority, abs((target_date - struct_date).days))

            cur = best[mrn]["race"]
            if cur is None or score < cur["score"]:
                best[mrn]["race"] = {
                    "value": race_val,
                    "score": score,
                    "source": row.get("STRUCT_SOURCE", ""),
                    "date_raw": row.get("STRUCT_DATE_RAW", "")
                }

        # -------- Ethnicity --------
        if eth_val:
            score = (priority, 999999999)
            if target_date is not None and struct_date is not None:
                score = (priority, abs((target_date - struct_date).days))

            cur = best[mrn]["ethnicity"]
            if cur is None or score < cur["score"]:
                best[mrn]["ethnicity"] = {
                    "value": eth_val,
                    "score": score,
                    "source": row.get("STRUCT_SOURCE", ""),
                    "date_raw": row.get("STRUCT_DATE_RAW", "")
                }

        # -------- Age --------
        if age_base is not None:
            age_floor = None
            age_round = None

            if target_date is not None and struct_date is not None:
                day_diff = (target_date - struct_date).days
                age_adjusted = float(age_base) + (float(day_diff) / 365.25)
                age_floor = int(math.floor(age_adjusted))
                age_round = round_half_up(age_adjusted)
            else:
                age_floor = int(math.floor(float(age_base)))
                age_round = round_half_up(float(age_base))

            score = (priority, 999999999)
            if target_date is not None and struct_date is not None:
                score = (priority, abs((target_date - struct_date).days))

            cur = best[mrn]["age"]
            if cur is None or score < cur["score"]:
                best[mrn]["age"] = {
                    "value_floor": age_floor,
                    "value_round": age_round,
                    "score": score,
                    "source": row.get("STRUCT_SOURCE", ""),
                    "date_raw": row.get("STRUCT_DATE_RAW", ""),
                    "target_date": target_date.strftime("%Y-%m-%d") if target_date else "",
                    "struct_date": struct_date.strftime("%Y-%m-%d") if struct_date else "",
                    "age_at_encounter": age_raw
                }

    return best


def enrich_master_with_structured_demo(master, notes_df, evidence_rows):
    print("Loading structured encounters for Race / Ethnicity / Age...")
    struct_df = load_structured_encounters()
    print("Structured encounter rows: {0}".format(len(struct_df)))

    target_dates = build_target_dates(notes_df, struct_df)
    best_struct = choose_structured_demo_values(struct_df, target_dates)

    print("Structured demo values found for MRNs: {0}".format(len(best_struct)))

    for mrn, info in best_struct.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        # Race
        race_info = info.get("race")
        if race_info is not None and clean_cell(race_info.get("value")):
            master.loc[mask, "Race"] = race_info["value"]
            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": "",
                "NOTE_DATE": race_info.get("date_raw", ""),
                "NOTE_TYPE": "STRUCTURED_{0}".format(race_info.get("source", "")).upper(),
                "FIELD": "Race",
                "VALUE": race_info.get("value", ""),
                "STATUS": "structured_fill",
                "CONFIDENCE": "1.0",
                "SECTION": "STRUCTURED_ENCOUNTER",
                "EVIDENCE": "Race from structured encounter"
            })

        # Ethnicity
        eth_info = info.get("ethnicity")
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

        # Age
        age_info = info.get("age")
        if age_info is not None:
            age_floor = age_info.get("value_floor")
            age_round = age_info.get("value_round")

            # keep both calculations internally; use round for final Age if available,
            # otherwise fall back to floor
            final_age = age_round if age_round is not None else age_floor

            if final_age is not None:
                master.loc[mask, "Age"] = final_age

                ev = (
                    "Age from structured encounter | "
                    "AGE_AT_ENCOUNTER={0} | STRUCT_DATE={1} | TARGET_DATE={2} | "
                    "AGE_FLOOR={3} | AGE_ROUND={4} | FINAL_USED={5}"
                ).format(
                    age_info.get("age_at_encounter", ""),
                    age_info.get("struct_date", ""),
                    age_info.get("target_date", ""),
                    age_floor,
                    age_round,
                    final_age
                )

                evidence_rows.append({
                    MERGE_KEY: mrn,
                    "NOTE_ID": "",
                    "NOTE_DATE": age_info.get("date_raw", ""),
                    "NOTE_TYPE": "STRUCTURED_{0}".format(age_info.get("source", "")).upper(),
                    "FIELD": "Age",
                    "VALUE": final_age,
                    "STATUS": "structured_fill",
                    "CONFIDENCE": "1.0",
                    "SECTION": "STRUCTURED_ENCOUNTER",
                    "EVIDENCE": ev
                })

    return master, evidence_rows


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

    # -----------------------
    # Structured fill first: Race / Ethnicity / Age
    # -----------------------
    master, evidence_rows = enrich_master_with_structured_demo(master, notes_df, evidence_rows)

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
            try:
                all_cands.extend(fn(snote))
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

    # Apply note-based predictions to master,
    # but DO NOT overwrite structured Race / Ethnicity / Age
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
                    master.loc[mask, "BMI"] = float(val)
                    master.loc[mask, "Obesity"] = 1 if float(val) >= 30.0 else 0
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
