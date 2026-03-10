#!/usr/bin/env python3
# update_bmi_smoking_only.py
#
# BMI + Smoking updater for the existing master file.
#
# IMPORTANT:
# - BMI logic is intentionally left unchanged.
# - Only Smoking selection / resolution logic has been updated.
#
# BMI logic:
#   Stage 1: anchor day only
#   Stage 2: +/- 7 days (only if nothing found in Stage 1)
#   Stage 3: +/- 14 days (only if nothing found in Stage 2)
#
# Smoking logic:
#   Current:
#       Stage 1: anchor day only
#       Stage 2: +/- 7 days
#       Stage 3: +/- 14 days
#   Historical fallback:
#       allow any note ON OR BEFORE reconstruction date
#       and now keep Current / Former / Never if extractor resolves it
#       from quit timing or explicit evidence
#
# Anchor logic:
#   - Primary anchor from structured RECONSTRUCTION_DATE + ADMIT_DATE
#   - Backup anchor from CPT/procedure/date logic if primary anchor missing
#
# Updates only:
#   - BMI
#   - Obesity
#   - SmokingStatus
#
# Outputs:
#   /home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv
#   /home/apokol/Breast_Restore/_outputs/bmi_smoking_only_evidence.csv
#
# Python 3.6.8 compatible

import os
import re
from glob import glob
from datetime import datetime
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"
MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_EVID = "{0}/_outputs/bmi_smoking_only_evidence.csv".format(BASE_DIR)

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

from models import SectionedNote  # noqa: E402
from extractors.bmi import extract_bmi  # noqa: E402
from extractors.smoking import extract_smoking  # noqa: E402


# -----------------------
# Utilities
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

def days_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    return (dt1.date() - dt2.date()).days


# -----------------------
# Sectionizer
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
# Structured anchor logic
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
        out["ADMIT_DATE_STRUCT"] = df[admit_col].astype(str) if admit_col else ""
        out["RECONSTRUCTION_DATE_STRUCT"] = df[recon_col].astype(str) if recon_col else ""
        out["CPT_CODE_STRUCT"] = df[cpt_col].astype(str) if cpt_col else ""
        out["PROCEDURE_STRUCT"] = df[proc_col].astype(str) if proc_col else ""
        out["REASON_FOR_VISIT_STRUCT"] = df[reason_col].astype(str) if reason_col else ""
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            MERGE_KEY, "STRUCT_SOURCE", "STRUCT_PRIORITY", "STRUCT_DATE_RAW",
            "ADMIT_DATE_STRUCT", "RECONSTRUCTION_DATE_STRUCT",
            "CPT_CODE_STRUCT", "PROCEDURE_STRUCT", "REASON_FOR_VISIT_STRUCT"
        ])

    return pd.concat(rows, ignore_index=True)

def _is_recon_like_row(row, has_preferred_cpt):
    preferred_cpts = set([
        "19357", "19340", "19342", "19361", "19364", "19367", "S2068"
    ])
    primary_exclude_cpts = set(["19325", "19330"])
    fallback_allowed_cpts = set(["19350", "19380"])

    cpt_code = clean_cell(row.get("CPT_CODE_STRUCT", "")).upper()
    procedure = clean_cell(row.get("PROCEDURE_STRUCT", "")).lower()
    reason_for_visit = clean_cell(row.get("REASON_FOR_VISIT_STRUCT", "")).lower()

    if cpt_code in primary_exclude_cpts:
        return False

    if cpt_code in preferred_cpts:
        return True

    if (not has_preferred_cpt) and (cpt_code in fallback_allowed_cpts):
        return True

    text = "{0} {1}".format(procedure, reason_for_visit)

    keywords = [
        "tissue expander",
        "breast recon",
        "implant on same day of mastectomy",
        "insert or replcmnt breast implnt on sep day from mastectomy",
        "latissimus",
        "diep",
        "tram",
        "flap",
        "free flap",
        "expander placmnt",
        "reconstruct",
        "reconstruction",
    ]

    for kw in keywords:
        if kw in text:
            return True

    return False

def choose_best_anchor_rows(struct_df):
    best = {}
    if len(struct_df) == 0:
        return best

    source_priority = {
        "clinic": 1,
        "operation": 2,
        "inpatient": 3
    }

    eligible_sources = struct_df[struct_df["STRUCT_SOURCE"].isin(["clinic", "operation", "inpatient"])].copy()
    if len(eligible_sources) == 0:
        return best

    has_preferred_cpt = {}
    preferred_cpts = set(["19357", "19340", "19342", "19361", "19364", "19367", "S2068"])
    for mrn, g in eligible_sources.groupby(MERGE_KEY):
        found = False
        for val in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist():
            if clean_cell(val).upper() in preferred_cpts:
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

        admit_date = parse_date_safe(row.get("ADMIT_DATE_STRUCT", ""))
        recon_date = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))

        if admit_date is None or recon_date is None:
            continue

        if not _is_recon_like_row(row, has_preferred_cpt.get(mrn, False)):
            continue

        score = (
            source_priority[source],
            recon_date,
            admit_date
        )

        current_best = best.get(mrn)
        if current_best is None or score < current_best["score"]:
            best[mrn] = {
                "anchor_type": "primary_recon_date",
                "anchor_date": recon_date.strftime("%Y-%m-%d"),
                "admit_date": admit_date.strftime("%Y-%m-%d"),
                "recon_date": recon_date.strftime("%Y-%m-%d"),
                "score": score,
                "source": source,
                "cpt_code": clean_cell(row.get("CPT_CODE_STRUCT", "")),
                "procedure": clean_cell(row.get("PROCEDURE_STRUCT", "")),
                "reason_for_visit": clean_cell(row.get("REASON_FOR_VISIT_STRUCT", ""))
            }

    return best

def choose_backup_anchor_rows(struct_df, existing_anchor_map):
    backup = {}
    if len(struct_df) == 0:
        return backup

    source_priority = {
        "operation": 1,
        "clinic": 2,
        "inpatient": 3,
        "other": 9
    }

    eligible = struct_df[struct_df["STRUCT_SOURCE"].isin(["clinic", "operation", "inpatient"])].copy()
    if len(eligible) == 0:
        return backup

    has_preferred_cpt = {}
    preferred_cpts = set(["19357", "19340", "19342", "19361", "19364", "19367", "S2068"])
    for mrn, g in eligible.groupby(MERGE_KEY):
        found = False
        for val in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist():
            if clean_cell(val).upper() in preferred_cpts:
                found = True
                break
        has_preferred_cpt[mrn] = found

    for _, row in eligible.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if (not mrn) or (mrn in existing_anchor_map):
            continue

        if not _is_recon_like_row(row, has_preferred_cpt.get(mrn, False)):
            continue

        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        pr = source_priority.get(source, 9)

        dt = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))
        anchor_type = "backup_recon_date_struct"

        if dt is None:
            dt = parse_date_safe(row.get("ADMIT_DATE_STRUCT", ""))
            anchor_type = "backup_admit_date_struct"

        if dt is None:
            dt = parse_date_safe(row.get("STRUCT_DATE_RAW", ""))
            anchor_type = "backup_struct_date_raw"

        if dt is None:
            continue

        score = (pr, dt)

        cur = backup.get(mrn)
        if cur is None or score < cur["score"]:
            backup[mrn] = {
                "anchor_type": anchor_type,
                "anchor_date": dt.strftime("%Y-%m-%d"),
                "admit_date": clean_cell(row.get("ADMIT_DATE_STRUCT", "")),
                "recon_date": dt.strftime("%Y-%m-%d"),
                "score": score,
                "source": source,
                "cpt_code": clean_cell(row.get("CPT_CODE_STRUCT", "")),
                "procedure": clean_cell(row.get("PROCEDURE_STRUCT", "")),
                "reason_for_visit": clean_cell(row.get("REASON_FOR_VISIT_STRUCT", ""))
            }

    return backup


# -----------------------
# Notes
# -----------------------
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


# -----------------------
# Selection logic
# -----------------------
def note_type_bucket(note_type, source_file):
    s = "{0} {1}".format(clean_cell(note_type).lower(), clean_cell(source_file).lower())

    if "brief op" in s:
        return "brief_op"
    if "operative" in s or "operation" in s or "op note" in s or "oper report" in s:
        return "operation"
    if "anesthesia" in s:
        return "anesthesia"
    if "pre-op" in s or "preop" in s:
        return "preop"
    if "h&p" in s or "history and physical" in s:
        return "hp"
    if "progress" in s:
        return "progress"
    if "clinic" in s or "office" in s:
        return "clinic"
    if "consult" in s:
        return "consult"
    return "other"

def candidate_stage_rank(cand, recon_dt, source_file):
    note_dt = parse_date_safe(getattr(cand, "note_date", ""))
    if note_dt is None or recon_dt is None:
        return None

    dd = days_between(note_dt, recon_dt)
    if dd is None:
        return None

    bucket = note_type_bucket(getattr(cand, "note_type", ""), source_file)
    status = clean_cell(getattr(cand, "status", "")).lower()
    explicit = (status != "computed")

    if explicit and dd == 0 and bucket in ("brief_op", "operation", "preop", "anesthesia", "hp"):
        bucket_ord = 0 if bucket == "brief_op" else 1 if bucket == "operation" else 2
        return (1, bucket_ord, 0, 0, 0)

    if explicit and bucket in ("brief_op", "operation", "preop", "anesthesia", "hp", "clinic", "progress", "consult"):
        bucket_ord = 0 if bucket == "brief_op" else 1 if bucket == "operation" else 2 if bucket in ("preop", "anesthesia", "hp") else 3
        post_penalty = 1 if dd > 0 else 0
        return (2, abs(dd), post_penalty, bucket_ord, 0)

    if (not explicit) and bucket in ("brief_op", "operation", "preop", "anesthesia", "hp", "clinic", "progress", "consult"):
        bucket_ord = 0 if bucket == "brief_op" else 1 if bucket == "operation" else 2 if bucket in ("preop", "anesthesia", "hp") else 3
        post_penalty = 1 if dd > 0 else 0
        return (3, abs(dd), post_penalty, bucket_ord, 0)

    return (9, abs(dd), 1 if dd > 0 else 0, 9, 0)

def smoking_value_priority(val):
    v = clean_cell(val)
    if v == "Current":
        return 0
    if v == "Former":
        return 1
    if v == "Never":
        return 2
    return 9

def choose_best_candidate(existing, new, recon_dt, source_file):
    # Generic selection used by BMI and any non-smoking fields.
    if existing is None:
        return new

    ex_rank = candidate_stage_rank(existing, recon_dt, source_file)
    nw_rank = candidate_stage_rank(new, recon_dt, source_file)

    if ex_rank is None:
        return new
    if nw_rank is None:
        return existing

    if nw_rank < ex_rank:
        return new

    if nw_rank == ex_rank:
        ex_conf = float(getattr(existing, "confidence", 0.0) or 0.0)
        nw_conf = float(getattr(new, "confidence", 0.0) or 0.0)
        if nw_conf > ex_conf:
            return new

    return existing

def choose_best_smoking_candidate(existing, new, recon_dt, source_file):
    # Smoking-only comparator.
    # BMI is not affected because BMI still uses choose_best_candidate().
    if existing is None:
        return new

    ex_rank = candidate_stage_rank(existing, recon_dt, source_file)
    nw_rank = candidate_stage_rank(new, recon_dt, source_file)

    if ex_rank is None:
        return new
    if nw_rank is None:
        return existing

    ex_val = clean_cell(getattr(existing, "value", ""))
    nw_val = clean_cell(getattr(new, "value", ""))

    ex_pri = smoking_value_priority(ex_val)
    nw_pri = smoking_value_priority(nw_val)

    # Prefer better temporal/note rank first.
    if nw_rank < ex_rank:
        return new
    if ex_rank < nw_rank:
        return existing

    # If same note/window rank, prefer Current > Former > Never.
    if nw_pri < ex_pri:
        return new
    if ex_pri < nw_pri:
        return existing

    # Then confidence.
    ex_conf = float(getattr(existing, "confidence", 0.0) or 0.0)
    nw_conf = float(getattr(new, "confidence", 0.0) or 0.0)
    if nw_conf > ex_conf:
        return new
    if ex_conf > nw_conf:
        return existing

    # Then prefer note closest to recon, pre-op favored over post-op.
    ex_note_dt = parse_date_safe(getattr(existing, "note_date", ""))
    nw_note_dt = parse_date_safe(getattr(new, "note_date", ""))

    ex_dd = days_between(ex_note_dt, recon_dt) if ex_note_dt is not None and recon_dt is not None else None
    nw_dd = days_between(nw_note_dt, recon_dt) if nw_note_dt is not None and recon_dt is not None else None

    if ex_dd is not None and nw_dd is not None:
        ex_post = 1 if ex_dd > 0 else 0
        nw_post = 1 if nw_dd > 0 else 0
        if nw_post < ex_post:
            return new
        if ex_post < nw_post:
            return existing
        if abs(nw_dd) < abs(ex_dd):
            return new
        if abs(ex_dd) < abs(nw_dd):
            return existing

    return existing

def note_in_window(note_dt, recon_dt, before_days, after_days):
    dd = days_between(note_dt, recon_dt)
    if dd is None:
        return False
    return (dd >= (-1 * before_days) and dd <= after_days)

def note_on_or_before_recon(note_dt, recon_dt):
    dd = days_between(note_dt, recon_dt)
    if dd is None:
        return False
    return dd <= 0

def collect_bmi_candidates_for_window(notes_df, anchor_map, stage_name, before_days, after_days, eligible_mrns, evidence_rows):
    best_by_mrn = {}
    notes_with_any_candidate = set()

    for _, row in notes_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue
        if mrn not in eligible_mrns:
            continue

        anchor = anchor_map.get(mrn)
        if anchor is None:
            continue

        recon_dt = parse_date_safe(anchor.get("recon_date", ""))
        note_dt = parse_date_safe(row.get("NOTE_DATE", ""))

        if recon_dt is None or note_dt is None:
            continue

        if not note_in_window(note_dt, recon_dt, before_days, after_days):
            continue

        snote = build_sectioned_note(
            note_text=row["NOTE_TEXT"],
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        try:
            candidates = extract_bmi(snote)
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
                "STAGE_USED": stage_name,
                "WINDOW_USED": "{0}_{1}".format(before_days, after_days),
                "ANCHOR_TYPE": anchor.get("anchor_type", ""),
                "ANCHOR_DATE": anchor.get("anchor_date", ""),
                "EVIDENCE": "extract_bmi failed: {0}".format(repr(e))
            })
            continue

        if not candidates:
            continue

        notes_with_any_candidate.add(mrn)

        for c in candidates:
            note_day_diff = days_between(parse_date_safe(getattr(c, "note_date", "")), recon_dt)
            evid = (
                "{0} | BMI_RECON_DATE={1} | BMI_NOTE_DAY_DIFF={2} | "
                "ANCHOR_TYPE={3} | ANCHOR_SOURCE={4} | ANCHOR_CPT={5} | ANCHOR_PROCEDURE={6}"
            ).format(
                getattr(c, "evidence", ""),
                anchor.get("recon_date", ""),
                note_day_diff,
                anchor.get("anchor_type", ""),
                anchor.get("source", ""),
                anchor.get("cpt_code", ""),
                anchor.get("procedure", "")
            )

            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": getattr(c, "note_id", row["NOTE_ID"]),
                "NOTE_DATE": getattr(c, "note_date", row["NOTE_DATE"]),
                "NOTE_TYPE": getattr(c, "note_type", row["NOTE_TYPE"]),
                "FIELD": "BMI",
                "VALUE": getattr(c, "value", ""),
                "STATUS": getattr(c, "status", ""),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "STAGE_USED": stage_name,
                "WINDOW_USED": "{0}_{1}".format(before_days, after_days),
                "ANCHOR_TYPE": anchor.get("anchor_type", ""),
                "ANCHOR_DATE": anchor.get("anchor_date", ""),
                "EVIDENCE": evid
            })

            existing = best_by_mrn.get(mrn)
            best_by_mrn[mrn] = choose_best_candidate(existing, c, recon_dt, row.get("SOURCE_FILE", ""))

    return best_by_mrn, notes_with_any_candidate, evidence_rows

def collect_smoking_current_candidates_for_window(notes_df, anchor_map, stage_name, before_days, after_days, eligible_mrns, evidence_rows):
    best_by_mrn = {}
    notes_with_any_candidate = set()

    for _, row in notes_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue
        if mrn not in eligible_mrns:
            continue

        anchor = anchor_map.get(mrn)
        if anchor is None:
            continue

        recon_dt = parse_date_safe(anchor.get("recon_date", ""))
        note_dt = parse_date_safe(row.get("NOTE_DATE", ""))

        if recon_dt is None or note_dt is None:
            continue

        if not note_in_window(note_dt, recon_dt, before_days, after_days):
            continue

        snote = build_sectioned_note(
            note_text=row["NOTE_TEXT"],
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        try:
            candidates = extract_smoking(snote)
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
                "STAGE_USED": stage_name,
                "WINDOW_USED": "{0}_{1}".format(before_days, after_days),
                "ANCHOR_TYPE": anchor.get("anchor_type", ""),
                "ANCHOR_DATE": anchor.get("anchor_date", ""),
                "EVIDENCE": "extract_smoking failed: {0}".format(repr(e))
            })
            continue

        if not candidates:
            continue

        current_candidates = [c for c in candidates if clean_cell(getattr(c, "value", "")) == "Current"]
        if not current_candidates:
            continue

        notes_with_any_candidate.add(mrn)

        for c in current_candidates:
            note_day_diff = days_between(parse_date_safe(getattr(c, "note_date", "")), recon_dt)
            evid = (
                "{0} | SMOKING_RECON_DATE={1} | SMOKING_NOTE_DAY_DIFF={2} | "
                "ANCHOR_TYPE={3} | ANCHOR_SOURCE={4} | ANCHOR_CPT={5} | ANCHOR_PROCEDURE={6}"
            ).format(
                getattr(c, "evidence", ""),
                anchor.get("recon_date", ""),
                note_day_diff,
                anchor.get("anchor_type", ""),
                anchor.get("source", ""),
                anchor.get("cpt_code", ""),
                anchor.get("procedure", "")
            )

            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": getattr(c, "note_id", row["NOTE_ID"]),
                "NOTE_DATE": getattr(c, "note_date", row["NOTE_DATE"]),
                "NOTE_TYPE": getattr(c, "note_type", row["NOTE_TYPE"]),
                "FIELD": "SmokingStatus",
                "VALUE": getattr(c, "value", ""),
                "STATUS": getattr(c, "status", ""),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "STAGE_USED": stage_name,
                "WINDOW_USED": "{0}_{1}".format(before_days, after_days),
                "ANCHOR_TYPE": anchor.get("anchor_type", ""),
                "ANCHOR_DATE": anchor.get("anchor_date", ""),
                "EVIDENCE": evid
            })

            existing = best_by_mrn.get(mrn)
            best_by_mrn[mrn] = choose_best_smoking_candidate(existing, c, recon_dt, row.get("SOURCE_FILE", ""))

    return best_by_mrn, notes_with_any_candidate, evidence_rows

def collect_smoking_historical_candidates(notes_df, anchor_map, eligible_mrns, evidence_rows):
    best_by_mrn = {}
    notes_with_any_candidate = set()

    for _, row in notes_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue
        if mrn not in eligible_mrns:
            continue

        anchor = anchor_map.get(mrn)
        if anchor is None:
            continue

        recon_dt = parse_date_safe(anchor.get("recon_date", ""))
        note_dt = parse_date_safe(row.get("NOTE_DATE", ""))

        if recon_dt is None or note_dt is None:
            continue

        if not note_on_or_before_recon(note_dt, recon_dt):
            continue

        snote = build_sectioned_note(
            note_text=row["NOTE_TEXT"],
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        try:
            candidates = extract_smoking(snote)
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
                "STAGE_USED": "historical_preop",
                "WINDOW_USED": "preop_any",
                "ANCHOR_TYPE": anchor.get("anchor_type", ""),
                "ANCHOR_DATE": anchor.get("anchor_date", ""),
                "EVIDENCE": "extract_smoking failed: {0}".format(repr(e))
            })
            continue

        if not candidates:
            continue

        # Updated: keep Current too, not just Former/Never.
        hist_candidates = []
        for c in candidates:
            val = clean_cell(getattr(c, "value", ""))
            if val in {"Current", "Former", "Never"}:
                hist_candidates.append(c)

        if not hist_candidates:
            continue

        notes_with_any_candidate.add(mrn)

        for c in hist_candidates:
            note_day_diff = days_between(parse_date_safe(getattr(c, "note_date", "")), recon_dt)
            evid = (
                "{0} | SMOKING_RECON_DATE={1} | SMOKING_NOTE_DAY_DIFF={2} | "
                "ANCHOR_TYPE={3} | ANCHOR_SOURCE={4} | ANCHOR_CPT={5} | ANCHOR_PROCEDURE={6}"
            ).format(
                getattr(c, "evidence", ""),
                anchor.get("recon_date", ""),
                note_day_diff,
                anchor.get("anchor_type", ""),
                anchor.get("source", ""),
                anchor.get("cpt_code", ""),
                anchor.get("procedure", "")
            )

            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": getattr(c, "note_id", row["NOTE_ID"]),
                "NOTE_DATE": getattr(c, "note_date", row["NOTE_DATE"]),
                "NOTE_TYPE": getattr(c, "note_type", row["NOTE_TYPE"]),
                "FIELD": "SmokingStatus",
                "VALUE": getattr(c, "value", ""),
                "STATUS": getattr(c, "status", ""),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "STAGE_USED": "historical_preop",
                "WINDOW_USED": "preop_any",
                "ANCHOR_TYPE": anchor.get("anchor_type", ""),
                "ANCHOR_DATE": anchor.get("anchor_date", ""),
                "EVIDENCE": evid
            })

            existing = best_by_mrn.get(mrn)
            best_by_mrn[mrn] = choose_best_smoking_candidate(existing, c, recon_dt, row.get("SOURCE_FILE", ""))

    return best_by_mrn, notes_with_any_candidate, evidence_rows


# -----------------------
# Main
# -----------------------
def main():
    print("Loading master...")
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)
    print("Master rows: {0}".format(len(master)))

    print("Loading structured encounters...")
    struct_df = load_structured_encounters()

    primary_anchor_map = choose_best_anchor_rows(struct_df)
    backup_anchor_map = choose_backup_anchor_rows(struct_df, primary_anchor_map)

    anchor_map = {}
    for mrn, info in primary_anchor_map.items():
        anchor_map[mrn] = info
    for mrn, info in backup_anchor_map.items():
        if mrn not in anchor_map:
            anchor_map[mrn] = info

    print("Primary structured anchors found: {0}".format(len(primary_anchor_map)))
    print("Backup anchors found from CPT/procedure/date logic: {0}".format(len(backup_anchor_map)))
    print("Total anchors available: {0}".format(len(anchor_map)))

    print("Loading and reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    evidence_rows = []
    final_best_bmi = {}
    final_best_smoking = {}

    # -----------------------
    # BMI staged extraction
    # -----------------------
    stage1_mrns = set(anchor_map.keys())
    print("Stage 1: searching BMI on anchor day only...")
    best_stage1, found_stage1, evidence_rows = collect_bmi_candidates_for_window(
        notes_df=notes_df,
        anchor_map=anchor_map,
        stage_name="day0",
        before_days=0,
        after_days=0,
        eligible_mrns=stage1_mrns,
        evidence_rows=evidence_rows
    )
    print("MRNs with any BMI candidate on anchor day: {0}".format(len(found_stage1)))
    for mrn, cand in best_stage1.items():
        final_best_bmi[mrn] = cand

    stage2_mrns = set([m for m in anchor_map.keys() if m not in final_best_bmi])
    print("Stage 2: searching BMI in +/-7 days ONLY for MRNs with no day-0 candidate...")
    print("Eligible MRNs: {0}".format(len(stage2_mrns)))
    best_stage2, found_stage2, evidence_rows = collect_bmi_candidates_for_window(
        notes_df=notes_df,
        anchor_map=anchor_map,
        stage_name="pm7",
        before_days=7,
        after_days=7,
        eligible_mrns=stage2_mrns,
        evidence_rows=evidence_rows
    )
    print("MRNs with any BMI candidate in +/-7 days: {0}".format(len(found_stage2)))
    for mrn, cand in best_stage2.items():
        if mrn not in final_best_bmi:
            final_best_bmi[mrn] = cand

    stage3_mrns = set([m for m in anchor_map.keys() if m not in final_best_bmi])
    print("Stage 3: searching BMI in +/-14 days ONLY for MRNs with no candidate in earlier stages...")
    print("Eligible MRNs: {0}".format(len(stage3_mrns)))
    best_stage3, found_stage3, evidence_rows = collect_bmi_candidates_for_window(
        notes_df=notes_df,
        anchor_map=anchor_map,
        stage_name="pm14",
        before_days=14,
        after_days=14,
        eligible_mrns=stage3_mrns,
        evidence_rows=evidence_rows
    )
    print("MRNs with any BMI candidate in +/-14 days: {0}".format(len(found_stage3)))
    for mrn, cand in best_stage3.items():
        if mrn not in final_best_bmi:
            final_best_bmi[mrn] = cand

    print("Final BMI predictions aggregated for MRNs: {0}".format(len(final_best_bmi)))

    # -----------------------
    # Smoking staged extraction
    # -----------------------
    smoke_stage1_mrns = set(anchor_map.keys())
    print("Stage 1: searching Current SmokingStatus on anchor day only...")
    smoke_best_stage1, smoke_found_stage1, evidence_rows = collect_smoking_current_candidates_for_window(
        notes_df=notes_df,
        anchor_map=anchor_map,
        stage_name="day0",
        before_days=0,
        after_days=0,
        eligible_mrns=smoke_stage1_mrns,
        evidence_rows=evidence_rows
    )
    print("MRNs with any Current SmokingStatus candidate on anchor day: {0}".format(len(smoke_found_stage1)))
    for mrn, cand in smoke_best_stage1.items():
        final_best_smoking[mrn] = cand

    smoke_stage2_mrns = set([m for m in anchor_map.keys() if m not in final_best_smoking])
    print("Stage 2: searching Current SmokingStatus in +/-7 days ONLY for MRNs with no day-0 current candidate...")
    print("Eligible MRNs: {0}".format(len(smoke_stage2_mrns)))
    smoke_best_stage2, smoke_found_stage2, evidence_rows = collect_smoking_current_candidates_for_window(
        notes_df=notes_df,
        anchor_map=anchor_map,
        stage_name="pm7",
        before_days=7,
        after_days=7,
        eligible_mrns=smoke_stage2_mrns,
        evidence_rows=evidence_rows
    )
    print("MRNs with any Current SmokingStatus candidate in +/-7 days: {0}".format(len(smoke_found_stage2)))
    for mrn, cand in smoke_best_stage2.items():
        if mrn not in final_best_smoking:
            final_best_smoking[mrn] = cand

    smoke_stage3_mrns = set([m for m in anchor_map.keys() if m not in final_best_smoking])
    print("Stage 3: searching Current SmokingStatus in +/-14 days ONLY for MRNs with no earlier current candidate...")
    print("Eligible MRNs: {0}".format(len(smoke_stage3_mrns)))
    smoke_best_stage3, smoke_found_stage3, evidence_rows = collect_smoking_current_candidates_for_window(
        notes_df=notes_df,
        anchor_map=anchor_map,
        stage_name="pm14",
        before_days=14,
        after_days=14,
        eligible_mrns=smoke_stage3_mrns,
        evidence_rows=evidence_rows
    )
    print("MRNs with any Current SmokingStatus candidate in +/-14 days: {0}".format(len(smoke_found_stage3)))
    for mrn, cand in smoke_best_stage3.items():
        if mrn not in final_best_smoking:
            final_best_smoking[mrn] = cand

    smoke_hist_mrns = set([m for m in anchor_map.keys() if m not in final_best_smoking])
    print("Stage 4: searching SmokingStatus in any note on or before reconstruction date...")
    print("Eligible MRNs: {0}".format(len(smoke_hist_mrns)))
    smoke_best_hist, smoke_found_hist, evidence_rows = collect_smoking_historical_candidates(
        notes_df=notes_df,
        anchor_map=anchor_map,
        eligible_mrns=smoke_hist_mrns,
        evidence_rows=evidence_rows
    )
    print("MRNs with any SmokingStatus candidate in historical preop notes: {0}".format(len(smoke_found_hist)))
    for mrn, cand in smoke_best_hist.items():
        if mrn not in final_best_smoking:
            final_best_smoking[mrn] = cand

    print("Final SmokingStatus predictions aggregated for MRNs: {0}".format(len(final_best_smoking)))

    if "BMI" not in master.columns:
        master["BMI"] = pd.NA
    if "Obesity" not in master.columns:
        master["Obesity"] = pd.NA
    if "SmokingStatus" not in master.columns:
        master["SmokingStatus"] = pd.NA

    for mrn, cand in final_best_bmi.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        val = getattr(cand, "value", pd.NA)
        if pd.isna(val):
            continue

        try:
            bmi_float = round(float(val), 1)
            master.loc[mask, "BMI"] = bmi_float
            master.loc[mask, "Obesity"] = 1 if bmi_float >= 30.0 else 0
        except Exception:
            pass

    for mrn, cand in final_best_smoking.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        val = getattr(cand, "value", pd.NA)
        if pd.isna(val):
            continue

        master.loc[mask, "SmokingStatus"] = val

    os.makedirs(os.path.dirname(MASTER_FILE), exist_ok=True)
    master.to_csv(MASTER_FILE, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\nDONE.")
    print("Updated master: {0}".format(MASTER_FILE))
    print("BMI + Smoking evidence: {0}".format(OUTPUT_EVID))
    print("\nRun:")
    print(" python update_bmi_smoking_only.py")

if __name__ == "__main__":
    main()
