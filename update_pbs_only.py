#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_pbs_only.py

PBS-only updater for:
- PastBreastSurgery
- PBS_Lumpectomy
- PBS_Breast Reduction
- PBS_Mastopexy
- PBS_Augmentation
- PBS_Other

Design:
- Uses reconstruction date as anchor
- Uses Recon_Laterality from master when available
- Falls back to anchor-row procedure text when possible
- Prioritizes operation notes first, then clinic-like notes
- Uses notes BEFORE recon first
- If no qualifying pre-recon match, then checks post-recon notes ONLY if the mention is clearly historical
- For unilateral recon:
    * ipsilateral = accept
    * contralateral = reject
    * unknown laterality = do not auto-count
- For bilateral recon:
    * unknown laterality can count

Outputs:
1) /home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD_PBS_ONLY.csv
2) /home/apokol/Breast_Restore/_outputs/pbs_only_evidence.csv

Python 3.6.8 compatible.
"""

import os
import re
from glob import glob
from datetime import datetime

import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_MASTER = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD_PBS_ONLY.csv".format(BASE_DIR)
OUTPUT_EVID = "{0}/_outputs/pbs_only_evidence.csv".format(BASE_DIR)

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
from extractors.pbs import extract_pbs  # noqa: E402


PBS_FIELDS = [
    "PastBreastSurgery",
    "PBS_Lumpectomy",
    "PBS_Breast Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "PBS_Other",
]


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
# Structured encounters and recon anchors
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
                "reason_for_visit": clean_cell(row.get("REASON_FOR_VISIT_STRUCT", "")),
            }

    return best


# -----------------------
# Note / laterality helpers
# -----------------------
LEFT_RX = re.compile(r"\b(left|lt|l)\b|\(left\)|\(lt\)|\(l\)|\bleft\s+breast\b", re.I)
RIGHT_RX = re.compile(r"\b(right|rt|r)\b|\(right\)|\(rt\)|\((?:r)\)|\bright\s+breast\b", re.I)
BILAT_RX = re.compile(r"\b(bilateral|bilat|both\s+breasts?)\b", re.I)
CONTRALAT_RX = re.compile(r"\bcontralateral\b", re.I)

HISTORY_CUE_RX = re.compile(
    r"\b(s/p|status\s+post|history\s+of|with\s+a\s+history\s+of|prior|previous|remote)\b",
    re.I
)

NEGATIVE_HISTORY_RX = re.compile(
    r"\b(no\s+prior\s+breast\s+surgery|no\s+history\s+of\s+breast\s+surgery|denies\s+prior\s+breast\s+surgery|never\s+had\s+breast\s+surgery)\b",
    re.I
)


def normalize_recon_laterality(x):
    s = clean_cell(x).lower()
    if not s:
        return ""
    if "bilat" in s or "bilateral" in s or "both" in s:
        return "bilateral"
    if s in {"left", "l"} or "left" in s:
        return "left"
    if s in {"right", "r"} or "right" in s:
        return "right"
    return ""


def extract_laterality_from_text(text):
    t = clean_cell(text)
    if not t:
        return ""

    has_b = BILAT_RX.search(t) is not None
    has_l = LEFT_RX.search(t) is not None
    has_r = RIGHT_RX.search(t) is not None

    if has_b:
        return "bilateral"
    if has_l and has_r:
        return "bilateral"
    if has_l:
        return "left"
    if has_r:
        return "right"
    return ""


def laterality_relation(recon_lat, proc_lat, context_text):
    recon_lat = normalize_recon_laterality(recon_lat)
    proc_lat = normalize_recon_laterality(proc_lat)

    if recon_lat == "bilateral":
        if proc_lat:
            return "accept"
        return "accept"

    if recon_lat in {"left", "right"}:
        if proc_lat == recon_lat:
            return "accept"
        if proc_lat == "bilateral":
            return "accept"
        if proc_lat in {"left", "right"} and proc_lat != recon_lat:
            return "reject_contralateral"

        ctx = clean_cell(context_text).lower()
        if "contralateral" in ctx:
            return "reject_contralateral"

        return "unknown_unilateral"

    return "unknown_recon"


def is_operation_note_type(note_type, source_file):
    s = "{0} {1}".format(clean_cell(note_type).lower(), clean_cell(source_file).lower())
    if "brief op" in s:
        return True
    if "op note" in s:
        return True
    if "operative" in s:
        return True
    if "operation" in s:
        return True
    if "oper report" in s:
        return True
    return False


def is_clinic_like_note(note_type, source_file):
    s = "{0} {1}".format(clean_cell(note_type).lower(), clean_cell(source_file).lower())
    patterns = [
        "progress",
        "clinic",
        "office",
        "follow up",
        "follow-up",
        "pre-op",
        "preop",
        "consult",
        "h&p",
        "history and physical"
    ]
    for p in patterns:
        if p in s:
            return True
    return False


def is_historical_context(text):
    t = clean_cell(text)
    if not t:
        return False
    if HISTORY_CUE_RX.search(t):
        return True
    return False


def has_negative_history(text):
    t = clean_cell(text)
    if not t:
        return False
    return NEGATIVE_HISTORY_RX.search(t) is not None


def stage_and_rank(note_type, source_file, note_dt, recon_dt, accepted_post_hist):
    dd = days_between(note_dt, recon_dt)

    is_op = is_operation_note_type(note_type, source_file)
    is_clinic = is_clinic_like_note(note_type, source_file)

    if dd is None:
        return (9, 9999, 9)

    if dd < 0:
        if is_op:
            return (0, abs(dd), 0)
        if is_clinic:
            return (1, abs(dd), 1)
        return (2, abs(dd), 2)

    if dd >= 0 and accepted_post_hist:
        if is_op:
            return (3, abs(dd), 0)
        if is_clinic:
            return (4, abs(dd), 1)
        return (5, abs(dd), 2)

    return (9, abs(dd), 9)


def candidate_score(c):
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    nt = str(getattr(c, "note_type", "") or "").lower()
    op_bonus = 0.05 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    date_bonus = 0.01 if clean_cell(getattr(c, "note_date", "")) else 0.0
    return conf + op_bonus + date_bonus


def choose_better_pbs(existing, new, recon_dt):
    if existing is None:
        return new

    ex_rank = stage_and_rank(
        getattr(existing, "note_type", ""),
        getattr(existing, "_source_file", ""),
        parse_date_safe(getattr(existing, "note_date", "")),
        recon_dt,
        getattr(existing, "_accepted_post_hist", False)
    )
    nw_rank = stage_and_rank(
        getattr(new, "note_type", ""),
        getattr(new, "_source_file", ""),
        parse_date_safe(getattr(new, "note_date", "")),
        recon_dt,
        getattr(new, "_accepted_post_hist", False)
    )

    if nw_rank < ex_rank:
        return new
    if ex_rank < nw_rank:
        return existing

    return new if candidate_score(new) > candidate_score(existing) else existing


# -----------------------
# Main PBS updater
# -----------------------
def main():
    print("Loading master...")
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)

    for c in PBS_FIELDS:
        if c not in master.columns:
            master[c] = pd.NA

    print("Master rows: {0}".format(len(master)))

    print("Loading notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    print("Loading structured encounters...")
    struct_df = load_structured_encounters()
    anchor_map = choose_best_anchor_rows(struct_df)
    print("Recon anchors found: {0}".format(len(anchor_map)))

    evidence_rows = []

    # initialize PBS fields to 0
    for c in PBS_FIELDS:
        master[c] = 0

    best_by_mrn = {}

    for _, row in notes_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        anchor = anchor_map.get(mrn)
        if anchor is None:
            continue

        recon_dt = parse_date_safe(anchor.get("recon_date", ""))
        note_dt = parse_date_safe(row.get("NOTE_DATE", ""))
        if recon_dt is None or note_dt is None:
            continue

        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        recon_lat = ""
        if "Recon_Laterality" in master.columns:
            recon_lat = normalize_recon_laterality(master.loc[mask, "Recon_Laterality"].astype(str).iloc[0])

        if not recon_lat:
            recon_lat = extract_laterality_from_text(
                "{0} {1}".format(anchor.get("procedure", ""), anchor.get("reason_for_visit", ""))
            )

        snote = build_sectioned_note(
            note_text=row["NOTE_TEXT"],
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        try:
            cands = extract_pbs(snote)
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
                "RECON_DATE": anchor.get("recon_date", ""),
                "RECON_LATERALITY": recon_lat,
                "PROC_LATERALITY": "",
                "RULE_DECISION": "extractor_failed",
                "EVIDENCE": repr(e)
            })
            continue

        if not cands:
            continue

        if mrn not in best_by_mrn:
            best_by_mrn[mrn] = {}

        for c in cands:
            logical = str(getattr(c, "field", ""))

            if logical not in {"PBS_Lumpectomy", "PBS_Breast Reduction", "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other"}:
                continue

            evid = clean_cell(getattr(c, "evidence", ""))
            if not evid:
                continue

            proc_lat = extract_laterality_from_text(evid)
            lat_decision = laterality_relation(recon_lat, proc_lat, evid)

            hist_context = is_historical_context(evid)
            neg_context = has_negative_history(evid)

            day_diff = days_between(note_dt, recon_dt)

            accept = False
            reason = ""

            if neg_context:
                accept = False
                reason = "reject_negative_history"
            elif day_diff is None:
                accept = False
                reason = "reject_missing_date_diff"
            elif day_diff < 0:
                if lat_decision == "accept":
                    accept = True
                    reason = "accept_pre_recon"
                elif lat_decision == "reject_contralateral":
                    accept = False
                    reason = "reject_contralateral"
                elif lat_decision == "unknown_unilateral":
                    accept = False
                    reason = "reject_unknown_laterality_unilateral"
                else:
                    accept = False
                    reason = "reject_unknown_recon_laterality"
            else:
                # post-recon fallback only if clearly historical
                if not hist_context:
                    accept = False
                    reason = "reject_post_recon_not_historical"
                else:
                    if lat_decision == "accept":
                        accept = True
                        reason = "accept_post_recon_historical"
                    elif lat_decision == "reject_contralateral":
                        accept = False
                        reason = "reject_contralateral"
                    elif lat_decision == "unknown_unilateral":
                        accept = False
                        reason = "reject_unknown_laterality_unilateral"
                    else:
                        accept = False
                        reason = "reject_unknown_recon_laterality"

            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": getattr(c, "note_id", row["NOTE_ID"]),
                "NOTE_DATE": getattr(c, "note_date", row["NOTE_DATE"]),
                "NOTE_TYPE": getattr(c, "note_type", row["NOTE_TYPE"]),
                "FIELD": logical,
                "VALUE": getattr(c, "value", True),
                "STATUS": getattr(c, "status", ""),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "RECON_DATE": anchor.get("recon_date", ""),
                "RECON_LATERALITY": recon_lat,
                "PROC_LATERALITY": proc_lat,
                "RULE_DECISION": reason,
                "EVIDENCE": evid
            })

            if not accept:
                continue

            # attach helper attrs for ranking
            setattr(c, "_source_file", row.get("SOURCE_FILE", ""))
            setattr(c, "_accepted_post_hist", bool(day_diff >= 0 and hist_context))

            existing = best_by_mrn[mrn].get(logical)
            best_by_mrn[mrn][logical] = choose_better_pbs(existing, c, recon_dt)

    print("Accepted PBS note-based predictions for MRNs: {0}".format(len(best_by_mrn)))

    # write final PBS fields
    for mrn, fields in best_by_mrn.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        any_positive = False

        for logical in ["PBS_Lumpectomy", "PBS_Breast Reduction", "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other"]:
            cand = fields.get(logical)
            if cand is None:
                continue
            master.loc[mask, logical] = 1
            any_positive = True

        master.loc[mask, "PastBreastSurgery"] = 1 if any_positive else 0

    # any MRN without subtype stays 0 for PastBreastSurgery
    subtype_cols = ["PBS_Lumpectomy", "PBS_Breast Reduction", "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other"]
    for idx in master.index:
        any_positive = False
        for c in subtype_cols:
            try:
                if int(float(str(master.at[idx, c]).strip())) == 1:
                    any_positive = True
                    break
            except Exception:
                pass
        master.at[idx, "PastBreastSurgery"] = 1 if any_positive else 0

    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    master.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\nDONE.")
    print("- Updated PBS-only master: {0}".format(OUTPUT_MASTER))
    print("- PBS evidence: {0}".format(OUTPUT_EVID))
    print("\nRun:")
    print(" python update_pbs_only.py")


if __name__ == "__main__":
    main()
