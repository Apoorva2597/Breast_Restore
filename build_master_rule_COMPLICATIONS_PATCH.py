#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_master_rule_COMPLICATIONS_PATCH.py

Purpose:
- Patch Stage1 / Stage2 complication outcome predictions into the existing master
- Uses STAGE2_DATE + PRED_HAS_STAGE2 already merged into master
- DOES NOT overwrite original master
- Writes a new patched master + evidence file

FIX (this version):
- assign_stage() now falls back to WINDOW_START when PRED_HAS_STAGE2=1
  but STAGE2_DATE is blank. This unlocks Stage2 complication assignment
  for confirmed Stage2 patients whose exact exchange date was not captured.

MinorComp derivation rule (unchanged):
    MinorComp = 1 only if:
        any complication signal for that stage
        AND Reoperation == 0
        AND Rehospitalization == 0
        AND Failure == 0

Input master:
    /home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds.csv

Outputs:
    /home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds_complications.csv
    /home/apokol/Breast_Restore/_outputs/complications_patch_evidence.csv

Python 3.6.8 compatible
"""

import os
import re
from glob import glob
from datetime import datetime

import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE   = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds.csv".format(BASE_DIR)
OUTPUT_MASTER = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds_complications.csv".format(BASE_DIR)
OUTPUT_EVID   = "{0}/_outputs/complications_patch_evidence.csv".format(BASE_DIR)

NOTE_GLOBS = [
    "{0}/**/HPI11526*Clinic Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation notes.csv".format(BASE_DIR),
]

MERGE_KEY = "MRN"

from models import SectionedNote                              # noqa: E402
from extractors.complications import extract_complication_outcomes  # noqa: E402

TARGET_FIELDS = [
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
]

BOOLEAN_FIELDS = set(TARGET_FIELDS)

# ============================================================
# Utilities
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
        raise RuntimeError("MRN column not found. Columns: {0}".format(list(df.columns)[:50]))
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


def to_bool01(x):
    s = clean_cell(x).lower()
    return 1 if s in {"1", "true", "t", "yes", "y"} else 0


def merge_boolean(existing, new):
    if existing is None:
        return new
    try:
        exv = bool(existing.value)
        nwv = bool(new.value)
    except Exception:
        return new
    if nwv and not exv:
        return new
    if exv and not nwv:
        return existing
    ex_conf = float(getattr(existing, "confidence", 0.0) or 0.0)
    nw_conf = float(getattr(new,      "confidence", 0.0) or 0.0)
    return new if nw_conf > ex_conf else existing


def _cand_to01(cand):
    if cand is None:
        return 0
    try:
        return 1 if bool(getattr(cand, "value", "")) else 0
    except Exception:
        return 0

# ============================================================
# Sectionizer
# ============================================================

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
# Note loading
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
                                  "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"], required=False)

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
        if line_col and line_col != "LINE":      tmp = tmp.rename(columns={line_col: "LINE"})
        if type_col and type_col != "NOTE_TYPE": tmp = tmp.rename(columns={type_col: "NOTE_TYPE"})
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
        mrn = str(mrn).strip(); nid = str(nid).strip()
        if not nid: continue
        full_text = join_note(g)
        if not full_text: continue
        note_type = g["NOTE_TYPE"].astype(str).iloc[0] if g["NOTE_TYPE"].astype(str).str.strip().any() else g["_SOURCE_FILE_"].astype(str).iloc[0]
        note_date = g["NOTE_DATE_OF_SERVICE"].astype(str).iloc[0] if g["NOTE_DATE_OF_SERVICE"].astype(str).str.strip().any() else ""
        reconstructed.append({
            MERGE_KEY: mrn, "NOTE_ID": nid, "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date, "SOURCE_FILE": g["_SOURCE_FILE_"].astype(str).iloc[0],
            "NOTE_TEXT": full_text
        })
    return pd.DataFrame(reconstructed)

# ============================================================
# Stage assignment
# ============================================================

def assign_stage(note_date, pred_has_stage2, stage2_date, window_start=None):
    """
    Assign a note to STAGE1 or STAGE2.

    FIX: If PRED_HAS_STAGE2=1 but STAGE2_DATE is blank,
    fall back to WINDOW_START before defaulting to STAGE1.
    This prevents Stage2 complications being silently dropped
    for confirmed Stage2 patients with missing exchange dates.
    """
    ndt  = parse_date_safe(note_date)
    s2dt = parse_date_safe(stage2_date)

    if ndt is None:
        return None

    if to_bool01(pred_has_stage2) != 1:
        return "STAGE1"

    # Fall back to WINDOW_START if STAGE2_DATE is missing
    if s2dt is None:
        s2dt = parse_date_safe(window_start)

    if s2dt is None:
        return "STAGE1"

    if ndt.date() < s2dt.date():
        return "STAGE1"

    return "STAGE2"


def field_for_stage(stage_label, base_field):
    if stage_label == "STAGE1":
        mapping = {
            "MinorComp":         "Stage1_MinorComp",
            "Reoperation":       "Stage1_Reoperation",
            "Rehospitalization": "Stage1_Rehospitalization",
            "MajorComp":         "Stage1_MajorComp",
            "Failure":           "Stage1_Failure",
            "Revision":          "Stage1_Revision",
        }
        return mapping.get(base_field)

    if stage_label == "STAGE2":
        mapping = {
            "MinorComp":         "Stage2_MinorComp",
            "Reoperation":       "Stage2_Reoperation",
            "Rehospitalization": "Stage2_Rehospitalization",
            "MajorComp":         "Stage2_MajorComp",
            "Failure":           "Stage2_Failure",
            "Revision":          "Stage2_Revision",
        }
        return mapping.get(base_field)

    return None

# ============================================================
# Master column setup
# ============================================================

def ensure_target_columns(master):
    for col in TARGET_FIELDS:
        if col not in master.columns:
            master[col] = 0
        master[col] = master[col].fillna(0)
        try:
            master[col] = (
                master[col].astype(str).str.strip()
                .replace({"": "0", "nan": "0", "None": "0",
                           "none": "0", "null": "0", "NA": "0", "na": "0"})
            )
        except Exception:
            pass
    return master

# ============================================================
# Main
# ============================================================

def main():
    if not os.path.exists(MASTER_FILE):
        raise FileNotFoundError("Master file not found: {0}".format(MASTER_FILE))

    print("Loading master:", MASTER_FILE)
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)
    master = ensure_target_columns(master)

    # Build per-MRN lookup including WINDOW_START
    master_lookup = {}
    for _, row in master.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if mrn:
            master_lookup[mrn] = {
                "PRED_HAS_STAGE2": clean_cell(row.get("PRED_HAS_STAGE2", "")),
                "STAGE2_DATE":     clean_cell(row.get("STAGE2_DATE", "")),
                "WINDOW_START":    clean_cell(row.get("WINDOW_START", "")),
            }

    print("Loading & reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes:", len(notes_df))

    evidence_rows      = []
    best_by_mrn        = {}
    comp_signal_by_mrn = {}   # tracks raw ComplicationSignal per stage

    for _, row in notes_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        master_info     = master_lookup.get(mrn, {})
        pred_has_stage2 = master_info.get("PRED_HAS_STAGE2", "")
        stage2_date     = master_info.get("STAGE2_DATE", "")
        window_start    = master_info.get("WINDOW_START", "")

        # FIX: pass window_start as fallback date
        stage_label = assign_stage(
            row.get("NOTE_DATE", ""),
            pred_has_stage2,
            stage2_date,
            window_start=window_start
        )

        if stage_label is None:
            continue

        snote = build_sectioned_note(
            note_text=row.get("NOTE_TEXT", ""),
            note_type=row.get("NOTE_TYPE", ""),
            note_id=row.get("NOTE_ID", ""),
            note_date=row.get("NOTE_DATE", "")
        )

        try:
            cands = extract_complication_outcomes(snote)
        except Exception as e:
            evidence_rows.append({
                MERGE_KEY:        mrn,
                "NOTE_ID":        row.get("NOTE_ID", ""),
                "NOTE_DATE":      row.get("NOTE_DATE", ""),
                "NOTE_TYPE":      row.get("NOTE_TYPE", ""),
                "STAGE_ASSIGNED": stage_label,
                "FIELD":          "EXTRACTOR_ERROR",
                "VALUE": "", "STATUS": "", "CONFIDENCE": "", "SECTION": "",
                "EVIDENCE":       repr(e)
            })
            continue

        if not cands:
            continue

        if mrn not in best_by_mrn:
            best_by_mrn[mrn] = {}
        if mrn not in comp_signal_by_mrn:
            comp_signal_by_mrn[mrn] = {"STAGE1": None, "STAGE2": None}

        for c in cands:
            raw_field = str(getattr(c, "field", ""))

            # ComplicationSignal tracked separately — feeds MinorComp derivation
            if raw_field == "ComplicationSignal":
                existing_sig = comp_signal_by_mrn[mrn].get(stage_label)
                comp_signal_by_mrn[mrn][stage_label] = merge_boolean(existing_sig, c)

                evidence_rows.append({
                    MERGE_KEY:        mrn,
                    "NOTE_ID":        getattr(c, "note_id",    row.get("NOTE_ID", "")),
                    "NOTE_DATE":      getattr(c, "note_date",  row.get("NOTE_DATE", "")),
                    "NOTE_TYPE":      getattr(c, "note_type",  row.get("NOTE_TYPE", "")),
                    "STAGE_ASSIGNED": stage_label,
                    "FIELD":          "RAW_{0}_ComplicationSignal".format(stage_label),
                    "VALUE":          getattr(c, "value",      ""),
                    "STATUS":         getattr(c, "status",     ""),
                    "CONFIDENCE":     getattr(c, "confidence", ""),
                    "SECTION":        getattr(c, "section",    ""),
                    "EVIDENCE":       getattr(c, "evidence",   "")
                })
                continue

            # Map raw extractor field to logical outcome column
            logical_fields = []

            if raw_field == "StageOutcome_Reoperation":
                logical_fields.append(field_for_stage(stage_label, "Reoperation"))
                logical_fields.append(field_for_stage(stage_label, "MajorComp"))

            elif raw_field == "StageOutcome_Rehospitalization":
                logical_fields.append(field_for_stage(stage_label, "Rehospitalization"))
                logical_fields.append(field_for_stage(stage_label, "MajorComp"))

            elif raw_field == "StageOutcome_Failure":
                logical_fields.append(field_for_stage(stage_label, "Failure"))

            elif raw_field == "StageOutcome_Revision":
                logical_fields.append(field_for_stage(stage_label, "Revision"))

            for logical in logical_fields:
                if not logical:
                    continue

                evidence_rows.append({
                    MERGE_KEY:        mrn,
                    "NOTE_ID":        getattr(c, "note_id",    row.get("NOTE_ID", "")),
                    "NOTE_DATE":      getattr(c, "note_date",  row.get("NOTE_DATE", "")),
                    "NOTE_TYPE":      getattr(c, "note_type",  row.get("NOTE_TYPE", "")),
                    "STAGE_ASSIGNED": stage_label,
                    "FIELD":          logical,
                    "VALUE":          getattr(c, "value",      ""),
                    "STATUS":         getattr(c, "status",     ""),
                    "CONFIDENCE":     getattr(c, "confidence", ""),
                    "SECTION":        getattr(c, "section",    ""),
                    "EVIDENCE":       getattr(c, "evidence",   "")
                })

                existing = best_by_mrn[mrn].get(logical)
                best_by_mrn[mrn][logical] = merge_boolean(existing, c)

    print("Aggregated complication predictions for MRNs:", len(best_by_mrn))

    # Write direct stage outcomes first
    for mrn, fields in best_by_mrn.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue
        for logical, cand in fields.items():
            master.loc[mask, logical] = _cand_to01(cand)

    # Derive MinorComp AFTER all other stage outcomes are finalized
    for mrn, stage_signals in comp_signal_by_mrn.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        # Stage 1
        s1_signal  = _cand_to01(stage_signals.get("STAGE1"))
        s1_reop    = to_bool01(master.loc[mask, "Stage1_Reoperation"].iloc[0])
        s1_rehosp  = to_bool01(master.loc[mask, "Stage1_Rehospitalization"].iloc[0])
        s1_failure = to_bool01(master.loc[mask, "Stage1_Failure"].iloc[0])
        master.loc[mask, "Stage1_MinorComp"] = (
            1 if (s1_signal == 1 and s1_reop == 0 and s1_rehosp == 0 and s1_failure == 0) else 0
        )

        # Stage 2
        s2_signal  = _cand_to01(stage_signals.get("STAGE2"))
        s2_reop    = to_bool01(master.loc[mask, "Stage2_Reoperation"].iloc[0])
        s2_rehosp  = to_bool01(master.loc[mask, "Stage2_Rehospitalization"].iloc[0])
        s2_failure = to_bool01(master.loc[mask, "Stage2_Failure"].iloc[0])
        master.loc[mask, "Stage2_MinorComp"] = (
            1 if (s2_signal == 1 and s2_reop == 0 and s2_rehosp == 0 and s2_failure == 0) else 0
        )

    # Zero MinorComp for patients with no raw signal at all
    known_signal_mrns = set(comp_signal_by_mrn.keys())
    for idx in master.index:
        mrn = clean_cell(master.at[idx, MERGE_KEY])
        if mrn not in known_signal_mrns:
            master.at[idx, "Stage1_MinorComp"] = 0
            master.at[idx, "Stage2_MinorComp"] = 0

    # Final cleanup — normalise all target fields to "0" / "1" strings
    for col in TARGET_FIELDS:
        master[col] = (
            master[col].fillna(0).astype(str).str.strip()
            .replace({"": "0", "nan": "0", "None": "0",
                       "none": "0", "null": "0", "NA": "0", "na": "0"})
        )

    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    master.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\nDONE.")
    print("Patched master:", OUTPUT_MASTER)
    print("Evidence file: ", OUTPUT_EVID)
    print("MinorComp derived after aggregation.")


if __name__ == "__main__":
    main()
