#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_master_rule_COMPLICATIONS_PATCH.py

Purpose:
- Patch Stage1 / Stage2 complication outcome predictions into the existing master
- Uses STAGE2_DATE + PRED_HAS_STAGE2 already merged into master
- DOES NOT overwrite original master
- Writes a new patched master + evidence file

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

MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds.csv".format(BASE_DIR)
OUTPUT_MASTER = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds_complications.csv".format(BASE_DIR)
OUTPUT_EVID = "{0}/_outputs/complications_patch_evidence.csv".format(BASE_DIR)

NOTE_GLOBS = [
    "{0}/**/HPI11526*Clinic Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation notes.csv".format(BASE_DIR),
]

MERGE_KEY = "MRN"

from models import SectionedNote  # noqa: E402
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


def to_bool01(x):
    s = clean_cell(x).lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    return 0


def assign_stage(note_date, pred_has_stage2, stage2_date):
    ndt = parse_date_safe(note_date)
    s2dt = parse_date_safe(stage2_date)

    if ndt is None:
        return None

    if to_bool01(pred_has_stage2) != 1 or s2dt is None:
        return "STAGE1"

    if ndt.date() < s2dt.date():
        return "STAGE1"

    return "STAGE2"


def field_for_stage(stage_label, base_field):
    if stage_label == "STAGE1":
        mapping = {
            "MinorComp": "Stage1_MinorComp",
            "Reoperation": "Stage1_Reoperation",
            "Rehospitalization": "Stage1_Rehospitalization",
            "MajorComp": "Stage1_MajorComp",
            "Failure": "Stage1_Failure",
            "Revision": "Stage1_Revision",
        }
        return mapping.get(base_field)
    if stage_label == "STAGE2":
        mapping = {
            "MinorComp": "Stage2_MinorComp",
            "Reoperation": "Stage2_Reoperation",
            "Rehospitalization": "Stage2_Rehospitalization",
            "MajorComp": "Stage2_MajorComp",
            "Failure": "Stage2_Failure",
            "Revision": "Stage2_Revision",
        }
        return mapping.get(base_field)
    return None


def ensure_target_columns(master):
    for col in TARGET_FIELDS:
        if col not in master.columns:
            master[col] = 0
        master[col] = master[col].fillna(0)

        try:
            master[col] = (
                master[col]
                .astype(str)
                .str.strip()
                .replace({
                    "": "0",
                    "nan": "0",
                    "None": "0",
                    "none": "0",
                    "null": "0",
                    "NA": "0",
                    "na": "0"
                })
            )
        except Exception:
            pass

    return master


def main():
    if not os.path.exists(MASTER_FILE):
        raise FileNotFoundError("Master file not found: {0}".format(MASTER_FILE))

    print("Loading master:", MASTER_FILE)
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)

    master = ensure_target_columns(master)

    master_lookup = {}
    for _, row in master.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if mrn:
            master_lookup[mrn] = {
                "PRED_HAS_STAGE2": clean_cell(row.get("PRED_HAS_STAGE2", "")),
                "STAGE2_DATE": clean_cell(row.get("STAGE2_DATE", "")),
            }

    print("Loading & reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes:", len(notes_df))

    evidence_rows = []
    best_by_mrn = {}

    for _, row in notes_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        master_info = master_lookup.get(mrn, {})
        pred_has_stage2 = master_info.get("PRED_HAS_STAGE2", "")
        stage2_date = master_info.get("STAGE2_DATE", "")

        stage_label = assign_stage(
            row.get("NOTE_DATE", ""),
            pred_has_stage2,
            stage2_date
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
                MERGE_KEY: mrn,
                "NOTE_ID": row.get("NOTE_ID", ""),
                "NOTE_DATE": row.get("NOTE_DATE", ""),
                "NOTE_TYPE": row.get("NOTE_TYPE", ""),
                "STAGE_ASSIGNED": stage_label,
                "FIELD": "EXTRACTOR_ERROR",
                "VALUE": "",
                "STATUS": "",
                "CONFIDENCE": "",
                "SECTION": "",
                "EVIDENCE": repr(e)
            })
            continue

        if not cands:
            continue

        if mrn not in best_by_mrn:
            best_by_mrn[mrn] = {}

        for c in cands:
            logical_fields = []

            if str(c.field) == "ComplicationSignal":
                logical_fields.append(field_for_stage(stage_label, "MinorComp"))

            elif str(c.field) == "StageOutcome_Reoperation":
                logical_fields.append(field_for_stage(stage_label, "Reoperation"))
                logical_fields.append(field_for_stage(stage_label, "MajorComp"))

            elif str(c.field) == "StageOutcome_Rehospitalization":
                logical_fields.append(field_for_stage(stage_label, "Rehospitalization"))
                logical_fields.append(field_for_stage(stage_label, "MajorComp"))

            elif str(c.field) == "StageOutcome_Failure":
                logical_fields.append(field_for_stage(stage_label, "Failure"))

            elif str(c.field) == "StageOutcome_Revision":
                logical_fields.append(field_for_stage(stage_label, "Revision"))

            for logical in logical_fields:
                if not logical:
                    continue

                evidence_rows.append({
                    MERGE_KEY: mrn,
                    "NOTE_ID": getattr(c, "note_id", row.get("NOTE_ID", "")),
                    "NOTE_DATE": getattr(c, "note_date", row.get("NOTE_DATE", "")),
                    "NOTE_TYPE": getattr(c, "note_type", row.get("NOTE_TYPE", "")),
                    "STAGE_ASSIGNED": stage_label,
                    "FIELD": logical,
                    "VALUE": getattr(c, "value", ""),
                    "STATUS": getattr(c, "status", ""),
                    "CONFIDENCE": getattr(c, "confidence", ""),
                    "SECTION": getattr(c, "section", ""),
                    "EVIDENCE": getattr(c, "evidence", "")
                })

                existing = best_by_mrn[mrn].get(logical)
                best_by_mrn[mrn][logical] = merge_boolean(existing, c)

    print("Aggregated complication predictions for MRNs:", len(best_by_mrn))

    for mrn, fields in best_by_mrn.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        for logical, cand in fields.items():
            val = getattr(cand, "value", "")
            try:
                val = 1 if bool(val) else 0
            except Exception:
                val = 0
            master.loc[mask, logical] = val

    # final cleanup so blanks never remain in outcome columns
    for col in TARGET_FIELDS:
        master[col] = (
            master[col]
            .fillna(0)
            .astype(str)
            .str.strip()
            .replace({
                "": "0",
                "nan": "0",
                "None": "0",
                "none": "0",
                "null": "0",
                "NA": "0",
                "na": "0"
            })
        )

    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    master.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\nDONE.")
    print("Patched master:", OUTPUT_MASTER)
    print("Evidence file:", OUTPUT_EVID)


if __name__ == "__main__":
    main()
