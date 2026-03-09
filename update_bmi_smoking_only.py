#!/usr/bin/env python3
# update_bmi_only.py
#
# BMI + Smoking updater for the existing master file.
#
# BMI logic:
#   Stage 1: anchor day only
#   Stage 2: +/- 7 days
#   Stage 3: +/- 14 days
#
# Smoking uses the SAME staged window and ranking logic.

import os
import re
from glob import glob
from datetime import datetime
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"
MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_EVID = "{0}/_outputs/bmi_only_evidence.csv".format(BASE_DIR)

MERGE_KEY = "MRN"

from models import SectionedNote
from extractors.bmi import extract_bmi
from extractors.smoking import extract_smoking


# -----------------------
# Utilities
# -----------------------
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
# Candidate ranking
# -----------------------
def note_type_bucket(note_type):

    s = clean_cell(note_type).lower()

    if "brief op" in s:
        return "brief_op"

    if "operative" in s or "operation" in s or "op note" in s:
        return "operation"

    if "pre-op" in s or "preop" in s:
        return "preop"

    if "clinic" in s:
        return "clinic"

    if "progress" in s:
        return "progress"

    return "other"


def candidate_stage_rank(cand, recon_dt):

    note_dt = parse_date_safe(getattr(cand, "note_date", ""))

    if note_dt is None or recon_dt is None:
        return None

    dd = days_between(note_dt, recon_dt)

    bucket = note_type_bucket(getattr(cand, "note_type", ""))

    if dd == 0 and bucket in ("brief_op", "operation"):
        return (1, 0)

    if bucket in ("brief_op", "operation", "preop", "clinic", "progress"):
        return (2, abs(dd))

    return (9, abs(dd))


def choose_best(existing, new, recon_dt):

    if existing is None:
        return new

    ex_rank = candidate_stage_rank(existing, recon_dt)
    nw_rank = candidate_stage_rank(new, recon_dt)

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


# -----------------------
# Candidate collection
# -----------------------
def collect_candidates(notes_df, anchor_map, extractor_fn, stage_name, before_days, after_days):

    best_by_mrn = {}

    for _, row in notes_df.iterrows():

        mrn = clean_cell(row.get("MRN", ""))

        if not mrn:
            continue

        anchor = anchor_map.get(mrn)

        if anchor is None:
            continue

        recon_dt = parse_date_safe(anchor.get("recon_date", ""))

        note_dt = parse_date_safe(row.get("NOTE_DATE", ""))

        if recon_dt is None or note_dt is None:
            continue

        dd = days_between(note_dt, recon_dt)

        if dd < -before_days or dd > after_days:
            continue

        snote = build_sectioned_note(
            note_text=row["NOTE_TEXT"],
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        try:
            candidates = extractor_fn(snote)
        except Exception:
            continue

        for c in candidates:

            existing = best_by_mrn.get(mrn)

            best_by_mrn[mrn] = choose_best(existing, c, recon_dt)

    return best_by_mrn


# -----------------------
# Main
# -----------------------
def main():

    print("Loading master...")
    master = pd.read_csv(MASTER_FILE, dtype=str)

    if "BMI" not in master.columns:
        master["BMI"] = pd.NA

    if "Obesity" not in master.columns:
        master["Obesity"] = pd.NA

    if "SmokingStatus" not in master.columns:
        master["SmokingStatus"] = pd.NA

    print("Loading notes...")
    notes_df = pd.read_csv("{0}/notes_reconstructed.csv".format(BASE_DIR), dtype=str)

    anchor_map = pd.read_csv("{0}/reconstruction_anchors.csv".format(BASE_DIR), dtype=str)
    anchor_map = anchor_map.set_index("MRN").to_dict("index")

    final_bmi = {}
    final_smoking = {}

    for stage, before, after in [
        ("day0", 0, 0),
        ("pm7", 7, 7),
        ("pm14", 14, 14),
    ]:

        print("Running stage:", stage)

        bmi_candidates = collect_candidates(notes_df, anchor_map, extract_bmi, stage, before, after)

        smoke_candidates = collect_candidates(notes_df, anchor_map, extract_smoking, stage, before, after)

        for mrn, cand in bmi_candidates.items():
            if mrn not in final_bmi:
                final_bmi[mrn] = cand

        for mrn, cand in smoke_candidates.items():
            if mrn not in final_smoking:
                final_smoking[mrn] = cand

    for mrn, cand in final_bmi.items():

        val = getattr(cand, "value", None)

        try:
            bmi = round(float(val), 1)
            mask = master["MRN"] == mrn
            master.loc[mask, "BMI"] = bmi
            master.loc[mask, "Obesity"] = 1 if bmi >= 30 else 0
        except Exception:
            pass

    for mrn, cand in final_smoking.items():

        val = getattr(cand, "value", None)

        mask = master["MRN"] == mrn

        master.loc[mask, "SmokingStatus"] = val

    master.to_csv(MASTER_FILE, index=False)

    print("DONE.")
    print("Updated master:", MASTER_FILE)


if __name__ == "__main__":
    main()
