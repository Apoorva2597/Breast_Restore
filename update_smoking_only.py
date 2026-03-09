#!/usr/bin/env python3
# update_smoking_only.py
#
# SmokingStatus updater using reconstruction-date anchoring
#
# Stages
#   Stage1: anchor day
#   Stage2: +/-7 days
#   Stage3: +/-14 days
#
# Output:
#   SmokingStatus
#
# Python 3.6.8 compatible

import pandas as pd
import os
from datetime import datetime
from extractors.smoking import extract_smoking
from models import SectionedNote

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = BASE_DIR + "/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
OUTPUT_EVID = BASE_DIR + "/_outputs/smoking_only_evidence.csv"

MERGE_KEY = "MRN"

STAGE_WINDOWS = [
    ("day0",0,0),
    ("pm7",7,7),
    ("pm14",14,14)
]

PREFERRED_NOTE_BUCKETS = [
    "brief_op",
    "operation",
    "preop",
    "anesthesia",
    "hp",
    "clinic",
    "progress",
]

# --------------------------
# Utilities
# --------------------------

def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"","nan","none","null","na"}:
        return ""
    return s


def parse_date_safe(x):
    try:
        return pd.to_datetime(x,errors="coerce")
    except:
        return None


def days_between(dt1,dt2):
    if pd.isna(dt1) or pd.isna(dt2):
        return None
    return (dt1.date()-dt2.date()).days


# --------------------------
# Candidate ranking
# --------------------------

def smoking_rank(cand,recon_dt):

    note_dt = parse_date_safe(cand.note_date)
    if note_dt is None:
        return None

    dd = days_between(note_dt,recon_dt)

    if dd is None:
        return None

    section = clean_cell(cand.section).lower()

    # prioritize social history
    if "social" in section:
        sec_rank = 0
    else:
        sec_rank = 1

    return (
        abs(dd),
        sec_rank,
        -float(cand.confidence or 0)
    )


def choose_best(existing,new,recon_dt):

    if existing is None:
        return new

    r1 = smoking_rank(existing,recon_dt)
    r2 = smoking_rank(new,recon_dt)

    if r1 is None:
        return new

    if r2 is None:
        return existing

    if r2 < r1:
        return new

    return existing


# --------------------------
# Window search
# --------------------------

def collect_smoking_candidates(notes_df,anchor_map,stage_name,before_days,after_days,eligible,evidence):

    best_by_mrn = {}

    for _,row in notes_df.iterrows():

        mrn = clean_cell(row.get(MERGE_KEY))

        if mrn not in eligible:
            continue

        anchor = anchor_map.get(mrn)

        if anchor is None:
            continue

        recon_dt = parse_date_safe(anchor["recon_date"])
        note_dt = parse_date_safe(row["NOTE_DATE"])

        if recon_dt is None or note_dt is None:
            continue

        dd = days_between(note_dt,recon_dt)

        if dd < -before_days or dd > after_days:
            continue

        snote = SectionedNote(
            sections=row["SECTIONS"],
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        try:
            candidates = extract_smoking(snote)
        except:
            continue

        if not candidates:
            continue

        for c in candidates:

            evidence.append({
                MERGE_KEY:mrn,
                "NOTE_ID":c.note_id,
                "NOTE_DATE":c.note_date,
                "NOTE_TYPE":c.note_type,
                "VALUE":c.value,
                "SECTION":c.section,
                "STAGE":stage_name,
                "ANCHOR_DATE":anchor["recon_date"],
                "EVIDENCE":c.evidence
            })

            best = best_by_mrn.get(mrn)

            best_by_mrn[mrn] = choose_best(best,c,recon_dt)

    return best_by_mrn,evidence


# --------------------------
# Main
# --------------------------

def main():

    master = pd.read_csv(MASTER_FILE,dtype=str)

    if "SmokingStatus" not in master.columns:
        master["SmokingStatus"] = pd.NA

    # anchor map reused from BMI script
    from update_bmi_only import (
        load_structured_encounters,
        choose_best_bmi_anchor_rows,
        choose_backup_bmi_anchor_rows,
        load_and_reconstruct_notes,
        build_sectioned_note
    )

    struct_df = load_structured_encounters()

    primary = choose_best_bmi_anchor_rows(struct_df)
    backup = choose_backup_bmi_anchor_rows(struct_df,primary)

    anchor_map = primary.copy()
    anchor_map.update(backup)

    notes_df = load_and_reconstruct_notes()

    evidence=[]

    final_best={}

    # Stage1
    eligible=set(anchor_map.keys())

    best,evidence = collect_smoking_candidates(
        notes_df,
        anchor_map,
        "day0",
        0,
        0,
        eligible,
        evidence
    )

    final_best.update(best)

    # Stage2
    eligible=set(anchor_map.keys())-set(final_best.keys())

    best,evidence = collect_smoking_candidates(
        notes_df,
        anchor_map,
        "pm7",
        7,
        7,
        eligible,
        evidence
    )

    for k,v in best.items():
        if k not in final_best:
            final_best[k]=v

    # Stage3
    eligible=set(anchor_map.keys())-set(final_best.keys())

    best,evidence = collect_smoking_candidates(
        notes_df,
        anchor_map,
        "pm14",
        14,
        14,
        eligible,
        evidence
    )

    for k,v in best.items():
        if k not in final_best:
            final_best[k]=v

    # update master
    for mrn,c in final_best.items():

        mask = master[MERGE_KEY].astype(str).str.strip()==mrn

        if mask.any():
            master.loc[mask,"SmokingStatus"]=c.value

    master.to_csv(MASTER_FILE,index=False)

    pd.DataFrame(evidence).to_csv(OUTPUT_EVID,index=False)

    print("DONE")
    print("Smoking predictions:",len(final_best))


if __name__=="__main__":
    main()
