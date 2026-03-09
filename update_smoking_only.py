#!/usr/bin/env python3
# update_smoking_only.py
#
# SmokingStatus updater anchored to reconstruction date
#
# Mirrors the BMI updater architecture and uses the same:
#   - file paths
#   - reconstruction anchor logic
#   - note reconstruction
#   - staged window search
#
# Smoking stages:
#   Stage 1: anchor day only
#   Stage 2: +/- 7 days (if none found in stage1)
#   Stage 3: +/- 14 days (if none found earlier)
#
# Output:
#   SmokingStatus column in master
#
# Evidence file:
#   smoking_only_evidence.csv
#
# Python 3.6.8 compatible

import os
from datetime import datetime
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = BASE_DIR + "/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
OUTPUT_EVID = BASE_DIR + "/_outputs/smoking_only_evidence.csv"

MERGE_KEY = "MRN"

# reuse the same infrastructure from the BMI updater
from update_bmi_only import (
    load_structured_encounters,
    choose_best_bmi_anchor_rows,
    choose_backup_bmi_anchor_rows,
    load_and_reconstruct_notes,
    build_sectioned_note,
    parse_date_safe,
    days_between
)

from extractors.smoking import extract_smoking


# ------------------------------------------------
# utilities
# ------------------------------------------------
def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


# ------------------------------------------------
# candidate ranking
# ------------------------------------------------
def smoking_candidate_rank(cand, recon_dt):
    """
    Ranking logic prioritizes:
      1. closest date to reconstruction
      2. SOCIAL HISTORY section
      3. higher confidence
    """

    note_dt = parse_date_safe(getattr(cand, "note_date", ""))
    if note_dt is None or recon_dt is None:
        return None

    dd = days_between(note_dt, recon_dt)
    if dd is None:
        return None

    section = clean_cell(getattr(cand, "section", "")).lower()

    if "social" in section:
        section_rank = 0
    else:
        section_rank = 1

    conf = float(getattr(cand, "confidence", 0.0) or 0.0)

    return (
        abs(dd),
        section_rank,
        -conf
    )


def choose_best(existing, new, recon_dt):

    if existing is None:
        return new

    ex_rank = smoking_candidate_rank(existing, recon_dt)
    nw_rank = smoking_candidate_rank(new, recon_dt)

    if ex_rank is None:
        return new
    if nw_rank is None:
        return existing

    if nw_rank < ex_rank:
        return new

    return existing


# ------------------------------------------------
# note window check
# ------------------------------------------------
def note_in_window(note_dt, recon_dt, before_days, after_days):

    dd = days_between(note_dt, recon_dt)

    if dd is None:
        return False

    return (dd >= (-1 * before_days) and dd <= after_days)


# ------------------------------------------------
# candidate collection
# ------------------------------------------------
def collect_smoking_candidates_for_window(
    notes_df,
    anchor_map,
    stage_name,
    before_days,
    after_days,
    eligible_mrns,
    evidence_rows
):

    best_by_mrn = {}

    for _, row in notes_df.iterrows():

        mrn = clean_cell(row.get(MERGE_KEY, ""))

        if not mrn or mrn not in eligible_mrns:
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
            smoking_candidates = extract_smoking(snote)
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
                "ANCHOR_DATE": anchor.get("recon_date", ""),
                "EVIDENCE": "extract_smoking failed: {0}".format(repr(e))
            })

            continue

        if not smoking_candidates:
            continue

        for c in smoking_candidates:

            note_day_diff = days_between(
                parse_date_safe(getattr(c, "note_date", "")),
                recon_dt
            )

            evid = (
                "{0} | RECON_DATE={1} | NOTE_DAY_DIFF={2}"
            ).format(
                getattr(c, "evidence", ""),
                anchor.get("recon_date", ""),
                note_day_diff
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
                "ANCHOR_DATE": anchor.get("recon_date", ""),
                "EVIDENCE": evid
            })

            existing = best_by_mrn.get(mrn)

            best_by_mrn[mrn] = choose_best(existing, c, recon_dt)

    return best_by_mrn, evidence_rows


# ------------------------------------------------
# main
# ------------------------------------------------
def main():

    print("Loading master...")

    master = pd.read_csv(MASTER_FILE, dtype=str)

    if "SmokingStatus" not in master.columns:
        master["SmokingStatus"] = pd.NA

    print("Loading structured encounters...")

    struct_df = load_structured_encounters()

    primary_anchor_map = choose_best_bmi_anchor_rows(struct_df)
    backup_anchor_map = choose_backup_bmi_anchor_rows(struct_df, primary_anchor_map)

    anchor_map = {}

    for mrn, info in primary_anchor_map.items():
        anchor_map[mrn] = info

    for mrn, info in backup_anchor_map.items():
        if mrn not in anchor_map:
            anchor_map[mrn] = info

    print("Total anchors available:", len(anchor_map))

    print("Loading reconstructed notes...")

    notes_df = load_and_reconstruct_notes()

    print("Reconstructed notes:", len(notes_df))

    evidence_rows = []

    final_best = {}

    # Stage 1
    stage1_mrns = set(anchor_map.keys())

    print("Stage1: anchor day search")

    best_stage1, evidence_rows = collect_smoking_candidates_for_window(
        notes_df,
        anchor_map,
        "day0",
        0,
        0,
        stage1_mrns,
        evidence_rows
    )

    for mrn, cand in best_stage1.items():
        final_best[mrn] = cand

    # Stage 2
    stage2_mrns = set([m for m in anchor_map.keys() if m not in final_best])

    print("Stage2: +/-7 days search")

    best_stage2, evidence_rows = collect_smoking_candidates_for_window(
        notes_df,
        anchor_map,
        "pm7",
        7,
        7,
        stage2_mrns,
        evidence_rows
    )

    for mrn, cand in best_stage2.items():
        if mrn not in final_best:
            final_best[mrn] = cand

    # Stage 3
    stage3_mrns = set([m for m in anchor_map.keys() if m not in final_best])

    print("Stage3: +/-14 days search")

    best_stage3, evidence_rows = collect_smoking_candidates_for_window(
        notes_df,
        anchor_map,
        "pm14",
        14,
        14,
        stage3_mrns,
        evidence_rows
    )

    for mrn, cand in best_stage3.items():
        if mrn not in final_best:
            final_best[mrn] = cand

    print("Final smoking predictions:", len(final_best))

    for mrn, cand in final_best.items():

        mask = master[MERGE_KEY].astype(str).str.strip() == mrn

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
    print("Updated master:", MASTER_FILE)
    print("Smoking evidence:", OUTPUT_EVID)


if __name__ == "__main__":
    main()
