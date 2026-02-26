#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deep_audit_stage2.py  (REVISED v3 - FIX LINKING)

Fix:
- Your validation outputs are MRN-based and do NOT carry ENCRYPTED_PAT_ID.
- stage_event_level.csv is ENCRYPTED_PAT_ID-based and does NOT carry MRN.
=> Need an MRN <-> ENCRYPTED_PAT_ID bridge.

This script builds the bridge from staging files (they contain BOTH MRN + ENCRYPTED_PAT_ID),
then maps mismatches (MRN) -> ENCRYPTED_PAT_ID -> stage_event_level to populate FP bucket files.

Inputs:
- ./_outputs/validation_mismatches_STAGE2_ANCHOR_FIXED.csv
- ./_outputs/stage_event_level.csv
- ./_staging_inputs/HPI11526 Operation Notes.csv
- ./_staging_inputs/HPI11526 Clinic Notes.csv   (optional; used if present)

Outputs (always written):
- ./_outputs/audit_bucket_summary.csv
- ./_outputs/audit_fp_by_bucket.csv
- ./_outputs/audit_bucket_noteType_breakdown.csv
- ./_outputs/audit_fn_patients.csv
- ./_outputs/audit_fp_events_sample.csv   (top evidence rows for quick QA)
"""

import os
import pandas as pd

ROOT = os.path.abspath(".")
OUT = os.path.join(ROOT, "_outputs")
STG = os.path.join(ROOT, "_staging_inputs")

def read_csv_robust(path):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, encoding=enc).fillna("")
        except Exception:
            continue
    raise IOError("Failed to read CSV: {}".format(path))

def pick_col(df, options):
    cols = set(df.columns)
    for c in options:
        if c in cols:
            return c
    # try case-insensitive
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in options:
        k = str(c).strip().lower()
        if k in lower_map:
            return lower_map[k]
    return None

def norm(x):
    s = "" if x is None else str(x).strip()
    if s.lower() == "nan":
        return ""
    # excel float artifact
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s

def build_mrn_pid_map():
    paths = []
    op = os.path.join(STG, "HPI11526 Operation Notes.csv")
    cl = os.path.join(STG, "HPI11526 Clinic Notes.csv")
    if os.path.isfile(op):
        paths.append(op)
    if os.path.isfile(cl):
        paths.append(cl)

    if not paths:
        raise IOError("Missing staging inputs for MRN<->ENCRYPTED_PAT_ID map in: {}".format(STG))

    frames = []
    for p in paths:
        df = read_csv_robust(p)
        mrn_col = pick_col(df, ["MRN", "mrn"])
        pid_col = pick_col(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
        if not mrn_col or not pid_col:
            continue
        tmp = df[[mrn_col, pid_col]].copy()
        tmp.columns = ["MRN", "ENCRYPTED_PAT_ID"]
        tmp["MRN"] = tmp["MRN"].map(norm)
        tmp["ENCRYPTED_PAT_ID"] = tmp["ENCRYPTED_PAT_ID"].map(norm)
        tmp = tmp[(tmp["MRN"] != "") & (tmp["ENCRYPTED_PAT_ID"] != "")]
        frames.append(tmp)

    if not frames:
        raise ValueError("Could not find MRN + ENCRYPTED_PAT_ID columns in staging inputs.")

    m = pd.concat(frames, ignore_index=True).drop_duplicates()
    # In rare cases MRN maps to multiple IDs; keep the most frequent mapping.
    m["n"] = 1
    m = (
        m.groupby(["MRN", "ENCRYPTED_PAT_ID"])["n"]
        .sum()
        .reset_index()
        .sort_values(["MRN", "n"], ascending=[True, False])
    )
    m = m.drop_duplicates(subset=["MRN"], keep="first")[["MRN", "ENCRYPTED_PAT_ID"]]
    return m

def main():
    mism_path = os.path.join(OUT, "validation_mismatches_STAGE2_ANCHOR_FIXED.csv")
    events_path = os.path.join(OUT, "stage_event_level.csv")

    mism = read_csv_robust(mism_path)
    events = read_csv_robust(events_path)

    # Required mismatch columns
    for c in ["GOLD_HAS_STAGE2", "PRED_HAS_STAGE2"]:
        if c not in mism.columns:
            raise ValueError("Mismatches missing required column: {}".format(c))

    mism_mrn_col = pick_col(mism, ["MRN", "mrn"])
    if not mism_mrn_col:
        raise ValueError("Mismatches file missing MRN column.")

    events_pid_col = pick_col(events, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not events_pid_col:
        raise ValueError("stage_event_level missing ENCRYPTED_PAT_ID column.")

    events = events.copy()
    events["ENCRYPTED_PAT_ID"] = events[events_pid_col].map(norm)

    # --- build MRN<->PID map from staging files
    mrn_pid = build_mrn_pid_map()

    # --- split FP / FN
    fp = mism[(mism["GOLD_HAS_STAGE2"].astype(str).str.strip() == "0") &
              (mism["PRED_HAS_STAGE2"].astype(str).str.strip() == "1")].copy()

    fn = mism[(mism["GOLD_HAS_STAGE2"].astype(str).str.strip() == "1") &
              (mism["PRED_HAS_STAGE2"].astype(str).str.strip() == "0")].copy()

    # Always write FN list
    fn_out = os.path.join(OUT, "audit_fn_patients.csv")
    fn.to_csv(fn_out, index=False)

    # --- global bucket distribution (all predicted stage2)
    bucket_all = pd.DataFrame(columns=["DETECTION_BUCKET", "Total_predictions"])
    if "STAGE" in events.columns and "DETECTION_BUCKET" in events.columns:
        all_stage2 = events[events["STAGE"] == "STAGE2"].copy()
        if not all_stage2.empty:
            bucket_all = (
                all_stage2.groupby("DETECTION_BUCKET")
                .size()
                .reset_index(name="Total_predictions")
                .sort_values("Total_predictions", ascending=False)
            )
    bucket_all_out = os.path.join(OUT, "audit_bucket_summary.csv")
    bucket_all.to_csv(bucket_all_out, index=False)

    # --- FP join: MRN -> PID -> stage_event_level
    fp_work = fp.copy()
    fp_work["MRN"] = fp_work[mism_mrn_col].map(norm)
    fp_work = fp_work.merge(mrn_pid, on="MRN", how="left")

    fp_ids = set([x for x in fp_work["ENCRYPTED_PAT_ID"].map(norm).tolist() if x])

    fp_events = pd.DataFrame()
    if fp_ids and ("STAGE" in events.columns):
        fp_events = events[(events["ENCRYPTED_PAT_ID"].isin(fp_ids)) & (events["STAGE"] == "STAGE2")].copy()

    # --- FP by bucket
    fp_by_bucket = pd.DataFrame(columns=["DETECTION_BUCKET", "FP_count"])
    if not fp_events.empty and "DETECTION_BUCKET" in fp_events.columns:
        fp_by_bucket = (
            fp_events.groupby("DETECTION_BUCKET")
            .size()
            .reset_index(name="FP_count")
            .sort_values("FP_count", ascending=False)
        )
    fp_bucket_out = os.path.join(OUT, "audit_fp_by_bucket.csv")
    fp_by_bucket.to_csv(fp_bucket_out, index=False)

    # --- Bucket x NOTE_TYPE breakdown
    bucket_note = pd.DataFrame(columns=["DETECTION_BUCKET", "NOTE_TYPE", "count"])
    if not fp_events.empty and "DETECTION_BUCKET" in fp_events.columns and "NOTE_TYPE" in fp_events.columns:
        bucket_note = (
            fp_events.groupby(["DETECTION_BUCKET", "NOTE_TYPE"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    bucket_note_out = os.path.join(OUT, "audit_bucket_noteType_breakdown.csv")
    bucket_note.to_csv(bucket_note_out, index=False)

    # --- quick sample for human review (top 200 rows)
    sample_cols = [c for c in [
        "ENCRYPTED_PAT_ID", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE",
        "DETECTION_BUCKET", "PATTERN_NAME", "IS_OPERATIVE_CONTEXT", "EVIDENCE_SNIPPET"
    ] if c in fp_events.columns]
    fp_sample = fp_events[sample_cols].head(200) if not fp_events.empty else pd.DataFrame(columns=sample_cols)
    fp_sample_out = os.path.join(OUT, "audit_fp_events_sample.csv")
    fp_sample.to_csv(fp_sample_out, index=False)

    print("Wrote:")
    print(" ", bucket_all_out)
    print(" ", fp_bucket_out)
    print(" ", bucket_note_out)
    print(" ", fp_sample_out)
    print(" ", fn_out)

if __name__ == "__main__":
    main()
