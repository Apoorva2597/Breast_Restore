#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deep_audit_stage2.py  (REVISED)

Fix:
- Your mismatches file is likely MRN-based (no ENCRYPTED_PAT_ID), so FP/FN event joins were skipped.
- This version:
  1) Ensures ALL audit CSVs are written every run (even if empty)
  2) If mismatches lacks ENCRYPTED_PAT_ID, it derives it by joining mismatches -> merged on MRN
  3) Then joins to stage_event_level.csv on ENCRYPTED_PAT_ID to produce FP-by-bucket outputs

Inputs (must exist):
- ./_outputs/validation_mismatches_STAGE2_ANCHOR_FIXED.csv
- ./_outputs/validation_merged_STAGE2_ANCHOR_FIXED.csv
- ./_outputs/stage_event_level.csv

Outputs (always written):
- ./_outputs/audit_bucket_summary.csv
- ./_outputs/audit_fp_by_bucket.csv
- ./_outputs/audit_fn_patients.csv
- ./_outputs/audit_bucket_noteType_breakdown.csv
"""

import os
import pandas as pd

ROOT = os.path.abspath(".")
OUT = os.path.join(ROOT, "_outputs")

def read_csv(path):
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")

def pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def norm(s):
    s = "" if s is None else str(s).strip()
    return "" if s.lower() == "nan" else s

def main():
    mism_path = os.path.join(OUT, "validation_mismatches_STAGE2_ANCHOR_FIXED.csv")
    merged_path = os.path.join(OUT, "validation_merged_STAGE2_ANCHOR_FIXED.csv")
    events_path = os.path.join(OUT, "stage_event_level.csv")

    mism = read_csv(mism_path)
    merged = read_csv(merged_path)
    events = read_csv(events_path)

    # --- identify key columns
    mism_mrn_col = pick_col(mism, ["MRN", "mrn"])
    merged_mrn_col = pick_col(merged, ["MRN", "mrn"])
    mism_pid_col = pick_col(mism, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    merged_pid_col = pick_col(merged, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    events_pid_col = pick_col(events, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])

    # --- split FP / FN
    need_cols = ["GOLD_HAS_STAGE2", "PRED_HAS_STAGE2"]
    for c in need_cols:
        if c not in mism.columns:
            raise ValueError("Mismatches missing required column: {}".format(c))

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

    # --- Ensure we have ENCRYPTED_PAT_ID on FP rows
    fp_work = fp.copy()

    if mism_pid_col:
        fp_work["ENCRYPTED_PAT_ID"] = fp_work[mism_pid_col].map(norm)
    else:
        # derive from merged via MRN join
        if (mism_mrn_col is None) or (merged_mrn_col is None) or (merged_pid_col is None):
            # cannot derive; will output empty FP breakdowns
            fp_work["ENCRYPTED_PAT_ID"] = ""
        else:
            tmp = merged[[merged_mrn_col, merged_pid_col]].copy()
            tmp["MRN_JOIN"] = tmp[merged_mrn_col].map(norm)
            tmp["ENCRYPTED_PAT_ID"] = tmp[merged_pid_col].map(norm)
            tmp = tmp[tmp["MRN_JOIN"] != ""].drop_duplicates(subset=["MRN_JOIN"])

            fp_work["MRN_JOIN"] = fp_work[mism_mrn_col].map(norm)
            fp_work = fp_work.merge(tmp[["MRN_JOIN", "ENCRYPTED_PAT_ID"]], on="MRN_JOIN", how="left")

    # --- FP bucket breakdowns (join to events on ENCRYPTED_PAT_ID)
    fp_by_bucket = pd.DataFrame(columns=["DETECTION_BUCKET", "FP_count"])
    bucket_note = pd.DataFrame(columns=["DETECTION_BUCKET", "NOTE_TYPE", "count"])

    if events_pid_col is None:
        # no join key in events; write empties
        pass
    else:
        events2 = events.copy()
        events2["ENCRYPTED_PAT_ID"] = events2[events_pid_col].map(norm)

        fp_ids = fp_work["ENCRYPTED_PAT_ID"].map(norm)
        fp_ids = set([x for x in fp_ids.tolist() if x])

        if fp_ids:
            fp_events = events2[(events2["ENCRYPTED_PAT_ID"].isin(fp_ids)) & (events2["STAGE"] == "STAGE2")].copy()

            if not fp_events.empty and "DETECTION_BUCKET" in fp_events.columns:
                fp_by_bucket = (
                    fp_events.groupby("DETECTION_BUCKET")
                    .size()
                    .reset_index(name="FP_count")
                    .sort_values("FP_count", ascending=False)
                )

                if "NOTE_TYPE" in fp_events.columns:
                    bucket_note = (
                        fp_events.groupby(["DETECTION_BUCKET", "NOTE_TYPE"])
                        .size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False)
                    )

    fp_bucket_out = os.path.join(OUT, "audit_fp_by_bucket.csv")
    fp_by_bucket.to_csv(fp_bucket_out, index=False)

    bucket_note_out = os.path.join(OUT, "audit_bucket_noteType_breakdown.csv")
    bucket_note.to_csv(bucket_note_out, index=False)

    print("Wrote:")
    print(" ", bucket_all_out)
    print(" ", fp_bucket_out)
    print(" ", bucket_note_out)
    print(" ", fn_out)

if __name__ == "__main__":
    main()
