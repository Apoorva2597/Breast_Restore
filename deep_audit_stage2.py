#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deep_audit_stage2.py

Purpose:
Break down FP and FN by detection bucket + note type
So we stop guessing and see exactly which rule is hurting us.

Inputs (must already exist):
- ./_outputs/validation_mismatches_STAGE2_ANCHOR_FIXED.csv
- ./_outputs/validation_merged_STAGE2_ANCHOR_FIXED.csv
- ./_outputs/stage_event_level.csv

Outputs:
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

def main():

    mism = read_csv(os.path.join(OUT, "validation_mismatches_STAGE2_ANCHOR_FIXED.csv"))
    merged = read_csv(os.path.join(OUT, "validation_merged_STAGE2_ANCHOR_FIXED.csv"))
    events = read_csv(os.path.join(OUT, "stage_event_level.csv"))

    # ----------------------------
    # Split FP / FN
    # ----------------------------
    fp = mism[(mism["GOLD_HAS_STAGE2"] == "0") & (mism["PRED_HAS_STAGE2"] == "1")].copy()
    fn = mism[(mism["GOLD_HAS_STAGE2"] == "1") & (mism["PRED_HAS_STAGE2"] == "0")].copy()

    # ----------------------------
    # FP bucket breakdown
    # ----------------------------
    if "ENCRYPTED_PAT_ID" in fp.columns:
        fp_events = fp.merge(
            events[events["STAGE"] == "STAGE2"],
            on="ENCRYPTED_PAT_ID",
            how="left"
        )
    else:
        fp_events = pd.DataFrame()

    if not fp_events.empty:
        bucket_summary = (
            fp_events.groupby("DETECTION_BUCKET")
            .size()
            .reset_index(name="FP_count")
            .sort_values("FP_count", ascending=False)
        )
        bucket_summary.to_csv(os.path.join(OUT, "audit_fp_by_bucket.csv"), index=False)

        bucket_note = (
            fp_events.groupby(["DETECTION_BUCKET", "NOTE_TYPE"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        bucket_note.to_csv(os.path.join(OUT, "audit_bucket_noteType_breakdown.csv"), index=False)

    # ----------------------------
    # FN patients (no Stage2 predicted)
    # ----------------------------
    fn.to_csv(os.path.join(OUT, "audit_fn_patients.csv"), index=False)

    # ----------------------------
    # Global bucket distribution (all predicted Stage2)
    # ----------------------------
    all_stage2 = events[events["STAGE"] == "STAGE2"].copy()
    bucket_all = (
        all_stage2.groupby("DETECTION_BUCKET")
        .size()
        .reset_index(name="Total_predictions")
        .sort_values("Total_predictions", ascending=False)
    )
    bucket_all.to_csv(os.path.join(OUT, "audit_bucket_summary.csv"), index=False)

    print("Audit files written to _outputs/")

if __name__ == "__main__":
    main()
