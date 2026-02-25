#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_validation_cases.py (Python 3.6.8 compatible)

Inputs:
- ./_outputs/validation_mismatches_STAGE2_ANCHOR_FIXED.csv   (from your validator)
- ./_outputs/stage_event_level.csv                          (from build_stage12_WITH_AUDIT.py)
- ./_outputs/validation_merged_STAGE2_ANCHOR_FIXED.csv      (optional, for extra context if present)

Outputs:
- ./_outputs/FP_cases_detailed.csv
- ./_outputs/FN_cases_with_recent_notes.csv   (last 2 notes per patient from stage_event_level)
"""

from __future__ import print_function
import os
import pandas as pd

OUT_DIR = os.path.join(os.path.abspath("."), "_outputs")

def _read(path):
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")

def main():
    mism_path = os.path.join(OUT_DIR, "validation_mismatches_STAGE2_ANCHOR_FIXED.csv")
    events_path = os.path.join(OUT_DIR, "stage_event_level.csv")

    if not os.path.isfile(mism_path):
        raise IOError("Missing mismatches file: {}".format(mism_path))
    if not os.path.isfile(events_path):
        raise IOError("Missing event-level file: {}".format(events_path))

    mism = _read(mism_path)
    events = _read(events_path)

    # Ensure numeric flags
    for c in ["GOLD_HAS_STAGE2", "PRED_HAS_STAGE2"]:
        if c in mism.columns:
            mism[c] = mism[c].astype(str).str.strip()
        else:
            raise ValueError("Mismatches file missing column: {}".format(c))

    # Key column for joining back to events: ENCRYPTED_PAT_ID may not exist in mismatches (MRN-based)
    # We join FP/FN summaries at MRN-level only (since mismatches are MRN-based).
    # So: FP file is just mismatches subset + any event-level hits for those patients if MRN exists in events (it doesn't).
    # Practical approach without extra data: create FP/FN lists from mismatches, and attach ALL stage_event_level rows
    # by ENCRYPTED_PAT_ID ONLY if mismatches also includes ENCRYPTED_PAT_ID. If not, still output FP/FN lists.
    has_pid = "ENCRYPTED_PAT_ID" in mism.columns and mism["ENCRYPTED_PAT_ID"].str.strip().any()

    # Split FP / FN
    fp = mism[(mism["GOLD_HAS_STAGE2"] == "0") & (mism["PRED_HAS_STAGE2"] == "1")].copy()
    fn = mism[(mism["GOLD_HAS_STAGE2"] == "1") & (mism["PRED_HAS_STAGE2"] == "0")].copy()

    # FP detailed: attach triggering evidence if we can join by ENCRYPTED_PAT_ID
    if has_pid and "ENCRYPTED_PAT_ID" in events.columns:
        fp = fp.merge(
            events[events["STAGE"] == "STAGE2"].copy(),
            on="ENCRYPTED_PAT_ID",
            how="left",
            suffixes=("", "_event")
        )

    fp_out = os.path.join(OUT_DIR, "FP_cases_detailed.csv")
    fp.to_csv(fp_out, index=False)

    # FN recent notes: if we can join by ENCRYPTED_PAT_ID, pull last 2 notes (any stage) per patient
    if has_pid and "ENCRYPTED_PAT_ID" in events.columns:
        fn_pids = fn["ENCRYPTED_PAT_ID"].astype(str).str.strip()
        fn_pids = set([x for x in fn_pids.tolist() if x])

        ev = events[events["ENCRYPTED_PAT_ID"].isin(fn_pids)].copy()
        # Sort by EVENT_DATE (string YYYY-MM-DD ok), then take last 2 per patient
        ev["EVENT_DATE_SORT"] = ev["EVENT_DATE"].astype(str)
        ev = ev.sort_values(["ENCRYPTED_PAT_ID", "EVENT_DATE_SORT", "NOTE_ID"], ascending=[True, False, False])
        ev = ev.groupby("ENCRYPTED_PAT_ID").head(2)

        fn_recent = fn.merge(ev, on="ENCRYPTED_PAT_ID", how="left", suffixes=("", "_event"))
    else:
        # Fallback: cannot attach note-level context without ENCRYPTED_PAT_ID in mismatches
        fn_recent = fn.copy()

    fn_out = os.path.join(OUT_DIR, "FN_cases_with_recent_notes.csv")
    fn_recent.to_csv(fn_out, index=False)

    print("OK: wrote", fp_out)
    print("OK: wrote", fn_out)

if __name__ == "__main__":
    main()
