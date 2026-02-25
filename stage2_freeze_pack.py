#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_freeze_pack.py  (Python 3.6.8 compatible)

Run from: ~/Breast_Restore
Purpose:
- "Freeze" (snapshot) current Stage 1/2 outputs + validation artifacts so nothing gets overwritten
- Normalize/standardize Stage 2 result columns into a clean, analysis-friendly CSV
- (Optional) create a compact Stage2-only review file for FN/FP manual review

Inputs (auto-detected):
- ./_outputs/patient_stage_summary.csv
- ./_outputs/stage_event_level.csv
- ./_outputs/validation_metrics.txt            (if exists)
- ./_outputs/validation_mismatches.csv         (if exists)
- ./_outputs/validation_merged.csv             (if exists)

Outputs:
- ./_frozen_stage2/<timestamp>/... (copies of all found artifacts)
- ./_frozen_stage2/<timestamp>/stage2_patient_clean.csv
- ./_frozen_stage2/<timestamp>/stage2_event_clean.csv
- ./_frozen_stage2/<timestamp>/stage2_review_fp_fn.csv  (if validation_merged present)

Notes:
- No downstream logic changes. This is ONLY organization + column cleanup.
"""

from __future__ import print_function

import os
import re
import shutil
from datetime import datetime

import pandas as pd


BASE = os.getcwd()
OUT_DIR = os.path.join(BASE, "_outputs")
FREEZE_BASE = os.path.join(BASE, "_frozen_stage2")

# Expected primary files
PAT_SUM = os.path.join(OUT_DIR, "patient_stage_summary.csv")
EVENTS = os.path.join(OUT_DIR, "stage_event_level.csv")

# Optional validation artifacts
VAL_METRICS = os.path.join(OUT_DIR, "validation_metrics.txt")
VAL_MISMATCH = os.path.join(OUT_DIR, "validation_mismatches.csv")
VAL_MERGED = os.path.join(OUT_DIR, "validation_merged.csv")


def safe_mkdir(p):
    if not os.path.isdir(p):
        os.makedirs(p)


def now_stamp():
    # filesystem-safe timestamp
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_csv_safely(path):
    """
    Robust read for mixed encodings you’ve seen in Cedar exports.
    """
    # Try utf-8 first
    try:
        return pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8")
    except Exception:
        pass

    # Try cp1252 (common when you saw 0xA0 / windows artifacts)
    try:
        return pd.read_csv(path, dtype=str, low_memory=False, encoding="cp1252")
    except Exception:
        pass

    # Last resort: latin-1 (never fails decode, but can mangle some chars)
    return pd.read_csv(path, dtype=str, low_memory=False, encoding="latin-1")


def standardize_date(s):
    """
    Standardize to YYYY-MM-DD when possible; otherwise keep as-is.
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return ""

    # Already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    # Try parsing common formats
    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%y %H:%M",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Pull a token if embedded
    m = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", s)
    if m:
        token = m.group(1)
        try:
            dt = datetime.strptime(token, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", s)
    if m:
        token = m.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                dt = datetime.strptime(token, fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    return s  # keep original if unknown


def copy_if_exists(src, dst_dir):
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        return True
    return False


def main():
    if not os.path.isfile(PAT_SUM):
        raise IOError("Missing required file: {0}".format(PAT_SUM))
    if not os.path.isfile(EVENTS):
        raise IOError("Missing required file: {0}".format(EVENTS))

    stamp = now_stamp()
    freeze_dir = os.path.join(FREEZE_BASE, stamp)
    safe_mkdir(freeze_dir)

    print("Freezing Stage 2 artifacts to:", freeze_dir)

    # 1) Copy primary outputs
    shutil.copy2(PAT_SUM, os.path.join(freeze_dir, os.path.basename(PAT_SUM)))
    shutil.copy2(EVENTS, os.path.join(freeze_dir, os.path.basename(EVENTS)))

    # 2) Copy optional validation artifacts (if present)
    found = []
    for f in (VAL_METRICS, VAL_MISMATCH, VAL_MERGED):
        if copy_if_exists(f, freeze_dir):
            found.append(os.path.basename(f))

    if found:
        print("Also copied validation artifacts:", ", ".join(found))
    else:
        print("No optional validation artifacts found in ./_outputs (that is OK).")

    # 3) Create clean patient-level Stage2 file
    ps = read_csv_safely(PAT_SUM)

    # Normalize expected columns (be resilient to slight header drift)
    # We prefer these canonical names:
    rename_map = {}
    for c in ps.columns:
        cc = c.strip()
        # common variants
        if cc.upper() == "ENCRYPTED_PAT_ID":
            rename_map[c] = "ENCRYPTED_PAT_ID"
        elif cc.upper() == "STAGE2_DATE":
            rename_map[c] = "STAGE2_DATE"
        elif cc.upper() == "STAGE2_NOTE_ID":
            rename_map[c] = "STAGE2_NOTE_ID"
        elif cc.upper() == "STAGE2_NOTE_TYPE":
            rename_map[c] = "STAGE2_NOTE_TYPE"
        elif cc.upper() == "STAGE2_MATCH_PATTERN":
            rename_map[c] = "STAGE2_MATCH_PATTERN"
        elif cc.upper() == "STAGE2_HITS":
            rename_map[c] = "STAGE2_HITS"
        elif cc.upper() == "HAS_STAGE2":
            rename_map[c] = "HAS_STAGE2"
        elif cc.upper() == "STAGE1_DATE":
            rename_map[c] = "STAGE1_DATE"
        elif cc.upper() == "HAS_STAGE1":
            rename_map[c] = "HAS_STAGE1"

    ps = ps.rename(columns=rename_map)

    required_patient_cols = ["ENCRYPTED_PAT_ID", "HAS_STAGE2", "STAGE2_DATE", "STAGE2_NOTE_ID", "STAGE2_NOTE_TYPE", "STAGE2_MATCH_PATTERN", "STAGE2_HITS"]
    missing = [c for c in required_patient_cols if c not in ps.columns]
    if missing:
        raise ValueError("patient_stage_summary.csv missing required columns: {0}. Found: {1}".format(missing, list(ps.columns)))

    # Clean fields
    ps["STAGE2_DATE"] = ps["STAGE2_DATE"].apply(standardize_date)
    ps["HAS_STAGE2"] = ps["HAS_STAGE2"].fillna("").astype(str).str.strip()
    ps["STAGE2_HITS"] = ps["STAGE2_HITS"].fillna("").astype(str).str.strip()

    # Derived convenience columns
    ps["PRED_HAS_STAGE2"] = ps["HAS_STAGE2"].apply(lambda x: 1 if x in ("1", "True", "true", "YES", "yes") else 0)
    ps["STAGE2_DATE_MISSING"] = ps["STAGE2_DATE"].apply(lambda x: 1 if (not x) else 0)

    patient_clean_cols = [
        "ENCRYPTED_PAT_ID",
        "PRED_HAS_STAGE2",
        "STAGE2_DATE",
        "STAGE2_NOTE_ID",
        "STAGE2_NOTE_TYPE",
        "STAGE2_MATCH_PATTERN",
        "STAGE2_HITS",
        "STAGE2_DATE_MISSING",
    ]
    stage2_patient_clean = ps[patient_clean_cols].copy()

    patient_clean_out = os.path.join(freeze_dir, "stage2_patient_clean.csv")
    stage2_patient_clean.to_csv(patient_clean_out, index=False)
    print("Wrote:", patient_clean_out)

    # 4) Create clean event-level Stage2 file
    ev = read_csv_safely(EVENTS)

    # Canonicalize event headers
    ev_rename = {}
    for c in ev.columns:
        cc = c.strip().upper()
        if cc == "ENCRYPTED_PAT_ID":
            ev_rename[c] = "ENCRYPTED_PAT_ID"
        elif cc == "STAGE":
            ev_rename[c] = "STAGE"
        elif cc in ("EVENT_DATE", "DATE", "NOTE_DATE", "OP_DATE"):
            ev_rename[c] = "EVENT_DATE"
        elif cc == "NOTE_ID":
            ev_rename[c] = "NOTE_ID"
        elif cc == "NOTE_TYPE":
            ev_rename[c] = "NOTE_TYPE"
        elif cc in ("MATCH_PATTERN", "PATTERN"):
            ev_rename[c] = "MATCH_PATTERN"
    ev = ev.rename(columns=ev_rename)

    required_event_cols = ["ENCRYPTED_PAT_ID", "STAGE", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE", "MATCH_PATTERN"]
    missing_ev = [c for c in required_event_cols if c not in ev.columns]
    if missing_ev:
        raise ValueError("stage_event_level.csv missing required columns: {0}. Found: {1}".format(missing_ev, list(ev.columns)))

    ev["EVENT_DATE"] = ev["EVENT_DATE"].apply(standardize_date)
    ev["STAGE"] = ev["STAGE"].fillna("").astype(str).str.strip()

    stage2_events = ev[ev["STAGE"] == "STAGE2"].copy()
    event_clean_out = os.path.join(freeze_dir, "stage2_event_clean.csv")
    stage2_events.to_csv(event_clean_out, index=False)
    print("Wrote:", event_clean_out)

    # 5) Optional: build a compact FP/FN review file if validation_merged exists
    if os.path.isfile(VAL_MERGED):
        vm = read_csv_safely(VAL_MERGED)

        # We expect these from your validator script:
        # - MRN (gold key)
        # - GOLD_HAS_STAGE2
        # - PRED_HAS_STAGE2
        # Some scripts used HAS_STAGE2 and/or PRED_HAS_STAGE2; handle both.
        cols = [c.strip() for c in vm.columns]
        vm.columns = cols

        # Identify gold/pred cols
        gold_col = "GOLD_HAS_STAGE2" if "GOLD_HAS_STAGE2" in vm.columns else None
        pred_col = "PRED_HAS_STAGE2" if "PRED_HAS_STAGE2" in vm.columns else ("HAS_STAGE2" if "HAS_STAGE2" in vm.columns else None)

        if not gold_col or not pred_col:
            print("NOTE: validation_merged.csv present but missing GOLD_HAS_STAGE2 or prediction column; skipping FP/FN review file.")
        else:
            def to01(x):
                x = "" if x is None else str(x).strip()
                return 1 if x in ("1", "True", "true", "YES", "yes") else 0

            vm["_gold"] = vm[gold_col].apply(to01)
            vm["_pred"] = vm[pred_col].apply(to01)

            fp = vm[(vm["_pred"] == 1) & (vm["_gold"] == 0)].copy()
            fn = vm[(vm["_pred"] == 0) & (vm["_gold"] == 1)].copy()

            # Keep a tight set of columns if available
            keep = []
            for c in ["MRN", "ENCRYPTED_PAT_ID", "STAGE2_DATE", "STAGE2_NOTE_TYPE", "STAGE2_NOTE_ID", "STAGE2_MATCH_PATTERN", gold_col, pred_col]:
                if c in vm.columns and c not in keep:
                    keep.append(c)

            # Fallback to all columns if none found (shouldn't happen)
            if not keep:
                keep = list(vm.columns)

            fp["_error_type"] = "FP"
            fn["_error_type"] = "FN"

            review = pd.concat([fp[keep + ["_error_type"]], fn[keep + ["_error_type"]]], axis=0, sort=False)
            # Sort for human review
            sort_cols = [c for c in ["_error_type", "MRN", "ENCRYPTED_PAT_ID", "STAGE2_DATE"] if c in review.columns]
            if sort_cols:
                review = review.sort_values(sort_cols)

            review_out = os.path.join(freeze_dir, "stage2_review_fp_fn.csv")
            review.to_csv(review_out, index=False)
            print("Wrote:", review_out)

    # 6) Print quick counts (sanity)
    n_pat = stage2_patient_clean.shape[0]
    n_pred_s2 = int(stage2_patient_clean["PRED_HAS_STAGE2"].sum())
    n_ev_s2 = stage2_events.shape[0]

    print("\nSanity checks:")
    print("Patients in patient_stage_summary:", n_pat)
    print("Pred Stage2 patients:", n_pred_s2)
    print("Stage2 events:", n_ev_s2)
    print("\nDone. You can now proceed to complication validation without overwriting Stage 2 artifacts.")


if __name__ == "__main__":
    main()
