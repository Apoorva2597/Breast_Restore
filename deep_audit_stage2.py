#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deep_audit_stage2.py  (REVISED v4 - FIX EMPTY FP BUCKETS)

Why your FP bucket files are empty:
- FP/FN live in validation_mismatches (usually MRN-based)
- stage_event_level is ENCRYPTED_PAT_ID-based
- Your MRN->ENCRYPTED_PAT_ID mapping is failing (format mismatch, leading zeros, non-digit chars, etc.)

This version:
- Builds TWO MRN keys everywhere:
  (1) MRN_RAW (trimmed)
  (2) MRN_DIGITS (digits-only)
- Tries ALL linking paths (in this order):
  A) mismatches already has ENCRYPTED_PAT_ID
  B) mismatches -> merged (MRN_RAW then MRN_DIGITS) to get ENCRYPTED_PAT_ID
  C) mismatches -> staging map (MRN_RAW then MRN_DIGITS) to get ENCRYPTED_PAT_ID
- Normalizes STAGE values in stage_event_level before filtering.

Inputs:
- ./_outputs/validation_mismatches_STAGE2_ANCHOR_FIXED.csv
- ./_outputs/validation_merged_STAGE2_ANCHOR_FIXED.csv   (used if present)
- ./_outputs/stage_event_level.csv
- ./_staging_inputs/HPI11526 Operation Notes.csv
- ./_staging_inputs/HPI11526 Clinic Notes.csv (optional)

Outputs (always written):
- ./_outputs/audit_bucket_summary.csv
- ./_outputs/audit_fp_by_bucket.csv
- ./_outputs/audit_bucket_noteType_breakdown.csv
- ./_outputs/audit_fn_patients.csv
- ./_outputs/audit_fp_events_sample.csv
- ./_outputs/audit_link_debug.csv   (counts + how many FP rows mapped)
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
    cols = list(df.columns)
    colset = set(cols)
    for c in options:
        if c in colset:
            return c
    low = {str(c).strip().lower(): c for c in cols}
    for c in options:
        k = str(c).strip().lower()
        if k in low:
            return low[k]
    return None

def norm(x):
    s = "" if x is None else str(x).strip()
    if s.lower() == "nan":
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s

def digits_only(x):
    s = norm(x)
    if not s:
        return ""
    return "".join([ch for ch in s if ch.isdigit()])

def ensure_mrn_keys(df, mrn_col, out_raw="MRN_RAW", out_digits="MRN_DIGITS"):
    df[out_raw] = df[mrn_col].map(norm) if mrn_col in df.columns else ""
    df[out_digits] = df[mrn_col].map(digits_only) if mrn_col in df.columns else ""
    return df

def ensure_pid(df, pid_col, out_pid="ENCRYPTED_PAT_ID"):
    df[out_pid] = df[pid_col].map(norm) if pid_col and pid_col in df.columns else ""
    return df

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
        tmp.columns = ["MRN_SRC", "PID_SRC"]
        tmp["MRN_RAW"] = tmp["MRN_SRC"].map(norm)
        tmp["MRN_DIGITS"] = tmp["MRN_SRC"].map(digits_only)
        tmp["ENCRYPTED_PAT_ID"] = tmp["PID_SRC"].map(norm)
        tmp = tmp[(tmp["ENCRYPTED_PAT_ID"] != "") & ((tmp["MRN_RAW"] != "") | (tmp["MRN_DIGITS"] != ""))]
        frames.append(tmp[["MRN_RAW", "MRN_DIGITS", "ENCRYPTED_PAT_ID"]])

    if not frames:
        raise ValueError("Could not find MRN + ENCRYPTED_PAT_ID columns in staging inputs.")

    m = pd.concat(frames, ignore_index=True).drop_duplicates()

    # resolve duplicates by frequency for each MRN key
    m["n"] = 1

    m_raw = (
        m[m["MRN_RAW"] != ""]
        .groupby(["MRN_RAW", "ENCRYPTED_PAT_ID"])["n"].sum()
        .reset_index()
        .sort_values(["MRN_RAW", "n"], ascending=[True, False])
        .drop_duplicates(subset=["MRN_RAW"], keep="first")[["MRN_RAW", "ENCRYPTED_PAT_ID"]]
    )

    m_dig = (
        m[m["MRN_DIGITS"] != ""]
        .groupby(["MRN_DIGITS", "ENCRYPTED_PAT_ID"])["n"].sum()
        .reset_index()
        .sort_values(["MRN_DIGITS", "n"], ascending=[True, False])
        .drop_duplicates(subset=["MRN_DIGITS"], keep="first")[["MRN_DIGITS", "ENCRYPTED_PAT_ID"]]
    )

    return m_raw, m_dig

def map_pid_for_mismatches(mism, merged, mrn_raw_map, mrn_dig_map):
    mism = mism.copy()

    # if mism already has pid
    mism_pid_col = pick_col(mism, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if mism_pid_col:
        mism = ensure_pid(mism, mism_pid_col, out_pid="ENCRYPTED_PAT_ID")
        return mism

    # need MRN
    mism_mrn_col = pick_col(mism, ["MRN", "mrn"])
    if not mism_mrn_col:
        raise ValueError("Mismatches file missing MRN column AND missing ENCRYPTED_PAT_ID.")

    mism = ensure_mrn_keys(mism, mism_mrn_col)

    # B) try mism -> merged if merged has pid + mrn
    if merged is not None and not merged.empty:
        merged_pid_col = pick_col(merged, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
        merged_mrn_col = pick_col(merged, ["MRN", "mrn"])
        if merged_pid_col and merged_mrn_col:
            m2 = merged[[merged_mrn_col, merged_pid_col]].copy()
            m2 = ensure_mrn_keys(m2, merged_mrn_col)
            m2 = ensure_pid(m2, merged_pid_col)
            # raw join
            j = m2[m2["MRN_RAW"] != ""][["MRN_RAW", "ENCRYPTED_PAT_ID"]].drop_duplicates(subset=["MRN_RAW"])
            mism = mism.merge(j, on="MRN_RAW", how="left", suffixes=("", "_m"))
            mism["ENCRYPTED_PAT_ID"] = mism["ENCRYPTED_PAT_ID"].fillna("")
            # digits join for still-unmapped
            need = (mism["ENCRYPTED_PAT_ID"] == "") & (mism["MRN_DIGITS"] != "")
            if need.any():
                j2 = m2[m2["MRN_DIGITS"] != ""][["MRN_DIGITS", "ENCRYPTED_PAT_ID"]].drop_duplicates(subset=["MRN_DIGITS"])
                mism2 = mism[need].merge(j2, on="MRN_DIGITS", how="left", suffixes=("", "_d"))
                mism.loc[need, "ENCRYPTED_PAT_ID"] = mism2["ENCRYPTED_PAT_ID"].fillna("").values

            return mism

    # C) fall back to staging map
    mism = mism.merge(mrn_raw_map, on="MRN_RAW", how="left", suffixes=("", "_rawmap"))
    mism["ENCRYPTED_PAT_ID"] = mism["ENCRYPTED_PAT_ID"].fillna("")

    need = (mism["ENCRYPTED_PAT_ID"] == "") & (mism["MRN_DIGITS"] != "")
    if need.any():
        mism2 = mism[need].merge(mrn_dig_map, on="MRN_DIGITS", how="left", suffixes=("", "_digmap"))
        mism.loc[need, "ENCRYPTED_PAT_ID"] = mism2["ENCRYPTED_PAT_ID"].fillna("").values

    mism["ENCRYPTED_PAT_ID"] = mism["ENCRYPTED_PAT_ID"].fillna("")
    return mism

def main():
    mism_path = os.path.join(OUT, "validation_mismatches_STAGE2_ANCHOR_FIXED.csv")
    merged_path = os.path.join(OUT, "validation_merged_STAGE2_ANCHOR_FIXED.csv")
    events_path = os.path.join(OUT, "stage_event_level.csv")

    mism = read_csv_robust(mism_path)

    merged = None
    if os.path.isfile(merged_path):
        merged = read_csv_robust(merged_path)
    else:
        merged = pd.DataFrame()

    events = read_csv_robust(events_path)

    # normalize events pid and stage
    events_pid_col = pick_col(events, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not events_pid_col:
        raise ValueError("stage_event_level.csv missing ENCRYPTED_PAT_ID")
    events = ensure_pid(events, events_pid_col, out_pid="ENCRYPTED_PAT_ID")
    if "STAGE" in events.columns:
        events["STAGE_N"] = events["STAGE"].map(lambda x: norm(x).upper())
    else:
        events["STAGE_N"] = ""

    # required columns
    for c in ["GOLD_HAS_STAGE2", "PRED_HAS_STAGE2"]:
        if c not in mism.columns:
            raise ValueError("Mismatches missing required column: {}".format(c))

    # maps from staging
    mrn_raw_map, mrn_dig_map = build_mrn_pid_map()

    # attach pid to mismatches robustly
    mism2 = map_pid_for_mismatches(mism, merged, mrn_raw_map, mrn_dig_map)

    # split FP / FN
    fp = mism2[(mism2["GOLD_HAS_STAGE2"].astype(str).str.strip() == "0") &
               (mism2["PRED_HAS_STAGE2"].astype(str).str.strip() == "1")].copy()

    fn = mism2[(mism2["GOLD_HAS_STAGE2"].astype(str).str.strip() == "1") &
               (mism2["PRED_HAS_STAGE2"].astype(str).str.strip() == "0")].copy()

    fn_out = os.path.join(OUT, "audit_fn_patients.csv")
    fn.to_csv(fn_out, index=False)

    # global bucket summary
    bucket_all = pd.DataFrame(columns=["DETECTION_BUCKET", "Total_predictions"])
    if "DETECTION_BUCKET" in events.columns:
        all_stage2 = events[events["STAGE_N"] == "STAGE2"].copy()
        if not all_stage2.empty:
            bucket_all = (
                all_stage2.groupby("DETECTION_BUCKET")
                .size()
                .reset_index(name="Total_predictions")
                .sort_values("Total_predictions", ascending=False)
            )
    bucket_all_out = os.path.join(OUT, "audit_bucket_summary.csv")
    bucket_all.to_csv(bucket_all_out, index=False)

    # FP events
    fp["ENCRYPTED_PAT_ID"] = fp["ENCRYPTED_PAT_ID"].map(norm)
    fp_ids = set([x for x in fp["ENCRYPTED_PAT_ID"].tolist() if x])

    fp_events = pd.DataFrame()
    if fp_ids:
        fp_events = events[(events["ENCRYPTED_PAT_ID"].isin(fp_ids)) & (events["STAGE_N"] == "STAGE2")].copy()

    # FP by bucket
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

    # bucket x note type
    bucket_note = pd.DataFrame(columns=["DETECTION_BUCKET", "NOTE_TYPE", "count"])
    if not fp_events.empty and ("DETECTION_BUCKET" in fp_events.columns) and ("NOTE_TYPE" in fp_events.columns):
        bucket_note = (
            fp_events.groupby(["DETECTION_BUCKET", "NOTE_TYPE"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    bucket_note_out = os.path.join(OUT, "audit_bucket_noteType_breakdown.csv")
    bucket_note.to_csv(bucket_note_out, index=False)

    # sample evidence rows
    sample_cols = [c for c in [
        "ENCRYPTED_PAT_ID", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE",
        "DETECTION_BUCKET", "PATTERN_NAME", "IS_OPERATIVE_CONTEXT", "EVIDENCE_SNIPPET"
    ] if c in fp_events.columns]
    fp_sample = fp_events[sample_cols].head(200) if not fp_events.empty else pd.DataFrame(columns=sample_cols)
    fp_sample_out = os.path.join(OUT, "audit_fp_events_sample.csv")
    fp_sample.to_csv(fp_sample_out, index=False)

    # debug counts
    debug = pd.DataFrame([{
        "mism_rows": len(mism),
        "mism_rows_with_pid_after_mapping": int((mism2["ENCRYPTED_PAT_ID"].map(norm) != "").sum()),
        "fp_rows": len(fp),
        "fp_rows_with_pid": int((fp["ENCRYPTED_PAT_ID"].map(norm) != "").sum()),
        "unique_fp_pids": len(fp_ids),
        "fp_events_rows_joined": len(fp_events),
        "events_total": len(events),
        "events_stage2_rows": int((events["STAGE_N"] == "STAGE2").sum())
    }])
    debug_out = os.path.join(OUT, "audit_link_debug.csv")
    debug.to_csv(debug_out, index=False)

    print("Wrote:")
    print(" ", bucket_all_out)
    print(" ", fp_bucket_out)
    print(" ", bucket_note_out)
    print(" ", fp_sample_out)
    print(" ", fn_out)
    print(" ", debug_out)

if __name__ == "__main__":
    main()
