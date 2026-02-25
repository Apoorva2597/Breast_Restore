#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_FINAL.py  (Python 3.6.8 compatible)

Run from: ~/Breast_Restore
Input:    ./_staging_inputs/HPI11526 Operation Notes.csv   (auto-detected if name differs)
Outputs:  ./_outputs/patient_stage_summary.csv
          ./_outputs/stage_event_level.csv

What it does:
- Reads staged OP notes CSV
- Detects Stage 1 vs Stage 2 signals from NOTE_TEXT
- Emits patient-level earliest Stage1 and Stage2 dates + evidence
- Emits event-level rows for every detected stage signal (for QA)
"""

from __future__ import print_function

import os
import csv
import re
import glob
from datetime import datetime

STAGING_DIR = os.path.join(os.getcwd(), "_staging_inputs")
OUT_DIR = os.path.join(os.getcwd(), "_outputs")

EXPECTED_COLS = [
    "MRN", "ENCRYPTED_PAT_ID", "ETHNICITY", "RACE", "PAT_ENC_CSN_ID",
    "ENCRYPTED_CSN", "OPERATION_DATE", "NOTE_DATE_OF_SERVICE", "NOTE_TYPE",
    "NOTE_ID", "LINE", "NOTE_TEXT"
]

# -------------------------
# Helpers
# -------------------------

def _safe_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def _pick_input_csv():
    """
    Prefer the exact expected file name; otherwise pick the first CSV in _staging_inputs.
    """
    preferred = os.path.join(STAGING_DIR, "HPI11526 Operation Notes.csv")
    if os.path.isfile(preferred):
        return preferred

    candidates = sorted(glob.glob(os.path.join(STAGING_DIR, "*.csv")))
    if not candidates:
        raise IOError("No CSV found in staging dir: {0}".format(STAGING_DIR))
    return candidates[0]

def _normalize_text(s):
    if s is None:
        return ""
    # Keep it simple and fast for large notes
    s = s.replace("\r", "\n").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _parse_date_any(s):
    """
    Accepts common formats seen in exports. Returns YYYY-MM-DD string or "".
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    # Some exports include time; some are pure date; some have weird spacing.
    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M",
        "%m/%d/%y %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Fallback: try to pull a date-like token out
    m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", s)
    if m:
        token = m.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                dt = datetime.strptime(token, fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    m = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", s)
    if m:
        token = m.group(1)
        try:
            dt = datetime.strptime(token, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    return ""

def _best_note_date(row):
    """
    Prefer OPERATION_DATE, else NOTE_DATE_OF_SERVICE.
    """
    op = _parse_date_any(row.get("OPERATION_DATE", ""))
    if op:
        return op
    dos = _parse_date_any(row.get("NOTE_DATE_OF_SERVICE", ""))
    return dos

# -------------------------
# Stage detection
# -------------------------

# Stage 2: expander removal + implant placement/exchange
STAGE2_PATTERNS = [
    # Strong, explicit
    r"\b(expander|tissue expander|te)\b.*\b(remov|explant|take out|removed|removal)\b.*\b(implant|silicone implant|permanent implant)\b",
    r"\b(implant)\b.*\b(exchange|exchang(e|ed)|replace|replacement)\b",
    r"\b(exchange)\b.*\b(implant|implants)\b",
    r"\b(removal of)\b.*\b(tissue expander|expanders|te)\b",
    r"\b(place|placement|insert|insertion)\b.*\b(permanent|silicone)\b.*\b(implant|implants)\b",

    # Common op-note phrasing
    r"\b(expander[- ]?to[- ]?implant)\b",
    r"\b(te/implant)\b",
    r"\b(second stage)\b.*\b(implant|exchange)\b",
    r"\b(stage\s*2)\b.*\b(implant|exchange)\b",
]

# Stage 1: mastectomy + expander placement (or first stage reconstruction)
STAGE1_PATTERNS = [
    # Explicit
    r"\b(mastectomy|nipple[- ]?sparing mastectomy|skin[- ]?sparing mastectomy)\b.*\b(tissue expander|expanders|te)\b.*\b(place|placement|insert|insertion)\b",
    r"\b(place|placement|insert|insertion)\b.*\b(tissue expander|expanders|te)\b",
    r"\b(first stage)\b.*\b(reconstruction|tissue expander|expander|te)\b",
    r"\b(stage\s*1)\b.*\b(reconstruction|tissue expander|expander|te)\b",

    # Backup: mastectomy mention (some notes omit explicit "placement" even when done)
    r"\b(mastectomy|nipple[- ]?sparing mastectomy|skin[- ]?sparing mastectomy)\b",
]

def detect_stage(note_text):
    """
    Returns: (stage_label, matched_pattern)
    stage_label in {"STAGE2","STAGE1",""}
    """
    t = _normalize_text(note_text)

    # Stage 2 first to avoid mislabeling (stage 2 notes often mention mastectomy history)
    for pat in STAGE2_PATTERNS:
        if re.search(pat, t):
            return "STAGE2", pat

    for pat in STAGE1_PATTERNS:
        if re.search(pat, t):
            return "STAGE1", pat

    return "", ""

# -------------------------
# Main
# -------------------------

def read_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        # Basic header sanity check (don’t hard-fail if extra cols exist)
        hdr = reader.fieldnames or []
        missing = [c for c in EXPECTED_COLS if c not in hdr]
        if missing:
            raise ValueError("Missing expected columns: {0}. Found: {1}".format(missing, hdr))

        for r in reader:
            rows.append(r)
    return rows

def build(outputs_dir, input_csv):
    _safe_mkdir(outputs_dir)

    rows = read_rows(input_csv)

    # patient_id -> dict
    patients = {}
    # event-level rows
    events = []

    for r in rows:
        pid = (r.get("ENCRYPTED_PAT_ID") or "").strip()
        if not pid:
            # skip malformed rows
            continue

        date = _best_note_date(r)
        note_id = (r.get("NOTE_ID") or "").strip()
        note_type = (r.get("NOTE_TYPE") or "").strip()
        text = r.get("NOTE_TEXT", "")

        stage, pat = detect_stage(text)
        if not stage:
            continue

        # Record event-level info for QA
        events.append({
            "ENCRYPTED_PAT_ID": pid,
            "STAGE": stage,
            "EVENT_DATE": date,
            "NOTE_ID": note_id,
            "NOTE_TYPE": note_type,
            "MATCH_PATTERN": pat
        })

        # Patient-level aggregation
        if pid not in patients:
            patients[pid] = {
                "stage1_date": "",
                "stage1_note_id": "",
                "stage1_note_type": "",
                "stage1_pattern": "",
                "stage2_date": "",
                "stage2_note_id": "",
                "stage2_note_type": "",
                "stage2_pattern": "",
                "stage1_hits": 0,
                "stage2_hits": 0,
            }

        p = patients[pid]
        if stage == "STAGE1":
            p["stage1_hits"] += 1
            # earliest stage1 date
            if date:
                if (not p["stage1_date"]) or (date < p["stage1_date"]):
                    p["stage1_date"] = date
                    p["stage1_note_id"] = note_id
                    p["stage1_note_type"] = note_type
                    p["stage1_pattern"] = pat
            else:
                # no date available; keep first evidence if empty
                if not p["stage1_note_id"]:
                    p["stage1_note_id"] = note_id
                    p["stage1_note_type"] = note_type
                    p["stage1_pattern"] = pat

        elif stage == "STAGE2":
            p["stage2_hits"] += 1
            # earliest stage2 date (optionally could enforce after stage1; we do not enforce here)
            if date:
                if (not p["stage2_date"]) or (date < p["stage2_date"]):
                    p["stage2_date"] = date
                    p["stage2_note_id"] = note_id
                    p["stage2_note_type"] = note_type
                    p["stage2_pattern"] = pat
            else:
                if not p["stage2_note_id"]:
                    p["stage2_note_id"] = note_id
                    p["stage2_note_type"] = note_type
                    p["stage2_pattern"] = pat

    # Write patient summary
    patient_out = os.path.join(outputs_dir, "patient_stage_summary.csv")
    with open(patient_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID",
            "STAGE1_DATE", "STAGE1_NOTE_ID", "STAGE1_NOTE_TYPE", "STAGE1_MATCH_PATTERN", "STAGE1_HITS",
            "STAGE2_DATE", "STAGE2_NOTE_ID", "STAGE2_NOTE_TYPE", "STAGE2_MATCH_PATTERN", "STAGE2_HITS",
            "HAS_STAGE1", "HAS_STAGE2"
        ])
        w.writeheader()
        for pid in sorted(patients.keys()):
            p = patients[pid]
            w.writerow({
                "ENCRYPTED_PAT_ID": pid,
                "STAGE1_DATE": p["stage1_date"],
                "STAGE1_NOTE_ID": p["stage1_note_id"],
                "STAGE1_NOTE_TYPE": p["stage1_note_type"],
                "STAGE1_MATCH_PATTERN": p["stage1_pattern"],
                "STAGE1_HITS": p["stage1_hits"],
                "STAGE2_DATE": p["stage2_date"],
                "STAGE2_NOTE_ID": p["stage2_note_id"],
                "STAGE2_NOTE_TYPE": p["stage2_note_type"],
                "STAGE2_MATCH_PATTERN": p["stage2_pattern"],
                "STAGE2_HITS": p["stage2_hits"],
                "HAS_STAGE1": 1 if (p["stage1_hits"] > 0 or p["stage1_note_id"]) else 0,
                "HAS_STAGE2": 1 if (p["stage2_hits"] > 0 or p["stage2_note_id"]) else 0,
            })

    # Write event-level file
    event_out = os.path.join(outputs_dir, "stage_event_level.csv")
    with open(event_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID", "STAGE", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE", "MATCH_PATTERN"
        ])
        w.writeheader()
        # sort by pid then date
        def _key(e):
            d = e.get("EVENT_DATE") or ""
            return (e.get("ENCRYPTED_PAT_ID") or "", d, e.get("STAGE") or "")
        for e in sorted(events, key=_key):
            w.writerow(e)

    # Console summary
    n_pat = len(patients)
    has_s1 = sum(1 for pid in patients if (patients[pid]["stage1_hits"] > 0 or patients[pid]["stage1_note_id"]))
    has_s2 = sum(1 for pid in patients if (patients[pid]["stage2_hits"] > 0 or patients[pid]["stage2_note_id"]))
    print("OK: input:", input_csv)
    print("OK: outputs:", patient_out)
    print("OK: outputs:", event_out)
    print("Patients with any stage signal:", n_pat)
    print("Patients with Stage 1 evidence:", has_s1)
    print("Patients with Stage 2 evidence:", has_s2)

def main():
    if not os.path.isdir(STAGING_DIR):
        raise IOError("Staging dir not found: {0}".format(STAGING_DIR))
    input_csv = _pick_input_csv()
    build(OUT_DIR, input_csv)

if __name__ == "__main__":
    main()
