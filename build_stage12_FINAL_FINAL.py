#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_FINAL_WITH_CLINIC.py (Python 3.6.8 compatible)

UPDATED:
- Now reads BOTH:
    - HPI11526 Operation Notes.csv
    - HPI11526 Clinic Notes.csv
  from ./_staging_inputs/
- Combines rows before stage detection
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
    "MRN", "ENCRYPTED_PAT_ID", "ETHNICITY", "RACE",
    "PAT_ENC_CSN_ID", "ENCRYPTED_CSN",
    "OPERATION_DATE", "NOTE_DATE_OF_SERVICE",
    "NOTE_TYPE", "NOTE_ID", "LINE", "NOTE_TEXT"
]

# -------------------------
# Helpers
# -------------------------

def _safe_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def _find_input_csvs():
    """
    Return list of relevant staging CSVs:
    - Operation Notes
    - Clinic Notes
    """
    files = []
    patterns = [
        "HPI11526 Operation Notes.csv",
        "HPI11526 Clinic Notes.csv"
    ]
    for name in patterns:
        p = os.path.join(STAGING_DIR, name)
        if os.path.isfile(p):
            files.append(p)

    if not files:
        raise IOError("No Operation/Clinic notes found in: {0}".format(STAGING_DIR))

    return files

def _normalize_text(s):
    if s is None:
        return ""
    s = s.replace("\r", "\n").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _parse_date_any(s):
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    fmts = [
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y",
        "%Y/%m/%d", "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M", "%m/%d/%y %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return ""

def _best_note_date(row):
    op = _parse_date_any(row.get("OPERATION_DATE", ""))
    if op:
        return op
    return _parse_date_any(row.get("NOTE_DATE_OF_SERVICE", ""))

# -------------------------
# Stage detection logic (unchanged core)
# -------------------------

RE_TE = re.compile(r"\b(expander|expanders|tissue expander|te)\b", re.I)
RE_REMOVE = re.compile(r"\b(remove(d|al)?|explant(ed|ation)?|take\s*out|takedown|retrieve)\b", re.I)
RE_IMPLANT = re.compile(r"\b(implant(s)?|prosthesis)\b", re.I)
RE_ACTION = re.compile(r"\b(place(d|ment)?|insert(ed|ion)?|exchange(d)?|replace(d|ment)?)\b", re.I)

RE_EXCHANGE = re.compile(
    r"\b(implant|expander)\b.*\b(exchange|replace|replacement)\b"
    r"|\b(exchange|replace|replacement)\b.*\b(implant|expander)\b",
    re.I
)

RE_SCHEDULE = re.compile(r"\b(schedule(d)?|planned|plan|will)\b", re.I)
RE_SURG = re.compile(r"\b(surgery|procedure|operation|or)\b", re.I)

STAGE1_PATTERNS = [
    r"\b(mastectomy)\b.*\b(tissue expander|expanders|te)\b.*\b(place|placement|insert|insertion)\b",
    r"\b(place|placement|insert|insertion)\b.*\b(tissue expander|expanders|te)\b",
    r"\b(first stage)\b.*\b(reconstruction|tissue expander|expander|te)\b",
    r"\b(stage\s*1)\b.*\b(reconstruction|tissue expander|expander|te)\b",
]

def _stage2_bucket(text_norm):

    # 1) Performed exchange
    if RE_EXCHANGE.search(text_norm):
        return True, "EXCHANGE"

    # 2) Performed TE removal + implant + action
    if RE_TE.search(text_norm) and RE_REMOVE.search(text_norm) and RE_IMPLANT.search(text_norm) and RE_ACTION.search(text_norm):
        return True, "EXPANDER_TO_IMPLANT"

    # 3) Scheduled Stage 2
    if RE_SCHEDULE.search(text_norm) and RE_IMPLANT.search(text_norm) and (RE_SURG.search(text_norm) or RE_TE.search(text_norm)):
        return True, "SCHEDULED_STAGE2"

    return False, ""

def detect_stage(note_text):
    t = _normalize_text(note_text)

    ok2, pat2 = _stage2_bucket(t)
    if ok2:
        return "STAGE2", pat2

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
        for r in reader:
            rows.append(r)
    return rows

def build(outputs_dir, input_csvs):
    _safe_mkdir(outputs_dir)

    all_rows = []
    for path in input_csvs:
        all_rows.extend(read_rows(path))

    patients = {}
    events = []

    for r in all_rows:
        pid = (r.get("ENCRYPTED_PAT_ID") or "").strip()
        if not pid:
            continue

        date = _best_note_date(r)
        note_id = (r.get("NOTE_ID") or "").strip()
        note_type = (r.get("NOTE_TYPE") or "").strip()
        text = r.get("NOTE_TEXT", "")

        stage, pat = detect_stage(text)
        if not stage:
            continue

        events.append({
            "ENCRYPTED_PAT_ID": pid,
            "STAGE": stage,
            "EVENT_DATE": date,
            "NOTE_ID": note_id,
            "NOTE_TYPE": note_type,
            "MATCH_PATTERN": pat
        })

        if pid not in patients:
            patients[pid] = {
                "stage1_date": "",
                "stage2_date": "",
                "stage1_hits": 0,
                "stage2_hits": 0,
            }

        p = patients[pid]

        if stage == "STAGE1":
            p["stage1_hits"] += 1
            if date and (not p["stage1_date"] or date < p["stage1_date"]):
                p["stage1_date"] = date

        elif stage == "STAGE2":
            p["stage2_hits"] += 1
            if date and (not p["stage2_date"] or date < p["stage2_date"]):
                p["stage2_date"] = date

    patient_out = os.path.join(outputs_dir, "patient_stage_summary.csv")
    with open(patient_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID",
            "STAGE1_DATE", "STAGE1_HITS",
            "STAGE2_DATE", "STAGE2_HITS",
            "HAS_STAGE1", "HAS_STAGE2"
        ])
        w.writeheader()
        for pid in sorted(patients.keys()):
            p = patients[pid]
            w.writerow({
                "ENCRYPTED_PAT_ID": pid,
                "STAGE1_DATE": p["stage1_date"],
                "STAGE1_HITS": p["stage1_hits"],
                "STAGE2_DATE": p["stage2_date"],
                "STAGE2_HITS": p["stage2_hits"],
                "HAS_STAGE1": 1 if p["stage1_hits"] > 0 else 0,
                "HAS_STAGE2": 1 if p["stage2_hits"] > 0 else 0,
            })

    print("OK: wrote", patient_out)

def main():
    if not os.path.isdir(STAGING_DIR):
        raise IOError("Staging dir not found: {0}".format(STAGING_DIR))
    input_csvs = _find_input_csvs()
    build(OUT_DIR, input_csvs)

if __name__ == "__main__":
    main()
