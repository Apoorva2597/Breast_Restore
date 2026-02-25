#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_FINAL_WITH_CLINIC_reduceFN_v2.py (Python 3.6.8 compatible)

Goal: reduce FN by adding NOTE_TYPE-aware performed Stage2 detection.
Key idea:
- In OP/operative-type notes, Stage2 can be documented without explicit "expander" wording.
- Add OPERATIVE-ONLY performed buckets:
  A) implant + (exchange/replace)  (even without expander)
  B) implant + (placed/inserted/placement) + (reconstruction OR stage2 hint)
  C) (capsulectomy/capsulotomy) + implant (often stage2 revision/exchange context)
Scheduled logic unchanged except it stays sentence-scoped + guards.

Inputs:
- ./_staging_inputs/HPI11526 Operation Notes.csv
- ./_staging_inputs/HPI11526 Clinic Notes.csv

Output:
- ./_outputs/patient_stage_summary.csv
"""

from __future__ import print_function
import os
import csv
import re
from datetime import datetime

STAGING_DIR = os.path.join(os.getcwd(), "_staging_inputs")
OUT_DIR = os.path.join(os.getcwd(), "_outputs")

# -------------------------
# Helpers
# -------------------------

def _safe_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def _find_input_csvs():
    files = []
    for name in ["HPI11526 Operation Notes.csv", "HPI11526 Clinic Notes.csv"]:
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
# Stage detection logic
# -------------------------

# NOTE TYPE heuristic: treat these as "operative context"
RE_OPERATIVE_TYPE = re.compile(
    r"\b(operative|op note|brief op|operation|surgical|procedure|or note)\b",
    re.I
)

RE_TE = re.compile(r"\b(expander|expanders|tissue expander|te)\b", re.I)
RE_REMOVE = re.compile(r"\b(remove(d|al)?|explant(ed|ation)?|take\s*out|takedown|retrieve)\b", re.I)

# broaden implant device terms (keep conservative, avoid "implant" only as future plan by using operative gating)
RE_IMPLANT = re.compile(
    r"\b(implant(s)?|prosthesis|silicone|saline|gel|mentor|allergan|sientra)\b",
    re.I
)

RE_ACTION = re.compile(r"\b(place(d|ment)?|insert(ed|ion)?|exchange(d)?|exchanged|replace(d|ment)?|replacement)\b", re.I)
RE_EXCH_WORD = re.compile(r"\b(exchange|exchanged|replace|replaced|replacement)\b", re.I)

RE_RECON = re.compile(r"\b(breast reconstruction|reconstruction)\b", re.I)
RE_STAGE2_HINT = re.compile(r"\b(second stage|stage\s*2)\b", re.I)

RE_CAPSULE = re.compile(r"\b(capsulectomy|capsulotomy)\b", re.I)

# Performed exchange (tight)
RE_EXCHANGE_TIGHT = re.compile(
    r"\b(implant|expander)\b.{0,50}\b(exchange|exchanged|replace|replaced|replacement)\b"
    r"|\b(exchange|exchanged|replace|replaced|replacement)\b.{0,50}\b(implant|expander)\b",
    re.I
)

# Scheduled (same as your tight version)
RE_SCHEDULE = re.compile(r"\b(schedule(d)?|planned|plan)\b", re.I)
RE_SCHEDULED_FOR = re.compile(r"\bscheduled\b.{0,12}\bfor\b", re.I)
RE_PROC_CUE = re.compile(r"\b(surgery|procedure|operation|or|operative)\b", re.I)

RE_NOT_SCHEDULED = re.compile(
    r"\b(not|no|never)\s+(scheduled|plan(ned)?|planning)\b|\bno plans\b|\bnot planning\b",
    re.I
)

RE_COUNSEL_ONLY = re.compile(
    r"\b(discuss(ed|ion)?|consider(ing)?|option(s)?|candidate|counsel(ing)?|risks? and benefits|review(ed)?)\b",
    re.I
)

RE_BAD_SCHED_CONTEXT = re.compile(
    r"\b(follow[- ]?up|f/u|clinic|appt|appointment|visit|pt|ot|imaging|mri|ct|us|ultrasound|mammo|labs?)\b",
    re.I
)

RE_STAGE2_PROC_PHRASE = re.compile(
    r"\b(expander[- ]?to[- ]?implant)\b"
    r"|\b(exchange)\b.{0,30}\b(expander|implant)\b"
    r"|\b(expander)\b.{0,30}\b(exchange)\b"
    r"|\b(second stage|stage\s*2)\b.{0,40}\b(reconstruction|exchange)\b"
    r"|\b(expander)\b.{0,40}\b(remove|removal|explant)\b.{0,40}\b(implant)\b",
    re.I
)

STAGE1_PATTERNS = [
    r"\b(mastectomy)\b.*\b(tissue expander|expanders|te)\b.*\b(place|placement|insert|insertion)\b",
    r"\b(place|placement|insert|insertion)\b.*\b(tissue expander|expanders|te)\b",
    r"\b(first stage)\b.*\b(reconstruction|tissue expander|expander|te)\b",
    r"\b(stage\s*1)\b.*\b(reconstruction|tissue expander|expander|te)\b",
]

def _scheduled_stage2_sentence_level(text_norm, proximity=50):
    if not RE_SCHEDULE.search(text_norm):
        return False

    parts = re.split(r"[.;]\s+|\n+", text_norm)
    for sent in parts:
        s = sent.strip()
        if not s:
            continue
        if not RE_SCHEDULE.search(s):
            continue
        if RE_NOT_SCHEDULED.search(s):
            continue
        if RE_BAD_SCHED_CONTEXT.search(s):
            continue
        if RE_COUNSEL_ONLY.search(s):
            continue

        m_proc = RE_STAGE2_PROC_PHRASE.search(s)
        if not m_proc:
            continue

        sched_positions = [m.start() for m in RE_SCHEDULE.finditer(s)]
        proc_start = m_proc.start()
        proc_end = m_proc.end()

        close = False
        for sp in sched_positions:
            if abs(sp - proc_start) <= proximity or abs(sp - proc_end) <= proximity:
                close = True
                break
        if not close:
            continue

        if not (RE_SCHEDULED_FOR.search(s) or RE_PROC_CUE.search(s)):
            continue

        return True

    return False

def _stage2_bucket(text_norm, note_type_norm):
    is_operative = bool(RE_OPERATIVE_TYPE.search(note_type_norm))

    # 1) Performed exchange (tight)
    if RE_EXCHANGE_TIGHT.search(text_norm):
        return True, "EXCHANGE_TIGHT"

    # 2) Performed TE removal + implant + action (classic)
    if RE_TE.search(text_norm) and RE_REMOVE.search(text_norm) and RE_IMPLANT.search(text_norm) and RE_ACTION.search(text_norm):
        return True, "EXPANDER_TO_IMPLANT"

    # 3) NEW (operative-only): implant + exchange/replace (no expander required)
    if is_operative and RE_IMPLANT.search(text_norm) and RE_EXCH_WORD.search(text_norm):
        return True, "OPONLY_IMPLANT_EXCHANGE"

    # 4) NEW (operative-only): implant placement + reconstruction/stage2 hint
    if is_operative and RE_IMPLANT.search(text_norm) and re.search(r"\b(place(d|ment)?|insert(ed|ion)?)\b", text_norm, re.I) and (RE_RECON.search(text_norm) or RE_STAGE2_HINT.search(text_norm)):
        return True, "OPONLY_IMPLANT_PLACEMENT_RECON"

    # 5) NEW (operative-only): capsule work + implant (common revision/exchange patterns)
    if is_operative and RE_CAPSULE.search(text_norm) and RE_IMPLANT.search(text_norm):
        return True, "OPONLY_CAPSULE_PLUS_IMPLANT"

    # 6) Scheduled Stage2 (tight, sentence scoped)
    if _scheduled_stage2_sentence_level(text_norm, proximity=50):
        return True, "SCHEDULED_STAGE2_TIGHT"

    return False, ""

def detect_stage(note_text, note_type):
    t = _normalize_text(note_text)
    nt = _normalize_text(note_type)

    ok2, pat2 = _stage2_bucket(t, nt)
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

    rows = []
    for p in input_csvs:
        rows.extend(read_rows(p))

    patients = {}

    for r in rows:
        pid = (r.get("ENCRYPTED_PAT_ID") or "").strip()
        if not pid:
            continue

        date = _best_note_date(r)
        text = r.get("NOTE_TEXT", "")
        note_type = r.get("NOTE_TYPE", "")

        stage, _ = detect_stage(text, note_type)
        if not stage:
            continue

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
