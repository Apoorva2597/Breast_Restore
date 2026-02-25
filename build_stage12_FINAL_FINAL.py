#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_FINAL_FINAL.py  (Python 3.6.8 compatible)

Run from: ~/Breast_Restore
Input:    ./_staging_inputs/HPI11526 Operation Notes.csv (or first CSV in _staging_inputs)
Outputs:  ./_outputs/patient_stage_summary_FINAL_FINAL.csv
          ./_outputs/stage_event_level_FINAL_FINAL.csv

Stage2 logic (FN-focused tweak):
Stage2 = strong exchange/expander->implant signal
AND one of:
  A) performed/operative context (status post / underwent / postoperative / POD / op-note cues), OR
  B) scheduled context near the exchange phrase (scheduled/schedule/scheduled for),
     with an explicit date token in the same local window.

Planning-only w/o date stays excluded.
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

def _safe_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def _pick_input_csv():
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

def _best_note_date(meta):
    op = _parse_date_any(meta.get("OPERATION_DATE", ""))
    if op:
        return op
    dos = _parse_date_any(meta.get("NOTE_DATE_OF_SERVICE", ""))
    return dos

def _safe_int(x, default=0):
    try:
        return int(str(x).strip())
    except Exception:
        return default

# -------------------------
# Context cues
# -------------------------

RE_PERFORMED = re.compile(
    r"\b(underwent|was performed|performed|completed|s/p|status post|post[- ]?op|postoperative|pod)\b",
    re.I
)

RE_OP_NOTE_CUES = re.compile(
    r"\b(operative report|op note|brief op note|procedure note|anesthesia|ebl|estimated blood loss|"
    r"specimen|drain|jp drain|implants?:|intraoperative|findings|complications|disposition|to pacu)\b",
    re.I
)

# Scheduling / planning cues
RE_SCHEDULE = re.compile(
    r"\b(scheduled|schedule|scheduled for|will schedule|plan to|planning to|plans to|will plan|we will|to be done|set up for)\b",
    re.I
)

# Any explicit date token in text
RE_DATE_TOKEN = re.compile(
    r"(\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b)",
    re.I
)

def has_performed_context(t):
    return True if (RE_PERFORMED.search(t) or RE_OP_NOTE_CUES.search(t)) else False

def has_scheduled_with_date(t):
    return True if (RE_SCHEDULE.search(t) and RE_DATE_TOKEN.search(t)) else False

def planning_only(t):
    if RE_SCHEDULE.search(t) and (not has_performed_context(t)) and (not has_scheduled_with_date(t)):
        return True
    return False

# -------------------------
# Stage detection
# -------------------------

RE_TE = re.compile(r"\b(tissue expander|tissue expanders|expander|expanders|\bte\b)\b", re.I)
RE_REMOVE = re.compile(r"\b(remov(e|al|ed)?|explant(ed)?|take out)\b", re.I)
RE_IMPLANT = re.compile(r"\bimplant(s)?\b", re.I)

RE_EXCHANGE_TE_FOR_IMPLANT = re.compile(
    r"\bexchange(d)?\b.{0,120}\b(tissue expander|tissue expanders|expanders|expander|\bte\b)\b.{0,220}\b(for|to)\b.{0,120}\bimplant(s)?\b"
    r"|\b(tissue expander|tissue expanders|expanders|expander|\bte\b)\b.{0,120}\bexchang(e|ed)\b.{0,120}\b(for|to)\b.{0,120}\bimplant(s)?\b",
    re.I
)

RE_IMPLANT_EXCHANGE = re.compile(
    r"\bimplant(s)?\b.*\b(exchange|exchang(e|ed)|replace|replaced|replacement)\b"
    r"|\b(exchange|exchang(e|ed)|replace|replaced|replacement)\b.*\bimplant(s)?\b",
    re.I
)

RE_EXPANDER_TO_IMPLANT = re.compile(r"\bexpander[- ]?to[- ]?implant\b", re.I)

RE_REMOVE_NO_IMPLANT = re.compile(
    r"\b(expander|expanders|tissue expander|tissue expanders|\bte\b)\b.*\b(remov(e|ed|al)?|explant(ed)?|take out)\b"
    r".{0,220}\b(without|no)\b.{0,120}\b(implant|implants)\b",
    re.I
)

# local window helpers
def _near_window(t, span_start, span_end, window=260):
    a = max(0, span_start - window)
    b = min(len(t), span_end + window)
    return t[a:b]

def _stage2_bucket(t):
    if RE_REMOVE_NO_IMPLANT.search(t):
        return False, ""

    # exclude weak planning-only notes globally
    if planning_only(t):
        return False, ""

    signal = ""
    m = None

    m = RE_EXCHANGE_TE_FOR_IMPLANT.search(t)
    if m:
        signal = "EXCHANGE: exchange (TE) for/to implant"
    else:
        m = RE_IMPLANT_EXCHANGE.search(t)
        if m:
            signal = "EXCHANGE: implant + (exchange|replace|replacement)"
        else:
            m = RE_EXPANDER_TO_IMPLANT.search(t)
            if m:
                signal = "PHRASE: expander-to-implant"
            else:
                if (RE_TE.search(t) and RE_REMOVE.search(t) and RE_IMPLANT.search(t)):
                    signal = "EXPANDER->IMPLANT: (TE) + (remove/explant/take out) + implant"

    if not signal:
        return False, ""

    # Require performed OR scheduled-with-date, but evaluate near the matched phrase first (better for FN capture)
    local_ctx = ""
    if m is not None:
        local_ctx = _near_window(t, m.start(), m.end(), window=320)

    if local_ctx:
        if has_performed_context(local_ctx) or has_scheduled_with_date(local_ctx):
            return True, signal

    if has_performed_context(t) or has_scheduled_with_date(t):
        return True, signal

    return False, ""

STAGE1_PATTERNS = [
    r"\b(mastectomy|nipple[- ]?sparing mastectomy|skin[- ]?sparing mastectomy)\b.*\b(tissue expander|expanders|\bte\b)\b.*\b(place|placement|insert|insertion)\b",
    r"\b(place|placement|insert|insertion)\b.*\b(tissue expander|expanders|\bte\b)\b",
    r"\b(first stage)\b.*\b(reconstruction|tissue expander|expander|\bte\b)\b",
    r"\b(stage\s*1)\b.*\b(reconstruction|tissue expander|expander|\bte\b)\b",
    r"\b(mastectomy|nipple[- ]?sparing mastectomy|skin[- ]?sparing mastectomy)\b",
]

def detect_stage(note_text):
    t = _normalize_text(note_text)

    ok2, pat2 = _stage2_bucket(t)
    if ok2:
        return "STAGE2", pat2

    for pat in STAGE1_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return "STAGE1", pat

    return "", ""

# -------------------------
# Read + aggregate NOTE_TEXT per NOTE_ID
# -------------------------

def read_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        hdr = reader.fieldnames or []
        missing = [c for c in EXPECTED_COLS if c not in hdr]
        if missing:
            raise ValueError("Missing expected columns: {0}. Found: {1}".format(missing, hdr))
        for r in reader:
            rows.append(r)
    return rows

def aggregate_notes(rows):
    notes = {}
    for r in rows:
        pid = (r.get("ENCRYPTED_PAT_ID") or "").strip()
        note_id = (r.get("NOTE_ID") or "").strip()
        if not pid or not note_id:
            continue
        key = (pid, note_id)
        if key not in notes:
            notes[key] = {
                "ENCRYPTED_PAT_ID": pid,
                "NOTE_ID": note_id,
                "NOTE_TYPE": (r.get("NOTE_TYPE") or "").strip(),
                "MRN": (r.get("MRN") or "").strip(),
                "OPERATION_DATE": (r.get("OPERATION_DATE") or "").strip(),
                "NOTE_DATE_OF_SERVICE": (r.get("NOTE_DATE_OF_SERVICE") or "").strip(),
                "_lines": []
            }
        line_no = _safe_int(r.get("LINE", ""), default=0)
        txt = r.get("NOTE_TEXT", "")
        notes[key]["_lines"].append((line_no, "" if txt is None else str(txt)))

    for key in list(notes.keys()):
        lines = notes[key]["_lines"]
        lines.sort(key=lambda x: x[0])
        notes[key]["NOTE_TEXT_FULL"] = "\n".join([t for _, t in lines])
        del notes[key]["_lines"]

    return notes

# -------------------------
# Main build
# -------------------------

def build(outputs_dir, input_csv):
    _safe_mkdir(outputs_dir)

    rows = read_rows(input_csv)
    notes = aggregate_notes(rows)

    patients = {}
    events = []

    for (pid, note_id), meta in notes.items():
        text_full = meta.get("NOTE_TEXT_FULL", "")
        stage, pat = detect_stage(text_full)
        if not stage:
            continue

        date = _best_note_date(meta)
        note_type = meta.get("NOTE_TYPE", "")

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
            if date:
                if (not p["stage1_date"]) or (date < p["stage1_date"]):
                    p["stage1_date"] = date
                    p["stage1_note_id"] = note_id
                    p["stage1_note_type"] = note_type
                    p["stage1_pattern"] = pat
            else:
                if not p["stage1_note_id"]:
                    p["stage1_note_id"] = note_id
                    p["stage1_note_type"] = note_type
                    p["stage1_pattern"] = pat

        elif stage == "STAGE2":
            p["stage2_hits"] += 1
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

    patient_out = os.path.join(outputs_dir, "patient_stage_summary_FINAL_FINAL.csv")
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

    event_out = os.path.join(outputs_dir, "stage_event_level_FINAL_FINAL.csv")
    with open(event_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID", "STAGE", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE", "MATCH_PATTERN"
        ])
        w.writeheader()

        def _key(e):
            d = e.get("EVENT_DATE") or ""
            return (e.get("ENCRYPTED_PAT_ID") or "", d, e.get("STAGE") or "")

        for e in sorted(events, key=_key):
            w.writerow(e)

    print("OK: input:", input_csv)
    print("OK: outputs:", patient_out)
    print("OK: outputs:", event_out)
    print("Patients with any stage signal:", len(patients))
    print("Patients with Stage 1 evidence:", sum(1 for pid in patients if (patients[pid]["stage1_hits"] > 0 or patients[pid]["stage1_note_id"])))
    print("Patients with Stage 2 evidence:", sum(1 for pid in patients if (patients[pid]["stage2_hits"] > 0 or patients[pid]["stage2_note_id"])))

def main():
    if not os.path.isdir(STAGING_DIR):
        raise IOError("Staging dir not found: {0}".format(STAGING_DIR))
    input_csv = _pick_input_csv()
    build(OUT_DIR, input_csv)

if __name__ == "__main__":
    main()
