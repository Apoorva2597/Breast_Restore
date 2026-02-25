#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_FINAL.py (Python 3.6.8 compatible) — revised to INCLUDE "scheduled" stage-2 signals.

Key change:
- We NO LONGER suppress detection just because the note contains words like "scheduled/planned/will".
- We ADD a "scheduled Stage 2" bucket that triggers when scheduling language co-occurs with clear Stage 2 procedure intent.
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

def _best_note_date(row):
    op = _parse_date_any(row.get("OPERATION_DATE", ""))
    if op:
        return op
    dos = _parse_date_any(row.get("NOTE_DATE_OF_SERVICE", ""))
    return dos

def _make_snippet(text_norm, start, end, width=200):
    lo = max(0, start - width)
    hi = min(len(text_norm), end + width)
    return text_norm[lo:hi].strip()

# -------------------------
# Stage detection
# -------------------------

# Core vocab
RE_TE = re.compile(r"\b(expander|expanders|tissue expander|te)\b", re.I)
RE_REMOVE = re.compile(r"\b(remove(d|al)?|explant(ed|ation)?|take\s*out|takedown|retrieve)\b", re.I)
RE_IMPLANT = re.compile(r"\b(implant(s)?|prosthesis|gel implant|silicone implant|saline implant|mentor|allergan|sientra)\b", re.I)

# Action verbs (helps avoid pure “device mention”)
RE_ACTION = re.compile(r"\b(place(d|ment)?|insert(ed|ion)?|implant(ed|ation)?|exchange(d)?|replace(d|ment)?)\b", re.I)

# Exchange bucket (performed)
RE_EXCHANGE = re.compile(
    r"\b(implant|expander)\b.*\b(exchange|exchang(e|ed)|replace|replaced|replacement)\b"
    r"|\b(exchange|exchang(e|ed)|replace|replaced|replacement)\b.*\b(implant|expander)\b",
    re.I
)

# Scheduled / planned bucket (NEW)
RE_SCHEDULE = re.compile(r"\b(schedule(d)?|planned|plan|will|to be|anticipat(ed|e))\b", re.I)
RE_SURG_WORDS = re.compile(r"\b(surgery|surgical|procedure|operation|or)\b", re.I)
RE_STAGE2_INTENT = re.compile(
    r"\b(expander[- ]?to[- ]?implant)\b"
    r"|\b(second stage|stage\s*2)\b.*\b(reconstruction|exchange|implant)\b"
    r"|\b(exchange|exchang(e|ed)|replace|replaced|replacement)\b"
    r"|\bimplant(s)?\b",
    re.I
)

# Historical-only guard (keep this conservative; DO NOT block scheduled/planned)
RE_HISTORY_ONLY = re.compile(r"\b(history of|hx of|previously|prior)\b", re.I)

STAGE2_HINT_PATTERNS = [
    r"\b(expander[- ]?to[- ]?implant)\b",
    r"\b(second stage)\b.*\b(reconstruction|exchange|implant)\b",
    r"\b(stage\s*2)\b.*\b(reconstruction|exchange|implant)\b",
]

STAGE1_PATTERNS = [
    r"\b(mastectomy|nipple[- ]?sparing mastectomy|skin[- ]?sparing mastectomy)\b.*\b(tissue expander|expanders|te)\b.*\b(place|placement|insert|insertion)\b",
    r"\b(place|placement|insert|insertion)\b.*\b(tissue expander|expanders|te)\b",
    r"\b(first stage)\b.*\b(reconstruction|tissue expander|expander|te)\b",
    r"\b(stage\s*1)\b.*\b(reconstruction|tissue expander|expander|te)\b",
]

def _stage2_bucket(text_norm):
    """
    Stage 2 triggers in this order:
      1) Performed exchange/replace language (implant/expander exchange)
      2) Performed expander removal + implant + action verb
      3) Scheduled/planned Stage 2 intent (NEW)
      4) Conservative explicit hints
    Returns: (True/False, matched_pattern_string, evidence_snippet)
    """

    # 1) Performed exchange
    m = RE_EXCHANGE.search(text_norm)
    if m:
        return True, "EXCHANGE: (implant/expander) + (exchange|replace|replacement)", _make_snippet(text_norm, m.start(), m.end())

    # 2) Performed TE removal + implant + action (action can be anywhere)
    if RE_TE.search(text_norm) and RE_REMOVE.search(text_norm) and RE_IMPLANT.search(text_norm) and RE_ACTION.search(text_norm):
        m2 = RE_REMOVE.search(text_norm) or RE_ACTION.search(text_norm)
        if m2:
            return True, "EXPANDER->IMPLANT: TE + remove/explant/take out + implant + action", _make_snippet(text_norm, m2.start(), m2.end())
        return True, "EXPANDER->IMPLANT: TE + remove/explant/take out + implant + action", _make_snippet(text_norm, 0, min(len(text_norm), 80))

    # 3) Scheduled/planned Stage 2 intent (NEW)
    # Require scheduling language + clear Stage2 intent, and some “procedure/surgery” cue OR TE mention.
    if RE_SCHEDULE.search(text_norm) and RE_STAGE2_INTENT.search(text_norm) and (RE_SURG_WORDS.search(text_norm) or RE_TE.search(text_norm)):
        # Avoid pure historical mentions (e.g., “history of scheduled ...”)
        if not RE_HISTORY_ONLY.search(text_norm):
            ms = RE_SCHEDULE.search(text_norm)
            return True, "SCHEDULED_STAGE2: scheduled/planned + stage2 intent", _make_snippet(text_norm, ms.start(), ms.end())

    # 4) Explicit hints
    for pat in STAGE2_HINT_PATTERNS:
        mh = re.search(pat, text_norm)
        if mh:
            return True, pat, _make_snippet(text_norm, mh.start(), mh.end())

    return False, "", ""

def detect_stage(note_text):
    t = _normalize_text(note_text)
    ok2, pat2, snip2 = _stage2_bucket(t)
    if ok2:
        return "STAGE2", pat2, snip2

    for pat in STAGE1_PATTERNS:
        m1 = re.search(pat, t)
        if m1:
            return "STAGE1", pat, _make_snippet(t, m1.start(), m1.end())

    return "", "", ""

# -------------------------
# Main
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

def build(outputs_dir, input_csv):
    _safe_mkdir(outputs_dir)
    rows = read_rows(input_csv)

    patients = {}
    events = []

    for r in rows:
        pid = (r.get("ENCRYPTED_PAT_ID") or "").strip()
        if not pid:
            continue

        date = _best_note_date(r)
        note_id = (r.get("NOTE_ID") or "").strip()
        note_type = (r.get("NOTE_TYPE") or "").strip()
        text = r.get("NOTE_TEXT", "")

        stage, pat, snippet = detect_stage(text)
        if not stage:
            continue

        events.append({
            "ENCRYPTED_PAT_ID": pid,
            "STAGE": stage,
            "EVENT_DATE": date,
            "NOTE_ID": note_id,
            "NOTE_TYPE": note_type,
            "MATCH_PATTERN": pat,
            "EVIDENCE_SNIPPET": snippet
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

    event_out = os.path.join(outputs_dir, "stage_event_level.csv")
    with open(event_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID", "STAGE", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE", "MATCH_PATTERN", "EVIDENCE_SNIPPET"
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

def main():
    if not os.path.isdir(STAGING_DIR):
        raise IOError("Staging dir not found: {0}".format(STAGING_DIR))
    input_csv = _pick_input_csv()
    build(OUT_DIR, input_csv)

if __name__ == "__main__":
    main()
