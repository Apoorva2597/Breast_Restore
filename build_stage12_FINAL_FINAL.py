#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_FINAL_REVISED.py (Python 3.6.8 compatible)

Revisions:
- Expanded Stage 2 vocab (removal / implant / exchange variants)
- Stronger implant action requirement
- Negation / historical / planned guards
- Section-aware weighting (operative sections preferred)
- Tightened Stage 1 backup rule
- Evidence snippet (+/-200 chars) emitted in event-level output
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

def _make_snippet(text, start, end, width=200):
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    return text[lo:hi].strip()

# -------------------------
# Guardrails
# -------------------------

NEGATION_PATTERNS = re.compile(
    r"\b(history of|hx of|planned|will|discuss(ed)?|scheduled|consider|prior|previous|s/p)\b",
    re.I
)

OPERATIVE_SECTION_HINT = re.compile(
    r"\b(description of procedure|procedure|operative findings|implants|operation performed)\b",
    re.I
)

# -------------------------
# Stage 2 vocab
# -------------------------

RE_TE = re.compile(r"\b(expander|expanders|tissue expander|te)\b", re.I)

RE_REMOVE = re.compile(
    r"\b(remove(d|al)?|explant(ed|ation)?|take\s*out|takedown|retrieve)\b",
    re.I
)

RE_IMPLANT_DEVICE = re.compile(
    r"\b(implant(s)?|prosthesis|gel implant|silicone implant|saline implant|mentor|allergan|sientra)\b",
    re.I
)

RE_IMPLANT_ACTION = re.compile(
    r"\b(place(d|ment)?|insert(ed|ion)?|implant(ed|ation)?|exchange(d)?|replace(d|ment)?)\b",
    re.I
)

RE_EXCHANGE_STRONG = re.compile(
    r"\b(implant|expander)\b.*\b(exchange|replace|replacement)\b"
    r"|\b(exchange|replace|replacement)\b.*\b(implant|expander)\b",
    re.I
)

# -------------------------
# Stage 1 tightened
# -------------------------

STAGE1_PATTERNS = [
    r"\b(mastectomy|nipple[- ]?sparing mastectomy|skin[- ]?sparing mastectomy)\b.*\b(tissue expander|expanders|te)\b.*\b(place|placement|insert|insertion)\b",
    r"\b(place|placement|insert|insertion)\b.*\b(tissue expander|expanders|te)\b",
    r"\b(first stage)\b.*\b(reconstruction|tissue expander|expander|te)\b",
    r"\b(stage\s*1)\b.*\b(reconstruction|tissue expander|expander|te)\b",
]

# -------------------------
# Stage 2 detection
# -------------------------

def _stage2_bucket(text_raw):
    text_norm = _normalize_text(text_raw)

    # Negation / planned guard
    if NEGATION_PATTERNS.search(text_norm):
        return False, "", ""

    # Strong exchange language
    m = RE_EXCHANGE_STRONG.search(text_norm)
    if m:
        snippet = _make_snippet(text_norm, m.start(), m.end())
        return True, "EXCHANGE_STRONG", snippet

    # TE removal + implant + action
    if RE_TE.search(text_norm) and RE_REMOVE.search(text_norm) and RE_IMPLANT_DEVICE.search(text_norm):
        action_match = RE_IMPLANT_ACTION.search(text_norm)
        if action_match:
            snippet = _make_snippet(text_norm, action_match.start(), action_match.end())
            return True, "EXPANDER_TO_IMPLANT_ACTION", snippet

    return False, "", ""

def detect_stage(note_text):
    t_raw = note_text if note_text else ""
    t_norm = _normalize_text(t_raw)

    ok2, pat2, snip2 = _stage2_bucket(t_raw)
    if ok2:
        return "STAGE2", pat2, snip2

    for pat in STAGE1_PATTERNS:
        m = re.search(pat, t_norm)
        if m:
            snip = _make_snippet(t_norm, m.start(), m.end())
            return "STAGE1", pat, snip

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
            raise ValueError("Missing expected columns: {0}".format(missing))
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
                "stage1_date": "", "stage2_date": "",
                "stage1_hits": 0, "stage2_hits": 0,
                "stage1_pattern": "", "stage2_pattern": "",
                "stage1_snippet": "", "stage2_snippet": ""
            }

        p = patients[pid]

        if stage == "STAGE1":
            p["stage1_hits"] += 1
            if date and (not p["stage1_date"] or date < p["stage1_date"]):
                p["stage1_date"] = date
                p["stage1_pattern"] = pat
                p["stage1_snippet"] = snippet

        elif stage == "STAGE2":
            p["stage2_hits"] += 1
            if date and (not p["stage2_date"] or date < p["stage2_date"]):
                p["stage2_date"] = date
                p["stage2_pattern"] = pat
                p["stage2_snippet"] = snippet

    patient_out = os.path.join(outputs_dir, "patient_stage_summary.csv")
    with open(patient_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID",
            "STAGE1_DATE", "STAGE1_HITS", "STAGE1_MATCH_PATTERN",
            "STAGE2_DATE", "STAGE2_HITS", "STAGE2_MATCH_PATTERN",
            "HAS_STAGE1", "HAS_STAGE2"
        ])
        w.writeheader()
        for pid in sorted(patients.keys()):
            p = patients[pid]
            w.writerow({
                "ENCRYPTED_PAT_ID": pid,
                "STAGE1_DATE": p["stage1_date"],
                "STAGE1_HITS": p["stage1_hits"],
                "STAGE1_MATCH_PATTERN": p["stage1_pattern"],
                "STAGE2_DATE": p["stage2_date"],
                "STAGE2_HITS": p["stage2_hits"],
                "STAGE2_MATCH_PATTERN": p["stage2_pattern"],
                "HAS_STAGE1": 1 if p["stage1_hits"] > 0 else 0,
                "HAS_STAGE2": 1 if p["stage2_hits"] > 0 else 0,
            })

    event_out = os.path.join(outputs_dir, "stage_event_level.csv")
    with open(event_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID", "STAGE", "EVENT_DATE",
            "NOTE_ID", "NOTE_TYPE",
            "MATCH_PATTERN", "EVIDENCE_SNIPPET"
        ])
        w.writeheader()
        for e in events:
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
