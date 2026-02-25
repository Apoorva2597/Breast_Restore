#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_WITH_AUDIT_tunedFP.py (Python 3.6.8 compatible)

Goal: reduce FP while keeping the audit outputs.

Key tuning (FP control):
- OPONLY_IMPLANT_PLACEMENT_RECON now EXCLUDES likely direct-to-implant/immediate mastectomy cases
  unless expander/TE is also mentioned.
- OPONLY_IMPLANT_EXCHANGE now requires a stronger Stage2 anchor:
  (expander/TE OR capsule-work) present somewhere in note.
- Adds "HISTORY/PRIOR" guard at sentence-level for OPONLY + SCHEDULED triggers:
  skips matches in sentences that look like past history rather than performed/current.
- Keeps audit outputs:
  - ./_outputs/patient_stage_summary.csv
  - ./_outputs/stage_event_level.csv

Inputs:
- ./_staging_inputs/HPI11526 Operation Notes.csv
- ./_staging_inputs/HPI11526 Clinic Notes.csv
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

def _make_snippet(text_norm, start, end, width=150):
    lo = max(0, start - width)
    hi = min(len(text_norm), end + width)
    return text_norm[lo:hi].strip()

def _sentences(text_norm):
    # crude but stable for normalized text
    return re.split(r"[.;]\s+|\n+", text_norm)

# -------------------------
# Stage detection logic (tuned)
# -------------------------

RE_OPERATIVE_TYPE = re.compile(
    r"\b(operative|op note|brief op|operation|surgical|procedure|or note)\b",
    re.I
)

RE_TE = re.compile(r"\b(expander|expanders|tissue expander|te)\b", re.I)
RE_REMOVE = re.compile(r"\b(remove(d|al)?|explant(ed|ation)?|take\s*out|takedown|retrieve)\b", re.I)

RE_IMPLANT = re.compile(
    r"\b(implant(s)?|prosthesis|silicone|saline|gel|mentor|allergan|sientra)\b",
    re.I
)

RE_ACTION = re.compile(r"\b(place(d|ment)?|insert(ed|ion)?|exchange(d)?|exchanged|replace(d|ment)?|replacement)\b", re.I)
RE_EXCH_WORD = re.compile(r"\b(exchange|exchanged|replace|replaced|replacement)\b", re.I)

RE_RECON = re.compile(r"\b(breast reconstruction|reconstruction)\b", re.I)
RE_STAGE2_HINT = re.compile(r"\b(second stage|stage\s*2)\b", re.I)
RE_CAPSULE = re.compile(r"\b(capsulectomy|capsulotomy)\b", re.I)

# guard against direct-to-implant (DTI) / immediate implant at mastectomy (common FP)
RE_MASTECTOMY = re.compile(r"\b(mastectomy)\b", re.I)
RE_IMMEDIATE = re.compile(r"\b(immediate|direct[- ]?to[- ]?implant|dti)\b", re.I)

# history / prior guard
RE_HISTORY_GUARD = re.compile(
    r"\b(history of|h/o|prior|previous(ly)?|s\/p|status post|had (an|a)|had (the)?|was|were)\b",
    re.I
)

# Performed exchange (tight)
RE_EXCHANGE_TIGHT = re.compile(
    r"\b(implant|expander)\b.{0,50}\b(exchange|exchanged|replace|replaced|replacement)\b"
    r"|\b(exchange|exchanged|replace|replaced|replacement)\b.{0,50}\b(implant|expander)\b",
    re.I
)

# Scheduled (as before)
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
        return False, "", "", 0, 0

    parts = _sentences(text_norm)
    offset = 0
    for sent in parts:
        s = sent.strip()
        if not s:
            offset += len(sent) + 1
            continue
        if not RE_SCHEDULE.search(s):
            offset += len(sent) + 1
            continue
        if RE_NOT_SCHEDULED.search(s):
            offset += len(sent) + 1
            continue
        if RE_BAD_SCHED_CONTEXT.search(s):
            offset += len(sent) + 1
            continue
        if RE_COUNSEL_ONLY.search(s):
            offset += len(sent) + 1
            continue
        if RE_HISTORY_GUARD.search(s):
            offset += len(sent) + 1
            continue

        m_proc = RE_STAGE2_PROC_PHRASE.search(s)
        if not m_proc:
            offset += len(sent) + 1
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
            offset += len(sent) + 1
            continue

        if not (RE_SCHEDULED_FOR.search(s) or RE_PROC_CUE.search(s)):
            offset += len(sent) + 1
            continue

        return True, "SCHEDULED_STAGE2_TIGHT", "SCHEDULED_SENTENCE", (offset + m_proc.start()), (offset + m_proc.end())

    return False, "", "", 0, 0

def _oponly_implant_exchange_ok(text_norm):
    # tighten: require implant + exchange AND (expander OR capsule-work) somewhere (strong stage2 anchor)
    if not (RE_IMPLANT.search(text_norm) and RE_EXCH_WORD.search(text_norm)):
        return False
    if RE_TE.search(text_norm) or RE_CAPSULE.search(text_norm):
        return True
    return False

def _oponly_implant_placement_recon_ok(text_norm):
    # tighten: block likely DTI/immediate mastectomy implant cases unless expander present
    if not (RE_IMPLANT.search(text_norm) and re.search(r"\b(place(d|ment)?|insert(ed|ion)?)\b", text_norm, re.I)):
        return False
    if not (RE_RECON.search(text_norm) or RE_STAGE2_HINT.search(text_norm)):
        return False

    # if mastectomy + immediate/DTI present AND no expander mention -> likely NOT stage2
    if RE_MASTECTOMY.search(text_norm) and RE_IMMEDIATE.search(text_norm) and (not RE_TE.search(text_norm)):
        return False

    # also require NOT dominated by history language (note-level)
    return True

def _stage2_bucket(text_norm, note_type_norm):
    is_operative = 1 if RE_OPERATIVE_TYPE.search(note_type_norm) else 0

    m = RE_EXCHANGE_TIGHT.search(text_norm)
    if m:
        return True, "EXCHANGE_TIGHT", "EXCHANGE_TIGHT", m.start(), m.end(), is_operative

    if RE_TE.search(text_norm) and RE_REMOVE.search(text_norm) and RE_IMPLANT.search(text_norm) and RE_ACTION.search(text_norm):
        m2 = RE_REMOVE.search(text_norm) or RE_ACTION.search(text_norm)
        st, en = (m2.start(), m2.end()) if m2 else (0, min(len(text_norm), 60))
        return True, "EXPANDER_TO_IMPLANT", "TE+REMOVE+IMPLANT+ACTION", st, en, is_operative

    # OPONLY (tightened)
    if is_operative and _oponly_implant_exchange_ok(text_norm):
        m3 = RE_EXCH_WORD.search(text_norm) or RE_IMPLANT.search(text_norm)
        st, en = (m3.start(), m3.end()) if m3 else (0, min(len(text_norm), 60))
        return True, "OPONLY_IMPLANT_EXCHANGE_TIGHT", "OPONLY_IMPLANT_EXCHANGE_TIGHT", st, en, is_operative

    if is_operative and _oponly_implant_placement_recon_ok(text_norm) and (not RE_HISTORY_GUARD.search(text_norm)):
        m4 = re.search(r"\b(place(d|ment)?|insert(ed|ion)?)\b", text_norm, re.I) or RE_IMPLANT.search(text_norm)
        st, en = (m4.start(), m4.end()) if m4 else (0, min(len(text_norm), 60))
        return True, "OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT", "OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT", st, en, is_operative

    if is_operative and RE_CAPSULE.search(text_norm) and RE_IMPLANT.search(text_norm) and (not RE_HISTORY_GUARD.search(text_norm)):
        m5 = RE_CAPSULE.search(text_norm)
        return True, "OPONLY_CAPSULE_PLUS_IMPLANT", "OPONLY_CAPSULE_PLUS_IMPLANT", m5.start(), m5.end(), is_operative

    ok, bucket, patname, st, en = _scheduled_stage2_sentence_level(text_norm, proximity=50)
    if ok:
        return True, bucket, patname, st, en, is_operative

    return False, "", "", 0, 0, is_operative

def detect_stage(note_text, note_type):
    t = _normalize_text(note_text)
    nt = _normalize_text(note_type)

    ok2, bucket, patname, st, en, isop = _stage2_bucket(t, nt)
    if ok2:
        return "STAGE2", bucket, patname, st, en, isop

    for pat in STAGE1_PATTERNS:
        m = re.search(pat, t)
        if m:
            return "STAGE1", "STAGE1", pat, m.start(), m.end(), 0

    return "", "", "", 0, 0, 0

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
    events = []
    stage2_match_counts = {}

    for r in rows:
        pid = (r.get("ENCRYPTED_PAT_ID") or "").strip()
        if not pid:
            continue

        date = _best_note_date(r)
        note_id = (r.get("NOTE_ID") or "").strip()
        note_type = (r.get("NOTE_TYPE") or "").strip()
        text = r.get("NOTE_TEXT", "")

        stage, bucket, patname, st, en, isop = detect_stage(text, note_type)
        if not stage:
            continue

        tnorm = _normalize_text(text)
        snippet = _make_snippet(tnorm, st, en, width=150) if tnorm else ""

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
            stage2_match_counts[pid] = stage2_match_counts.get(pid, 0) + 1
            if date and (not p["stage2_date"] or date < p["stage2_date"]):
                p["stage2_date"] = date

        events.append({
            "ENCRYPTED_PAT_ID": pid,
            "STAGE": stage,
            "EVENT_DATE": date,
            "NOTE_ID": note_id,
            "NOTE_TYPE": note_type,
            "DETECTION_BUCKET": bucket,
            "PATTERN_NAME": patname,
            "IS_OPERATIVE_CONTEXT": int(isop),
            "EVIDENCE_SNIPPET": snippet
        })

    for pid, p in patients.items():
        p["stage2_match_count"] = stage2_match_counts.get(pid, 0)
        if p["stage1_date"] and p["stage2_date"]:
            p["has_stage1_before_stage2"] = 1 if p["stage1_date"] <= p["stage2_date"] else 0
        else:
            p["has_stage1_before_stage2"] = 0

    patient_out = os.path.join(outputs_dir, "patient_stage_summary.csv")
    with open(patient_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID",
            "STAGE1_DATE", "STAGE1_HITS",
            "STAGE2_DATE", "STAGE2_HITS",
            "HAS_STAGE1", "HAS_STAGE2",
            "HAS_STAGE1_BEFORE_STAGE2",
            "STAGE2_MATCH_COUNT_PER_PATIENT"
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
                "HAS_STAGE1_BEFORE_STAGE2": p["has_stage1_before_stage2"],
                "STAGE2_MATCH_COUNT_PER_PATIENT": p["stage2_match_count"],
            })

    event_out = os.path.join(outputs_dir, "stage_event_level.csv")
    with open(event_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID", "STAGE", "EVENT_DATE",
            "NOTE_ID", "NOTE_TYPE",
            "DETECTION_BUCKET", "PATTERN_NAME",
            "IS_OPERATIVE_CONTEXT",
            "EVIDENCE_SNIPPET"
        ])
        w.writeheader()
        for e in events:
            w.writerow(e)

    print("OK: wrote", patient_out)
    print("OK: wrote", event_out)

def main():
    if not os.path.isdir(STAGING_DIR):
        raise IOError("Staging dir not found: {0}".format(STAGING_DIR))
    input_csvs = _find_input_csvs()
    build(OUT_DIR, input_csvs)

if __name__ == "__main__":
    main()
