#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_FINAL_FINAL.py

Stage2 now REQUIRES:
  (exchange/remove + implant language)
AND
  (operative/performed context evidence)

Planning-only notes are excluded.
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
        raise IOError("No CSV found in staging dir")
    return candidates[0]


def _normalize_text(s):
    if s is None:
        return ""
    s = s.replace("\r", "\n").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_date_any(s):
    if not s:
        return ""
    s = str(s).strip()
    fmts = [
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y",
        "%Y/%m/%d", "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except:
            pass
    return ""


def _best_note_date(meta):
    d = _parse_date_any(meta.get("OPERATION_DATE"))
    if d:
        return d
    return _parse_date_any(meta.get("NOTE_DATE_OF_SERVICE"))


# -------- CONTEXT FILTERS --------

RE_OP_CONTEXT = re.compile(
    r"\b(operative report|op note|procedure note|anesthesia|ebl|"
    r"specimen|drain|jp drain|intraoperative|findings|disposition|to pacu)\b",
    re.I
)

RE_DONE = re.compile(
    r"\b(underwent|was performed|performed|s/p|status post|post[- ]?op|pod)\b",
    re.I
)

RE_PLAN = re.compile(
    r"\b(plan|planned|planning|will|schedule|discussed|candidate for|consent)\b",
    re.I
)

RE_TE = re.compile(r"\b(tissue expander|expanders|\bte\b)\b", re.I)
RE_REMOVE = re.compile(r"\b(remov|explant|take out)\w*\b", re.I)
RE_IMPLANT = re.compile(r"\bimplant(s)?\b", re.I)
RE_EXCHANGE = re.compile(r"\b(exchange|replace|replacement)\b", re.I)


def has_op_context(t):
    return bool(RE_OP_CONTEXT.search(t) or RE_DONE.search(t))


def planning_only(t):
    if RE_PLAN.search(t) and not has_op_context(t):
        return True
    return False


def detect_stage(note_text):
    t = _normalize_text(note_text)

    # ---- STAGE2 ----
    if not planning_only(t):

        if has_op_context(t):

            # Exchange language
            if RE_EXCHANGE.search(t) and RE_IMPLANT.search(t):
                return "STAGE2"

            # TE removed + implant placed
            if RE_TE.search(t) and RE_REMOVE.search(t) and RE_IMPLANT.search(t):
                return "STAGE2"

    # ---- STAGE1 ----
    if re.search(r"\bmastectomy\b.*\b(tissue expander|expanders|\bte\b)\b", t):
        return "STAGE1"

    if re.search(r"\b(place|placement|insert)\b.*\b(tissue expander|expanders|\bte\b)\b", t):
        return "STAGE1"

    return ""


def read_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
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
                "NOTE_TYPE": r.get("NOTE_TYPE", ""),
                "OPERATION_DATE": r.get("OPERATION_DATE", ""),
                "NOTE_DATE_OF_SERVICE": r.get("NOTE_DATE_OF_SERVICE", ""),
                "_lines": []
            }
        notes[key]["_lines"].append((int(r.get("LINE") or 0), r.get("NOTE_TEXT") or ""))

    for key in notes:
        lines = sorted(notes[key]["_lines"], key=lambda x: x[0])
        notes[key]["NOTE_TEXT_FULL"] = "\n".join([t for _, t in lines])
        del notes[key]["_lines"]

    return notes


def build():
    _safe_mkdir(OUT_DIR)
    input_csv = _pick_input_csv()
    rows = read_rows(input_csv)
    notes = aggregate_notes(rows)

    patients = {}

    for (pid, note_id), meta in notes.items():
        stage = detect_stage(meta["NOTE_TEXT_FULL"])
        if not stage:
            continue

        date = _best_note_date(meta)

        if pid not in patients:
            patients[pid] = {
                "STAGE1_DATE": "",
                "STAGE2_DATE": ""
            }

        if stage == "STAGE1":
            if not patients[pid]["STAGE1_DATE"] or date < patients[pid]["STAGE1_DATE"]:
                patients[pid]["STAGE1_DATE"] = date

        if stage == "STAGE2":
            if not patients[pid]["STAGE2_DATE"] or date < patients[pid]["STAGE2_DATE"]:
                patients[pid]["STAGE2_DATE"] = date

    out = os.path.join(OUT_DIR, "patient_stage_summary_FINAL_FINAL.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ENCRYPTED_PAT_ID", "STAGE1_DATE", "STAGE2_DATE", "HAS_STAGE1", "HAS_STAGE2"])
        for pid in sorted(patients.keys()):
            s1 = patients[pid]["STAGE1_DATE"]
            s2 = patients[pid]["STAGE2_DATE"]
            w.writerow([pid, s1, s2, 1 if s1 else 0, 1 if s2 else 0])

    print("OK:", out)


if __name__ == "__main__":
    build()
          
