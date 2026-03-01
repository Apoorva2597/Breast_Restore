#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE2 STAGING ONLY
- No validation
- Produces:
    1) stage_event_level.csv
    2) patient_stage_summary.csv
    3) audit_bucket_counts.csv
"""

import os
import csv
import re

# -------------------------
# Config
# -------------------------

STAGING_DIR = os.path.join(os.getcwd(), "_staging_inputs")
OUT_DIR = os.path.join(os.getcwd(), "_outputs")

# -------------------------
# Regex
# -------------------------

RE_OPERATIVE_TYPE = re.compile(r"\b(operative|op note|brief op|operation|surgical)\b", re.I)

INTRAOP_SIGNALS = re.compile(
    r"\b(estimated blood loss|ebl|drains? placed|specimens? removed|anesthesia|operating room|intraoperative|incision was made|pre-op|post-op)\b",
    re.I,
)

RE_TE = re.compile(r"\b(expander|tissue expander|te)\b", re.I)
RE_IMPLANT = re.compile(r"\b(implant|prosthesis|silicone|saline|gel|mentor|allergan|sientra)\b", re.I)

RE_EXCHANGE_TIGHT = re.compile(
    r"\b(implant|expander)\b.{0,50}\b(exchange|exchanged|replace|replaced|replacement)\b"
    r"|\b(exchange|exchanged|replace|replaced|replacement)\b.{0,50}\b(implant|expander)\b",
    re.I,
)

RE_ACTION_OBJECT = re.compile(
    r"\b(remove(d)?|explant(ed)?|exchange(d)?|replace(d)?|revision)\b.{0,80}?\b(implant|expander)\b",
    re.I,
)

RE_STAGE1 = [
    r"\b(mastectomy)\b.*\b(expander|te)\b.*\b(place|insert)\b",
    r"\b(place|insert)\b.*\b(expander|te)\b",
]

# -------------------------
# Helpers
# -------------------------

def _safe_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def _normalize_text(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).lower()).strip()

def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

# -------------------------
# Detection
# -------------------------

def detect_stage(note_text, note_type):

    text = _normalize_text(note_text)
    note_type_norm = _normalize_text(note_type)

    is_op = 1 if RE_OPERATIVE_TYPE.search(note_type_norm) else 0

    if is_op and RE_EXCHANGE_TIGHT.search(text) and INTRAOP_SIGNALS.search(text):
        return ("STAGE2", "EXCHANGE_TIGHT")

    if is_op and RE_TE.search(text) and RE_IMPLANT.search(text):
        if RE_ACTION_OBJECT.search(text) and INTRAOP_SIGNALS.search(text):
            return ("STAGE2", "EXPANDER_TO_IMPLANT")

    for pat in RE_STAGE1:
        if re.search(pat, text):
            return ("STAGE1", "STAGE1")

    return ("", "")

# -------------------------
# Main
# -------------------------

def main():

    _safe_mkdir(OUT_DIR)

    note_paths = [
        os.path.join(STAGING_DIR, "HPI11526 Operation Notes.csv"),
        os.path.join(STAGING_DIR, "HPI11526 Clinic Notes.csv"),
    ]

    note_paths = [p for p in note_paths if os.path.isfile(p)]
    if not note_paths:
        raise IOError("No note files found.")

    note_rows = []
    for p in note_paths:
        note_rows.extend(_read_csv_rows(p))

    patients = {}
    events = []
    bucket_counts = {}

    for r in note_rows:

        pid = str(r.get("ENCRYPTED_PAT_ID","")).strip()
        if not pid:
            continue

        stage, bucket = detect_stage(
            r.get("NOTE_TEXT",""),
            r.get("NOTE_TYPE",""),
        )

        if not stage:
            continue

        if pid not in patients:
            patients[pid] = {"stage1":0,"stage2":0}

        if stage == "STAGE1":
            patients[pid]["stage1"] += 1
        elif stage == "STAGE2":
            patients[pid]["stage2"] += 1

        events.append({
            "ENCRYPTED_PAT_ID": pid,
            "NOTE_ID": r.get("NOTE_ID",""),
            "NOTE_TYPE": r.get("NOTE_TYPE",""),
            "STAGE": stage,
            "DETECTION_BUCKET": bucket,
        })

        bucket_counts[bucket] = bucket_counts.get(bucket,0) + 1

    # -------------------------
    # Write Outputs
    # -------------------------

    event_out = os.path.join(OUT_DIR, "stage_event_level.csv")
    with open(event_out,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=["ENCRYPTED_PAT_ID","NOTE_ID","NOTE_TYPE","STAGE","DETECTION_BUCKET"])
        w.writeheader()
        for e in events:
            w.writerow(e)

    patient_out = os.path.join(OUT_DIR,"patient_stage_summary.csv")
    with open(patient_out,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=["ENCRYPTED_PAT_ID","HAS_STAGE1","HAS_STAGE2"])
        w.writeheader()
        for pid in patients:
            w.writerow({
                "ENCRYPTED_PAT_ID": pid,
                "HAS_STAGE1": 1 if patients[pid]["stage1"]>0 else 0,
                "HAS_STAGE2": 1 if patients[pid]["stage2"]>0 else 0,
            })

    audit_out = os.path.join(OUT_DIR,"audit_bucket_counts.csv")
    with open(audit_out,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["DETECTION_BUCKET","count"])
        for k,v in sorted(bucket_counts.items()):
            w.writerow([k,v])

    print("Staging complete.")
    print("Patients:", len(patients))
    print("Events:", len(events))

if __name__ == "__main__":
    main()
