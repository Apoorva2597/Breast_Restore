#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_stage2_anchor_with_audit.py (REFINED)

Key Improvements:
- EXCHANGE_TIGHT requires intraoperative signals
- Strict action + object pairing
- Planning / counseling suppression
- Historical-only suppression
- Output file names unchanged
"""

from __future__ import print_function
import os
import csv
import re
import sys
import random
from datetime import datetime

# -------------------------
# Config
# -------------------------

STAGING_DIR = os.path.join(os.getcwd(), "_staging_inputs")
OUT_DIR = os.path.join(os.getcwd(), "_outputs")
DEFAULT_GOLD = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
GOLD_STAGE2_COL = "Stage2_Applicable"

# -------------------------
# Regex Improvements
# -------------------------

RE_OPERATIVE_TYPE = re.compile(r"\b(operative|op note|brief op|operation|surgical)\b", re.I)

INTRAOP_SIGNALS = re.compile(
    r"\b(estimated blood loss|ebl|drains? placed|specimens? removed|anesthesia|operating room|intraoperative|incision was made|pre-op|post-op)\b",
    re.I,
)

RE_TE = re.compile(r"\b(expander|tissue expander|te)\b", re.I)
RE_IMPLANT = re.compile(r"\b(implant|prosthesis|silicone|saline|gel|mentor|allergan|sientra)\b", re.I)

RE_ACTION_OBJECT = re.compile(
    r"\b(remove(d)?|explant(ed)?|exchange(d)?|replace(d)?|revision)\b.{0,80}?\b(implant|expander)\b",
    re.I,
)

RE_EXCHANGE_TIGHT = re.compile(
    r"\b(implant|expander)\b.{0,50}\b(exchange|exchanged|replace|replaced|replacement)\b"
    r"|\b(exchange|exchanged|replace|replaced|replacement)\b.{0,50}\b(implant|expander)\b",
    re.I,
)

RE_PLANNING = re.compile(
    r"\b(possible|discuss(ed|ion)?|candidate for|plan(ned)?|considering|may need|would like|risk of)\b",
    re.I,
)

RE_HISTORY = re.compile(
    r"\b(status post|s/p|history of|previously|prior to|in \d{4})\b",
    re.I,
)

RE_REMOVE = re.compile(r"\b(remove(d)?|explant(ed)?|takedown|retrieve)\b", re.I)

RE_STAGE1 = [
    r"\b(mastectomy)\b.*\b(expander|te)\b.*\b(place|insert)\b",
    r"\b(place|insert)\b.*\b(expander|te)\b",
    r"\b(stage\s*1)\b.*\b(reconstruction|expander|te)\b",
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
    s = str(s).replace("\r", "\n").lower()
    return re.sub(r"\s+", " ", s).strip()

def _truthy(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in ("1", "y", "yes", "true", "t"):
        return 1
    return 0

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

    # ---- Strict EXCHANGE_TIGHT ----
    if is_op and RE_EXCHANGE_TIGHT.search(text):
        if INTRAOP_SIGNALS.search(text):
            if not RE_PLANNING.search(text) and not RE_HISTORY.search(text):
                return ("STAGE2", "EXCHANGE_TIGHT", 1)

    # ---- Expander → Implant ----
    if is_op and RE_TE.search(text) and RE_IMPLANT.search(text):
        if RE_ACTION_OBJECT.search(text):
            if INTRAOP_SIGNALS.search(text):
                if not RE_HISTORY.search(text):
                    return ("STAGE2", "EXPANDER_TO_IMPLANT", 1)

    # ---- Stage1 ----
    for pat in RE_STAGE1:
        if re.search(pat, text):
            return ("STAGE1", "STAGE1", 0)

    return ("", "", 0)

# -------------------------
# Build + Validate
# -------------------------

def build_and_validate(note_paths, gold_path, out_dir):

    _safe_mkdir(out_dir)

    note_rows = []
    for p in note_paths:
        note_rows.extend(_read_csv_rows(p))

    gold_rows = _read_csv_rows(gold_path)

    gold_stage2 = {}
    for r in gold_rows:
        pid = str(r.get("ENCRYPTED_PAT_ID", "")).strip()
        gold_stage2[pid] = _truthy(r.get(GOLD_STAGE2_COL))

    patients = {}
    events = []

    for r in note_rows:

        pid = str(r.get("ENCRYPTED_PAT_ID", "")).strip()
        if not pid:
            continue

        stage, bucket, isop = detect_stage(
            r.get("NOTE_TEXT", ""),
            r.get("NOTE_TYPE", ""),
        )

        if not stage:
            continue

        if pid not in patients:
            patients[pid] = {"stage1":0, "stage2":0}

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

    # -------------------------
    # Write Outputs (unchanged names)
    # -------------------------

    event_out = os.path.join(out_dir, "stage_event_level.csv")
    with open(event_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ENCRYPTED_PAT_ID","NOTE_ID","NOTE_TYPE","STAGE","DETECTION_BUCKET"])
        w.writeheader()
        for e in events:
            w.writerow(e)

    patient_out = os.path.join(out_dir, "patient_stage_summary.csv")
    with open(patient_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ENCRYPTED_PAT_ID","HAS_STAGE1","HAS_STAGE2"])
        w.writeheader()
        for pid in patients:
            w.writerow({
                "ENCRYPTED_PAT_ID": pid,
                "HAS_STAGE1": 1 if patients[pid]["stage1"]>0 else 0,
                "HAS_STAGE2": 1 if patients[pid]["stage2"]>0 else 0,
            })

    # -------------------------
    # Validation
    # -------------------------

    TP=FP=FN=TN=0
    all_pids = set(gold_stage2.keys()) | set(patients.keys())

    for pid in all_pids:
        gold = gold_stage2.get(pid,0)
        pred = 1 if (pid in patients and patients[pid]["stage2"]>0) else 0

        if pred==1 and gold==1: TP+=1
        elif pred==1 and gold==0: FP+=1
        elif pred==0 and gold==1: FN+=1
        else: TN+=1

    print("Stage2 Anchor:")
    print("TP={0} FP={1} FN={2} TN={3}".format(TP,FP,FN,TN))
    print("Precision={0:.3f} Recall={1:.3f}".format(
        float(TP)/(TP+FP) if TP+FP else 0,
        float(TP)/(TP+FN) if TP+FN else 0
    ))

# -------------------------
# Main
# -------------------------

def main():

    note_paths = [
        os.path.join(STAGING_DIR, "HPI11526 Operation Notes.csv"),
        os.path.join(STAGING_DIR, "HPI11526 Clinic Notes.csv"),
    ]

    note_paths = [p for p in note_paths if os.path.isfile(p)]

    if not note_paths:
        raise IOError("No note files found.")

    build_and_validate(note_paths, DEFAULT_GOLD, OUT_DIR)

if __name__ == "__main__":
    main()
