#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage2_anchor_with_audit.py (Python 3.6.8 compatible)

- Uses gold label: Stage2_Applicable
- Produces:
  _outputs/stage_event_level.csv
  _outputs/patient_stage_summary.csv
  _outputs/validation_metrics.txt
  _outputs/validation_mismatches.csv
  _outputs/audit_bucket_summary.csv
  _outputs/audit_fp_by_bucket.csv
  _outputs/audit_bucket_noteType_breakdown.csv
  _outputs/audit_fp_events_sample.csv

Defaults:
- Notes: ./_staging_inputs/HPI11526 Operation Notes.csv and ./_staging_inputs/HPI11526 Clinic Notes.csv (if present)
- Gold:  /home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv
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

GOLD_STAGE2_COL = "Stage2_Applicable"  # <-- confirmed by you

# -------------------------
# Helpers
# -------------------------

def _safe_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def _normalize_text(s):
    if s is None:
        return ""
    s = str(s).replace("\r", "\n").lower()
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
    # Accept multiple possible column names
    for k in ["OPERATION_DATE", "NOTE_DATE_OF_SERVICE", "EVENT_DATE", "NOTE_DATE", "DATE_OF_SERVICE"]:
        d = _parse_date_any(row.get(k, ""))
        if d:
            return d
    return ""

def _make_snippet(text_norm, start, end, width=160):
    if not text_norm:
        return ""
    lo = max(0, start - width)
    hi = min(len(text_norm), end + width)
    return text_norm[lo:hi].strip()

def _truthy(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in ("1", "y", "yes", "true", "t"):
        return 1
    if s in ("0", "n", "no", "false", "f", ""):
        return 0
    # numeric fallback
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0

def _read_csv_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def _find_default_note_csvs():
    files = []
    for name in ["HPI11526 Operation Notes.csv", "HPI11526 Clinic Notes.csv"]:
        p = os.path.join(STAGING_DIR, name)
        if os.path.isfile(p):
            files.append(p)
    if not files:
        raise IOError("No default note CSVs found in: {0}. Provide --notes <csv1,csv2,...>".format(STAGING_DIR))
    return files

def _detect_id_col(cols, preferred):
    # preferred: list of likely names in order
    colset = set(cols)
    for c in preferred:
        if c in colset:
            return c
    # heuristic: any column containing these tokens
    for c in cols:
        cl = c.lower()
        if "encrypted" in cl and "pat" in cl:
            return c
    for c in cols:
        cl = c.lower()
        if cl in ("patientid", "patient_id", "pat_id", "mrn"):
            return c
    return ""

# -------------------------
# Detection logic (keep current buckets + audit)
# -------------------------

RE_OPERATIVE_TYPE = re.compile(r"\b(operative|op note|brief op|operation|surgical|procedure|or note)\b", re.I)

RE_TE = re.compile(r"\b(expander|expanders|tissue expander|te)\b", re.I)
RE_REMOVE = re.compile(r"\b(remove(d|al)?|explant(ed|ation)?|take\s*out|takedown|retrieve)\b", re.I)

RE_IMPLANT = re.compile(r"\b(implant(s)?|prosthesis|silicone|saline|gel|mentor|allergan|sientra)\b", re.I)

RE_ACTION = re.compile(r"\b(place(d|ment)?|insert(ed|ion)?|exchange(d)?|exchanged|replace(d|ment)?|replacement)\b", re.I)
RE_EXCH_WORD = re.compile(r"\b(exchange|exchanged|replace|replaced|replacement)\b", re.I)

RE_RECON = re.compile(r"\b(breast reconstruction|reconstruction)\b", re.I)
RE_STAGE2_HINT = re.compile(r"\b(second stage|stage\s*2)\b", re.I)

RE_CAPSULE = re.compile(r"\b(capsulectomy|capsulotomy)\b", re.I)

RE_EXCHANGE_TIGHT = re.compile(
    r"\b(implant|expander)\b.{0,50}\b(exchange|exchanged|replace|replaced|replacement)\b"
    r"|\b(exchange|exchanged|replace|replaced|replacement)\b.{0,50}\b(implant|expander)\b",
    re.I
)

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
        return (False, "", "", 0, 0)

    parts = re.split(r"[.;]\s+|\n+", text_norm)
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

        return (True, "SCHEDULED_STAGE2_TIGHT", "SCHEDULED_SENTENCE", offset + m_proc.start(), offset + m_proc.end())

    return (False, "", "", 0, 0)

def _stage2_bucket(text_norm, note_type_norm):
    isop = 1 if RE_OPERATIVE_TYPE.search(note_type_norm or "") else 0

    m = RE_EXCHANGE_TIGHT.search(text_norm)
    if m:
        return (True, "EXCHANGE_TIGHT", "EXCHANGE_TIGHT", m.start(), m.end(), isop)

    if RE_TE.search(text_norm) and RE_REMOVE.search(text_norm) and RE_IMPLANT.search(text_norm) and RE_ACTION.search(text_norm):
        m2 = RE_REMOVE.search(text_norm) or RE_ACTION.search(text_norm)
        st, en = (m2.start(), m2.end()) if m2 else (0, min(len(text_norm), 60))
        return (True, "EXPANDER_TO_IMPLANT", "TE+REMOVE+IMPLANT+ACTION", st, en, isop)

    if isop and RE_IMPLANT.search(text_norm) and RE_EXCH_WORD.search(text_norm):
        m3 = RE_EXCH_WORD.search(text_norm) or RE_IMPLANT.search(text_norm)
        st, en = (m3.start(), m3.end()) if m3 else (0, min(len(text_norm), 60))
        return (True, "OPONLY_IMPLANT_EXCHANGE", "OPONLY_IMPLANT_EXCHANGE", st, en, isop)

    if isop and RE_IMPLANT.search(text_norm) and re.search(r"\b(place(d|ment)?|insert(ed|ion)?)\b", text_norm, re.I) and (RE_RECON.search(text_norm) or RE_STAGE2_HINT.search(text_norm)):
        m4 = re.search(r"\b(place(d|ment)?|insert(ed|ion)?)\b", text_norm, re.I) or RE_IMPLANT.search(text_norm)
        st, en = (m4.start(), m4.end()) if m4 else (0, min(len(text_norm), 60))
        return (True, "OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT", "OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT", st, en, isop)

    if isop and RE_CAPSULE.search(text_norm) and RE_IMPLANT.search(text_norm):
        m5 = RE_CAPSULE.search(text_norm)
        return (True, "OPONLY_CAPSULE_PLUS_IMPLANT", "OPONLY_CAPSULE_PLUS_IMPLANT", m5.start(), m5.end(), isop)

    ok, bucket, patname, st, en = _scheduled_stage2_sentence_level(text_norm, proximity=50)
    if ok:
        return (True, bucket, patname, st, en, isop)

    return (False, "", "", 0, 0, isop)

def detect_stage(note_text, note_type):
    t = _normalize_text(note_text)
    nt = _normalize_text(note_type)

    ok2, bucket, patname, st, en, isop = _stage2_bucket(t, nt)
    if ok2:
        return ("STAGE2", bucket, patname, st, en, isop)

    for pat in STAGE1_PATTERNS:
        m = re.search(pat, t)
        if m:
            return ("STAGE1", "STAGE1", pat, m.start(), m.end(), 0)

    return ("", "", "", 0, 0, 0)

# -------------------------
# Build + Validate
# -------------------------

def _parse_args(argv):
    # Minimal args to avoid argparse dependency
    args = {
        "notes": "",
        "gold": DEFAULT_GOLD,
        "out": OUT_DIR,
        "seed": "13",
        "fp_sample_n": "250",
    }
    for a in argv[1:]:
        if a.startswith("--notes="):
            args["notes"] = a.split("=", 1)[1].strip()
        elif a.startswith("--gold="):
            args["gold"] = a.split("=", 1)[1].strip()
        elif a.startswith("--out="):
            args["out"] = a.split("=", 1)[1].strip()
        elif a.startswith("--seed="):
            args["seed"] = a.split("=", 1)[1].strip()
        elif a.startswith("--fp_sample_n="):
            args["fp_sample_n"] = a.split("=", 1)[1].strip()
    return args

def build_and_validate(note_paths, gold_path, out_dir, fp_sample_n=250, seed=13):
    _safe_mkdir(out_dir)
    random.seed(seed)

    # Load notes
    note_rows = []
    for p in note_paths:
        note_rows.extend(_read_csv_rows(p))
    if not note_rows:
        raise ValueError("No note rows loaded.")

    note_cols = list(note_rows[0].keys())
    note_id_col = _detect_id_col(note_cols, ["ENCRYPTED_PAT_ID", "PAT_ID", "PATIENT_ID", "PatientID"])
    if not note_id_col:
        raise ValueError("Could not detect patient id column in notes. FOUND COLS (first 40): {0}".format(note_cols[:40]))

    # Column mapping for required fields in notes
    # NOTE_TEXT is required; accept NOTE_TEXT_DEID fallback
    def get_note_text(r):
        if "NOTE_TEXT" in r and (r.get("NOTE_TEXT") is not None):
            return r.get("NOTE_TEXT", "")
        if "NOTE_TEXT_DEID" in r:
            return r.get("NOTE_TEXT_DEID", "")
        return ""

    # Load gold
    gold_rows = _read_csv_rows(gold_path)
    if not gold_rows:
        raise ValueError("Gold file empty: {0}".format(gold_path))
    gold_cols = list(gold_rows[0].keys())
    gold_id_col = _detect_id_col(gold_cols, ["ENCRYPTED_PAT_ID", "PAT_ID", "PATIENT_ID", "PatientID"])
    if not gold_id_col:
        raise ValueError("Could not detect patient id column in gold. FOUND COLS (first 60): {0}".format(gold_cols[:60]))

    if GOLD_STAGE2_COL not in gold_cols:
        raise ValueError("ERROR: Could not find gold Stage2 flag column in gold: {0}. FOUND COLS (first 80): {1}".format(
            GOLD_STAGE2_COL, gold_cols[:80]
        ))

    gold_stage2 = {}
    for r in gold_rows:
        pid = str(r.get(gold_id_col, "")).strip()
        if not pid:
            continue
        gold_stage2[pid] = _truthy(r.get(GOLD_STAGE2_COL))

    # Run detection
    patients = {}
    events = []

    for r in note_rows:
        pid = str(r.get(note_id_col, "")).strip()
        if not pid:
            continue

        note_type = r.get("NOTE_TYPE", "") or ""
        note_id = str(r.get("NOTE_ID", "") or "").strip()
        event_date = _best_note_date(r)
        text = get_note_text(r)

        stage, bucket, patname, st, en, isop = detect_stage(text, note_type)
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
            if event_date and (not p["stage1_date"] or event_date < p["stage1_date"]):
                p["stage1_date"] = event_date
        elif stage == "STAGE2":
            p["stage2_hits"] += 1
            if event_date and (not p["stage2_date"] or event_date < p["stage2_date"]):
                p["stage2_date"] = event_date

        tnorm = _normalize_text(text)
        snippet = _make_snippet(tnorm, st, en, width=160)

        events.append({
            "ENCRYPTED_PAT_ID": pid if note_id_col == "ENCRYPTED_PAT_ID" else pid,
            "EVENT_DATE": event_date,
            "NOTE_ID": note_id,
            "NOTE_TYPE": note_type,
            "STAGE": stage,
            "DETECTION_BUCKET": bucket,
            "PATTERN_NAME": patname,
            "IS_OPERATIVE_CONTEXT": int(isop),
            "EVIDENCE_SNIPPET": snippet,
        })

    # Write stage_event_level.csv
    event_out = os.path.join(out_dir, "stage_event_level.csv")
    with open(event_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE",
            "STAGE", "DETECTION_BUCKET", "PATTERN_NAME", "IS_OPERATIVE_CONTEXT",
            "EVIDENCE_SNIPPET"
        ])
        w.writeheader()
        for e in events:
            w.writerow(e)

    # Patient summary
    patient_out = os.path.join(out_dir, "patient_stage_summary.csv")
    with open(patient_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID",
            "STAGE1_DATE", "STAGE1_HITS", "HAS_STAGE1",
            "STAGE2_DATE", "STAGE2_HITS", "HAS_STAGE2",
        ])
        w.writeheader()
        for pid in sorted(patients.keys()):
            p = patients[pid]
            w.writerow({
                "ENCRYPTED_PAT_ID": pid,
                "STAGE1_DATE": p["stage1_date"],
                "STAGE1_HITS": p["stage1_hits"],
                "HAS_STAGE1": 1 if p["stage1_hits"] > 0 else 0,
                "STAGE2_DATE": p["stage2_date"],
                "STAGE2_HITS": p["stage2_hits"],
                "HAS_STAGE2": 1 if p["stage2_hits"] > 0 else 0,
            })

    # Validation: Stage2 Anchor = predicted HAS_STAGE2 vs gold Stage2_Applicable
    all_pids = set(gold_stage2.keys()) | set(patients.keys())

    TP = FP = FN = TN = 0
    mismatches = []
    fp_bucket_counts = {}
    fp_noteType_counts = {}  # (bucket, note_type)
    bucket_total_predictions = {}

    # Build quick index for FP sampling: events by pid where stage2
    stage2_events_by_pid = {}
    for e in events:
        if e["STAGE"] == "STAGE2":
            stage2_events_by_pid.setdefault(e["ENCRYPTED_PAT_ID"], []).append(e)
            b = e["DETECTION_BUCKET"]
            bucket_total_predictions[b] = bucket_total_predictions.get(b, 0) + 1

    for pid in sorted(all_pids):
        gold = gold_stage2.get(pid, 0)
        pred = 1 if (pid in patients and patients[pid]["stage2_hits"] > 0) else 0

        if pred == 1 and gold == 1:
            TP += 1
        elif pred == 1 and gold == 0:
            FP += 1
            # count buckets + note types for FP events (use all stage2 events for this pid)
            for e in stage2_events_by_pid.get(pid, []):
                b = e["DETECTION_BUCKET"]
                nt = (e["NOTE_TYPE"] or "").strip()
                fp_bucket_counts[b] = fp_bucket_counts.get(b, 0) + 1
                key = (b, nt)
                fp_noteType_counts[key] = fp_noteType_counts.get(key, 0) + 1
        elif pred == 0 and gold == 1:
            FN += 1
        else:
            TN += 1

        if pred != gold:
            mismatches.append({
                "ENCRYPTED_PAT_ID": pid,
                "GOLD_STAGE2_APPLICABLE": gold,
                "PRED_HAS_STAGE2": pred,
                "PRED_STAGE2_HITS": patients.get(pid, {}).get("stage2_hits", 0),
                "PRED_STAGE2_DATE": patients.get(pid, {}).get("stage2_date", ""),
            })

    # Metrics
    precision = float(TP) / float(TP + FP) if (TP + FP) else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics_out = os.path.join(out_dir, "validation_metrics.txt")
    with open(metrics_out, "w", encoding="utf-8") as f:
        f.write("Validation complete.\n")
        f.write("Stage2 Anchor (gold={0}):\n".format(GOLD_STAGE2_COL))
        f.write("TP={0} FP={1} FN={2} TN={3}\n".format(TP, FP, FN, TN))
        f.write("Precision={0:.3f} Recall={1:.3f} F1={2:.3f}\n".format(precision, recall, f1))

    # Mismatches CSV
    mismatch_out = os.path.join(out_dir, "validation_mismatches.csv")
    with open(mismatch_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID",
            "GOLD_STAGE2_APPLICABLE",
            "PRED_HAS_STAGE2",
            "PRED_STAGE2_HITS",
            "PRED_STAGE2_DATE",
        ])
        w.writeheader()
        for r in mismatches:
            w.writerow(r)

    # Audit bucket summary (total predictions per bucket)
    bucket_summary_out = os.path.join(out_dir, "audit_bucket_summary.csv")
    with open(bucket_summary_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["DETECTION_BUCKET", "Total_predictions"])
        w.writeheader()
        for b in sorted(bucket_total_predictions.keys(), key=lambda x: (-bucket_total_predictions[x], x)):
            w.writerow({"DETECTION_BUCKET": b, "Total_predictions": bucket_total_predictions[b]})

    # Audit FP by bucket
    fp_by_bucket_out = os.path.join(out_dir, "audit_fp_by_bucket.csv")
    with open(fp_by_bucket_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["DETECTION_BUCKET", "FP_count"])
        w.writeheader()
        for b in sorted(fp_bucket_counts.keys(), key=lambda x: (-fp_bucket_counts[x], x)):
            w.writerow({"DETECTION_BUCKET": b, "FP_count": fp_bucket_counts[b]})

    # Audit bucket x note_type breakdown (FP only)
    fp_noteType_out = os.path.join(out_dir, "audit_bucket_noteType_breakdown.csv")
    with open(fp_noteType_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["DETECTION_BUCKET", "NOTE_TYPE", "Count"])
        w.writeheader()
        items = list(fp_noteType_counts.items())
        items.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        for (b, nt), c in items:
            w.writerow({"DETECTION_BUCKET": b, "NOTE_TYPE": nt, "Count": c})

    # FP event samples (with snippet)
    fp_event_samples = []
    for pid in sorted(all_pids):
        gold = gold_stage2.get(pid, 0)
        pred = 1 if (pid in patients and patients[pid]["stage2_hits"] > 0) else 0
        if pred == 1 and gold == 0:
            for e in stage2_events_by_pid.get(pid, []):
                fp_event_samples.append({
                    "ENCRYPTED_PAT_ID": e["ENCRYPTED_PAT_ID"],
                    "EVENT_DATE": e["EVENT_DATE"],
                    "NOTE_ID": e["NOTE_ID"],
                    "NOTE_TYPE": e["NOTE_TYPE"],
                    "DETECTION_BUCKET": e["DETECTION_BUCKET"],
                    "PATTERN_NAME": e["PATTERN_NAME"],
                    "IS_OPERATIVE_CONTEXT": e["IS_OPERATIVE_CONTEXT"],
                    "EVIDENCE_SNIPPET": e["EVIDENCE_SNIPPET"],
                })

    random.shuffle(fp_event_samples)
    fp_event_samples = fp_event_samples[:max(0, int(fp_sample_n))]

    fp_events_out = os.path.join(out_dir, "audit_fp_events_sample.csv")
    with open(fp_events_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ENCRYPTED_PAT_ID", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE",
            "DETECTION_BUCKET", "PATTERN_NAME", "IS_OPERATIVE_CONTEXT", "EVIDENCE_SNIPPET"
        ])
        w.writeheader()
        for r in fp_event_samples:
            w.writerow(r)

    print("Validation complete.")
    print("Stage2 Anchor (gold={0}):".format(GOLD_STAGE2_COL))
    print("TP={0} FP={1} FN={2} TN={3}".format(TP, FP, FN, TN))
    print("Precision={0:.3f} Recall={1:.3f} F1={2:.3f}".format(precision, recall, f1))
    print("WROTE:", os.path.basename(event_out))
    print("WROTE:", os.path.basename(patient_out))
    print("WROTE:", os.path.basename(metrics_out))
    print("WROTE:", os.path.basename(mismatch_out))
    print("WROTE:", os.path.basename(bucket_summary_out))
    print("WROTE:", os.path.basename(fp_by_bucket_out))
    print("WROTE:", os.path.basename(fp_noteType_out))
    print("WROTE:", os.path.basename(fp_events_out))

def main():
    args = _parse_args(sys.argv)

    out_dir = args["out"]
    gold_path = args["gold"]
    fp_sample_n = int(args["fp_sample_n"])
    seed = int(args["seed"])

    if args["notes"]:
        note_paths = [p.strip() for p in args["notes"].split(",") if p.strip()]
    else:
        note_paths = _find_default_note_csvs()

    for p in note_paths:
        if not os.path.isfile(p):
            raise IOError("Notes CSV not found: {0}".format(p))
    if not os.path.isfile(gold_path):
        raise IOError("Gold CSV not found: {0}".format(gold_path))

    build_and_validate(note_paths, gold_path, out_dir, fp_sample_n=fp_sample_n, seed=seed)

if __name__ == "__main__":
    main()

# -------------------------
# RUN
# -------------------------
# Default (uses _staging_inputs/HPI11526 Operation Notes.csv + Clinic Notes.csv):
#   python build_stage2_anchor_with_audit.py --gold=/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv
#
# If your notes are a different CSV (or multiple CSVs):
#   python build_stage2_anchor_with_audit.py --notes=/path/notes1.csv,/path/notes2.csv --gold=/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv
#
# Outputs land in:
#   ./_outputs/
