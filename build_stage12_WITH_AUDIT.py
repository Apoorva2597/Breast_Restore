#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage2_anchor_with_audit.py (Python 3.6.8 compatible)

Revised logic to reduce FP based on audit:
- EXCHANGE_TIGHT was firing heavily in PROGRESS NOTES/H&P -> now requires operative context OR strong performed cues
- EXPANDER_TO_IMPLANT now sentence-scoped + excludes non-performed/counseling/plan language
- OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT caused direct-to-implant FP -> now requires stage2-specific cues (exchange/replace OR TE/remove OR stage2 hint)
- Capsule+implant now requires exchange/replace OR stage2 hint OR TE/remove

Gold:
- Stage2_Applicable in /home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv

Outputs:
- _outputs/stage_event_level.csv
- _outputs/patient_stage_summary.csv
- _outputs/validation_metrics.txt
- _outputs/validation_mismatches.csv
- _outputs/audit_bucket_summary.csv
- _outputs/audit_fp_by_bucket.csv
- _outputs/audit_bucket_noteType_breakdown.csv
- _outputs/audit_fp_events_sample.csv
"""

from __future__ import print_function
import os
import csv
import re
import sys
import random
from datetime import datetime

STAGING_DIR = os.path.join(os.getcwd(), "_staging_inputs")
OUT_DIR = os.path.join(os.getcwd(), "_outputs")

DEFAULT_GOLD = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
GOLD_STAGE2_COL = "Stage2_Applicable"

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
    colset = set(cols)
    for c in preferred:
        if c in colset:
            return c
    for c in cols:
        cl = c.lower()
        if "encrypted" in cl and "pat" in cl:
            return c
    for c in cols:
        cl = c.lower()
        if cl in ("patientid", "patient_id", "pat_id", "mrn"):
            return c
    return ""

def _split_sentences(text_norm):
    # conservative splitter
    if not text_norm:
        return []
    parts = re.split(r"[.;]\s+|\n+", text_norm)
    out = []
    offset = 0
    for p in parts:
        s = p.strip()
        if s:
            out.append((s, offset))
        offset += len(p) + 1
    return out

# -------------------------
# Detection logic
# -------------------------

RE_OPERATIVE_TYPE = re.compile(
    r"\b(operative|op note|brief op|operation|surgical|procedure|or note|operative report)\b", re.I
)

RE_TE = re.compile(r"\b(expander|expanders|tissue expander|te)\b", re.I)
RE_REMOVE = re.compile(r"\b(remove(d|al)?|explant(ed|ation)?|take\s*out|takedown|retrieve)\b", re.I)

RE_IMPLANT = re.compile(r"\b(implant(s)?|prosthesis|silicone|saline|gel|mentor|allergan|sientra)\b", re.I)
RE_ACTION = re.compile(r"\b(place(d|ment)?|insert(ed|ion)?|exchange(d)?|exchanged|replace(d|ment)?|replacement)\b", re.I)
RE_EXCH_WORD = re.compile(r"\b(exchange|exchanged|replace|replaced|replacement)\b", re.I)

RE_RECON = re.compile(r"\b(breast reconstruction|reconstruction)\b", re.I)
RE_STAGE2_HINT = re.compile(r"\b(second stage|stage\s*2)\b", re.I)

RE_CAPSULE = re.compile(r"\b(capsulectomy|capsulotomy)\b", re.I)

# performed-context cues (to allow Stage2 mentions outside NOTE_TYPE, but still reduce PROGRESS NOTE FP)
RE_PERFORMED_CUE = re.compile(
    r"\b(underwent|was performed|performed|taken to (the )?or|in (the )?or\b|operating room|"
    r"incision|estimated blood loss|ebl\b|drains?\b|specimens?\b|implants?\s*:|"
    r"procedure\s*:|brief op note|operative report|post[- ]?op( course)?\b)\b",
    re.I
)

# non-performed / counseling / planning cues (common in PROGRESS NOTES / H&P)
RE_NONPERFORMED = re.compile(
    r"\b(plan(ned)?|planning|scheduled|schedule(d)?|consider(ing)?|option(s)?|"
    r"discuss(ed|ion)?|counsel(ing)?|candidate|recommend(ed|s)?|would like|"
    r"may|might|could|would)\b",
    re.I
)
RE_NEGATE = re.compile(r"\b(no|not|never|denies?)\b.{0,12}\b(plan(ned)?|scheduled|planning|exchange|replace|implant)\b", re.I)

# Exchange pattern (tight) but now will be sentence-filtered
RE_EXCHANGE_TIGHT = re.compile(
    r"\b(implant|expander)\b.{0,50}\b(exchange|exchanged|replace|replaced|replacement)\b"
    r"|\b(exchange|exchanged|replace|replaced|replacement)\b.{0,50}\b(implant|expander)\b",
    re.I
)

# Scheduled Stage2 (unchanged)
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

def _is_operative(note_type_norm):
    return 1 if RE_OPERATIVE_TYPE.search(note_type_norm or "") else 0

def _sentence_allows_performed(sent_norm, isop):
    # If operative note -> allow unless explicit negation of procedure intent
    if isop:
        return False if RE_NEGATE.search(sent_norm) else True

    # Non-operative notes must have a performed cue to count as performed Stage2
    if not RE_PERFORMED_CUE.search(sent_norm):
        return False

    # Exclude clearly counseling/planning sentences in non-op notes
    if RE_NEGATE.search(sent_norm):
        return False
    if RE_NONPERFORMED.search(sent_norm) and not re.search(r"\b(post[- ]?op|status post|s/p|underwent|was performed)\b", sent_norm, re.I):
        return False

    return True

def _scheduled_stage2_sentence_level(text_norm, proximity=50):
    if not RE_SCHEDULE.search(text_norm):
        return (False, "", "", 0, 0)

    parts = _split_sentences(text_norm)
    for (s, offset) in parts:
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

        return (True, "SCHEDULED_STAGE2_TIGHT", "SCHEDULED_SENTENCE", offset + m_proc.start(), offset + m_proc.end())

    return (False, "", "", 0, 0)

def _stage2_bucket(text_norm, note_type_norm):
    isop = _is_operative(note_type_norm)

    # sentence-scoped evaluation to reduce PROGRESS NOTE FP
    sents = _split_sentences(text_norm)
    if not sents:
        sents = [(text_norm, 0)]

    # 1) EXCHANGE_TIGHT (revised): must be operative OR performed-cue sentence
    for (s, off) in sents:
        m = RE_EXCHANGE_TIGHT.search(s)
        if not m:
            continue
        if not _sentence_allows_performed(s, isop):
            continue
        return (True, "EXCHANGE_TIGHT", "EXCHANGE_TIGHT_SENTENCE", off + m.start(), off + m.end(), isop)

    # 2) EXPANDER_TO_IMPLANT (revised): TE + REMOVE + IMPLANT + ACTION in same sentence + performed allowed
    for (s, off) in sents:
        if not (RE_TE.search(s) and RE_REMOVE.search(s) and RE_IMPLANT.search(s) and RE_ACTION.search(s)):
            continue
        if not _sentence_allows_performed(s, isop):
            continue
        m2 = RE_REMOVE.search(s) or RE_ACTION.search(s)
        st, en = (m2.start(), m2.end()) if m2 else (0, min(len(s), 60))
        return (True, "EXPANDER_TO_IMPLANT", "TE+REMOVE+IMPLANT+ACTION_SENTENCE", off + st, off + en, isop)

    # 3) OPONLY_IMPLANT_EXCHANGE (keep, but sentence-performed gating)
    if isop:
        for (s, off) in sents:
            if RE_IMPLANT.search(s) and RE_EXCH_WORD.search(s):
                if not _sentence_allows_performed(s, isop):
                    continue
                m3 = RE_EXCH_WORD.search(s) or RE_IMPLANT.search(s)
                st, en = (m3.start(), m3.end()) if m3 else (0, min(len(s), 60))
                return (True, "OPONLY_IMPLANT_EXCHANGE", "OPONLY_IMPLANT_EXCHANGE_SENTENCE", off + st, off + en, isop)

    # 4) OPONLY_IMPLANT_PLACEMENT_RECON (revised to avoid direct-to-implant FP):
    # require operative AND implant placement AND (exchange/replace OR TE/remove OR explicit stage2 hint)
    if isop:
        RE_PLACE = re.compile(r"\b(place(d|ment)?|insert(ed|ion)?)\b", re.I)
        for (s, off) in sents:
            if not (RE_IMPLANT.search(s) and RE_PLACE.search(s)):
                continue
            stage2_specific = False
            if RE_EXCH_WORD.search(s):
                stage2_specific = True
            if RE_STAGE2_HINT.search(s):
                stage2_specific = True
            if RE_TE.search(s) or RE_REMOVE.search(s):
                stage2_specific = True
            if not stage2_specific:
                continue
            if not _sentence_allows_performed(s, isop):
                continue
            m4 = RE_EXCH_WORD.search(s) or RE_STAGE2_HINT.search(s) or RE_REMOVE.search(s) or RE_PLACE.search(s)
            st, en = (m4.start(), m4.end()) if m4 else (0, min(len(s), 60))
            return (True, "OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT", "OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT_SENTENCE", off + st, off + en, isop)

    # 5) OPONLY_CAPSULE_PLUS_IMPLANT (revised): require stage2-specific cue to avoid generic capsule work
    if isop:
        for (s, off) in sents:
            if not (RE_CAPSULE.search(s) and RE_IMPLANT.search(s)):
                continue
            stage2_specific = False
            if RE_EXCH_WORD.search(s) or RE_STAGE2_HINT.search(s) or RE_TE.search(s) or RE_REMOVE.search(s):
                stage2_specific = True
            if not stage2_specific:
                continue
            if not _sentence_allows_performed(s, isop):
                continue
            m5 = RE_CAPSULE.search(s)
            return (True, "OPONLY_CAPSULE_PLUS_IMPLANT", "OPONLY_CAPSULE_PLUS_IMPLANT_SENTENCE", off + m5.start(), off + m5.end(), isop)

    # 6) Scheduled Stage2 (tight, sentence scoped)
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
# Build + Validate + Audit
# -------------------------

def _parse_args(argv):
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

    note_rows = []
    for p in note_paths:
        note_rows.extend(_read_csv_rows(p))
    if not note_rows:
        raise ValueError("No note rows loaded.")

    note_cols = list(note_rows[0].keys())
    note_id_col = _detect_id_col(note_cols, ["ENCRYPTED_PAT_ID", "PAT_ID", "PATIENT_ID", "PatientID"])
    if not note_id_col:
        raise ValueError("Could not detect patient id column in notes. FOUND COLS (first 40): {0}".format(note_cols[:40]))

    def get_note_text(r):
        if "NOTE_TEXT" in r and (r.get("NOTE_TEXT") is not None):
            return r.get("NOTE_TEXT", "")
        if "NOTE_TEXT_DEID" in r:
            return r.get("NOTE_TEXT_DEID", "")
        return ""

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
            "ENCRYPTED_PAT_ID": pid,
            "EVENT_DATE": event_date,
            "NOTE_ID": note_id,
            "NOTE_TYPE": note_type,
            "STAGE": stage,
            "DETECTION_BUCKET": bucket,
            "PATTERN_NAME": patname,
            "IS_OPERATIVE_CONTEXT": int(isop),
            "EVIDENCE_SNIPPET": snippet,
        })

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

    all_pids = set(gold_stage2.keys()) | set(patients.keys())

    TP = FP = FN = TN = 0
    mismatches = []
    fp_bucket_counts = {}
    fp_noteType_counts = {}
    bucket_total_predictions = {}

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

    precision = float(TP) / float(TP + FP) if (TP + FP) else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics_out = os.path.join(out_dir, "validation_metrics.txt")
    with open(metrics_out, "w", encoding="utf-8") as f:
        f.write("Validation complete.\n")
        f.write("Stage2 Anchor (gold={0}):\n".format(GOLD_STAGE2_COL))
        f.write("TP={0} FP={1} FN={2} TN={3}\n".format(TP, FP, FN, TN))
        f.write("Precision={0:.3f} Recall={1:.3f} F1={2:.3f}\n".format(precision, recall, f1))

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

    bucket_summary_out = os.path.join(out_dir, "audit_bucket_summary.csv")
    with open(bucket_summary_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["DETECTION_BUCKET", "Total_predictions"])
        w.writeheader()
        for b in sorted(bucket_total_predictions.keys(), key=lambda x: (-bucket_total_predictions[x], x)):
            w.writerow({"DETECTION_BUCKET": b, "Total_predictions": bucket_total_predictions[b]})

    fp_by_bucket_out = os.path.join(out_dir, "audit_fp_by_bucket.csv")
    with open(fp_by_bucket_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["DETECTION_BUCKET", "FP_count"])
        w.writeheader()
        for b in sorted(fp_bucket_counts.keys(), key=lambda x: (-fp_bucket_counts[x], x)):
            w.writerow({"DETECTION_BUCKET": b, "FP_count": fp_bucket_counts[b]})

    fp_noteType_out = os.path.join(out_dir, "audit_bucket_noteType_breakdown.csv")
    with open(fp_noteType_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["DETECTION_BUCKET", "NOTE_TYPE", "Count"])
        w.writeheader()
        items = list(fp_noteType_counts.items())
        items.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        for (b, nt), c in items:
            w.writerow({"DETECTION_BUCKET": b, "NOTE_TYPE": nt, "Count": c})

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
    print("Stage2 Anchor:")
    print("  TP={0} FP={1} FN={2} TN={3}".format(TP, FP, FN, TN))
    print("  Precision={0:.3f} Recall={1:.3f} F1={2:.3f}".format(precision, recall, f1))

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
