#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage2_with_audit_vNEXT.py  (Python 3.6.8 compatible)

- Reads ONE notes CSV (auto-discovered) with columns like:
  ENCRYPTED_PAT_ID, NOTE_ID, NOTE_TYPE, NOTE_TEXT, NOTE_DATE_OF_SERVICE, OPERATION_DATE
  (also supports extra columns like file_tag, NOTE_TEXT_DEID, ROWS)
- Builds:
  - patient_stage_summary_FINAL_FINAL.csv
  - stage_event_level_FINAL_FINAL.csv
  - audit_bucket_summary.csv
  - audit_bucket_noteType_breakdown.csv
  - audit_fp_by_bucket.csv
  - audit_fp_events_sample.csv
  - audit_link_debug.csv

- Validates vs gold:
  default gold = /home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv
  expects ENCRYPTED_PAT_ID + one gold stage2 flag column (auto-detected)
"""

from __future__ import print_function
import os
import csv
import re
from datetime import datetime
from collections import defaultdict

# -------------------------
# Paths (robust defaults)
# -------------------------

CWD = os.getcwd()
if os.path.basename(CWD) == "_outputs":
    OUT_DIR = CWD
    PROJECT_ROOT = os.path.dirname(CWD)
else:
    OUT_DIR = os.path.join(CWD, "_outputs")
    PROJECT_ROOT = CWD

STAGING_DIR = os.path.join(PROJECT_ROOT, "_staging_inputs")
DEFAULT_GOLD = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"

# -------------------------
# Helpers
# -------------------------

def _safe_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def _norm(s):
    if s is None:
        return ""
    s = str(s)
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

def _best_event_date(row):
    # Prefer OPERATION_DATE; fallback NOTE_DATE_OF_SERVICE; then NOTE_DATE (if present)
    op = _parse_date_any(row.get("OPERATION_DATE", ""))
    if op:
        return op
    dos = _parse_date_any(row.get("NOTE_DATE_OF_SERVICE", ""))
    if dos:
        return dos
    nd = _parse_date_any(row.get("NOTE_DATE", ""))
    return nd

def _make_snippet(text_norm, start, end, width=160):
    if not text_norm:
        return ""
    lo = max(0, start - width)
    hi = min(len(text_norm), end + width)
    return text_norm[lo:hi].strip()

def _read_csv_dicts(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def _list_csv_candidates(search_dirs):
    cands = []
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(".csv"):
                cands.append(os.path.join(d, fn))
    # also allow cwd csvs
    for fn in os.listdir(CWD):
        if fn.lower().endswith(".csv"):
            cands.append(os.path.join(CWD, fn))
    # de-dupe
    out = []
    seen = set()
    for p in cands:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            seen.add(ap)
            out.append(ap)
    return out

def _has_min_notes_cols(fieldnames):
    if not fieldnames:
        return False
    f = set([x.strip() for x in fieldnames if x])
    return ("ENCRYPTED_PAT_ID" in f) and ("NOTE_TEXT" in f) and ("NOTE_ID" in f) and ("NOTE_TYPE" in f)

def _discover_notes_csv():
    # Prefer staging inputs if present; else search project root + outputs
    search_dirs = [STAGING_DIR, OUT_DIR, PROJECT_ROOT]
    for p in _list_csv_candidates(search_dirs):
        try:
            with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
            if _has_min_notes_cols(header):
                return p
        except Exception:
            continue
    return ""

def _detect_gold_label_column(fieldnames):
    # Try common possibilities (case sensitive to match DictReader keys)
    if not fieldnames:
        return ""
    f = list(fieldnames)

    preferred = [
        "GOLD_HAS_STAGE2",
        "GOLD_STAGE2",
        "GOLD_STAGE2_FLAG",
        "GOLD_HAS_STAGE2_FLAG",
        "GOLD_STAGE2_PRESENT",
        "GOLD",
        "STAGE2_GOLD",
        "GOLD_LABEL_STAGE2",
        "GOLD_STAGE2_LABEL",
        "GOLD_STAGE2_YN",
        "HAS_STAGE2_GOLD",
        "HAS_STAGE2",
        "GOLD_HAS_STAGE2?",
        "GOLD_HAS_STAGE2 ",
    ]
    for c in preferred:
        if c in f:
            return c

    # last resort: any column containing both 'gold' and 'stage2'
    for c in f:
        cn = c.lower().replace(" ", "")
        if ("gold" in cn) and ("stage2" in cn):
            return c

    # last resort: any column exactly 'GOLD_HAS_STAGE2' variant ignoring case/spaces/underscores
    canon = {}
    for c in f:
        canon_key = re.sub(r"[^a-z0-9]+", "", c.lower())
        canon[canon_key] = c
    for key in ["goldhasstage2", "goldstage2", "stage2gold", "hasstage2gold"]:
        if key in canon:
            return canon[key]

    return ""

def _to_int01(x):
    if x is None:
        return 0
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "false", "f", "no", "n", ""):
        return 0
    # handle 'Y'/'N' and other junk
    if "yes" in s:
        return 1
    return 0

# -------------------------
# Detection logic (updated to reduce FP drivers seen in breakdown)
# -------------------------

# Operative note-type heuristic
RE_OPERATIVE_TYPE = re.compile(r"\b(operative|op note|brief op|operation|surgical|procedure|or note|intraop|intra-op)\b", re.I)

# Devices
RE_TE = re.compile(r"\b(expander|expanders|tissue expander|te)\b", re.I)
RE_REMOVE = re.compile(r"\b(remove(d|al)?|explant(ed|ation)?|take\s*out|takedown|retrieve)\b", re.I)
RE_IMPLANT = re.compile(r"\b(implant(s)?|prosthesis|silicone|saline|gel|mentor|allergan|sientra)\b", re.I)

# Actions
RE_ACTION = re.compile(r"\b(place(d|ment)?|insert(ed|ion)?|exchange(d)?|exchanged|replace(d|ment)?|replacement)\b", re.I)
RE_EXCH_WORD = re.compile(r"\b(exchange|exchanged|replace|replaced|replacement)\b", re.I)

# Context cues
RE_RECON = re.compile(r"\b(breast reconstruction|reconstruction)\b", re.I)
RE_STAGE2_HINT = re.compile(r"\b(second stage|stage\s*2)\b", re.I)
RE_CAPSULE = re.compile(r"\b(capsulectomy|capsulotomy)\b", re.I)

# Scheduled / counseling guards
RE_SCHEDULE = re.compile(r"\b(schedule(d)?|planned|plan)\b", re.I)
RE_SCHEDULED_FOR = re.compile(r"\bscheduled\b.{0,12}\bfor\b", re.I)
RE_PROC_CUE = re.compile(r"\b(surgery|procedure|operation|or|operative)\b", re.I)

RE_NOT_SCHEDULED = re.compile(
    r"\b(not|no|never)\s+(scheduled|plan(ned)?|planning)\b|\bno plans\b|\bnot planning\b",
    re.I
)
RE_COUNSEL_ONLY = re.compile(
    r"\b(discuss(ed|ion)?|consider(ing)?|option(s)?|candidate|counsel(ing)?|risks? and benefits|review(ed)?|would like to)\b",
    re.I
)

# Performed cues to allow exchange language in non-operative notes (reduce FP from plans/history)
RE_PERFORMED_CUE = re.compile(
    r"\b(underwent|s\/p|status post|post[- ]?op|postoperative|was taken to the or|returned to the or|procedure performed|operation performed|procedures?:)\b",
    re.I
)
RE_HISTORY_CUE = re.compile(r"\b(hx|history of|previous(ly)?|in \d{4}|years? ago|back in)\b", re.I)

# Stage2 "performed" phrase for scheduled detection
RE_STAGE2_PROC_PHRASE = re.compile(
    r"\b(expander[- ]?to[- ]?implant)\b"
    r"|\b(exchange)\b.{0,30}\b(expander|implant)\b"
    r"|\b(expander)\b.{0,30}\b(exchange)\b"
    r"|\b(second stage|stage\s*2)\b.{0,40}\b(reconstruction|exchange)\b"
    r"|\b(expander)\b.{0,40}\b(remove|removal|explant)\b.{0,40}\b(implant)\b",
    re.I
)

# Exchange tight (string-level) — but we will *gate* it by note-type / performed cues
RE_EXCHANGE_TIGHT = re.compile(
    r"\b(implant|expander)\b.{0,50}\b(exchange|exchanged|replace|replaced|replacement)\b"
    r"|\b(exchange|exchanged|replace|replaced|replacement)\b.{0,50}\b(implant|expander)\b",
    re.I
)

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

        return (True, "SCHEDULED_STAGE2_TIGHT", "SCHEDULED_SENTENCE", (offset + m_proc.start()), (offset + m_proc.end()))

    return (False, "", "", 0, 0)

def _stage2_bucket(text_norm, note_type_norm):
    is_operative = 1 if RE_OPERATIVE_TYPE.search(note_type_norm) else 0

    # (A) EXCHANGE_TIGHT — main FP driver in progress notes; gate it:
    m = RE_EXCHANGE_TIGHT.search(text_norm)
    if m:
        # Reject if clearly scheduled/counseling
        if RE_SCHEDULE.search(text_norm) or RE_COUNSEL_ONLY.search(text_norm):
            pass
        else:
            if is_operative:
                return (True, "EXCHANGE_TIGHT", "EXCHANGE_TIGHT", m.start(), m.end(), is_operative)
            # Non-operative: require a performed cue in the SAME text AND avoid "history" dominated mentions
            # This keeps true postop follow-ups while dropping "candidate/discuss/plan/history"
            if RE_PERFORMED_CUE.search(text_norm) and (not RE_HISTORY_CUE.search(text_norm) or RE_PERFORMED_CUE.search(text_norm)):
                return (True, "EXCHANGE_TIGHT", "EXCHANGE_TIGHT_NONOP_GATED", m.start(), m.end(), is_operative)

    # (B) Classic TE removal + implant + action (performed)
    if RE_TE.search(text_norm) and RE_REMOVE.search(text_norm) and RE_IMPLANT.search(text_norm) and RE_ACTION.search(text_norm):
        m2 = RE_REMOVE.search(text_norm) or RE_ACTION.search(text_norm)
        st, en = (m2.start(), m2.end()) if m2 else (0, min(len(text_norm), 80))
        return (True, "EXPANDER_TO_IMPLANT", "TE+REMOVE+IMPLANT+ACTION", st, en, is_operative)

    # (C) Operative-only: implant exchange/replace (no expander required)
    if is_operative and RE_IMPLANT.search(text_norm) and RE_EXCH_WORD.search(text_norm):
        m3 = RE_EXCH_WORD.search(text_norm) or RE_IMPLANT.search(text_norm)
        st, en = (m3.start(), m3.end()) if m3 else (0, min(len(text_norm), 80))
        return (True, "OPONLY_IMPLANT_EXCHANGE", "OPONLY_IMPLANT_EXCHANGE", st, en, is_operative)

    # (D) Operative-only: implant placement + recon/stage2 hint, but guard against pure "plan"
    if is_operative and RE_IMPLANT.search(text_norm) and re.search(r"\b(place(d|ment)?|insert(ed|ion)?)\b", text_norm, re.I) and (RE_RECON.search(text_norm) or RE_STAGE2_HINT.search(text_norm)):
        if not (RE_SCHEDULE.search(text_norm) or RE_COUNSEL_ONLY.search(text_norm)):
            m4 = re.search(r"\b(place(d|ment)?|insert(ed|ion)?)\b", text_norm, re.I) or RE_IMPLANT.search(text_norm)
            st, en = (m4.start(), m4.end()) if m4 else (0, min(len(text_norm), 80))
            return (True, "OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT", "OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT", st, en, is_operative)

    # (E) Operative-only: capsule work + implant, but require exchange/replace OR explicit "revision"
    if is_operative and RE_CAPSULE.search(text_norm) and RE_IMPLANT.search(text_norm):
        if RE_EXCH_WORD.search(text_norm) or re.search(r"\b(revision|revisional)\b", text_norm, re.I):
            m5 = RE_CAPSULE.search(text_norm)
            return (True, "OPONLY_CAPSULE_PLUS_IMPLANT_TIGHT", "OPONLY_CAPSULE_PLUS_IMPLANT_TIGHT", m5.start(), m5.end(), is_operative)

    # (F) Scheduled stage2 (tight)
    ok, bucket, patname, st, en = _scheduled_stage2_sentence_level(text_norm, proximity=50)
    if ok:
        return (True, bucket, patname, st, en, is_operative)

    return (False, "", "", 0, 0, is_operative)

def detect_stage2(note_text, note_type):
    t = _norm(note_text)
    nt = _norm(note_type)
    ok2, bucket, patname, st, en, isop = _stage2_bucket(t, nt)
    if ok2:
        return (1, bucket, patname, st, en, isop)
    return (0, "", "", 0, 0, 0)

# -------------------------
# Validation + Audits
# -------------------------

def _metrics(tp, fp, fn, tn):
    # safe float formatting
    prec = (float(tp) / float(tp + fp)) if (tp + fp) > 0 else 0.0
    rec  = (float(tp) / float(tp + fn)) if (tp + fn) > 0 else 0.0
    f1   = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return (prec, rec, f1)

def _write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def build(notes_csv_path, gold_csv_path, outputs_dir):
    _safe_mkdir(outputs_dir)

    # ---- Load notes
    note_rows = _read_csv_dicts(notes_csv_path)
    if not note_rows:
        raise ValueError("No rows in notes: {0}".format(notes_csv_path))

    # sanity: required columns
    note_fields = list(note_rows[0].keys())
    required = ["ENCRYPTED_PAT_ID", "NOTE_ID", "NOTE_TYPE", "NOTE_TEXT"]
    missing = [c for c in required if c not in note_fields]
    if missing:
        raise ValueError("ERROR: Missing columns in notes: {0}\nFOUND COLUMNS (first 80): {1}".format(missing, ", ".join(note_fields[:80])))

    # ---- Build patient-level pred + event-level audit
    patients = {}  # pid -> summary
    events = []    # event-level stage2 hits
    patient_pred_has_stage2 = defaultdict(int)

    bucket_counts = defaultdict(int)
    bucket_noteType_counts = defaultdict(int)

    for r in note_rows:
        pid = (r.get("ENCRYPTED_PAT_ID") or "").strip()
        if not pid:
            continue

        note_id = (r.get("NOTE_ID") or "").strip()
        note_type = (r.get("NOTE_TYPE") or "").strip()
        text = r.get("NOTE_TEXT", "")

        event_date = _best_event_date(r)

        pred2, bucket, patname, st, en, isop = detect_stage2(text, note_type)
        if not pred2:
            continue

        patient_pred_has_stage2[pid] = 1

        tnorm = _norm(text)
        snippet = _make_snippet(tnorm, st, en, width=160)

        bucket_counts[bucket] += 1
        bucket_noteType_counts[(bucket, (note_type or "").strip())] += 1

        events.append({
            "ENCRYPTED_PAT_ID": pid,
            "EVENT_DATE": event_date,
            "NOTE_ID": note_id,
            "NOTE_TYPE": note_type,
            "DETECTION_BUCKET": bucket,
            "PATTERN_NAME": patname,
            "IS_OPERATIVE_CONTEXT": int(isop),
            "EVIDENCE_SNIPPET": snippet
        })

    # Patient summary
    for pid, pred in patient_pred_has_stage2.items():
        patients[pid] = {
            "ENCRYPTED_PAT_ID": pid,
            "PRED_HAS_STAGE2": int(pred)
        }

    # ---- Load gold + detect label column
    gold_rows = _read_csv_dicts(gold_csv_path)
    if not gold_rows:
        raise ValueError("No rows in gold: {0}".format(gold_csv_path))
    gold_fields = list(gold_rows[0].keys())
    gold_col = _detect_gold_label_column(gold_fields)
    if not gold_col:
        raise ValueError("ERROR: Could not find gold Stage2 flag column in gold.\nFOUND COLUMNS (first 80): {0}".format(", ".join(gold_fields[:80])))

    gold_by_pid = {}
    for gr in gold_rows:
        pid = (gr.get("ENCRYPTED_PAT_ID") or "").strip()
        if not pid:
            continue
        gold_by_pid[pid] = _to_int01(gr.get(gold_col, 0))

    # ---- Merge + compute confusion
    merged_rows = []
    tp = fp = fn = tn = 0

    # union of ids (so FN show up)
    all_pids = set(gold_by_pid.keys()) | set(patient_pred_has_stage2.keys())

    for pid in sorted(all_pids):
        g = int(gold_by_pid.get(pid, 0))
        p = int(patient_pred_has_stage2.get(pid, 0))
        if p == 1 and g == 1:
            tp += 1
        elif p == 1 and g == 0:
            fp += 1
        elif p == 0 and g == 1:
            fn += 1
        else:
            tn += 1

        merged_rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "GOLD_COL_USED": gold_col,
            "GOLD_HAS_STAGE2": g,
            "PRED_HAS_STAGE2": p
        })

    prec, rec, f1 = _metrics(tp, fp, fn, tn)

    # ---- Write core outputs
    _write_csv(os.path.join(outputs_dir, "patient_stage_summary_FINAL_FINAL.csv"),
               ["ENCRYPTED_PAT_ID", "PRED_HAS_STAGE2"],
               [patients[pid] for pid in sorted(patients.keys())])

    _write_csv(os.path.join(outputs_dir, "stage_event_level_FINAL_FINAL.csv"),
               ["ENCRYPTED_PAT_ID", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE", "DETECTION_BUCKET", "PATTERN_NAME", "IS_OPERATIVE_CONTEXT", "EVIDENCE_SNIPPET"],
               events)

    _write_csv(os.path.join(outputs_dir, "validation_merged_STAGE2_ANCHOR_FINAL_FINAL.csv"),
               ["ENCRYPTED_PAT_ID", "GOLD_COL_USED", "GOLD_HAS_STAGE2", "PRED_HAS_STAGE2"],
               merged_rows)

    # ---- Bucket audits (overall)
    bucket_summary_rows = []
    for b, cnt in sorted(bucket_counts.items(), key=lambda x: (-x[1], x[0])):
        bucket_summary_rows.append({"DETECTION_BUCKET": b, "Total_predictions": cnt})
    _write_csv(os.path.join(outputs_dir, "audit_bucket_summary.csv"),
               ["DETECTION_BUCKET", "Total_predictions"],
               bucket_summary_rows)

    # ---- Bucket x note_type breakdown
    bnt_rows = []
    for (b, nt), cnt in sorted(bucket_noteType_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        bnt_rows.append({"DETECTION_BUCKET": b, "NOTE_TYPE": nt, "Count": cnt})
    _write_csv(os.path.join(outputs_dir, "audit_bucket_noteType_breakdown.csv"),
               ["DETECTION_BUCKET", "NOTE_TYPE", "Count"],
               bnt_rows)

    # ---- FP by bucket + FP event samples
    gold_set = set(gold_by_pid.keys())
    fp_pids = set([pid for pid in all_pids if patient_pred_has_stage2.get(pid, 0) == 1 and gold_by_pid.get(pid, 0) == 0])

    fp_bucket_counts = defaultdict(int)
    fp_events = []

    for e in events:
        pid = e.get("ENCRYPTED_PAT_ID", "")
        if pid in fp_pids:
            fp_bucket_counts[e.get("DETECTION_BUCKET", "")] += 1
            fp_events.append({
                "ENCRYPTED_PAT_ID": pid,
                "EVENT_DATE": e.get("EVENT_DATE", ""),
                "NOTE_ID": e.get("NOTE_ID", ""),
                "NOTE_TYPE": e.get("NOTE_TYPE", ""),
                "DETECTION_BUCKET": e.get("DETECTION_BUCKET", ""),
                "PATTERN_NAME": e.get("PATTERN_NAME", ""),
                "IS_OPERATIVE_CONTEXT": e.get("IS_OPERATIVE_CONTEXT", 0),
                "EVIDENCE_SNIPPET": e.get("EVIDENCE_SNIPPET", "")
            })

    fp_by_bucket_rows = []
    for b, cnt in sorted(fp_bucket_counts.items(), key=lambda x: (-x[1], x[0])):
        fp_by_bucket_rows.append({"DETECTION_BUCKET": b, "FP_count": cnt})
    _write_csv(os.path.join(outputs_dir, "audit_fp_by_bucket.csv"),
               ["DETECTION_BUCKET", "FP_count"],
               fp_by_bucket_rows)

    # sample: cap at 5000 rows to keep manageable
    fp_events_sample = fp_events[:5000]
    _write_csv(os.path.join(outputs_dir, "audit_fp_events_sample.csv"),
               ["ENCRYPTED_PAT_ID", "EVENT_DATE", "NOTE_ID", "NOTE_TYPE", "DETECTION_BUCKET", "PATTERN_NAME", "IS_OPERATIVE_CONTEXT", "EVIDENCE_SNIPPET"],
               fp_events_sample)

    # ---- Link debug
    dbg = [{
        "notes_csv": notes_csv_path,
        "gold_csv": gold_csv_path,
        "gold_label_col_used": gold_col,
        "notes_row_count": len(note_rows),
        "gold_row_count": len(gold_rows),
        "unique_note_patients": len(set([r.get("ENCRYPTED_PAT_ID","").strip() for r in note_rows if r.get("ENCRYPTED_PAT_ID")])),
        "unique_gold_patients": len(set(gold_by_pid.keys())),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": "{0:.3f}".format(prec),
        "Recall": "{0:.3f}".format(rec),
        "F1": "{0:.3f}".format(f1),
    }]
    _write_csv(os.path.join(outputs_dir, "audit_link_debug.csv"),
               list(dbg[0].keys()),
               dbg)

    # ---- Console summary (keep short)
    print("Validation complete.")
    print("Stage2 Anchor:")
    print("  TP={0} FP={1} FN={2} TN={3}".format(tp, fp, fn, tn))
    print("  Precision={0:.3f} Recall={1:.3f} F1={2:.3f}".format(prec, rec, f1))

def main():
    notes_csv = _discover_notes_csv()
    if not notes_csv:
        raise IOError("Could not auto-discover notes CSV with required columns. Put it in _staging_inputs/ or _outputs/ or run from its folder.")

    gold_csv = DEFAULT_GOLD
    if not os.path.isfile(gold_csv):
        raise IOError("Gold file not found at: {0}".format(gold_csv))

    build(notes_csv, gold_csv, OUT_DIR)

if __name__ == "__main__":
    main()
