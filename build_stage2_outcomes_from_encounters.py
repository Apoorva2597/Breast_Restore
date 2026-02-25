#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage2_outcomes_from_encounters.py (Python 3.6.8 compatible)

Run from: ~/Breast_Restore

Inputs (auto-detected):
- Stage2 anchors:
    Preferred: ./_frozen_stage2/<latest_run>/stage2_patient_clean.csv
    Fallback:  ./_outputs/patient_stage_summary.csv
- Encounters (any that exist in the HPI folder you staged from, OR copied into _staging_inputs):
    HPI11526 Clinic Encounters.csv
    HPI11526 Inpatient Encounters.csv
    HPI11526 Operation Encounters.csv

Outputs:
- ./_outputs/stage2_outcomes_pred.csv

What it does:
- For patients with Stage2 date, builds 1-year window post Stage2 date
- Uses encounter PROCEDURE / CPT_CODE / REASON_FOR_VISIT text to flag:
    Stage2_Reoperation_pred
    Stage2_Rehospitalization_pred
    Stage2_MajorComp_pred = OR(Reop, Rehosp)
    Stage2_Failure_pred
    Stage2_Revision_pred
- Writes patient-level predictions + basic evidence (first matched encounter/date/source)
"""

from __future__ import print_function

import os
import re
import csv
import glob
from datetime import datetime, timedelta

# -------------------------
# Paths / Auto-detect
# -------------------------

ROOT = os.getcwd()
OUT_DIR = os.path.join(ROOT, "_outputs")
FROZEN_DIR = os.path.join(ROOT, "_frozen_stage2")
STAGING_DIR = os.path.join(ROOT, "_staging_inputs")

def _safe_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def _exists(p):
    return p and os.path.isfile(p)

def _latest_frozen_run_dir():
    if not os.path.isdir(FROZEN_DIR):
        return ""
    run_dirs = sorted([d for d in glob.glob(os.path.join(FROZEN_DIR, "*")) if os.path.isdir(d)])
    return run_dirs[-1] if run_dirs else ""

def pick_stage2_anchor_csv():
    # Prefer frozen artifacts
    latest = _latest_frozen_run_dir()
    if latest:
        cand = os.path.join(latest, "stage2_patient_clean.csv")
        if _exists(cand):
            return cand

    # Fallback to current outputs
    cand = os.path.join(ROOT, "_outputs", "patient_stage_summary.csv")
    if _exists(cand):
        return cand

    raise IOError("Could not find Stage2 anchor CSV. Expected in _frozen_stage2/*/stage2_patient_clean.csv or _outputs/patient_stage_summary.csv")

def _guess_hpi_dir():
    """
    Best-effort find where encounter CSVs are, by looking:
    1) _staging_inputs
    2) any /home/*/my_data_Breast/.../HPI11526 folder mentioned earlier is not discoverable reliably,
       so we only use local relative discovery.
    """
    # If user copied encounter files into _staging_inputs, use that
    if os.path.isdir(STAGING_DIR):
        return STAGING_DIR
    return ROOT

def pick_encounter_files():
    base = _guess_hpi_dir()

    # Try common locations: staging dir, then current dir, then recursive under ./_staging_inputs and ./my_data*
    patterns = [
        os.path.join(base, "HPI11526 Clinic Encounters.csv"),
        os.path.join(base, "HPI11526 Inpatient Encounters.csv"),
        os.path.join(base, "HPI11526 Operation Encounters.csv"),
        os.path.join(ROOT, "HPI11526 Clinic Encounters.csv"),
        os.path.join(ROOT, "HPI11526 Inpatient Encounters.csv"),
        os.path.join(ROOT, "HPI11526 Operation Encounters.csv"),
    ]

    found = {}
    for p in patterns:
        name = os.path.basename(p)
        if _exists(p):
            found[name] = p

    # If not found, try broad search (bounded)
    if len(found) < 1:
        for name in ["HPI11526 Clinic Encounters.csv", "HPI11526 Inpatient Encounters.csv", "HPI11526 Operation Encounters.csv"]:
            hits = glob.glob(os.path.join(ROOT, "**", name), recursive=True)
            if hits:
                found[name] = hits[0]

    return found

# -------------------------
# Parsing helpers
# -------------------------

def _normalize(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", "\n").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _parse_date_any(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%y %H:%M",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass

    # fallback: pull date token
    m = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            pass

    m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", s)
    if m:
        token = m.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(token, fmt).date()
            except Exception:
                pass

    return None

def _to_int_safe(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def read_csv_rows(path, encoding_list=None):
    if encoding_list is None:
        encoding_list = ["utf-8", "latin-1", "cp1252"]

    last_err = None
    for enc in encoding_list:
        try:
            with open(path, "r", encoding=enc, errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            last_err = e

    raise last_err

# -------------------------
# Signal definitions (fast, conservative defaults)
# Adjust later if needed.
# -------------------------

# Rehospitalization: inpatient encounter in window
# We'll use presence of Inpatient file + date within window as signal.

# Reoperation / Revision / Failure: keyword + optional CPT hints
REVISION_KWS = [
    r"\brevision\b", r"\bscar revision\b", r"\bcontour\b", r"\basymmetr(y|ies)\b",
    r"\bcapsulotom(y|ies)\b", r"\bcapsulectom(y|ies)\b", r"\bmalposition\b",
]
FAILURE_KWS = [
    r"\bexplant\b", r"\bimplant removal\b", r"\bremove(d)? implant\b",
    r"\bexpander removal\b", r"\bremove(d)? expander\b",
    r"\bextrusion\b", r"\bexposed implant\b", r"\bflap loss\b", r"\bnecrosis\b.*\bimplant\b",
]
REOP_KWS = [
    r"\breturn to (the )?or\b", r"\bre-?operation\b", r"\breoperation\b", r"\bwashout\b",
    r"\bdebridement\b", r"\bdrainage\b", r"\bhematoma evacuation\b",
    r"\bimplant exchange\b", r"\bexchange\b.*\bimplant\b", r"\breplace(ment)?\b.*\bimplant\b",
    r"\bremov(al|e)\b.*\b(expander|implant)\b",
]

# CPT hints (very light-touch; you can expand later)
# Common breast implant insertion/exchange/removal codes include 19340 (insert), 19342 (delayed insert),
# 19357 (tissue expander), 19328 (remove), etc. We use broad matching by integer when present.
CPT_REOP_HINT = set([19328, 19340, 19342, 19357, 19370])  # conservative starter set
CPT_REVISION_HINT = set([19370])  # revision of reconstructed breast
CPT_FAILURE_HINT = set([19328])   # removal of implant material

def any_kw_match(text, patterns):
    for pat in patterns:
        if re.search(pat, text):
            return pat
    return ""

# -------------------------
# Core logic
# -------------------------

def load_stage2_anchors(stage2_csv):
    rows = read_csv_rows(stage2_csv)

    # Accept either frozen "stage2_patient_clean.csv" or patient_stage_summary.csv
    # We need: ENCRYPTED_PAT_ID and STAGE2_DATE.
    anchors = {}  # pid -> date
    for r in rows:
        pid = (r.get("ENCRYPTED_PAT_ID") or "").strip()
        if not pid:
            continue

        # possible column names
        d = (r.get("STAGE2_DATE") or r.get("stage2_date") or "").strip()
        dt = _parse_date_any(d)
        if dt:
            anchors[pid] = dt

    if not anchors:
        raise ValueError("No Stage2 dates parsed from anchor file: {0}".format(stage2_csv))
    return anchors

def encounter_row_date(row):
    # different files have different date columns
    for k in ["ADMIT_DATE", "HOSP_ADMSN_TIME", "OPERATION_DATE", "RECONSTRUCTION_DATE", "CHECKOUT_TIME", "DISCHARGE_DATE_DT", "HOSP_DISCHRG_TIME"]:
        if k in row:
            dt = _parse_date_any(row.get(k))
            if dt:
                return dt
    return None

def encounter_text_blob(row):
    parts = []
    for k in ["PROCEDURE", "REASON_FOR_VISIT", "CPT_CODE", "ENCOUNTER_TYPE", "OP_DEPARTMENT", "DEPARTMENT"]:
        if k in row and row.get(k):
            parts.append(str(row.get(k)))
    return _normalize(" | ".join(parts))

def encounter_cpt(row):
    if "CPT_CODE" not in row:
        return None
    return _to_int_safe(row.get("CPT_CODE"))

def in_window(dt, start, end):
    return (dt is not None) and (dt >= start) and (dt <= end)

def build_predictions(anchors, encounter_files):
    """
    Returns dict pid -> pred + evidence
    """
    # Initialize output structure
    out = {}
    for pid, s2_date in anchors.items():
        out[pid] = {
            "ENCRYPTED_PAT_ID": pid,
            "STAGE2_DATE": s2_date.strftime("%Y-%m-%d"),
            "WINDOW_START": s2_date.strftime("%Y-%m-%d"),
            "WINDOW_END": (s2_date + timedelta(days=365)).strftime("%Y-%m-%d"),

            "Stage2_Reoperation_pred": 0,
            "Stage2_Rehospitalization_pred": 0,
            "Stage2_MajorComp_pred": 0,
            "Stage2_Failure_pred": 0,
            "Stage2_Revision_pred": 0,

            # evidence
            "reop_evidence_date": "",
            "reop_evidence_source": "",
            "reop_evidence_pattern": "",
            "rehosp_evidence_date": "",
            "rehosp_evidence_source": "",
            "failure_evidence_date": "",
            "failure_evidence_source": "",
            "failure_evidence_pattern": "",
            "revision_evidence_date": "",
            "revision_evidence_source": "",
            "revision_evidence_pattern": "",
        }

    # Helper to set evidence only once (earliest)
    def _set_evidence(pid, key_prefix, dt, source, pattern=""):
        rec = out[pid]
        date_key = key_prefix + "_evidence_date"
        if not rec[date_key] or (dt and rec[date_key] and dt.strftime("%Y-%m-%d") < rec[date_key]):
            rec[date_key] = dt.strftime("%Y-%m-%d") if dt else rec[date_key]
            rec[key_prefix + "_evidence_source"] = source
            if pattern:
                rec[key_prefix + "_evidence_pattern"] = pattern

    # Process each encounter file
    for fname, fpath in encounter_files.items():
        rows = read_csv_rows(fpath)
        source = os.path.basename(fpath)

        is_inpatient = ("Inpatient Encounters" in fname)

        for r in rows:
            pid = (r.get("ENCRYPTED_PAT_ID") or "").strip()
            if not pid or pid not in out:
                continue

            s2 = anchors[pid]
            start = s2
            end = s2 + timedelta(days=365)

            dt = encounter_row_date(r)
            if not in_window(dt, start, end):
                continue

            blob = encounter_text_blob(r)
            cpt = encounter_cpt(r)

            # Rehospitalization: any inpatient encounter in window
            if is_inpatient:
                if out[pid]["Stage2_Rehospitalization_pred"] == 0:
                    out[pid]["Stage2_Rehospitalization_pred"] = 1
                    _set_evidence(pid, "rehosp", dt, source)

            # Failure
            pat = any_kw_match(blob, FAILURE_KWS)
            if (not pat) and (cpt in CPT_FAILURE_HINT):
                pat = "CPT_HINT_{0}".format(cpt)
            if pat and out[pid]["Stage2_Failure_pred"] == 0:
                out[pid]["Stage2_Failure_pred"] = 1
                _set_evidence(pid, "failure", dt, source, pat)

            # Revision
            pat = any_kw_match(blob, REVISION_KWS)
            if (not pat) and (cpt in CPT_REVISION_HINT):
                pat = "CPT_HINT_{0}".format(cpt)
            if pat and out[pid]["Stage2_Revision_pred"] == 0:
                out[pid]["Stage2_Revision_pred"] = 1
                _set_evidence(pid, "revision", dt, source, pat)

            # Reoperation
            pat = any_kw_match(blob, REOP_KWS)
            if (not pat) and (cpt in CPT_REOP_HINT):
                pat = "CPT_HINT_{0}".format(cpt)
            if pat and out[pid]["Stage2_Reoperation_pred"] == 0:
                out[pid]["Stage2_Reoperation_pred"] = 1
                _set_evidence(pid, "reop", dt, source, pat)

    # Derive MajorComp
    for pid in out:
        rec = out[pid]
        rec["Stage2_MajorComp_pred"] = 1 if (rec["Stage2_Reoperation_pred"] == 1 or rec["Stage2_Rehospitalization_pred"] == 1) else 0

    return out

def write_output(preds, out_path):
    fieldnames = [
        "ENCRYPTED_PAT_ID", "STAGE2_DATE", "WINDOW_START", "WINDOW_END",
        "Stage2_Reoperation_pred", "Stage2_Rehospitalization_pred", "Stage2_MajorComp_pred",
        "Stage2_Failure_pred", "Stage2_Revision_pred",
        "reop_evidence_date", "reop_evidence_source", "reop_evidence_pattern",
        "rehosp_evidence_date", "rehosp_evidence_source",
        "failure_evidence_date", "failure_evidence_source", "failure_evidence_pattern",
        "revision_evidence_date", "revision_evidence_source", "revision_evidence_pattern",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pid in sorted(preds.keys()):
            w.writerow(preds[pid])

def main():
    _safe_mkdir(OUT_DIR)

    stage2_csv = pick_stage2_anchor_csv()
    encounter_files = pick_encounter_files()

    if not encounter_files:
        raise IOError("No encounter files found. Copy encounter CSVs into _staging_inputs or keep them somewhere under Breast_Restore so the script can find them.")

    print("Using Stage2 anchors:", stage2_csv)
    print("Using encounter files:")
    for k in sorted(encounter_files.keys()):
        print(" -", k, "=>", encounter_files[k])

    anchors = load_stage2_anchors(stage2_csv)
    print("Stage2 anchors loaded:", len(anchors))

    preds = build_predictions(anchors, encounter_files)

    # Quick counts
    n = len(preds)
    reop = sum(1 for pid in preds if preds[pid]["Stage2_Reoperation_pred"] == 1)
    reh = sum(1 for pid in preds if preds[pid]["Stage2_Rehospitalization_pred"] == 1)
    maj = sum(1 for pid in preds if preds[pid]["Stage2_MajorComp_pred"] == 1)
    fail = sum(1 for pid in preds if preds[pid]["Stage2_Failure_pred"] == 1)
    rev = sum(1 for pid in preds if preds[pid]["Stage2_Revision_pred"] == 1)

    print("Counts (pred):")
    print(" - Reoperation:", reop)
    print(" - Rehospitalization:", reh)
    print(" - MajorComp (OR):", maj)
    print(" - Failure:", fail)
    print(" - Revision:", rev)

    out_path = os.path.join(OUT_DIR, "stage2_outcomes_pred.csv")
    write_output(preds, out_path)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
