#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_WITH_AUDIT.py (Python 3.6.8 compatible)

FIX for your validation crash:
- validation_stage2_anchor_FINAL_FINAL.py expects the stage prediction file (_outputs/patient_stage_summary.csv)
  to contain a column named exactly "MRN" after it merges ENCRYPTED_PAT_ID -> MRN using op-notes id_map.
- Your current summary sometimes lacks MRN (or writes it as blank/other-case), causing KeyError: 'MRN'.

This version guarantees:
- A real "MRN" column ALWAYS exists in patient_stage_summary.csv.
- If MRN is missing in the staging input file, we build MRN using the SAME op-notes bridge file that validation uses:
    ./_staging_inputs/HPI11526 Operation Notes.csv
  (so validation sees MRN and proceeds).

Outputs:
- ./_outputs/patient_stage_summary.csv
- ./_outputs/stage2_audit_event_hits.csv
- ./_outputs/stage2_audit_bucket_counts.csv
- ./_outputs/stage2_candidate_planning_hits.csv
"""

from __future__ import print_function

import os
import re
import sys
import pandas as pd


# -------------------------
# Config (paths)
# -------------------------

ROOT = os.path.abspath(".")
IN_DIR = os.path.join(ROOT, "_staging_inputs")
OUT_DIR = os.path.join(ROOT, "_outputs")

# Default staging notes input (override by CLI arg 1)
DEFAULT_INPUT_NOTES = os.path.join(IN_DIR, "HPI11526 Operation Notes.csv")

# Op-notes bridge for ENCRYPTED_PAT_ID <-> MRN (keep identical to validation)
OP_BRIDGE_PATH = os.path.join(IN_DIR, "HPI11526 Operation Notes.csv")

OUT_SUMMARY = os.path.join(OUT_DIR, "patient_stage_summary.csv")
OUT_AUDIT_HITS = os.path.join(OUT_DIR, "stage2_audit_event_hits.csv")
OUT_AUDIT_BUCKETS = os.path.join(OUT_DIR, "stage2_audit_bucket_counts.csv")
OUT_PLANNING_HITS = os.path.join(OUT_DIR, "stage2_candidate_planning_hits.csv")

# Promotion behavior
PROMOTE_PLANNING_TO_HAS_STAGE2 = True


# -------------------------
# Helpers
# -------------------------

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))

def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df

def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def norm_str(x):
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s

def normalize_id(x):
    s = norm_str(x).strip()
    if not s:
        return ""
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit():
            return head
    return s

def normalize_mrn(x):
    return normalize_id(x)

def safe_snip(text, m, window=140):
    if text is None:
        return ""
    t = norm_str(text)
    if not t:
        return ""
    try:
        a = max(m.start() - window, 0)
        b = min(m.end() + window, len(t))
        return t[a:b].replace("\n", " ").replace("\r", " ").strip()
    except Exception:
        return (t[:2*window] + "...") if len(t) > 2*window else t

def coalesce_service_date(row, date_cols):
    for c in date_cols:
        if c and c in row and norm_str(row[c]).strip():
            return norm_str(row[c]).strip()
    return ""


# -------------------------
# Patterns
# -------------------------

INTRAOP_CONTEXT = [
    r"\bebl\b",
    r"\bestimated blood loss\b",
    r"\banesthesia\b",
    r"\bintubat(ed|ion)\b",
    r"\bdrain(s)?\b",
    r"\bjp drain(s)?\b",
    r"\bblake drain(s)?\b",
    r"\b(specimen|specimens)\b",
    r"\bsent (to )?path(ology)?\b",
    r"\bcounts? (were )?(correct|accurate)\b",
    r"\bprocedure\b",
    r"\bincision\b",
    r"\boperative (note|report)\b",
    r"\bbrief op note\b",
    r"\bfindings\b",
]

STAGE2_CORE = [
    r"\bexchange\b",
    r"\bexchanged\b",
    r"\bimplant (placement|placed|inserted)\b",
    r"\bimplant exchange\b",
    r"\btissue expander (removal|removed|remove)\b",
    r"\bexpander (removal|removed|remove)\b",
    r"\bremove(d)? (the )?tissue expander(s)?\b",
    r"\bpermanent implant(s)?\b",
]

EXCHANGE_TIGHT = r"\b(exchange (of )?(the )?(tissue )?expander(s)? (for|to) (a )?(permanent )?(silicone|saline)? ?implant(s)?|expander[- ]?to[- ]?implant exchange|exchange of tissue expander)\b"

PLANNING_ANY = [
    r"\bplan(ned|s)?\b",
    r"\bschedule(d|)\b",
    r"\bwill (proceed|undergo|have)\b",
    r"\bto be (scheduled|done)\b",
    r"\bconsent(ed|)\b",
    r"\bpre[- ]?op\b",
    r"\bpreoperative\b",
]

PLANNING_HIGHCONF = [
    r"\b(scheduled|schedule)\b.*\b(exchange|implant exchange|expander exchange|remove(d)? (the )?(tissue )?expander)\b",
    r"\bwill\b.*\b(exchange|implant exchange|expander exchange|remove(d)? (the )?(tissue )?expander)\b",
    r"\bplan(ned|)\b.*\b(exchange|implant exchange|expander exchange|remove(d)? (the )?(tissue )?expander)\b",
    r"\bconsent(ed|)\b.*\b(exchange|implant exchange|expander exchange)\b",
    r"\bto (the )?(operating room|or)\b.*\b(exchange|implant exchange|expander exchange)\b",
    r"\bprocedure\b.*\b(exchange|implant exchange|expander exchange)\b",
]

NEGATIONS = [
    r"\bno (plan|plans)\b",
    r"\bnot (planning|scheduled)\b",
    r"\bdecline(s|d)?\b.*\b(exchange|implant)\b",
    r"\bdefer(s|red)?\b.*\b(exchange|implant)\b",
]

RE_INTRAOP_CTX = re.compile("(" + "|".join(INTRAOP_CONTEXT) + ")", re.I)
RE_EXCHANGE_TIGHT = re.compile(EXCHANGE_TIGHT, re.I)
RE_STAGE2_CORE = re.compile("(" + "|".join(STAGE2_CORE) + ")", re.I)
RE_PLAN_ANY = re.compile("(" + "|".join(PLANNING_ANY) + ")", re.I)
RE_PLAN_HI = [re.compile(p, re.I) for p in PLANNING_HIGHCONF]
RE_NEG = re.compile("(" + "|".join(NEGATIONS) + ")", re.I)


# -------------------------
# Bridge: ENCRYPTED_PAT_ID <-> MRN (same file as validation)
# -------------------------

def build_id_map(op_path):
    op = normalize_cols(read_csv_robust(op_path, dtype=str, low_memory=False))
    op_mrn_col = pick_first_existing(op, ["MRN", "mrn"])
    op_enc_col = pick_first_existing(op, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not op_mrn_col or not op_enc_col:
        raise ValueError("Bridge op-notes must contain MRN and ENCRYPTED_PAT_ID. Found: {}".format(list(op.columns)))

    op["MRN"] = op[op_mrn_col].map(normalize_mrn)
    op["ENCRYPTED_PAT_ID"] = op[op_enc_col].map(normalize_id)
    id_map = op[["ENCRYPTED_PAT_ID", "MRN"]].drop_duplicates()
    id_map = id_map[(id_map["ENCRYPTED_PAT_ID"] != "") & (id_map["MRN"] != "")].copy()
    return id_map


# -------------------------
# Detection logic
# -------------------------

def detect_strict_stage2(text):
    t = norm_str(text)
    if not t:
        return (False, "", None)

    if RE_NEG.search(t):
        return (False, "", None)

    if not RE_INTRAOP_CTX.search(t):
        return (False, "", None)

    m = RE_EXCHANGE_TIGHT.search(t)
    if m:
        return (True, "INTRAOP_EXCHANGE_TIGHT", m)

    m2 = RE_STAGE2_CORE.search(t)
    if m2:
        return (True, "INTRAOP_STAGE2_CORE", m2)

    return (False, "", None)

def detect_planning(text):
    t = norm_str(text)
    if not t:
        return (False, False, "", None)

    if RE_NEG.search(t):
        return (False, False, "", None)

    if not RE_PLAN_ANY.search(t):
        return (False, False, "", None)

    core = RE_STAGE2_CORE.search(t)
    if not core:
        return (False, False, "", None)

    for r in RE_PLAN_HI:
        m = r.search(t)
        if m:
            return (True, True, "PLAN_HIGHCONF", m)

    return (True, False, "PLAN_LOWERCONF", core)


# -------------------------
# Main
# -------------------------

def main():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    input_notes = DEFAULT_INPUT_NOTES
    if len(sys.argv) > 1 and sys.argv[1].strip():
        input_notes = sys.argv[1].strip()

    if not os.path.isfile(input_notes):
        raise IOError("Input notes CSV not found: {}".format(input_notes))
    if not os.path.isfile(OP_BRIDGE_PATH):
        raise IOError("Bridge op-notes CSV not found: {}".format(OP_BRIDGE_PATH))

    id_map = build_id_map(OP_BRIDGE_PATH)

    df = normalize_cols(read_csv_robust(input_notes, dtype=str, low_memory=False))

    # Required: ENCRYPTED_PAT_ID
    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not enc_col:
        raise ValueError("Missing ENCRYPTED_PAT_ID in input notes. Found: {}".format(list(df.columns)))
    if enc_col != "ENCRYPTED_PAT_ID":
        df = df.rename(columns={enc_col: "ENCRYPTED_PAT_ID"})
    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)

    # MRN: if missing in df, we'll fill from id_map later; but ensure column exists
    mrn_col = pick_first_existing(df, ["MRN", "mrn"])
    if mrn_col and mrn_col != "MRN":
        df = df.rename(columns={mrn_col: "MRN"})
    if "MRN" not in df.columns:
        df["MRN"] = ""

    # Optional cols
    note_id_col = pick_first_existing(df, ["NOTE_ID", "NOTEID", "NOTE_ID_NUM", "NOTEID_NUM", "ENCOUNTER_NOTE_ID"])
    note_type_col = pick_first_existing(df, ["NOTE_TYPE", "NOTE_TYPE_DESC", "NOTECLASS", "DOC_TYPE", "TYPE"])
    date_col = pick_first_existing(df, ["SERVICE_DATE", "NOTE_DATE", "DATE", "VISIT_DATE", "ENCOUNTER_DATE"])

    # Text columns
    text_col = pick_first_existing(df, ["NOTE_TEXT", "TEXT", "NOTE", "DOCUMENT_TEXT", "CONTENT"])
    snip_cols = [c for c in df.columns if c.upper().startswith("SNIP_")]
    if not text_col and not snip_cols:
        raise ValueError("Missing note text. Expected NOTE_TEXT/TEXT/NOTE or SNIP_* columns. Found: {}".format(list(df.columns)))

    def get_text(row):
        if text_col and text_col in row:
            return norm_str(row[text_col])
        parts = []
        for c in snip_cols:
            v = norm_str(row.get(c, ""))
            if v.strip():
                parts.append(v.strip())
        return "\n".join(parts)

    # Patient aggregation
    audit_rows = []
    planning_rows = []
    pat = {}

    for idx in range(len(df)):
        row = df.iloc[idx]
        enc = normalize_id(row.get("ENCRYPTED_PAT_ID", ""))
        if not enc:
            continue

        mrn = normalize_mrn(row.get("MRN", ""))
        note_id = norm_str(row.get(note_id_col, "")).strip() if note_id_col else ""
        note_type = norm_str(row.get(note_type_col, "")).strip() if note_type_col else ""
        svc_date = coalesce_service_date(row, [date_col])

        text = get_text(row)

        if enc not in pat:
            pat[enc] = {
                "ENCRYPTED_PAT_ID": enc,
                "MRN": mrn,
                "HAS_STAGE2_STRICT": 0,
                "HAS_STAGE2": 0,
                "STAGE2_DATE": "",
                "STAGE2_NOTE_ID": "",
                "STAGE2_NOTE_TYPE": "",
                "STAGE2_MATCH_PATTERN": "",
                "STAGE2_HITS": 0,
                "CANDIDATE_PLANNING": 0,
                "PLANNING_HITS": 0,
                "PLANNING_HIGHCONF_HITS": 0,
                "BEST_STRICT_KEY": None,  # (date, note_id)
            }
        else:
            # keep any non-empty MRN encountered
            if (not pat[enc].get("MRN")) and mrn:
                pat[enc]["MRN"] = mrn

        # STRICT
        is_strict, bucket, m = detect_strict_stage2(text)
        if is_strict:
            pat[enc]["HAS_STAGE2_STRICT"] = 1
            pat[enc]["STAGE2_HITS"] += 1

            key = (svc_date or "9999-12-31", note_id or "ZZZ")
            best = pat[enc]["BEST_STRICT_KEY"]
            if (best is None) or (key < best):
                pat[enc]["BEST_STRICT_KEY"] = key
                pat[enc]["STAGE2_DATE"] = svc_date
                pat[enc]["STAGE2_NOTE_ID"] = note_id
                pat[enc]["STAGE2_NOTE_TYPE"] = note_type
                pat[enc]["STAGE2_MATCH_PATTERN"] = bucket

            audit_rows.append({
                "ENCRYPTED_PAT_ID": enc,
                "MRN": mrn,
                "NOTE_ID": note_id,
                "NOTE_TYPE": note_type,
                "SERVICE_DATE": svc_date,
                "BUCKET": bucket,
                "MATCH_TERM": m.group(0) if m else "",
                "SNIPPET": safe_snip(text, m) if m else (norm_str(text)[:300] if text else ""),
                "SOURCE_FILE": os.path.basename(input_notes),
            })

        # PLANNING
        is_plan, is_hi, plan_bucket, pm = detect_planning(text)
        if is_plan:
            pat[enc]["CANDIDATE_PLANNING"] = 1
            pat[enc]["PLANNING_HITS"] += 1
            if is_hi:
                pat[enc]["PLANNING_HIGHCONF_HITS"] += 1

            planning_rows.append({
                "ENCRYPTED_PAT_ID": enc,
                "MRN": mrn,
                "NOTE_ID": note_id,
                "NOTE_TYPE": note_type,
                "SERVICE_DATE": svc_date,
                "PLAN_BUCKET": plan_bucket,
                "IS_HIGHCONF": 1 if is_hi else 0,
                "MATCH_TERM": pm.group(0) if pm else "",
                "SNIPPET": safe_snip(text, pm) if pm else (norm_str(text)[:300] if text else ""),
                "SOURCE_FILE": os.path.basename(input_notes),
            })

    # Ensure MRN exists for every patient by merging id_map (same bridge as validation)
    summary_rows = []
    for enc, rec in pat.items():
        summary_rows.append({
            "ENCRYPTED_PAT_ID": rec.get("ENCRYPTED_PAT_ID", ""),
            "MRN": rec.get("MRN", ""),
            "HAS_STAGE2_STRICT": int(rec.get("HAS_STAGE2_STRICT", 0)),
            "STAGE2_DATE": rec.get("STAGE2_DATE", ""),
            "STAGE2_NOTE_ID": rec.get("STAGE2_NOTE_ID", ""),
            "STAGE2_NOTE_TYPE": rec.get("STAGE2_NOTE_TYPE", ""),
            "STAGE2_MATCH_PATTERN": rec.get("STAGE2_MATCH_PATTERN", ""),
            "STAGE2_HITS": int(rec.get("STAGE2_HITS", 0)),
            "CANDIDATE_PLANNING": int(rec.get("CANDIDATE_PLANNING", 0)),
            "PLANNING_HITS": int(rec.get("PLANNING_HITS", 0)),
            "PLANNING_HIGHCONF_HITS": int(rec.get("PLANNING_HIGHCONF_HITS", 0)),
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary[summary["ENCRYPTED_PAT_ID"] != ""].copy()

    # Fill MRN from bridge where missing/blank
    summary["MRN"] = summary["MRN"].fillna("").map(normalize_mrn)
    summary = summary.merge(id_map, on="ENCRYPTED_PAT_ID", how="left", suffixes=("", "_BRIDGE"))
    # Prefer existing MRN, else bridge
    summary["MRN"] = summary["MRN"].where(summary["MRN"] != "", summary["MRN_BRIDGE"].fillna("").map(normalize_mrn))
    summary = summary.drop(["MRN_BRIDGE"], axis=1)

    # Final HAS_STAGE2 with optional promotion
    if PROMOTE_PLANNING_TO_HAS_STAGE2:
        summary["HAS_STAGE2"] = summary.apply(
            lambda r: 1 if int(r["HAS_STAGE2_STRICT"]) == 1 or int(r["PLANNING_HIGHCONF_HITS"]) > 0 else 0,
            axis=1
        )
        # Fill metadata for promoted cases if strict metadata absent
        def fill_promoted_pattern(r):
            if int(r["HAS_STAGE2_STRICT"]) == 0 and int(r["PLANNING_HIGHCONF_HITS"]) > 0:
                if not norm_str(r["STAGE2_MATCH_PATTERN"]).strip():
                    return "PROMOTED_PLAN_HIGHCONF"
            return r["STAGE2_MATCH_PATTERN"]
        summary["STAGE2_MATCH_PATTERN"] = summary.apply(fill_promoted_pattern, axis=1)
    else:
        summary["HAS_STAGE2"] = summary["HAS_STAGE2_STRICT"].astype(int)

    # Audit dfs
    audit_df = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame(
        columns=["ENCRYPTED_PAT_ID", "MRN", "NOTE_ID", "NOTE_TYPE", "SERVICE_DATE", "BUCKET", "MATCH_TERM", "SNIPPET", "SOURCE_FILE"]
    )
    plan_df = pd.DataFrame(planning_rows) if planning_rows else pd.DataFrame(
        columns=["ENCRYPTED_PAT_ID", "MRN", "NOTE_ID", "NOTE_TYPE", "SERVICE_DATE", "PLAN_BUCKET", "IS_HIGHCONF", "MATCH_TERM", "SNIPPET", "SOURCE_FILE"]
    )

    # Bucket counts (pandas 0.24-friendly)
    bucket_df = pd.DataFrame(columns=["BUCKET", "COUNT"])
    if len(audit_df) > 0:
        bc = audit_df.groupby("BUCKET").size().reset_index(name="COUNT")
        bucket_df = bc.sort_values("COUNT", ascending=False)

    # Write outputs
    summary.to_csv(OUT_SUMMARY, index=False)
    audit_df.to_csv(OUT_AUDIT_HITS, index=False)
    bucket_df.to_csv(OUT_AUDIT_BUCKETS, index=False)
    plan_df.to_csv(OUT_PLANNING_HITS, index=False)

    # Counts
    patients = int(summary["ENCRYPTED_PAT_ID"].nunique())
    events = int(len(audit_df))
    strict_ct = int((summary["HAS_STAGE2_STRICT"] == 1).sum())
    final_ct = int((summary["HAS_STAGE2"] == 1).sum())
    planning_ct = int((summary["CANDIDATE_PLANNING"] == 1).sum())

    print("Staging complete.")
    print("Patients: {}".format(patients))
    print("Events: {}".format(events))
    print("HAS_STAGE2=1 (strict): {}".format(strict_ct))
    print("HAS_STAGE2=1 (final):  {}".format(final_ct))
    print("CANDIDATE_PLANNING=1:  {}".format(planning_ct))
    print("Wrote:")
    print("  {}".format(OUT_SUMMARY))
    print("  {}".format(OUT_AUDIT_HITS))
    print("  {}".format(OUT_AUDIT_BUCKETS))
    print("  {}".format(OUT_PLANNING_HITS))


if __name__ == "__main__":
    main()
