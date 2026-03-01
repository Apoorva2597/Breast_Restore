#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_WITH_AUDIT.py (Python 3.6.8 compatible)

Staging ONLY (+audit). No validation.

Goal:
- Produce patient_stage_summary.csv with columns expected by downstream tooling:
  ENCRYPTED_PAT_ID, HAS_STAGE2, STAGE2_DATE, STAGE2_NOTE_ID, STAGE2_NOTE_TYPE,
  STAGE2_MATCH_PATTERN, STAGE2_HITS, CANDIDATE_PLANNING

- Produce audit artifacts:
  _outputs/stage2_audit_event_hits.csv
  _outputs/stage2_audit_bucket_counts.csv
  _outputs/stage2_candidate_planning_hits.csv

Default input:
  _staging_inputs/HPI11526 Operation Notes.csv
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

# Default input (override by CLI arg 1)
DEFAULT_INPUT_NOTES = os.path.join(IN_DIR, "HPI11526 Operation Notes.csv")

OUT_SUMMARY = os.path.join(OUT_DIR, "patient_stage_summary.csv")
OUT_AUDIT_HITS = os.path.join(OUT_DIR, "stage2_audit_event_hits.csv")
OUT_AUDIT_BUCKETS = os.path.join(OUT_DIR, "stage2_audit_bucket_counts.csv")
OUT_PLANNING_HITS = os.path.join(OUT_DIR, "stage2_candidate_planning_hits.csv")

# Promotion behavior:
# - strict intraop signals => HAS_STAGE2=1
# - promote ONLY high-confidence planning phrases => HAS_STAGE2=1 (optional)
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
    # drop ".0" if numeric
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit():
            return head
    return s

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

def lower(text):
    return norm_str(text).lower()


# -------------------------
# Patterns
# -------------------------

# Intraoperative signals (to keep EXCHANGE_TIGHT precise)
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

# Core stage2 concepts (strict)
STAGE2_CORE = [
    # exchange / removal / implant placement
    r"\bexchange\b",
    r"\bexchanged\b",
    r"\bimplant (placement|placed|inserted)\b",
    r"\bimplant exchange\b",
    r"\btissue expander (removal|removed|remove)\b",
    r"\bexpander (removal|removed|remove)\b",
    r"\bremove(d)? (the )?tissue expander(s)?\b",
    r"\bpermanent implant(s)?\b",
]

# Tight exchange phrase + intraop context required
EXCHANGE_TIGHT = r"\b(exchange (of )?(the )?(tissue )?expander(s)? (for|to) (a )?(permanent )?(silicone|saline)? ?implant(s)?|expander[- ]?to[- ]?implant exchange|exchange of tissue expander)\b"

# Planning (looser) patterns
PLANNING_ANY = [
    r"\bplan(ned|s)?\b",
    r"\bschedule(d|)\b",
    r"\bwill (proceed|undergo|have)\b",
    r"\bto be (scheduled|done)\b",
    r"\bconsent(ed|)\b",
    r"\bpre[- ]?op\b",
    r"\bpreoperative\b",
    r"\bpost[- ]?op\b",
]

# High-confidence planning patterns (eligible for promotion)
PLANNING_HIGHCONF = [
    r"\b(scheduled|schedule)\b.*\b(exchange|implant exchange|expander exchange|remove(d)? (the )?(tissue )?expander)\b",
    r"\bwill\b.*\b(exchange|implant exchange|expander exchange|remove(d)? (the )?(tissue )?expander)\b",
    r"\bplan(ned|)\b.*\b(exchange|implant exchange|expander exchange|remove(d)? (the )?(tissue )?expander)\b",
    r"\bconsent(ed|)\b.*\b(exchange|implant exchange|expander exchange)\b",
    r"\bto (the )?(operating room|or)\b.*\b(exchange|implant exchange|expander exchange)\b",
    r"\bprocedure\b.*\b(exchange|implant exchange|expander exchange)\b",
]

# Negative / de-emphasis (avoid some common false positives)
NEGATIONS = [
    r"\bno (plan|plans)\b",
    r"\bnot (planning|scheduled)\b",
    r"\bdecline(s|d)?\b.*\b(exchange|implant)\b",
    r"\bdefer(s|red)?\b.*\b(exchange|implant)\b",
]


# Compile regex
RE_INTRAOP_CTX = re.compile("(" + "|".join(INTRAOP_CONTEXT) + ")", re.I)
RE_EXCHANGE_TIGHT = re.compile(EXCHANGE_TIGHT, re.I)
RE_STAGE2_CORE = re.compile("(" + "|".join(STAGE2_CORE) + ")", re.I)
RE_PLAN_ANY = re.compile("(" + "|".join(PLANNING_ANY) + ")", re.I)
RE_PLAN_HI = [re.compile(p, re.I) for p in PLANNING_HIGHCONF]
RE_NEG = re.compile("(" + "|".join(NEGATIONS) + ")", re.I)


# -------------------------
# Detection logic
# -------------------------

def detect_strict_stage2(text):
    """
    STRICT stage2: require an exchange/removal/implant pattern AND intraop context.
    Returns: (is_hit, bucket_name, match_obj)
    """
    t = norm_str(text)
    if not t:
        return (False, "", None)

    # Quick negatives
    if RE_NEG.search(t):
        # still allow intraop hits if clearly operative; but for simplicity,
        # treat as block in strict path
        return (False, "", None)

    # Must have intraop context
    if not RE_INTRAOP_CTX.search(t):
        return (False, "", None)

    # Prefer tight exchange phrase
    m = RE_EXCHANGE_TIGHT.search(t)
    if m:
        return (True, "INTRAOP_EXCHANGE_TIGHT", m)

    # Otherwise, require a core stage2 signal + intraop context
    m2 = RE_STAGE2_CORE.search(t)
    if m2:
        return (True, "INTRAOP_STAGE2_CORE", m2)

    return (False, "", None)

def detect_planning(text):
    """
    PLANNING candidate: clinic-note intent language mentioning exchange/removal/implant.
    Returns:
      (is_any_planning, is_highconf_planning, bucket_name, match_obj)
    """
    t = norm_str(text)
    if not t:
        return (False, False, "", None)

    # If explicit negative planning, skip
    if RE_NEG.search(t):
        return (False, False, "", None)

    # Must have some planning language
    if not RE_PLAN_ANY.search(t):
        return (False, False, "", None)

    # Must mention stage2 concept somewhere
    core = RE_STAGE2_CORE.search(t)
    if not core:
        return (False, False, "", None)

    # High-confidence patterns
    for r in RE_PLAN_HI:
        m = r.search(t)
        if m:
            return (True, True, "PLAN_HIGHCONF", m)

    # Otherwise low-confidence planning
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

    df = normalize_cols(read_csv_robust(input_notes, dtype=str, low_memory=False))

    # Required ID
    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not enc_col:
        raise ValueError("Missing ENCRYPTED_PAT_ID in input notes. Found: {}".format(list(df.columns)))
    if enc_col != "ENCRYPTED_PAT_ID":
        df = df.rename(columns={enc_col: "ENCRYPTED_PAT_ID"})
    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)

    # Optional cols
    mrn_col = pick_first_existing(df, ["MRN", "mrn"])
    note_id_col = pick_first_existing(df, ["NOTE_ID", "NOTEID", "NOTE_ID_NUM", "NOTEID_NUM", "ENCOUNTER_NOTE_ID"])
    note_type_col = pick_first_existing(df, ["NOTE_TYPE", "NOTE_TYPE_DESC", "NOTECLASS", "DOC_TYPE", "TYPE"])
    date_col = pick_first_existing(df, ["SERVICE_DATE", "NOTE_DATE", "DATE", "VISIT_DATE", "ENCOUNTER_DATE"])

    # Text columns: prefer NOTE_TEXT / TEXT, else stitch SNIP_* / HPI / etc.
    text_col = pick_first_existing(df, ["NOTE_TEXT", "TEXT", "NOTE", "DOCUMENT_TEXT", "CONTENT"])
    snip_cols = [c for c in df.columns if c.upper().startswith("SNIP_")]
    if not text_col and not snip_cols:
        raise ValueError("Missing note text. Expected NOTE_TEXT/TEXT/NOTE or SNIP_* columns. Found: {}".format(list(df.columns)))

    def get_text(row):
        if text_col and text_col in row:
            return norm_str(row[text_col])
        # stitch snips
        parts = []
        for c in snip_cols:
            v = norm_str(row.get(c, ""))
            if v.strip():
                parts.append(v.strip())
        return "\n".join(parts)

    # Iterate rows and build hit logs
    audit_rows = []
    planning_rows = []

    # patient-level aggregation
    # Keep earliest (date, note_id) among hits for STAGE2_DATE / NOTE_ID
    pat = {}

    for idx in range(len(df)):
        row = df.iloc[idx]
        enc = normalize_id(row.get("ENCRYPTED_PAT_ID", ""))
        if not enc:
            continue

        mrn = norm_str(row.get(mrn_col, "")).strip() if mrn_col else ""
        note_id = norm_str(row.get(note_id_col, "")).strip() if note_id_col else ""
        note_type = norm_str(row.get(note_type_col, "")).strip() if note_type_col else ""
        svc_date = coalesce_service_date(row, [date_col])

        text = get_text(row)

        # init patient
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
                "BEST_STRICT_KEY": None,   # (date, note_id)
                "BEST_PLAN_KEY": None,     # (date, note_id)
                "BEST_PLAN_BUCKET": "",
            }

        # STRICT detection
        is_strict, bucket, m = detect_strict_stage2(text)
        if is_strict:
            pat[enc]["HAS_STAGE2_STRICT"] = 1
            pat[enc]["STAGE2_HITS"] += 1

            # choose earliest hit as representative
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

        # PLANNING detection (independent logging)
        is_plan, is_hi, plan_bucket, pm = detect_planning(text)
        if is_plan:
            pat[enc]["CANDIDATE_PLANNING"] = 1
            pat[enc]["PLANNING_HITS"] += 1
            if is_hi:
                pat[enc]["PLANNING_HIGHCONF_HITS"] += 1

            keyp = (svc_date or "9999-12-31", note_id or "ZZZ")
            bestp = pat[enc]["BEST_PLAN_KEY"]
            # store earliest highconf planning as representative (else earliest planning)
            if is_hi:
                if (bestp is None) or (keyp < bestp) or (pat[enc]["BEST_PLAN_BUCKET"] != "PLAN_HIGHCONF"):
                    pat[enc]["BEST_PLAN_KEY"] = keyp
                    pat[enc]["BEST_PLAN_BUCKET"] = "PLAN_HIGHCONF"
            else:
                if (bestp is None) and (pat[enc]["BEST_PLAN_BUCKET"] == ""):
                    pat[enc]["BEST_PLAN_KEY"] = keyp
                    pat[enc]["BEST_PLAN_BUCKET"] = plan_bucket

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

    # Finalize HAS_STAGE2 with promotion rule
    has_stage2_ct = 0
    strict_ct = 0
    planning_ct = 0

    for enc, rec in pat.items():
        strict = 1 if rec.get("HAS_STAGE2_STRICT", 0) else 0
        plan_hi = 1 if rec.get("PLANNING_HIGHCONF_HITS", 0) > 0 else 0

        rec["HAS_STAGE2"] = 1 if strict else 0
        if PROMOTE_PLANNING_TO_HAS_STAGE2 and (not strict) and plan_hi:
            # Promote: represent using earliest highconf planning hit if no strict hit exists
            rec["HAS_STAGE2"] = 1
            # If we have no STAGE2_DATE from strict, set placeholder metadata so validation sees HAS_STAGE2=1
            if not rec.get("STAGE2_MATCH_PATTERN"):
                rec["STAGE2_MATCH_PATTERN"] = "PROMOTED_PLAN_HIGHCONF"
            if not rec.get("STAGE2_DATE"):
                # represent with earliest planning key if known
                kp = rec.get("BEST_PLAN_KEY")
                if kp:
                    rec["STAGE2_DATE"] = kp[0] if kp[0] != "9999-12-31" else ""
                    rec["STAGE2_NOTE_ID"] = rec.get("STAGE2_NOTE_ID") or ""
            if not rec.get("STAGE2_NOTE_TYPE"):
                rec["STAGE2_NOTE_TYPE"] = rec.get("STAGE2_NOTE_TYPE") or ""

        if rec["HAS_STAGE2"]:
            has_stage2_ct += 1
        if strict:
            strict_ct += 1
        if rec.get("CANDIDATE_PLANNING", 0):
            planning_ct += 1

    # Build summary DF (keep MRN if present, but validation only needs ENCRYPTED_PAT_ID + HAS_STAGE2 or STAGE2_DATE)
    summary_rows = []
    for enc, rec in pat.items():
        summary_rows.append({
            "ENCRYPTED_PAT_ID": rec.get("ENCRYPTED_PAT_ID", ""),
            "MRN": rec.get("MRN", ""),
            "HAS_STAGE2": int(rec.get("HAS_STAGE2", 0)),
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

    # Audit hits
    audit_df = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame(
        columns=["ENCRYPTED_PAT_ID", "MRN", "NOTE_ID", "NOTE_TYPE", "SERVICE_DATE", "BUCKET", "MATCH_TERM", "SNIPPET", "SOURCE_FILE"]
    )
    plan_df = pd.DataFrame(planning_rows) if planning_rows else pd.DataFrame(
        columns=["ENCRYPTED_PAT_ID", "MRN", "NOTE_ID", "NOTE_TYPE", "SERVICE_DATE", "PLAN_BUCKET", "IS_HIGHCONF", "MATCH_TERM", "SNIPPET", "SOURCE_FILE"]
    )

    # Bucket counts
    bucket_counts = []
    if len(audit_df) > 0:
        bc = audit_df.groupby("BUCKET", as_index=False).size()
        # pandas 0.24 compatibility: 'size' comes as column named 0 sometimes; normalize
        if "size" not in bc.columns:
            # find the size column
            size_col = [c for c in bc.columns if c != "BUCKET"][0]
            bc = bc.rename(columns={size_col: "COUNT"})
        else:
            bc = bc.rename(columns={"size": "COUNT"})
        bc = bc.sort_values("COUNT", ascending=False)
        for _, r in bc.iterrows():
            bucket_counts.append({"BUCKET": r["BUCKET"], "COUNT": int(r["COUNT"])})
    bucket_df = pd.DataFrame(bucket_counts) if bucket_counts else pd.DataFrame(columns=["BUCKET", "COUNT"])

    # Write outputs
    summary.to_csv(OUT_SUMMARY, index=False)
    audit_df.to_csv(OUT_AUDIT_HITS, index=False)
    bucket_df.to_csv(OUT_AUDIT_BUCKETS, index=False)
    plan_df.to_csv(OUT_PLANNING_HITS, index=False)

    # Print run summary
    print("Staging complete.")
    print("Patients: {}".format(int(summary["ENCRYPTED_PAT_ID"].nunique())))
    print("Events: {}".format(int(len(audit_df))))
    print("HAS_STAGE2=1 (strict): {}".format(int(strict_ct)))
    print("HAS_STAGE2=1 (final):  {}".format(int(has_stage2_ct)))
    print("CANDIDATE_PLANNING=1:  {}".format(int(planning_ct)))
    print("Wrote:")
    print("  {}".format(OUT_SUMMARY))
    print("  {}".format(OUT_AUDIT_HITS))
    print("  {}".format(OUT_AUDIT_BUCKETS))
    print("  {}".format(OUT_PLANNING_HITS))


if __name__ == "__main__":
    main()
