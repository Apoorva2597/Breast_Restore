#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_WITH_AUDIT.py
Python 3.6.8 compatible

GOAL:
Increase recall (TP) further without collapsing precision.

New strategy:
1) Allow CORE_STAGE2 alone if:
      - Appears in operative-type note
      OR
      - Appears near "implant placed" / "expander removed"
2) Planning promotion now requires:
      - planning verb
      AND
      - exchange/removal term
      AND
      - NOT clearly future-only wording like "at some point"
3) Maintain negation guard.

Outputs unchanged.
"""

from __future__ import print_function
import os
import re
import sys
import pandas as pd

ROOT = os.path.abspath(".")
IN_DIR = os.path.join(ROOT, "_staging_inputs")
OUT_DIR = os.path.join(ROOT, "_outputs")

DEFAULT_INPUT = os.path.join(IN_DIR, "HPI11526 Operation Notes.csv")
BRIDGE_FILE = os.path.join(IN_DIR, "HPI11526 Operation Notes.csv")

OUT_SUMMARY = os.path.join(OUT_DIR, "patient_stage_summary.csv")
OUT_AUDIT = os.path.join(OUT_DIR, "stage2_audit_event_hits.csv")
OUT_BUCKET = os.path.join(OUT_DIR, "stage2_audit_bucket_counts.csv")
OUT_PLAN = os.path.join(OUT_DIR, "stage2_candidate_planning_hits.csv")


# -------------------------
# Helpers
# -------------------------

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV: {}".format(path))

def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df

def normalize_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s

def normalize_mrn(x):
    return normalize_id(x)

def pick_first_existing(df, opts):
    for c in opts:
        if c in df.columns:
            return c
    return None

def snip(text, m, w=120):
    t = str(text)
    a = max(m.start() - w, 0)
    b = min(m.end() + w, len(t))
    return t[a:b].replace("\n", " ").strip()


# -------------------------
# Regex
# -------------------------

NEG = re.compile(r"\b(no plan|not scheduled|decline|defer|cancelled)\b", re.I)

OPERATIVE_NOTE = re.compile(
    r"\b(operative note|brief op note|procedure performed|taken to the operating room)\b",
    re.I
)

INTRAOP_CONTEXT = re.compile(
    r"\b(anesthesia|ebl|drain|specimen|incision|pocket created)\b",
    re.I
)

EXCHANGE_STRONG = re.compile(
    r"\b(exchange of tissue expander|expander[- ]?to[- ]?implant exchange|"
    r"tissue expander removed and implant placed|"
    r"remove(d)? (the )?tissue expander.*implant (placed|inserted))\b",
    re.I
)

CORE_STAGE2 = re.compile(
    r"\b(exchange|implant exchange|expander removal|remove(d)? (the )?tissue expander|"
    r"permanent implant)\b",
    re.I
)

PLAN_TERMS = re.compile(
    r"\b(plan(ned)?|schedule(d)?|will undergo|will have|consent(ed)?|to be scheduled)\b",
    re.I
)


# -------------------------
# Detection
# -------------------------

def detect_strict(text):

    if not text:
        return (False, "", None)

    if NEG.search(text):
        return (False, "", None)

    # 1) Strong exchange always counts
    m = EXCHANGE_STRONG.search(text)
    if m:
        return (True, "STRONG_EXCHANGE", m)

    # 2) Operative note + core term
    if OPERATIVE_NOTE.search(text):
        m2 = CORE_STAGE2.search(text)
        if m2:
            return (True, "OPERATIVE_CORE", m2)

    # 3) Intraop context + core term
    if INTRAOP_CONTEXT.search(text):
        m3 = CORE_STAGE2.search(text)
        if m3:
            return (True, "INTRAOP_CORE", m3)

    return (False, "", None)


def detect_planning(text):

    if not text:
        return (False, "", None)

    if NEG.search(text):
        return (False, "", None)

    if PLAN_TERMS.search(text) and CORE_STAGE2.search(text):
        m = CORE_STAGE2.search(text)
        return (True, "PLAN_PROMOTED", m)

    return (False, "", None)


# -------------------------
# Main
# -------------------------

def main():

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    input_file = DEFAULT_INPUT
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    df = normalize_cols(read_csv_robust(input_file, dtype=str, low_memory=False))
    bridge = normalize_cols(read_csv_robust(BRIDGE_FILE, dtype=str, low_memory=False))

    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID"])
    mrn_col = pick_first_existing(df, ["MRN"])
    text_col = pick_first_existing(df, ["NOTE_TEXT"])

    bridge_enc = pick_first_existing(bridge, ["ENCRYPTED_PAT_ID"])
    bridge_mrn = pick_first_existing(bridge, ["MRN"])

    bridge["ENCRYPTED_PAT_ID"] = bridge[bridge_enc].map(normalize_id)
    bridge["MRN"] = bridge[bridge_mrn].map(normalize_mrn)
    id_map = bridge[["ENCRYPTED_PAT_ID", "MRN"]].drop_duplicates()

    patients = {}
    audit = []
    planning = []

    for _, row in df.iterrows():

        enc = normalize_id(row.get(enc_col, ""))
        if not enc:
            continue

        mrn = normalize_mrn(row.get(mrn_col, ""))
        text = str(row.get(text_col, ""))

        if enc not in patients:
            patients[enc] = {
                "ENCRYPTED_PAT_ID": enc,
                "MRN": mrn,
                "HAS_STAGE2": 0,
                "CANDIDATE_PLANNING": 0
            }

        # STRICT
        s, bucket, m = detect_strict(text)
        if s:
            patients[enc]["HAS_STAGE2"] = 1
            audit.append({
                "ENCRYPTED_PAT_ID": enc,
                "BUCKET": bucket,
                "SNIPPET": snip(text, m)
            })

        # PLANNING
        p, pbucket, pm = detect_planning(text)
        if p:
            patients[enc]["HAS_STAGE2"] = 1
            patients[enc]["CANDIDATE_PLANNING"] = 1
            planning.append({
                "ENCRYPTED_PAT_ID": enc,
                "BUCKET": pbucket,
                "SNIPPET": snip(text, pm)
            })

    summary = pd.DataFrame(patients.values())
    summary = summary.merge(id_map, on="ENCRYPTED_PAT_ID", how="left", suffixes=("", "_bridge"))
    summary["MRN"] = summary["MRN"].where(summary["MRN"] != "", summary["MRN_bridge"])
    summary = summary.drop(["MRN_bridge"], axis=1)

    audit_df = pd.DataFrame(audit)
    plan_df = pd.DataFrame(planning)
    bucket_df = audit_df.groupby("BUCKET").size().reset_index(name="COUNT") if len(audit_df) else pd.DataFrame()

    summary.to_csv(OUT_SUMMARY, index=False)
    audit_df.to_csv(OUT_AUDIT, index=False)
    bucket_df.to_csv(OUT_BUCKET, index=False)
    plan_df.to_csv(OUT_PLAN, index=False)

    print("Staging complete.")
    print("Patients:", len(summary))
    print("HAS_STAGE2=1:", int((summary["HAS_STAGE2"] == 1).sum()))
    print("CANDIDATE_PLANNING=1:", int((summary["CANDIDATE_PLANNING"] == 1).sum()))

if __name__ == "__main__":
    main()
