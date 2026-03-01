#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_stage12_WITH_AUDIT.py
Revised strict Stage2 logic (captures definitive implant state
even in progress notes).

Python 3.6.8 compatible
"""

from __future__ import print_function
import os
import re
import pandas as pd


# ==============================
# PATHS
# ==============================

ROOT = os.path.abspath(".")
INPUT_NOTES = os.path.join(ROOT, "_staging_inputs", "HPI11526 Operation Notes.csv")

OUT_DIR = os.path.join(ROOT, "_outputs")
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

OUT_SUMMARY = os.path.join(OUT_DIR, "patient_stage_summary.csv")
OUT_AUDIT = os.path.join(OUT_DIR, "stage2_audit_event_hits.csv")
OUT_BUCKET = os.path.join(OUT_DIR, "stage2_audit_bucket_counts.csv")
OUT_PLANNING = os.path.join(OUT_DIR, "stage2_candidate_planning_hits.csv")


# ==============================
# HELPERS
# ==============================

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Could not read CSV: {}".format(path))


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


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


# ==============================
# REGEX PATTERNS
# ==============================

# High confidence procedural patterns
PATTERN_EXCHANGE = re.compile(r"(exchange|exchanged).*?(expander).*?(implant)", re.I)
PATTERN_EXPANDER_REMOVED = re.compile(r"(expander).*?(removed|removal)", re.I)
PATTERN_SECOND_STAGE = re.compile(r"(second stage reconstruction|stage\s*2 reconstruction)", re.I)
PATTERN_IMPLANT_PLACEMENT = re.compile(r"(implant).*?(placement|inserted|insertion)", re.I)
PATTERN_SP_EXPANDER = re.compile(r"s/p.*expander.*implant", re.I)

# Definitive implant-state language (allowed in progress notes)
PATTERN_IMPLANTS_IN = re.compile(
    r"(silicone implants? (are|were) in|"
    r"implants? (are|were) in|"
    r"now with silicone implants|"
    r"status post exchange.*implant)",
    re.I
)

# Planning exclusion
PATTERN_PLANNING = re.compile(
    r"(planning|considering|candidate for).*?(implant|reconstruction)",
    re.I
)


# ==============================
# STAGE 2 DETECTION
# ==============================

def detect_stage2_strict(text):
    if not text:
        return 0

    if (
        PATTERN_EXCHANGE.search(text)
        or PATTERN_EXPANDER_REMOVED.search(text)
        or PATTERN_SECOND_STAGE.search(text)
        or PATTERN_IMPLANT_PLACEMENT.search(text)
        or PATTERN_SP_EXPANDER.search(text)
        or PATTERN_IMPLANTS_IN.search(text)
    ):
        return 1

    return 0


def detect_candidate_planning(text):
    if not text:
        return 0
    return 1 if PATTERN_PLANNING.search(text) else 0


# ==============================
# MAIN
# ==============================

def main():

    if not os.path.isfile(INPUT_NOTES):
        raise IOError("Missing input notes file: {}".format(INPUT_NOTES))

    df = normalize_cols(read_csv_robust(INPUT_NOTES, dtype=str, low_memory=False))

    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID"])
    mrn_col = pick_first_existing(df, ["MRN", "mrn"])
    text_col = pick_first_existing(df, ["NOTE_TEXT", "NOTE_TEXT_DEID", "TEXT"])
    type_col = pick_first_existing(df, ["NOTE_TYPE", "NOTE TYPE"])

    if not enc_col or not mrn_col or not text_col:
        raise ValueError("Missing required columns.")

    df["ENCRYPTED_PAT_ID"] = df[enc_col].map(normalize_id)
    df["MRN"] = df[mrn_col].map(normalize_id)
    df["NOTE_TEXT"] = df[text_col].fillna("")
    df["NOTE_TYPE"] = df[type_col] if type_col else ""

    strict_flags = []
    planning_flags = []
    audit_rows = []

    for _, row in df.iterrows():
        pid = row["ENCRYPTED_PAT_ID"]
        mrn = row["MRN"]
        text = row["NOTE_TEXT"]

        strict = detect_stage2_strict(text)
        planning = detect_candidate_planning(text)

        # Exclude pure planning if no definitive implant state
        if planning == 1 and strict == 0:
            strict = 0

        strict_flags.append(strict)
        planning_flags.append(planning)

        if strict == 1:
            audit_rows.append({
                "ENCRYPTED_PAT_ID": pid,
                "MRN": mrn,
                "NOTE_TYPE": row["NOTE_TYPE"],
                "SNIPPET": text[:600]
            })

    df["HAS_STAGE2"] = strict_flags
    df["CANDIDATE_PLANNING"] = planning_flags

    summary = df.groupby(["ENCRYPTED_PAT_ID", "MRN"], as_index=False).agg({
        "HAS_STAGE2": "max",
        "CANDIDATE_PLANNING": "max"
    })

    summary.to_csv(OUT_SUMMARY, index=False)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUT_AUDIT, index=False)

    bucket_counts = pd.DataFrame({
        "Metric": [
            "Patients",
            "HAS_STAGE2=1",
            "CANDIDATE_PLANNING=1"
        ],
        "Count": [
            len(summary),
            int(summary["HAS_STAGE2"].sum()),
            int(summary["CANDIDATE_PLANNING"].sum())
        ]
    })

    bucket_counts.to_csv(OUT_BUCKET, index=False)

    planning_hits = df[df["CANDIDATE_PLANNING"] == 1][
        ["ENCRYPTED_PAT_ID", "MRN", "NOTE_TYPE"]
    ]
    planning_hits.to_csv(OUT_PLANNING, index=False)

    print("Staging complete.")
    print("Patients:", len(summary))
    print("HAS_STAGE2=1:", int(summary["HAS_STAGE2"].sum()))
    print("CANDIDATE_PLANNING=1:", int(summary["CANDIDATE_PLANNING"].sum()))
    print("Wrote:")
    print(" ", OUT_SUMMARY)
    print(" ", OUT_AUDIT)
    print(" ", OUT_BUCKET)
    print(" ", OUT_PLANNING)


if __name__ == "__main__":
    main()
