#!/usr/bin/env python3
# stage2_implant_from_text.py
# Python 3.6 compatible

from __future__ import print_function
import os
import re
import pandas as pd

BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"
ENCOUNTER_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

MASTER_IDS_FILE = os.path.join(BREAST_RESTORE_DIR, "MASTER__IDS_ONLY__vNEW.csv")

CLINIC_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Clinic Encounters.csv")
INPATIENT_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Inpatient Encounters.csv")
OP_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Operation Encounters.csv")

OUT_DIR = os.path.join(BREAST_RESTORE_DIR, "STAGE2_TEXT_IMPLANT")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def read_csv_robust(path):
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")

def norm(s):
    if s is None:
        return ""
    return str(s)

def norm_lower(s):
    return norm(s).lower()

def detect_col(df, candidates):
    cols = list(df.columns)
    upper_map = {c.upper(): c for c in cols}
    for cand in candidates:
        if cand.upper() in upper_map:
            return upper_map[cand.upper()]
    return None

# Tier A: high-confidence implant exchange phrases
TIER_A = [
    r"\bexpander\s+to\s+implant\b",
    r"\btissue\s+expander\s+exchange\b",
    r"\bexchange\s+(the\s+)?(tissue\s+)?expander\b",
    r"\bexpander\s+exchange\b",
    r"\bremove\s+(the\s+)?(tissue\s+)?expander\b.*\bimplant\b",
    r"\bimplant\b.*\bexchange\b",
]

# Tier B: use only if you need more sensitivity later
TIER_B = [
    r"\bimplant\s+placement\b",
    r"\binsertion\s+of\s+breast\s+implant\b",
    r"\bsecond\s+stage\b.*\bimplant\b",
]

PATTERNS = [(re.compile(p, re.I), "A") for p in TIER_A] + [(re.compile(p, re.I), "B") for p in TIER_B]

def main():
    master = read_csv_robust(MASTER_IDS_FILE)
    master_pid = detect_col(master, ["patient_id", "ENCRYPTED_PAT_ID"])
    if master_pid is None:
        raise RuntimeError("Could not find patient_id column in master file")
    cohort = set(master[master_pid].astype(str).str.strip())
    cohort = set([p for p in cohort if p and p.lower() not in ["nan", "none", "null"]])

    enc_files = [
        ("clinic", CLINIC_ENC),
        ("inpatient", INPATIENT_ENC),
        ("operation", OP_ENC),
    ]

    rows = []
    for tag, path in enc_files:
        df = read_csv_robust(path)

        pid_col = detect_col(df, ["ENCRYPTED_PAT_ID", "patient_id"])
        rov_col = detect_col(df, ["REASON_FOR_VISIT"])
        proc_col = detect_col(df, ["PROCEDURE", "PROCEDURE_NAME"])

        if pid_col is None:
            print("WARNING: no patient id col in", path)
            continue

        df["_PID_"] = df[pid_col].astype(str).str.strip()
        df = df[df["_PID_"].isin(cohort)].copy()
        if len(df) == 0:
            continue

        # Build a searchable text blob
        df["_ROV_"] = df[rov_col].astype(str) if rov_col else ""
        df["_PROC_"] = df[proc_col].astype(str) if proc_col else ""
        df["_TEXT_"] = (df["_ROV_"].fillna("").astype(str) + " | " + df["_PROC_"].fillna("").astype(str))

        # Match patterns
        match_tier = []
        match_pat = []
        for txt in df["_TEXT_"].fillna("").astype(str).tolist():
            tier = ""
            pat = ""
            t = norm(txt)
            for rx, tr in PATTERNS:
                if rx.search(t):
                    tier = tr
                    pat = rx.pattern
                    break
            match_tier.append(tier)
            match_pat.append(pat)

        df["match_tier"] = match_tier
        df["match_pattern"] = match_pat
        hits = df[df["match_tier"] != ""].copy()
        if len(hits) == 0:
            continue

        hits["source_file_tag"] = tag
        hits["source_file"] = os.path.basename(path)
        keep = ["_PID_", "match_tier", "match_pattern", "source_file_tag", "source_file"]
        if rov_col:
            keep.append(rov_col)
        if proc_col:
            keep.append(proc_col)
        if "RECONSTRUCTION_DATE" in df.columns:
            keep.append("RECONSTRUCTION_DATE")
        if "OPERATION_DATE" in df.columns:
            keep.append("OPERATION_DATE")
        if "ADMIT_DATE" in df.columns:
            keep.append("ADMIT_DATE")
        if "CHECKOUT_TIME" in df.columns:
            keep.append("CHECKOUT_TIME")
        if "PAT_ENC_CSN_ID" in df.columns:
            keep.append("PAT_ENC_CSN_ID")
        if "ENCOUNTER_TYPE" in df.columns:
            keep.append("ENCOUNTER_TYPE")
        if "DEPARTMENT" in df.columns:
            keep.append("DEPARTMENT")

        rows.append(hits[keep])

    if not rows:
        print("No text-based stage2 implant hits found.")
        return

    all_hits = pd.concat(rows, axis=0, ignore_index=True)
    all_hits.rename(columns={"_PID_": "patient_id"}, inplace=True)

    all_hits.to_csv(os.path.join(OUT_DIR, "stage2_implant_text_hits_rows.csv"), index=False, encoding="utf-8")

    # patient-level flags
    pt = (all_hits.groupby("patient_id")["match_tier"]
          .apply(lambda s: "A" if ("A" in set(s)) else "B")
          .reset_index())
    pt["has_stage2_implant_text"] = True
    pt.rename(columns={"match_tier": "best_tier"}, inplace=True)

    pt.to_csv(os.path.join(OUT_DIR, "stage2_implant_text_patient_level.csv"), index=False, encoding="utf-8")

    print("WROTE:", os.path.join(OUT_DIR, "stage2_implant_text_hits_rows.csv"))
    print("WROTE:", os.path.join(OUT_DIR, "stage2_implant_text_patient_level.csv"))
    print("Patients with implant-exchange text signal:", len(pt))

if __name__ == "__main__":
    main()
