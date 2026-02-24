#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build MRN <-> patient_id crosswalk from encounter files.

Assumptions based on your header:
- Encounters contain: MRN, ENCRYPTED_PAT_ID
- Your pipeline staging file uses: patient_id (same value-space as ENCRYPTED_PAT_ID)

Outputs (in Breast_Restore/CROSSWALK/):
- CROSSWALK__MRN_to_patient_id__vNEW.csv
- issues__oneMRN_manyPIDs.csv
- issues__onePID_manyMRNs.csv
- crosswalk_build_summary.txt
"""

from __future__ import print_function
import os
import re
import pandas as pd

# ----------------------------
# HARD-CODED PATHS (edit if needed)
# ----------------------------
BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"
ENCOUNTER_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

CLINIC_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Clinic Encounters.csv")
INPATIENT_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Inpatient Encounters.csv")
OP_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Operation Encounters.csv")

OUT_DIR = os.path.join(BREAST_RESTORE_DIR, "CROSSWALK")
OUT_XWALK = os.path.join(OUT_DIR, "CROSSWALK__MRN_to_patient_id__vNEW.csv")
OUT_ISSUE_MRN = os.path.join(OUT_DIR, "issues__oneMRN_manyPIDs.csv")
OUT_ISSUE_PID = os.path.join(OUT_DIR, "issues__onePID_manyMRNs.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "crosswalk_build_summary.txt")
# ----------------------------

def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def normalize_colname(c):
    return re.sub(r"\s+", "_", _safe_str(c).strip()).upper()

def read_csv_robust(path):
    # Py3.6/Pandas older: avoid read_csv(errors=...)
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")

def detect_col(columns, want_norm_names):
    norm_map = {c: normalize_colname(c) for c in columns}
    for want in want_norm_names:
        for orig, norm in norm_map.items():
            if norm == want:
                return orig
    return None

def clean_id_series(s):
    s = s.fillna("").astype(str).str.strip()
    s = s[s != ""]
    s = s[~s.str.lower().isin(["nan", "none", "null"])]
    return s

def load_pairs_from_file(path, tag):
    df = read_csv_robust(path)

    mrn_col = detect_col(df.columns, ["MRN"])
    pid_col = detect_col(df.columns, ["ENCRYPTED_PAT_ID", "PATIENT_ID"])

    if mrn_col is None:
        raise RuntimeError("Could not detect MRN column in: {}".format(path))
    if pid_col is None:
        raise RuntimeError("Could not detect ENCRYPTED_PAT_ID / PATIENT_ID column in: {}".format(path))

    tmp = df[[mrn_col, pid_col]].copy()
    tmp.columns = ["MRN", "patient_id"]

    tmp["MRN"] = clean_id_series(tmp["MRN"])
    tmp["patient_id"] = clean_id_series(tmp["patient_id"])

    # rows where both exist
    tmp = tmp[(tmp["MRN"] != "") & (tmp["patient_id"] != "")].copy()
    tmp["source_file_tag"] = tag
    tmp["source_file"] = os.path.basename(path)

    # de-dup exact rows
    tmp = tmp.drop_duplicates(subset=["MRN", "patient_id", "source_file"])
    return tmp

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    sources = [
        ("clinic_encounters", CLINIC_ENC),
        ("inpatient_encounters", INPATIENT_ENC),
        ("operation_encounters", OP_ENC),
    ]

    all_pairs = []
    for tag, path in sources:
        if not os.path.exists(path):
            raise RuntimeError("Missing encounter file: {}".format(path))
        p = load_pairs_from_file(path, tag)
        print("Loaded pairs:", len(p), "|", os.path.basename(path))
        all_pairs.append(p)

    pairs = pd.concat(all_pairs, axis=0, ignore_index=True)
    total_rows = len(pairs)

    # unique mapping rows
    uniq = pairs.drop_duplicates(subset=["MRN", "patient_id"]).copy()

    # aggregate provenance (which files contributed)
    prov = (
        pairs.groupby(["MRN", "patient_id"])["source_file"]
        .apply(lambda x: "|".join(sorted(set([_safe_str(v) for v in x if _safe_str(v)]))))
        .reset_index()
        .rename(columns={"source_file": "source_files"})
    )

    crosswalk = prov.copy()

    # issues
    mrn_counts = crosswalk.groupby("MRN")["patient_id"].nunique().reset_index().rename(columns={"patient_id": "n_patient_ids"})
    pid_counts = crosswalk.groupby("patient_id")["MRN"].nunique().reset_index().rename(columns={"MRN": "n_mrns"})

    oneMRN_manyPID = mrn_counts[mrn_counts["n_patient_ids"] > 1].copy()
    onePID_manyMRN = pid_counts[pid_counts["n_mrns"] > 1].copy()

    if len(oneMRN_manyPID) > 0:
        issue_mrn = crosswalk.merge(oneMRN_manyPID[["MRN"]], on="MRN", how="inner").sort_values(["MRN", "patient_id"])
        issue_mrn.to_csv(OUT_ISSUE_MRN, index=False, encoding="utf-8")
    else:
        issue_mrn = pd.DataFrame(columns=crosswalk.columns.tolist())

    if len(onePID_manyMRN) > 0:
        issue_pid = crosswalk.merge(onePID_manyMRN[["patient_id"]], on="patient_id", how="inner").sort_values(["patient_id", "MRN"])
        issue_pid.to_csv(OUT_ISSUE_PID, index=False, encoding="utf-8")
    else:
        issue_pid = pd.DataFrame(columns=crosswalk.columns.tolist())

    # write crosswalk
    crosswalk.to_csv(OUT_XWALK, index=False, encoding="utf-8")

    # summary
    lines = []
    lines.append("==== CROSSWALK BUILD SUMMARY ====")
    lines.append("Total raw pair-rows loaded (with provenance rows): {}".format(total_rows))
    lines.append("Unique MRN-patient_id pairs: {}".format(len(uniq)))
    lines.append("Unique MRNs in crosswalk: {}".format(crosswalk["MRN"].nunique()))
    lines.append("Unique patient_ids in crosswalk: {}".format(crosswalk["patient_id"].nunique()))
    lines.append("")
    lines.append("Issues: one MRN -> multiple patient_ids: {}".format(len(oneMRN_manyPID)))
    lines.append("Issues: one patient_id -> multiple MRNs: {}".format(len(onePID_manyMRN)))
    lines.append("")
    lines.append("WROTE: {}".format(OUT_XWALK))
    lines.append("WROTE: {} (rows={})".format(OUT_ISSUE_MRN, len(issue_mrn)))
    lines.append("WROTE: {} (rows={})".format(OUT_ISSUE_PID, len(issue_pid)))
    lines.append("")

    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))

if __name__ == "__main__":
    main()
