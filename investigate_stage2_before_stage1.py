#!/usr/bin/env python3
# investigate_stage2_before_stage1.py
# Python 3.6.8 compatible

from __future__ import print_function
import os
import re
import pandas as pd

BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"
ENCOUNTER_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

STAGING_FILE = os.path.join(BREAST_RESTORE_DIR, "MASTER__STAGING_PATHWAY__vNEW.csv")

CLINIC_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Clinic Encounters.csv")
INPATIENT_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Inpatient Encounters.csv")
OP_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Operation Encounters.csv")

OUT_DIR = os.path.join(BREAST_RESTORE_DIR, "VALIDATION_REPORTS")
OUT_CSV = os.path.join(OUT_DIR, "stage2_before_stage1__encounter_rows.csv")

CPT_OF_INTEREST = set(["19357", "19364", "19380", "19350"])


def read_csv_robust(path):
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")


def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def normalize_colname(c):
    return re.sub(r"\s+", "_", _safe_str(c).strip()).upper()


def detect_pid_col(columns):
    norm_map = {c: normalize_colname(c) for c in columns}
    preferred = ["PATIENT_ID", "ENCRYPTED_PAT_ID"]
    for want in preferred:
        for orig, norm in norm_map.items():
            if norm == want:
                return orig
    for orig, norm in norm_map.items():
        if "PAT" in norm and "ID" in norm:
            return orig
    return None


def detect_cpt_col(columns):
    norm_map = {c: normalize_colname(c) for c in columns}
    preferred = ["CPT_CODE", "CPT", "CPTCD"]
    for want in preferred:
        for orig, norm in norm_map.items():
            if norm == want:
                return orig
    for orig, norm in norm_map.items():
        if "CPT" in norm and "CODE" in norm:
            return orig
    for orig, norm in norm_map.items():
        if norm == "CPT":
            return orig
    return None


def detect_date_cols(columns):
    norm_map = {c: normalize_colname(c) for c in columns}
    ranked_keys = [
        "OPERATION_DATE",
        "RECONSTRUCTION_DATE",
        "ADMIT_DATE",
        "DISCHARGE_DATE_DT",
        "CHECKOUT_TIME",
        "HOSP_ADMSN_TIME",
        "HOSP_DISCHRG_TIME",
        "ENCOUNTER_DATE",
        "VISIT_DATE",
        "DATE",
        "DATETIME",
        "TIME",
    ]
    date_like = []
    for orig, norm in norm_map.items():
        if any(k in norm for k in ["DATE", "TIME", "DT", "DATETIME"]):
            date_like.append(orig)

    ordered = []
    for k in ranked_keys:
        for orig in date_like:
            if k in normalize_colname(orig) and orig not in ordered:
                ordered.append(orig)
    for orig in date_like:
        if orig not in ordered:
            ordered.append(orig)
    return ordered


def pick_best_event_dt_row(row, date_cols):
    for c in date_cols:
        v = _safe_str(row.get(c, "")).strip()
        if v and v.lower() not in ["nan", "none", "na", "null"]:
            return v, c
    return "", ""


def parse_dt(series):
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def std_cpt(x):
    s = _safe_str(x).replace('"', "").replace("'", "").strip()
    return s


def load_encounter_file(tag, path, cohort_pids):
    df = read_csv_robust(path)

    pid_col = detect_pid_col(df.columns)
    cpt_col = detect_cpt_col(df.columns)
    date_cols = detect_date_cols(df.columns)

    if pid_col is None:
        raise RuntimeError("Could not detect patient id column in: {}".format(path))
    if cpt_col is None:
        raise RuntimeError("Could not detect CPT column in: {}".format(path))

    df["_PID_"] = df[pid_col].astype(str).str.strip()
    df = df[df["_PID_"].isin(cohort_pids)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    date_cols_effective = [c for c in date_cols if c in df.columns]

    best_dt_raw = []
    best_dt_src = []
    for _, r in df.iterrows():
        raw, src = pick_best_event_dt_row(r, date_cols_effective)
        best_dt_raw.append(raw)
        best_dt_src.append(src)

    df["BEST_EVENT_DT_RAW"] = best_dt_raw
    df["BEST_EVENT_DT_SRC"] = best_dt_src
    df["BEST_EVENT_DT_PARSED"] = parse_dt(df["BEST_EVENT_DT_RAW"])

    df["CPT_CODE_STD"] = df[cpt_col].apply(std_cpt)
    df["SOURCE_FILE_TAG"] = tag
    df["SOURCE_FILE"] = os.path.basename(path)

    keep = [
        "_PID_", "CPT_CODE_STD",
        "BEST_EVENT_DT_RAW", "BEST_EVENT_DT_SRC", "BEST_EVENT_DT_PARSED",
        "SOURCE_FILE_TAG", "SOURCE_FILE"
    ]
    # keep helpful context if present
    for extra in ["PAT_ENC_CSN_ID", "ENCOUNTER_TYPE", "DEPARTMENT", "OP_DEPARTMENT", "REASON_FOR_VISIT", "PROCEDURE"]:
        if extra in df.columns:
            keep.append(extra)

    return df[keep].copy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    stg = read_csv_robust(STAGING_FILE)
    # expected columns in your staging output:
    # patient_id, stage1_date, stage2_date, has_expander, has_stage2_definitive
    for col in ["patient_id", "stage1_date", "stage2_date", "has_expander", "has_stage2_definitive"]:
        if col not in stg.columns:
            raise RuntimeError("Missing expected column in staging file: {}".format(col))

    stg["stage1_dt"] = pd.to_datetime(stg["stage1_date"], errors="coerce")
    stg["stage2_dt"] = pd.to_datetime(stg["stage2_date"], errors="coerce")

    bad = stg[
        (stg["has_expander"].astype(str).str.lower() == "true") &
        (stg["has_stage2_definitive"].astype(str).str.lower() == "true") &
        (stg["stage1_dt"].notnull()) &
        (stg["stage2_dt"].notnull()) &
        (stg["stage2_dt"] < stg["stage1_dt"])
    ].copy()

    print("Stage2 before Stage1 patients:", len(bad))
    if len(bad) == 0:
        print("None found. Exiting.")
        return

    cohort_pids = set(bad["patient_id"].astype(str).str.strip().tolist())

    # load encounters for these 4 only
    enc_parts = []
    enc_parts.append(load_encounter_file("clinic_encounters", CLINIC_ENC, cohort_pids))
    enc_parts.append(load_encounter_file("inpatient_encounters", INPATIENT_ENC, cohort_pids))
    enc_parts.append(load_encounter_file("operation_encounters", OP_ENC, cohort_pids))

    enc = pd.concat([x for x in enc_parts if len(x) > 0], ignore_index=True)

    # filter to CPTs of interest
    enc = enc[enc["CPT_CODE_STD"].isin(CPT_OF_INTEREST)].copy()

    # attach staging dates for convenience
    keep_stg = bad[["patient_id", "stage1_date", "stage2_date"]].copy()
    keep_stg = keep_stg.rename(columns={"patient_id": "_PID_"})
    enc = enc.merge(keep_stg, on="_PID_", how="left")

    # sort to make review easy
    enc = enc.sort_values(by=["_PID_", "BEST_EVENT_DT_PARSED", "CPT_CODE_STD", "SOURCE_FILE_TAG"], ascending=True)

    enc.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("WROTE:", OUT_CSV)
    print("\nPatients:")
    for pid in sorted(cohort_pids):
        s1 = bad.loc[bad["patient_id"] == pid, "stage1_date"].iloc[0]
        s2 = bad.loc[bad["patient_id"] == pid, "stage2_date"].iloc[0]
        print("  {} | stage1={} | stage2={}".format(pid, s1, s2))

    print("\nNext step: open the CSV and look for BEST_EVENT_DT_SRC differences causing ordering.")


if __name__ == "__main__":
    main()
