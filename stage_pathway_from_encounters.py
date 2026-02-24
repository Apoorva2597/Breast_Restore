#!/usr/bin/env python3
# stage_pathway_from_encounters.py
# Python 3.6.8 compatible
#
# Purpose:
#   Build NEW patient-level pathway staging using encounter files (not note text).
#   Stage 1 anchor: 19357 (tissue expander placement)
#   Stage 2 definitive anchor: 19364 (free flap reconstruction)
#   Secondary/QA flags: 19380 (revision), 19350 (nipple/areola reconstruction)
#
# Inputs (hardcoded per your paths):
#   - Master cohort ids (patient_id only): ~/Breast_Restore/<MASTER_IDS_FILE>
#   - Encounter files: /home/apokol/my_data_Breast/HPI-11526/HPI11256/
#
# Output:
#   - ~/Breast_Restore/MASTER__STAGING_PATHWAY__vNEW.csv

from __future__ import print_function
import os
import re
import pandas as pd


# ----------------------------
# HARD-CODED PATHS (edit if needed)
# ----------------------------
BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"
ENCOUNTER_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

MASTER_IDS_FILE = os.path.join(BREAST_RESTORE_DIR, "MASTER__IDS_ONLY__vNEW.csv")

CLINIC_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Clinic Encounters.csv")
INPATIENT_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Inpatient Encounters.csv")
OP_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Operation Encounters.csv")

OUT_FILE = os.path.join(BREAST_RESTORE_DIR, "MASTER__STAGING_PATHWAY__vNEW.csv")


# ----------------------------
# CPT definitions (current best option)
# ----------------------------
CPT_STAGE1_EXPANDER = set(["19357"])
CPT_STAGE2_DEFINITIVE = set(["19364"])     # conservative + defensible for your dataset
CPT_REVISION = set(["19380"])
CPT_NIPPLE = set(["19350"])


# ----------------------------
# Helpers
# ----------------------------
def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def read_csv_robust(path):
    # Pandas versions on py3.6 often don't support read_csv(errors=...)
    # Try utf-8 first, then latin1 as fallback.
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")

def normalize_colname(c):
    return re.sub(r"\s+", "_", _safe_str(c).strip()).upper()

def detect_pid_col(columns):
    # Prefer patient_id (your new master file), but support common alternatives
    norm_map = {c: normalize_colname(c) for c in columns}

    # Strong candidates in order
    preferred = ["PATIENT_ID", "ENCRYPTED_PAT_ID"]
    for want in preferred:
        for orig, norm in norm_map.items():
            if norm == want:
                return orig

    # Heuristics
    for orig, norm in norm_map.items():
        if "PAT" in norm and "ID" in norm:
            return orig
    for orig, norm in norm_map.items():
        if "PATIENT" in norm and "ID" in norm:
            return orig
    return None

def detect_cpt_col(columns):
    norm_map = {c: normalize_colname(c) for c in columns}
    preferred = ["CPT_CODE", "CPT", "CPTCD"]
    for want in preferred:
        for orig, norm in norm_map.items():
            if norm == want:
                return orig
    # heuristics
    for orig, norm in norm_map.items():
        if "CPT" in norm and "CODE" in norm:
            return orig
    for orig, norm in norm_map.items():
        if norm == "CPT":
            return orig
    return None

def detect_procedure_col(columns):
    norm_map = {c: normalize_colname(c) for c in columns}
    preferred = ["PROCEDURE", "PROC", "PROCEDURE_NAME"]
    for want in preferred:
        for orig, norm in norm_map.items():
            if norm == want:
                return orig
    for orig, norm in norm_map.items():
        if "PROCED" in norm:
            return orig
    return None

def detect_date_cols(columns):
    # We will choose best available per-row:
    # OPERATION_DATE > RECONSTRUCTION_DATE > ADMIT_DATE > DISCHARGE_DATE_DT > CHECKOUT_TIME > others
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

    # collect date-like cols
    date_like = []
    for orig, norm in norm_map.items():
        if any(k in norm for k in ["DATE", "TIME", "DT", "DATETIME"]):
            date_like.append(orig)

    # order them by ranked_keys appearance
    ordered = []
    for k in ranked_keys:
        for orig in date_like:
            if k in normalize_colname(orig):
                if orig not in ordered:
                    ordered.append(orig)

    # append remaining
    for orig in date_like:
        if orig not in ordered:
            ordered.append(orig)

    return ordered

def pick_best_event_dt_row(row, date_cols):
    # returns best raw date string for that row from ordered date_cols
    for c in date_cols:
        v = _safe_str(row.get(c, "")).strip()
        if v and v.lower() not in ["nan", "none", "na", "null"]:
            return v, c
    return "", ""

def parse_dt(series):
    # robust datetime parse; returns pandas datetime64[ns]
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def std_cpt(x):
    s = _safe_str(x).strip()
    # keep only digits + leading 'S' if present (you had S2068)
    s = s.replace('"', "").replace("'", "").strip()
    # common cases: '19364', 'S2068'
    return s


# ----------------------------
# Main staging build
# ----------------------------
def main():
    # 1) Load master ids
    if not os.path.exists(MASTER_IDS_FILE):
        raise RuntimeError("Master IDs file not found: {}".format(MASTER_IDS_FILE))

    master = read_csv_robust(MASTER_IDS_FILE)
    pid_col_master = detect_pid_col(master.columns)
    if pid_col_master is None:
        raise RuntimeError("Could not detect patient id column in master ids file.")

    cohort_pids = set(master[pid_col_master].astype(str).str.strip().tolist())
    cohort_pids = set([p for p in cohort_pids if p and p.lower() not in ["nan", "none", "null"]])

    print("Loaded cohort pids:", len(cohort_pids))

    # 2) Load encounter files and stack
    enc_files = [
        ("clinic_encounters", CLINIC_ENC),
        ("inpatient_encounters", INPATIENT_ENC),
        ("operation_encounters", OP_ENC),
    ]

    all_enc = []
    for tag, path in enc_files:
        if not os.path.exists(path):
            raise RuntimeError("Encounter file not found: {}".format(path))

        df = read_csv_robust(path)

        pid_col = detect_pid_col(df.columns)
        cpt_col = detect_cpt_col(df.columns)
        proc_col = detect_procedure_col(df.columns)
        date_cols = detect_date_cols(df.columns)

        if pid_col is None:
            raise RuntimeError("Could not detect patient id column in: {}".format(path))
        if cpt_col is None:
            raise RuntimeError("Could not detect CPT column in: {}".format(path))

        # Filter to cohort early
        df["_PID_"] = df[pid_col].astype(str).str.strip()
        df = df[df["_PID_"].isin(cohort_pids)].copy()

        if len(df) == 0:
            print("WARNING: 0 rows for cohort in", os.path.basename(path))
            continue

        # Best event dt per row
        best_dt_raw = []
        best_dt_src = []
        # use dict-access row to avoid KeyError
        cols_for_row = list(df.columns)
        date_cols_effective = [c for c in date_cols if c in cols_for_row]

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

        if proc_col is not None:
            df["PROCEDURE_STD"] = df[proc_col].astype(str)
        else:
            df["PROCEDURE_STD"] = ""

        keep_cols = [
            "_PID_", "CPT_CODE_STD", "PROCEDURE_STD",
            "BEST_EVENT_DT_RAW", "BEST_EVENT_DT_SRC", "BEST_EVENT_DT_PARSED",
            "SOURCE_FILE_TAG", "SOURCE_FILE"
        ]
        # add any useful operational columns if present (non-fatal)
        for extra in ["PAT_ENC_CSN_ID", "ENCOUNTER_TYPE", "DEPARTMENT", "OP_DEPARTMENT", "REASON_FOR_VISIT"]:
            if extra in df.columns:
                keep_cols.append(extra)

        df_small = df[keep_cols].copy()
        all_enc.append(df_small)

        print("Loaded", len(df_small), "rows from", os.path.basename(path), "| pid_col:", pid_col, "| cpt_col:", cpt_col)

    if not all_enc:
        raise RuntimeError("No encounter rows loaded after cohort filtering.")

    enc = pd.concat(all_enc, axis=0, ignore_index=True)

    # 3) Stage anchors per patient
    # Guard: require a parsable date for anchors; if missing, keep NaT
    def min_dt_for_codes(pid, codeset):
        sub = enc[(enc["_PID_"] == pid) & (enc["CPT_CODE_STD"].isin(codeset))].copy()
        if len(sub) == 0:
            return pd.NaT
        sub = sub[sub["BEST_EVENT_DT_PARSED"].notnull()]
        if len(sub) == 0:
            return pd.NaT
        return sub["BEST_EVENT_DT_PARSED"].min()

    out_rows = []
    for pid in sorted(cohort_pids):
        sub = enc[enc["_PID_"] == pid]
        if len(sub) == 0:
            # patient has no encounters in these files; keep row anyway
            out_rows.append({
                "patient_id": pid,
                "encounter_rows_loaded": 0,
                "has_expander": False,
                "stage1_date": "",
                "has_stage2_definitive": False,
                "stage2_date": "",
                "revision_only_flag": False,
                "first_revision_date": "",
                "first_nipple_date": "",
            })
            continue

        stage1_dt = min_dt_for_codes(pid, CPT_STAGE1_EXPANDER)
        stage2_dt = min_dt_for_codes(pid, CPT_STAGE2_DEFINITIVE)
        rev_dt = min_dt_for_codes(pid, CPT_REVISION)
        nip_dt = min_dt_for_codes(pid, CPT_NIPPLE)

        has_stage1 = pd.notnull(stage1_dt)
        has_stage2 = pd.notnull(stage2_dt)

        # Revision-only flag: has revision and/or nipple but no definitive stage2
        revision_only = (not has_stage2) and (pd.notnull(rev_dt) or pd.notnull(nip_dt))

        out_rows.append({
            "patient_id": pid,
            "encounter_rows_loaded": int(len(sub)),
            "has_expander": bool(has_stage1),
            "stage1_date": stage1_dt.strftime("%Y-%m-%d") if pd.notnull(stage1_dt) else "",
            "has_stage2_definitive": bool(has_stage2),
            "stage2_date": stage2_dt.strftime("%Y-%m-%d") if pd.notnull(stage2_dt) else "",
            "stage2_def_cpt_set": "19364_only",
            "revision_only_flag": bool(revision_only),
            "first_revision_date": rev_dt.strftime("%Y-%m-%d") if pd.notnull(rev_dt) else "",
            "first_nipple_date": nip_dt.strftime("%Y-%m-%d") if pd.notnull(nip_dt) else "",
            "counts_19357": int((sub["CPT_CODE_STD"] == "19357").sum()),
            "counts_19364": int((sub["CPT_CODE_STD"] == "19364").sum()),
            "counts_19380": int((sub["CPT_CODE_STD"] == "19380").sum()),
            "counts_19350": int((sub["CPT_CODE_STD"] == "19350").sum()),
        })

    out = pd.DataFrame(out_rows)

    # 4) Write output
    out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print("\nWROTE:", OUT_FILE)
    print("Rows:", len(out))
    print("Stage1 present:", int(out["has_expander"].sum()))
    print("Stage2 definitive present:", int(out["has_stage2_definitive"].sum()))
    print("Revision-only flagged:", int(out["revision_only_flag"].sum()))


if __name__ == "__main__":
    main()
