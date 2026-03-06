#!/usr/bin/env python3

import os
import math
import pandas as pd
from glob import glob
from datetime import datetime

BASE_DIR = "/home/apokol/Breast_Restore"

STRUCT_GLOBS = [
    f"{BASE_DIR}/**/HPI11526*Clinic Encounters.csv",
    f"{BASE_DIR}/**/HPI11526*Operation Encounters.csv",
]

NOTE_GLOBS = [
    f"{BASE_DIR}/**/HPI11526*Clinic Notes.csv",
    f"{BASE_DIR}/**/HPI11526*Inpatient Notes.csv",
    f"{BASE_DIR}/**/HPI11526*Operation Notes.csv",
]

OUTPUT_MASTER = f"{BASE_DIR}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"

MERGE_KEY = "MRN"

RECON_CPT_CODES = {
    "19357", "19340", "19342",
    "19361", "19364", "19367",
    "S2068"
}


def clean(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    return s


def parse_date(x):
    try:
        return pd.to_datetime(x, errors="coerce")
    except:
        return None


# -----------------------
# Race Normalization
# -----------------------

def normalize_race_token(x):

    s = clean(x).lower()

    if s in {"white", "white or caucasian", "caucasian"}:
        return "White"

    if s in {"black", "black or african american", "african american"}:
        return "Black or African American"

    if s in {"asian", "filipino", "chinese", "korean", "japanese", "other asian"}:
        return "Asian"

    if s == "american indian or alaska native":
        return "American Indian or Alaska Native"

    if s == "other":
        return "Other"

    if s in {"unknown", "declined", "choose not to disclose"}:
        return "Unknown"

    return ""


def normalize_race_list(values):

    real = []
    unknown = False

    for v in values:
        r = normalize_race_token(v)

        if not r:
            continue

        if r == "Unknown":
            unknown = True
            continue

        if r not in real:
            real.append(r)

    if len(real) == 0:
        return "Unknown" if unknown else ""

    if len(real) == 1:
        return real[0]

    return "Multiracial"


# -----------------------
# Load structured data
# -----------------------

def load_structured():

    rows = []

    files = []
    for g in STRUCT_GLOBS:
        files.extend(glob(g, recursive=True))

    for fp in files:

        df = pd.read_csv(fp, dtype=str)
        df.columns = [c.strip() for c in df.columns]

        if "MRN" not in df.columns:
            continue

        df["MRN"] = df["MRN"].astype(str)

        df["SOURCE_FILE"] = os.path.basename(fp)

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# -----------------------
# Race extraction
# -----------------------

def compute_race(struct_df):

    race_map = {}

    for mrn, g in struct_df.groupby("MRN"):

        races = g["RACE"].dropna().astype(str).tolist() if "RACE" in g else []

        race_map[mrn] = normalize_race_list(races)

    return race_map


# -----------------------
# Age computation
# -----------------------

def compute_age(struct_df):

    age_map = {}

    op_df = struct_df[struct_df["SOURCE_FILE"].str.contains("Operation", case=False)]

    clinic_df = struct_df[struct_df["SOURCE_FILE"].str.contains("Clinic", case=False)]

    for mrn, g in op_df.groupby("MRN"):

        recon_rows = g[g["CPT_CODE"].isin(RECON_CPT_CODES)] if "CPT_CODE" in g else g

        if len(recon_rows) == 0:
            continue

        recon_rows["OP_DATE"] = recon_rows["OPERATION_DATE"].apply(parse_date)

        recon_rows = recon_rows.dropna(subset=["OP_DATE"])

        if len(recon_rows) == 0:
            continue

        op_date = recon_rows.sort_values("OP_DATE").iloc[0]["OP_DATE"]

        clinic_rows = clinic_df[clinic_df["MRN"] == mrn]

        if len(clinic_rows) == 0:
            continue

        clinic_rows["CLINIC_DATE"] = clinic_rows["ENCOUNTER_DATE"].apply(parse_date)

        clinic_rows = clinic_rows.dropna(subset=["CLINIC_DATE"])

        clinic_rows["AGE_VAL"] = clinic_rows["AGE_AT_ENCOUNTER"].apply(lambda x: float(x) if clean(x) else None)

        clinic_rows = clinic_rows.dropna(subset=["AGE_VAL"])

        if len(clinic_rows) == 0:
            continue

        clinic_rows["DIST"] = (clinic_rows["CLINIC_DATE"] - op_date).abs()

        best = clinic_rows.sort_values("DIST").iloc[0]

        age_base = float(best["AGE_VAL"])

        day_diff = (op_date - best["CLINIC_DATE"]).days

        adj_age = age_base + (day_diff / 365.25)

        age_final = int(round(adj_age))

        age_map[mrn] = age_final

    return age_map


# -----------------------
# Main builder
# -----------------------

def main():

    struct_df = load_structured()

    if len(struct_df) == 0:
        raise RuntimeError("No structured encounter files found")

    mrns = sorted(struct_df["MRN"].dropna().unique())

    master = pd.DataFrame({"MRN": mrns})

    race_map = compute_race(struct_df)

    age_map = compute_age(struct_df)

    master["Race"] = master["MRN"].map(race_map)

    master["Age"] = master["MRN"].map(age_map)

    master.to_csv(OUTPUT_MASTER, index=False)

    print("DONE")
    print("Output:", OUTPUT_MASTER)


if __name__ == "__main__":
    main()
