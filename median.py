# explore_all_procedures_and_cpt.py
# Purpose: FULL inventory of procedure names and CPT codes in OP encounters
# No staging logic — pure data inspection

import re
import sys
import pandas as pd

OP_ENCOUNTERS_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv"

COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_PROC = "PROCEDURE"
COL_CPT = "CPT_CODE"
COL_OP_DATE = "OPERATION_DATE"
COL_ALT_DATE = "DISCHARGE_DATE_DT"


def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")


def norm_text(x):
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main():
    df = read_csv_fallback(OP_ENCOUNTERS_CSV)

    for c in [COL_PATIENT, COL_PROC]:
        if c not in df.columns:
            raise RuntimeError("Missing column: {}".format(c))

    df["patient_id"] = df[COL_PATIENT].fillna("").astype(str)
    df["proc_norm"] = df[COL_PROC].apply(norm_text)
    df["cpt"] = df[COL_CPT].fillna("").astype(str) if COL_CPT in df.columns else ""

    # Remove empty
    df = df[df["proc_norm"].str.len() > 0].copy()

    print("\nTOTAL ENCOUNTER ROWS:", len(df))
    print("TOTAL UNIQUE PATIENTS:", df["patient_id"].nunique())

    # ---------------------------
    # ALL PROCEDURE NAMES
    # ---------------------------
    print("\n=== ALL UNIQUE PROCEDURE NAMES ===")

    proc_summary = (
        df.groupby("proc_norm")
          .agg(
              encounter_rows=("proc_norm", "count"),
              unique_patients=("patient_id", "nunique"),
              unique_cpt_codes=("cpt", lambda x: len(set(x.dropna())))
          )
          .reset_index()
          .sort_values("unique_patients", ascending=False)
    )

    proc_summary.to_csv("ALL_procedure_names_inventory.csv", index=False)
    print("Wrote: ALL_procedure_names_inventory.csv")
    print("Total unique procedure names:", len(proc_summary))

    # ---------------------------
    # ALL CPT CODES
    # ---------------------------
    print("\n=== ALL UNIQUE CPT CODES ===")

    cpt_summary = (
        df[df["cpt"].str.len() > 0]
          .groupby("cpt")
          .agg(
              encounter_rows=("cpt", "count"),
              unique_patients=("patient_id", "nunique"),
              unique_procedures=("proc_norm", lambda x: len(set(x)))
          )
          .reset_index()
          .sort_values("unique_patients", ascending=False)
    )

    cpt_summary.to_csv("ALL_cpt_codes_inventory.csv", index=False)
    print("Wrote: ALL_cpt_codes_inventory.csv")
    print("Total unique CPT codes:", len(cpt_summary))

    # ---------------------------
    # POSSIBLE STAGE-2 KEYWORD SCAN (for discovery only)
    # ---------------------------
    print("\n=== POSSIBLE STAGE-2-LIKE PROCEDURE STRINGS ===")

    stage2_keywords = [
        "exchange", "remove", "removal", "replacement",
        "separate day", "sep day", "delayed",
        "implant on separate", "tissue expander removal"
    ]

    pattern = re.compile("|".join(stage2_keywords), re.I)

    df["stage2_like"] = df["proc_norm"].apply(lambda x: bool(pattern.search(x)))

    stage2_proc = (
        df[df["stage2_like"]]
        .groupby("proc_norm")
        .agg(
            encounter_rows=("proc_norm", "count"),
            unique_patients=("patient_id", "nunique")
        )
        .reset_index()
        .sort_values("unique_patients", ascending=False)
    )

    stage2_proc.to_csv("POSSIBLE_stage2_like_procedures.csv", index=False)
    print("Wrote: POSSIBLE_stage2_like_procedures.csv")
    print("Unique Stage-2-like procedure strings found:", len(stage2_proc))

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
