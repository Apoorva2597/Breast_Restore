# qa_operation_encounter_procedures.py
# Python 3.6 compatible
#
# PURPOSE (Option A):
#   Discover the real procedure vocabulary + CPT codes in the OPERATION ENCOUNTERS file
#   BEFORE we write Stage1/Stage2 staging rules.
#
# INPUT:
#   Operation encounters CSV with columns like:
#   MRN, ENCRYPTED_PAT_ID, ETHNICITY, RACE, PAT_ENC_CSN_ID, ENCRYPTED_CSN,
#   OPERATION_DATE, DISCHARGE_DATE_DT, AGE_AT_ENCOUNTER, OP_DEPARTMENT,
#   ENCOUNTER_TYPE, FINANCIAL_CLASS_NAME, REASON_FOR_VISIT, CPT_CODE, PROCEDURE
#
# OUTPUT (NO NOTE TEXT, no evidence snippets):
#   1) qa_op_enc_procedure_counts.csv  (procedure string -> count, unique patients)
#   2) qa_op_enc_cpt_counts.csv        (CPT -> count, unique patients)
#   3) qa_op_enc_cpt_procedure_pairs.csv (CPT x procedure -> count)  [helps mapping]
#
# NOTE:
#   This script should not write patient IDs to outputs (to reduce PHI/ID exposure).
#   It only writes aggregated counts.

import re
import sys
import pandas as pd


# -----------------------
# CONFIG
# -----------------------
INPUT_CSV = "HPI11526 Operation Encounters.csv"  
TOP_N_PRINT = 40

OUT_PROCEDURE = "qa_op_enc_procedure_counts.csv"
OUT_CPT = "qa_op_enc_cpt_counts.csv"
OUT_CPT_PROC = "qa_op_enc_cpt_procedure_pairs.csv"


# -----------------------
# HELPERS
# -----------------------
def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")


def norm_ws(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def norm_cpt(x):
    s = norm_ws(x)
    if not s or s.lower() == "nan":
        return ""
    return s.upper()


def norm_proc(x):
    s = norm_ws(x)
    if not s or s.lower() == "nan":
        return ""
    # Optional: keep original case? For grouping, normalize case.
    return s.lower()


def main():
    df = read_csv_fallback(INPUT_CSV)

    # Flexible column name handling
    # We’ll accept a few common variants just in case.
    col_map = {c.lower(): c for c in df.columns}

    def need(name, alts):
        for a in alts:
            if a.lower() in col_map:
                return col_map[a.lower()]
        raise RuntimeError("Missing required column: {} (tried {})".format(name, alts))

    COL_PID = need("ENCRYPTED_PAT_ID", ["ENCRYPTED_PAT_ID", "patient_id", "PATIENT_ID"])
    COL_CPT = need("CPT_CODE", ["CPT_CODE", "CPT", "CPT CODE"])
    COL_PROC = need("PROCEDURE", ["PROCEDURE", "Procedure", "PROC_DESC", "PROC_DESCRIPTION"])

    # Clean
    df["_pid"] = df[COL_PID].fillna("").astype(str).map(norm_ws)
    df["_cpt"] = df[COL_CPT].fillna("").astype(str).map(norm_cpt)
    df["_proc"] = df[COL_PROC].fillna("").astype(str).map(norm_proc)

    # Drop empty rows (no CPT and no procedure)
    df = df[(df["_cpt"] != "") | (df["_proc"] != "")].copy()

    # -----------------------
    # PROCEDURE COUNTS
    # -----------------------
    proc_grp = (
        df[df["_proc"] != ""]
        .groupby("_proc")
        .agg(
            encounter_rows=("__dummy__", "size") if "__dummy__" in df.columns else ("_proc", "size"),
            unique_patients=("_pid", pd.Series.nunique),
            unique_cpt_codes=("_cpt", pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"_proc": "procedure_norm"})
        .sort_values(["unique_patients", "encounter_rows"], ascending=False)
    )

    # -----------------------
    # CPT COUNTS
    # -----------------------
    cpt_grp = (
        df[df["_cpt"] != ""]
        .groupby("_cpt")
        .agg(
            encounter_rows=("_cpt", "size"),
            unique_patients=("_pid", pd.Series.nunique),
            unique_procedures=("_proc", pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"_cpt": "cpt_code"})
        .sort_values(["unique_patients", "encounter_rows"], ascending=False)
    )

    # -----------------------
    # CPT x PROCEDURE PAIRS (helps mapping Stage1 vs Stage2 later)
    # -----------------------
    pair_grp = (
        df[(df["_cpt"] != "") & (df["_proc"] != "")]
        .groupby(["_cpt", "_proc"])
        .agg(
            encounter_rows=("_cpt", "size"),
            unique_patients=("_pid", pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"_cpt": "cpt_code", "_proc": "procedure_norm"})
        .sort_values(["unique_patients", "encounter_rows"], ascending=False)
    )

    # Write outputs (aggregated only)
    proc_grp.to_csv(OUT_PROCEDURE, index=False)
    cpt_grp.to_csv(OUT_CPT, index=False)
    pair_grp.to_csv(OUT_CPT_PROC, index=False)

    # Console preview
    print("\n=== DONE: wrote outputs ===")
    print("1) {}".format(OUT_PROCEDURE))
    print("2) {}".format(OUT_CPT))
    print("3) {}".format(OUT_CPT_PROC))

    print("\n=== TOP PROCEDURES (by unique_patients) ===")
    print(proc_grp.head(TOP_N_PRINT).to_string(index=False))

    print("\n=== TOP CPT CODES (by unique_patients) ===")
    print(cpt_grp.head(TOP_N_PRINT).to_string(index=False))

    print("\nNOTE: Outputs contain only aggregated counts (no patient IDs, no note text).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
