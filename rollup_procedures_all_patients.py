# rollup_procedures_all_patients.py
# Python 3.6.8 compatible
#
# Goal:
#   For ALL patients in Operation Encounters, summarize procedures:
#   - patient-level: top CPT, top procedure string, counts, first/last dates
#   - cohort-level: top CPT codes
#
# Outputs:
#   patient_procedure_rollup.csv
#   cohort_top_cpt_codes.csv

from __future__ import print_function
import pandas as pd

OP_ENC_FILE = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv"

OUT_PATIENT = "patient_procedure_rollup.csv"
OUT_COHORT_CPT = "cohort_top_cpt_codes.csv"

BLANK_TOKENS = set(["", "nan", "none", "null", "na", "n/a", ".", "-", "--", "unknown"])

def read_csv_safe(path):
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            f = open(path, "r", encoding=enc, errors="replace")
            try:
                return pd.read_csv(f, engine="python", dtype=object)
            finally:
                try:
                    f.close()
                except Exception:
                    pass
        except Exception:
            continue
    return pd.read_csv(path, engine="python", dtype=object)

def norm_str(x):
    if x is None:
        return ""
    s = str(x)
    try:
        s = s.replace("\xa0", " ")
    except Exception:
        pass
    s = s.strip()
    if s.lower() in BLANK_TOKENS:
        return ""
    return s

def dt_to_ymd(series):
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d").fillna("")

def pick_top(series):
    # series should be a list of strings (already normalized)
    s = [x for x in series if x != ""]
    if not s:
        return ""
    vc = pd.Series(s).value_counts()
    return vc.index[0]

def count_nonempty(series):
    return int(sum([1 for x in series if x != ""]))

def main():
    df = read_csv_safe(OP_ENC_FILE)

    required = ["ENCRYPTED_PAT_ID", "CPT_CODE", "PROCEDURE"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError("Missing required column in OP encounters: {}".format(c))

    df["patient_id"] = df["ENCRYPTED_PAT_ID"].map(norm_str)
    df["CPT_CODE_norm"] = df["CPT_CODE"].map(norm_str)
    df["PROCEDURE_norm"] = df["PROCEDURE"].map(norm_str)

    # pick best date column
    date_col = None
    for c in ["OPERATION_DATE", "RECONSTRUCTION_DATE", "OP_DATE", "DATE"]:
        if c in df.columns:
            date_col = c
            break

    if date_col:
        df["op_date_ymd"] = dt_to_ymd(df[date_col])
    else:
        df["op_date_ymd"] = ""

    # cohort-level CPT frequency
    cohort_cpt = (
        df[df["CPT_CODE_norm"] != ""]
        .groupby("CPT_CODE_norm")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    cohort_cpt.to_csv(OUT_COHORT_CPT, index=False, encoding="utf-8")

    # patient-level rollup
    rows = []
    for pid, g in df[df["patient_id"] != ""].groupby("patient_id", sort=False):
        cpts = g["CPT_CODE_norm"].tolist()
        procs = g["PROCEDURE_norm"].tolist()
        dates = [d for d in g["op_date_ymd"].tolist() if d != ""]

        top_cpt = pick_top(cpts)
        top_proc = pick_top(procs)
        first_dt = min(dates) if dates else ""
        last_dt = max(dates) if dates else ""

        rows.append({
            "patient_id": pid,
            "op_rows": int(len(g)),
            "cpt_rows_nonempty": count_nonempty(cpts),
            "proc_rows_nonempty": count_nonempty(procs),
            "top_cpt": top_cpt,
            "top_procedure": (top_proc[:180] + "...") if len(top_proc) > 180 else top_proc,
            "first_op_date": first_dt,
            "last_op_date": last_dt,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATIENT, index=False, encoding="utf-8")

    print("Wrote:", OUT_PATIENT, "rows=", out.shape[0])
    print("Wrote:", OUT_COHORT_CPT, "rows=", cohort_cpt.shape[0])

if __name__ == "__main__":
    main()
