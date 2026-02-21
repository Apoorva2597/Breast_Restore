# make_pred_spine_stage1_stage2.py
# Build final patient-level prediction spine
# Python 3.6.8 compatible

from __future__ import print_function
import sys
import pandas as pd

SPINE_FILE = "patient_recon_staging_refined.csv"
STAGE1_FILE = "stage1_outcomes_patient_level.csv"
STAGE2_FILE = "stage2_outcomes_full_ab.csv"

OUT_FILE = "pred_spine_stage1_stage2.csv"

def read_csv_safe(path):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python")
    finally:
        try:
            f.close()
        except Exception:
            pass

def ensure_binary(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df[c] = (df[c] != 0).astype(int)
        else:
            df[c] = 0
    return df

def main():
    # Load files
    spine = read_csv_safe(SPINE_FILE)
    s1 = read_csv_safe(STAGE1_FILE)
    s2 = read_csv_safe(STAGE2_FILE)

    # Ensure patient_id present
    for df, name in [(spine,"SPINE"), (s1,"STAGE1"), (s2,"STAGE2")]:
        if "patient_id" not in df.columns:
            raise RuntimeError("Missing patient_id in " + name)

    spine["patient_id"] = spine["patient_id"].fillna("").astype(str)
    s1["patient_id"] = s1["patient_id"].fillna("").astype(str)
    s2["patient_id"] = s2["patient_id"].fillna("").astype(str)

    # Merge Stage 1
    m = spine.merge(s1, on="patient_id", how="left")

    stage1_cols = [
        "Stage1_MinorComp_pred",
        "Stage1_MajorComp_pred",
        "Stage1_Reoperation_pred",
        "Stage1_Rehospitalization_pred"
    ]
    m = ensure_binary(m, stage1_cols)

    # Merge Stage 2
    m = m.merge(s2, on="patient_id", how="left")

    stage2_cols = [
        "Stage2_MinorComp",
        "Stage2_MajorComp",
        "Stage2_Reoperation",
        "Stage2_Rehospitalization",
        "Stage2_Failure",
        "Stage2_Revision"
    ]
    m = ensure_binary(m, stage2_cols)

    # Basic integrity checks
    n_total = m["patient_id"].nunique()
    n_stage2 = int(m["Stage2_MinorComp"].sum() + 
                   m["Stage2_MajorComp"].sum() +
                   m["Stage2_Reoperation"].sum() +
                   m["Stage2_Rehospitalization"].sum() +
                   m["Stage2_Failure"].sum() +
                   m["Stage2_Revision"].sum() > 0)

    print("Total patients in spine:", n_total)
    print("Stage1 Major:", int(m["Stage1_MajorComp_pred"].sum()))
    print("Stage1 Minor:", int(m["Stage1_MinorComp_pred"].sum()))
    print("Stage2 Major:", int(m["Stage2_MajorComp"].sum()))
    print("Stage2 Minor:", int(m["Stage2_MinorComp"].sum()))
    print("Stage2 Failure:", int(m["Stage2_Failure"].sum()))
    print("Stage2 Revision:", int(m["Stage2_Revision"].sum()))

    m.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print("Wrote:", OUT_FILE)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
