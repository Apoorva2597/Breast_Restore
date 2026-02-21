# stage1_make_outcomes_from_complications.py
# Purpose: derive Stage1 patient-level outcome flags from S1_Comp1..3 fields
# Python 3.6.8+, pandas

from __future__ import print_function
import pandas as pd

INFILE = "stage1_complications_patient_level.csv"
OUTFILE = "stage1_outcomes_patient_level.csv"

df = pd.read_csv(INFILE, encoding="latin1")

# basic checks
if "patient_id" not in df.columns:
    raise RuntimeError("Missing patient_id in " + INFILE)

df["patient_id"] = df["patient_id"].fillna("").astype(str)

# columns we expect (but we won't crash if one is missing)
treat_cols = [c for c in df.columns if c.startswith("S1_Comp") and c.endswith("_Treatment")]
class_cols = [c for c in df.columns if c.startswith("S1_Comp") and c.endswith("_Classification")]

for c in treat_cols + class_cols:
    df[c] = df[c].fillna("").astype(str).str.strip().str.upper()

# flags
reop = pd.Series([0] * len(df))
rehosp = pd.Series([0] * len(df))
has_major = pd.Series([0] * len(df))
has_minor = pd.Series([0] * len(df))

# treatment-derived
for c in treat_cols:
    reop = reop | (df[c] == "REOPERATION")
    rehosp = rehosp | (df[c] == "REHOSPITALIZATION")

# classification-derived
for c in class_cols:
    has_major = has_major | (df[c] == "MAJOR")
    has_minor = has_minor | (df[c] == "MINOR")

df_out = pd.DataFrame()
df_out["patient_id"] = df["patient_id"]

df_out["Stage1_Reoperation_pred"] = reop.astype(int)
df_out["Stage1_Rehospitalization_pred"] = rehosp.astype(int)

# protocol-style major: MAJOR OR reop/rehosp
df_out["Stage1_MajorComp_pred"] = (has_major | reop | rehosp).astype(int)

# minor only if minor present and NOT major (mutually exclusive)
df_out["Stage1_MinorComp_pred"] = ((has_minor.astype(bool)) & (df_out["Stage1_MajorComp_pred"] == 0)).astype(int)

# quick summary
n = int(df_out["patient_id"].nunique())
print("Patients:", n)
print("Stage1_MinorComp_pred:", int(df_out["Stage1_MinorComp_pred"].sum()))
print("Stage1_MajorComp_pred:", int(df_out["Stage1_MajorComp_pred"].sum()))
print("Stage1_Reoperation_pred:", int(df_out["Stage1_Reoperation_pred"].sum()))
print("Stage1_Rehospitalization_pred:", int(df_out["Stage1_Rehospitalization_pred"].sum()))

df_out.to_csv(OUTFILE, index=False, encoding="utf-8")
print("Wrote:", OUTFILE)
