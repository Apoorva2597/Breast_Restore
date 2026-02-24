import pandas as pd

cohort = "/home/apokol/Breast_Restore/cohort_all_patient_level_final.csv"
out    = "/home/apokol/Breast_Restore/MASTER__IDS_ONLY__vNEW.csv"

df = pd.read_csv(cohort, dtype=object, engine="python", encoding="latin1", errors="replace")

# Keep only identifier-ish columns that actually exist
keep_candidates = ["patient_id","ENCRYPTED_PAT_ID","MRN","mrn","PAT_ID","PATIENT_ID","ENCRYPTED_PATID"]
keep = [c for c in keep_candidates if c in df.columns]

# Guarantee patient_id exists
assert "patient_id" in df.columns, "patient_id missing from cohort file"

ids = df[keep].drop_duplicates(subset=["patient_id"], keep="first")
ids.to_csv(out, index=False)

print("Wrote:", out)
print("Rows:", len(ids), "Unique patient_id:", ids["patient_id"].nunique())
print("Columns kept:", keep)
