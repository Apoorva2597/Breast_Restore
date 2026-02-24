
import pandas as pd

master = "/home/apokol/Breast_Restore/MASTER__STAGING_PATHWAY__vNEW.csv"
out = "/home/apokol/Breast_Restore/qa_sample_20_EXP_S2def.csv"

df = pd.read_csv(master, dtype=str)
flag = df["has_stage2_definitive"].astype(str).str.lower().isin(["true","1","yes"])
sample = df.loc[flag, ["patient_id"]].dropna().drop_duplicates().sample(n=20, random_state=42)

sample.to_csv(out, index=False)
print("WROTE:", out)
print("Rows:", len(sample), "Unique patients:", sample["patient_id"].nunique())
