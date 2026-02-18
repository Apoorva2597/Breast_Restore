import pandas as pd

df = pd.read_csv("patient_recon_staging_refined.csv")

df["has_expander_bool"] = df["has_expander"].astype(str).str.lower().isin(["true","1","yes","y"])
exp = df[df["has_expander_bool"]]

print("Total expander patients:", len(exp))
print("With structured Stage2:", exp["stage2_date"].notnull().sum())
print("Missing structured Stage2:", exp["stage2_date"].isnull().sum())
