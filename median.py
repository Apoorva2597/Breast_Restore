import pandas as pd

df = pd.read_csv("patient_level_phase1_p50.csv")

print("Chemo positive:", df["Chemo"].sum())
print("Radiation positive:", df["Radiation"].sum())
print("Total patients:", len(df))
