import pandas as pd

EVID = "_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv"

evid = pd.read_csv(EVID, dtype=str)

for c in evid.columns:
    evid[c] = evid[c].astype(str).str.strip()

sel = evid[evid["FIELD"] == "BMI_NOTE_SELECTION"].copy()
bmi = evid[evid["FIELD"] == "BMI"].copy()

merged = sel.merge(
    bmi[["MRN", "NOTE_ID", "VALUE", "EVIDENCE"]],
    on=["MRN", "NOTE_ID"],
    how="left",
    suffixes=("_sel", "_bmi")
)

print(merged[["MRN", "NOTE_ID", "NOTE_TYPE", "NOTE_DATE", "VALUE", "EVIDENCE_bmi"]].head(100).to_string(index=False))
print("\nSelected BMI notes:", len(sel))
print("Selected notes with extracted BMI:", merged["VALUE"].notna().sum())
