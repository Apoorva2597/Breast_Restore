import pandas as pd

EVID = "_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv"

evid = pd.read_csv(EVID, dtype=str)

for c in evid.columns:
    evid[c] = evid[c].astype(str).str.strip()

sel = evid[evid["FIELD"] == "BMI_NOTE_SELECTION"].copy()
bmi = evid[evid["FIELD"] == "BMI"].copy()

print("\nSelected BMI-note rows:", len(sel))
print("BMI extraction rows:", len(bmi))

# 1) note-type breakdown of selected notes
print("\nSELECTED NOTE TYPES\n")
print(sel["NOTE_TYPE"].value_counts(dropna=False).head(20).to_string())

# 2) selected notes that actually produced BMI
hit = sel.merge(
    bmi[["MRN", "NOTE_ID", "NOTE_TYPE", "VALUE"]],
    on=["MRN", "NOTE_ID"],
    how="left",
    suffixes=("_sel", "_bmi")
)

hit["HAS_BMI"] = hit["VALUE"].notna()

print("\nSELECTED NOTE TYPES WITH BMI HIT RATE\n")
summary = hit.groupby("NOTE_TYPE_sel")["HAS_BMI"].agg(["count", "sum"])
summary = summary.rename(columns={"count": "selected_notes", "sum": "notes_with_bmi"})
summary["hit_rate"] = summary["notes_with_bmi"] / summary["selected_notes"]
print(summary.sort_values("hit_rate", ascending=False).head(25).to_string())

print("\nEXAMPLES OF SELECTED NOTES WITH NO BMI EXTRACTION\n")
misses = hit[hit["HAS_BMI"] == False][["MRN", "NOTE_ID", "NOTE_DATE", "NOTE_TYPE_sel"]].head(30)
print(misses.to_string(index=False))

summary.to_csv("_outputs/bmi_note_type_hit_rate.csv")
hit.to_csv("_outputs/bmi_selected_vs_extracted.csv", index=False)

print("\nSaved:")
print("_outputs/bmi_note_type_hit_rate.csv")
print("_outputs/bmi_selected_vs_extracted.csv")
