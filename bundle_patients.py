import pandas as pd

df = pd.read_csv("patient_note_index.csv", encoding="utf-8", engine="python")

# Parse dates safely (keeps strings if messy; we just need sortable values)
# If NOTE_DATE_OF_SERVICE is consistent, this will work well.
df["note_date_parsed"] = pd.to_datetime(df["note_date"], errors="coerce")

# Notes per patient
g = df.groupby("patient_id")

manifest = g.agg(
    n_notes=("note_id", "count"),
    min_date=("note_date_parsed", "min"),
    max_date=("note_date_parsed", "max")
).reset_index()

manifest = manifest.sort_values("n_notes", ascending=False)

manifest.to_csv("patient_bundle_manifest.csv", index=False)
print("Wrote patient_bundle_manifest.csv")
print("Patients:", manifest.shape[0])
print("Notes per patient -> min:", int(manifest["n_notes"].min()),
      "median:", float(manifest["n_notes"].median()),
      "max:", int(manifest["n_notes"].max()))
print("\nTop 10 patients by note count:")
print(manifest.head(10))
