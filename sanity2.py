import pandas as pd

mrn = input("Enter MRN: ").strip()

def read_csv_safe(path):
    try:
        return pd.read_csv(path, dtype=str)
    except:
        return pd.read_csv(path, dtype=str, encoding="latin1")

clinic = read_csv_safe("/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Clinic Encounters.csv")
opnotes = read_csv_safe("/home/apokol/Breast_Restore/_staging_inputs/HPI11526 Operation Notes.csv")

clinic.columns=[str(c).strip() for c in clinic.columns]
opnotes.columns=[str(c).strip() for c in opnotes.columns]

clinic["MRN"]=clinic["MRN"].astype(str).str.strip()
opnotes["MRN"]=opnotes["MRN"].astype(str).str.strip()

c=clinic[clinic["MRN"]==mrn]
o=opnotes[opnotes["MRN"]==mrn]

print("\n=== CLINIC ENCOUNTERS ===")
if len(c)==0:
    print("No clinic rows found")
else:
    cols=[x for x in ["MRN","AGE_AT_ENCOUNTER","ADMIT_DATE","RECONSTRUCTION_DATE","CPT_CODE","PROCEDURE","REASON_FOR_VISIT"] if x in c.columns]
    print(c[cols].fillna("").to_string(index=False))

print("\n=== OPERATION NOTES ===")
if len(o)==0:
    print("No operation notes found")
else:
    cols=[x for x in ["MRN","NOTE_ID","NOTE_TYPE","NOTE_DATE_OF_SERVICE"] if x in o.columns]
    print(o[cols].drop_duplicates().fillna("").to_string(index=False))
