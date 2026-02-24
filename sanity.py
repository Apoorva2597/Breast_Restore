
import os
import pandas as pd

BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"
ENCOUNTER_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"
MASTER_IDS_FILE = os.path.join(BREAST_RESTORE_DIR, "MASTER__IDS_ONLY__vNEW.csv")

CLINIC_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Clinic Encounters.csv")
INPATIENT_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Inpatient Encounters.csv")
OP_ENC = os.path.join(ENCOUNTER_DIR, "HPI11526 Operation Encounters.csv")

def read_csv_robust(path):
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")

def detect_col(df, candidates):
    cols = list(df.columns)
    upper_map = {c.upper(): c for c in cols}
    for cand in candidates:
        if cand.upper() in upper_map:
            return upper_map[cand.upper()]
    return None

# load cohort
master = read_csv_robust(MASTER_IDS_FILE)
pid_col = detect_col(master, ["patient_id","ENCRYPTED_PAT_ID"])
cohort = set(master[pid_col].astype(str).str.strip())

all_rows = []

for path in [CLINIC_ENC, INPATIENT_ENC, OP_ENC]:
    df = read_csv_robust(path)
    pid_col = detect_col(df, ["ENCRYPTED_PAT_ID","patient_id"])
    if pid_col is None:
        continue
    df["_PID_"] = df[pid_col].astype(str).str.strip()
    df = df[df["_PID_"].isin(cohort)].copy()

    rov_col = detect_col(df, ["REASON_FOR_VISIT"])
    proc_col = detect_col(df, ["PROCEDURE","PROCEDURE_NAME"])

    if rov_col:
        tmp = df[[rov_col]].copy()
        tmp.columns = ["TEXT"]
        tmp["SOURCE"] = "ROV"
        all_rows.append(tmp)

    if proc_col:
        tmp = df[[proc_col]].copy()
        tmp.columns = ["TEXT"]
        tmp["SOURCE"] = "PROCEDURE"
        all_rows.append(tmp)

if not all_rows:
    print("No ROV/PROCEDURE columns detected.")
    raise SystemExit

stack = pd.concat(all_rows, ignore_index=True)
stack["TEXT"] = stack["TEXT"].astype(str).str.strip()

# remove blanks
stack = stack[stack["TEXT"]!=""]

# value counts
vc = stack.groupby(["SOURCE","TEXT"]).size().reset_index(name="COUNT")
vc = vc.sort_values("COUNT", ascending=False)

out = os.path.join(BREAST_RESTORE_DIR, "STAGE2_TEXT_IMPLANT", "unique_ROV_PROCEDURE_values.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
vc.to_csv(out, index=False, encoding="utf-8")

print("WROTE:", out)
print("\nTop 50 values:")
print(vc.head(50))
