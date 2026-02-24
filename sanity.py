
import pandas as pd

PIPE="/home/apokol/Breast_Restore/MASTER__STAGING_PATHWAY__vNEW.csv"
# If you saved the long stacked encounter-level extract anywhere, point to that.
# Otherwise re-create a small extract from your staging builder (recommended).
# For now, we’ll just inspect the 3 raw encounter files directly.

CLIN="/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Encounters.csv"
INP ="/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Encounters.csv"
OP  ="/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv"

def read(path):
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")

def norm(s):
    return str(s).strip().upper().replace(" ", "_")

def find_col(df, wants):
    m={c:norm(c) for c in df.columns}
    for w in wants:
        for c,n in m.items():
            if n==w:
                return c
    return None

for tag, path in [("CLINIC",CLIN),("INPATIENT",INP),("OPERATION",OP)]:
    df=read(path)
    cpt=find_col(df, ["CPT_CODE","CPT","CPTCD"])
    proc=find_col(df, ["PROCEDURE","PROC","PROCEDURE_NAME"])
    rov=find_col(df, ["REASON_FOR_VISIT","ROV"])
    print("\n===",tag,"===")
    print("rows:",len(df))
    print("CPT col:",cpt,"| PROC col:",proc,"| ROV col:",rov)

    if cpt:
        top = df[cpt].astype(str).str.strip().value_counts().head(30)
        print("\nTop CPT (30):")
        print(top.to_string())

    if proc:
        top = df[proc].astype(str).str.strip().value_counts().head(30)
        print("\nTop PROCEDURE (30):")
        print(top.to_string())

    if rov:
        top = df[rov].astype(str).str.strip().value_counts().head(30)
        print("\nTop ROV (30):")
        print(top.to_string())
