import pandas as pd

# Update this path to your actual Clinic Encounters file
CLINIC_ENC_FILE = "../my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Encounters.csv"

OUT_FILE = "patient_demographics.csv"

def read_csv_safely(path):
    # Your raw Epic exports often need cp1252
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")

def pick_first_nonempty(series):
    for v in series:
        if pd.notnull(v):
            s = str(v).strip()
            if s and s.lower() not in ("nan", "none", "unknown"):
                return s
    return None

def main():
    df = read_csv_safely(CLINIC_ENC_FILE)

    # Required columns based on your headers
    required = ["ENCRYPTED_PAT_ID", "RACE", "ETHNICITY"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError("Missing column in Clinic Encounters: {}".format(c))

    # Keep only what we need (PHI-minimizing)
    slim = df[["ENCRYPTED_PAT_ID", "RACE", "ETHNICITY"]].copy()
    slim = slim.rename(columns={"ENCRYPTED_PAT_ID": "patient_id", "RACE": "Race", "ETHNICITY": "Ethnicity"})

    # Aggregate to patient level (first non-empty seen in file order)
    out_rows = []
    for pid, g in slim.groupby("patient_id", sort=False):
        race = pick_first_nonempty(g["Race"].tolist())
        eth = pick_first_nonempty(g["Ethnicity"].tolist())
        out_rows.append({"patient_id": pid, "Race": race, "Ethnicity": eth})

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_FILE, index=False)
    print("Wrote {} (rows={})".format(OUT_FILE, out.shape[0]))

if __name__ == "__main__":
    main()
