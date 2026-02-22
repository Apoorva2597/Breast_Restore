# merge_patient_level_add_demo_and_recon.py

import pandas as pd

def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")

def nonempty(s):
    if s is None:
        return False
    try:
        return str(s).strip() != "" and str(s).lower().strip() not in ("nan", "none")
    except:
        return False

def main():
    base = read_csv_safely("patient_level_phase1_p50.csv")
    demo = read_csv_safely("patient_demographics.csv")
    recon = read_csv_safely("patient_recon_structured.csv")

    out = base.merge(demo, on="patient_id", how="left").merge(recon, on="patient_id", how="left")

    # Override logic: structured recon wins if present
    if "Recon_Performed" in out.columns and "Recon_Performed_structured" in out.columns:
        out["Recon_Performed_final"] = out["Recon_Performed"]
        mask = out["Recon_Performed_structured"].notnull()
        out.loc[mask, "Recon_Performed_final"] = out.loc[mask, "Recon_Performed_structured"]

    if "Recon_Type" in out.columns and "Recon_Type_structured" in out.columns:
        out["Recon_Type_final"] = out["Recon_Type"]
        mask = out["Recon_Type_structured"].notnull() & out["Recon_Type_structured"].apply(nonempty)
        out.loc[mask, "Recon_Type_final"] = out.loc[mask, "Recon_Type_structured"]

    out.to_csv("patient_level_phase1_p50_plus_demo_plus_recon.csv", index=False)

    print("Wrote patient_level_phase1_p50_plus_demo_plus_recon.csv (rows={})".format(out.shape[0]))
    print("Race non-null:", int(out["Race"].notnull().sum()) if "Race" in out.columns else 0)
    print("Ethnicity non-null:", int(out["Ethnicity"].notnull().sum()) if "Ethnicity" in out.columns else 0)
    print("Recon_Type_structured present:", int(out["Recon_Type_structured"].notnull().sum()) if "Recon_Type_structured" in out.columns else 0)

if __name__ == "__main__":
    main()
