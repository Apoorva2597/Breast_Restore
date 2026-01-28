import pandas as pd

def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")

def main():
    base = read_csv_safely("patient_level_phase1_p50.csv")
    recon = read_csv_safely("patient_recon_structured.csv")

    out = base.merge(recon, on="patient_id", how="left")

    # Override logic (structured wins if present)
    out["Recon_Performed_final"] = out["Recon_Performed"]
    mask_perf = out["Recon_Performed_structured"].notnull()
    out.loc[mask_perf, "Recon_Performed_final"] = out.loc[mask_perf, "Recon_Performed_structured"]

    out["Recon_Type_final"] = out["Recon_Type"]
    mask_type = out["Recon_Type_structured"].notnull() & (out["Recon_Type_structured"].astype(str).str.strip() != "")
    out.loc[mask_type, "Recon_Type_final"] = out.loc[mask_type, "Recon_Type_structured"]

    out.to_csv("patient_level_phase1_p50_plus_demo_plus_recon.csv", index=False)

    print("Wrote patient_level_phase1_p50_plus_demo_plus_recon.csv (rows={})".format(out.shape[0]))
    print("Structured Recon_Type present:", int(out["Recon_Type_structured"].notnull().sum()))
    print("Structured Recon_Performed present:", int(out["Recon_Performed_structured"].notnull().sum()))

if __name__ == "__main__":
    main()
