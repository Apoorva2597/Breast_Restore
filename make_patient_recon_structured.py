# make_patient_recon_structured.py

import pandas as pd
import re

OP_ENC_FILE = "../my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv"
OUT_FILE = "patient_recon_structured.csv"

def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")

# Very lightweight procedure-to-type mapping (good enough for now; refine later)
def infer_recon_type(proc_text, cpt):
    s = (proc_text or "")
    s_low = s.lower()
    c = str(cpt) if cpt is not None else ""
    c_low = c.lower()

    # Common keywords
    if "diep" in s_low or "free flap" in s_low or "flap" in s_low:
        return "autologous_flap"
    if "expander" in s_low or "tissue expander" in s_low:
        return "tissue_expander"
    if "implant" in s_low:
        return "implant"
    if "latissimus" in s_low:
        return "latissimus_flap"

    # CPT fallback (optional; you can extend later)
    if re.search(r"\b19357\b", c_low):  # tissue expander
        return "tissue_expander"
    if re.search(r"\b19340\b", c_low) or re.search(r"\b19342\b", c_low):  # implant-ish
        return "implant"

    return None

def main():
    df = read_csv_safely(OP_ENC_FILE)

    required = ["ENCRYPTED_PAT_ID", "PROCEDURE", "CPT_CODE"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError("Missing column in Operation Encounters: {}".format(c))

    slim_cols = ["ENCRYPTED_PAT_ID", "PROCEDURE", "CPT_CODE"]
    # Keep reconstruction date if present (internal-only; PHI)
    if "RECONSTRUCTION_DATE" in df.columns:
        slim_cols.append("RECONSTRUCTION_DATE")

    slim = df[slim_cols].copy()
    slim = slim.rename(columns={"ENCRYPTED_PAT_ID": "patient_id"})

    out_rows = []
    for pid, g in slim.groupby("patient_id", sort=False):
        # Determine if any recon signal exists (procedure text or cpt present)
        recon_types = []
        recon_dates = []

        for _, row in g.iterrows():
            proc = row.get("PROCEDURE", None)
            cpt = row.get("CPT_CODE", None)

            rtype = infer_recon_type(proc, cpt)
            if rtype:
                recon_types.append(rtype)

            if "RECONSTRUCTION_DATE" in row and pd.notnull(row["RECONSTRUCTION_DATE"]):
                recon_dates.append(str(row["RECONSTRUCTION_DATE"]))

        performed = True if (len(recon_types) > 0 or len(g) > 0) else False

        # pick the first inferred type (you can change to "most common" later)
        rtype_best = recon_types[0] if recon_types else None
        rdate_best = recon_dates[0] if recon_dates else None

        out_rows.append({
            "patient_id": pid,
            "Recon_Performed_structured": performed,
            "Recon_Type_structured": rtype_best,
            "Recon_Date_structured": rdate_best,
            "Recon_Source": "operation_encounters"
        })

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_FILE, index=False)
    print("Wrote {} (rows={})".format(OUT_FILE, out.shape[0]))

if __name__ == "__main__":
    main()
