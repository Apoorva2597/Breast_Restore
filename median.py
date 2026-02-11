# qa_cpt_reconstruction_candidates.py
# Python 3.6+ (pandas required)
#
# Goal:
#   Inventory unique CPT codes across ALL encounter files, and identify
#   "reconstruction-likely" CPTs based on the PROCEDURE strings they appear with.
#
# Outputs:
#   - qa_cpt_inventory_all_files.csv
#   - qa_cpt_top_procedures_per_cpt.csv
#   - qa_cpt_reconstruction_candidates.csv

import re
import sys
import pandas as pd


# -------------------------
# CONFIG
# -------------------------
BASE = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/"

ENCOUNTER_FILES = [
    ("clinic_encounters",    BASE + "HPI11526 Clinic Encounters.csv"),
    ("inpatient_encounters", BASE + "HPI11526 Inpatient Encounters.csv"),
    ("operation_encounters", BASE + "HPI11526 Operation Encounters.csv"),
]

COL_PAT = "ENCRYPTED_PAT_ID"
COL_CPT = "CPT_CODE"
COL_PROC = "PROCEDURE"
COL_MRN = "MRN"


# -------------------------
# Helpers
# -------------------------
def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")


def norm_text(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cpt_norm(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)  # "19342.0" -> "19342"
    s = s.replace(" ", "")
    return s


# Reconstruction-ish keyword screen (based on YOUR procedure text)
RECON_PROC_PAT = re.compile(
    r"\b(recon|reconst|reconstruction|implant|implnt|prost|expander|expandr|flap|diep|tram|siea|latissimus|"
    r"capsulotomy|capsulectomy|nipple|areola|mastopexy|augmentation)\b",
    re.I
)


def main():
    rows = []
    for file_tag, path in ENCOUNTER_FILES:
        df = read_csv_fallback(path)

        missing = [c for c in [COL_PAT, COL_CPT, COL_PROC] if c not in df.columns]
        if missing:
            print("WARN: {} missing columns {} -> skipping {}".format(file_tag, missing, path))
            continue

        df["file_tag"] = file_tag
        df["patient_id"] = df[COL_PAT].fillna("").astype(str)
        df["mrn"] = df[COL_MRN].fillna("").astype(str) if COL_MRN in df.columns else ""
        df["cpt"] = df[COL_CPT].apply(cpt_norm)
        df["procedure"] = df[COL_PROC].apply(norm_text)

        df = df[(df["patient_id"].str.len() > 0) & (df["cpt"].str.len() > 0)].copy()
        rows.append(df[["file_tag", "patient_id", "mrn", "cpt", "procedure"]])

    if not rows:
        raise RuntimeError("No encounter data loaded (check paths/columns).")

    all_df = pd.concat(rows, ignore_index=True)

    # 1) Global CPT inventory (per file + overall)
    inv_by_file = (all_df.groupby(["file_tag", "cpt"])
                         .agg(encounter_rows=("cpt", "size"),
                              unique_patients=("patient_id", pd.Series.nunique),
                              unique_procedures=("procedure", pd.Series.nunique))
                         .reset_index())

    inv_overall = (all_df.groupby(["cpt"])
                         .agg(encounter_rows=("cpt", "size"),
                              unique_patients=("patient_id", pd.Series.nunique),
                              unique_files=("file_tag", pd.Series.nunique),
                              unique_procedures=("procedure", pd.Series.nunique))
                         .reset_index())
    inv_overall["file_tag"] = "ALL_FILES"

    inv_all = pd.concat(
        [inv_by_file,
         inv_overall[["file_tag", "cpt", "encounter_rows", "unique_patients", "unique_procedures"]]],
        ignore_index=True
    )

    inv_all.to_csv("qa_cpt_inventory_all_files.csv", index=False)

    # 2) For each CPT, list top procedures it appears with (overall)
    top_proc = (all_df.groupby(["cpt", "procedure"])
                      .agg(encounter_rows=("procedure", "size"),
                           unique_patients=("patient_id", pd.Series.nunique),
                           files=("file_tag", lambda x: ",".join(sorted(set(x)))))
                      .reset_index()
                      .sort_values(["cpt", "encounter_rows"], ascending=[True, False]))

    # keep top N procedures per CPT
    N = 10
    top_proc["rank_within_cpt"] = top_proc.groupby("cpt").cumcount() + 1
    top_proc_out = top_proc[top_proc["rank_within_cpt"] <= N].copy()
    top_proc_out.to_csv("qa_cpt_top_procedures_per_cpt.csv", index=False)

    # 3) Reconstruction candidate CPTs:
    # Flag CPTs where ANY associated procedure string matches recon-ish vocabulary.
    proc_flag = top_proc.copy()
    proc_flag["recon_like_proc"] = proc_flag["procedure"].apply(lambda s: bool(RECON_PROC_PAT.search(s)))

    # CPT-level rollup of recon-likeness (overall)
    cand = (proc_flag.groupby("cpt")
                    .agg(total_rows=("encounter_rows", "sum"),
                         total_unique_patients=("unique_patients", "sum"),  # not perfect but useful
                         n_proc_pairs=("procedure", "size"),
                         n_recon_proc_pairs=("recon_like_proc", lambda x: int(x.sum())),
                         any_recon_proc=("recon_like_proc", "max"))
                    .reset_index())

    # attach file coverage + true unique patients across all files
    unique_pat = all_df.groupby("cpt")["patient_id"].nunique().reset_index().rename(columns={"patient_id":"unique_patients_all_files"})
    unique_files = all_df.groupby("cpt")["file_tag"].nunique().reset_index().rename(columns={"file_tag":"n_files"})
    cand = cand.merge(unique_pat, on="cpt", how="left").merge(unique_files, on="cpt", how="left")

    # keep only CPTs with any recon-like procedure text
    cand = cand[cand["any_recon_proc"] == True].copy()
    cand = cand.sort_values(["unique_patients_all_files", "total_rows"], ascending=False)

    cand.to_csv("qa_cpt_reconstruction_candidates.csv", index=False)

    print("Wrote:")
    print(" - qa_cpt_inventory_all_files.csv")
    print(" - qa_cpt_top_procedures_per_cpt.csv")
    print(" - qa_cpt_reconstruction_candidates.csv")
    print("")
    print("Recon-candidate CPTs found: {}".format(int(cand.shape[0])))

    # Quick peek: top 15 candidate CPTs
    if not cand.empty:
        print("\nTop recon-candidate CPTs (by unique patients across all files):")
        print(cand.head(15)[["cpt","unique_patients_all_files","n_files","n_recon_proc_pairs","total_rows"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
