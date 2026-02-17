# refine_expander_cohort.py
# Python 3.6.8+ (pandas)
#
# Purpose:
#   Refine "has_expander" cohort using OPERATION NOTES evidence near stage1 date (±14d).
#   Outputs a refined staging CSV you can use as the denominator for Stage2 extraction.
#
# Inputs:
#   - patient_recon_staging.csv
#   - HPI11526 Operation Notes.csv
#
# Output:
#   - patient_recon_staging_refined.csv
#   - qa_expander_refined_counts.txt

from __future__ import print_function

import re
import sys
import pandas as pd

PATIENT_STAGING_CSV = "patient_recon_staging.csv"
OP_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"

OUT_REFINED = "patient_recon_staging_refined.csv"
OUT_SUMMARY = "qa_expander_refined_counts.txt"

COL_PID_STG = "patient_id"
COL_HAS_EXP = "has_expander"
COL_STAGE1 = "stage1_date"

COL_PAT = "ENCRYPTED_PAT_ID"
COL_TXT = "NOTE_TEXT"
COL_DOS = "NOTE_DATE_OF_SERVICE"
COL_OPD = "OPERATION_DATE"
COL_NTYPE = "NOTE_TYPE"

WINDOW_DAYS = 14
CHUNKSIZE = 120000

def to_bool(x):
    s = str(x).strip().lower()
    return s in ["true", "1", "yes", "y"]

def to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def norm_text(x):
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ").replace(u"\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def iter_csv_latin1(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        for chunk in pd.read_csv(f, engine="python", **kwargs):
            yield chunk
    finally:
        try:
            f.close()
        except Exception:
            pass

# --- Evidence patterns (keep conservative) ---
RX = {
    # DTI evidence
    "DTI": re.compile(r"\b(direct[\s-]*to[\s-]*implant|DTI)\b", re.I),

    # Tissue expander evidence (placement)
    "TE_PLACEMENT": re.compile(
        r"\b(tissue\s*expander)\b.{0,80}\b(placement|placed|inserted|insertion)\b|"
        r"\b(placement|placed|inserted|insertion)\b.{0,80}\b(tissue\s*expander)\b|"
        r"\bsubpectoral\b.{0,80}\b(tissue\s*expander)\b|"
        r"\b(tissue\s*expander)\b.{0,80}\bsubpectoral\b",
        re.I
    ),

    # Expander tokens (for implants sections)
    "EXPANDER_TOKEN": re.compile(r"\b(tissue\s*expander|expander|\bte\b)\b", re.I),

    # Implant tokens (permanent implant)
    "IMPLANT_TOKEN": re.compile(r"\b(permanent\s+implant|implant)\b", re.I),
}

def main():
    stg = pd.read_csv(PATIENT_STAGING_CSV, engine="python")
    for c in [COL_PID_STG, COL_HAS_EXP, COL_STAGE1]:
        if c not in stg.columns:
            raise RuntimeError("Missing required column '{}' in {}".format(c, PATIENT_STAGING_CSV))

    stg[COL_PID_STG] = stg[COL_PID_STG].fillna("").astype(str)
    stg["has_expander_bool"] = stg[COL_HAS_EXP].apply(to_bool)
    stg["stage1_dt"] = to_dt(stg[COL_STAGE1])

    exp0 = stg[stg["has_expander_bool"]].copy()
    exp0_ids = set(exp0[COL_PID_STG].tolist())

    if not exp0_ids:
        raise RuntimeError("No has_expander==True rows found in patient_recon_staging.csv")

    # maps
    stage1_map = dict(zip(exp0[COL_PID_STG], exp0["stage1_dt"]))

    # tracking evidence per patient
    ev_dti = set()
    ev_te_place = set()
    ev_implant_no_te = set()

    usecols = [COL_PAT, COL_TXT, COL_DOS, COL_OPD]
    # NOTE_TYPE optional; don't require it
    # validate columns quickly
    head = pd.read_csv(open(OP_NOTES_CSV, "r", encoding="latin1", errors="replace"), engine="python", nrows=5)
    for c in usecols:
        if c not in head.columns:
            raise RuntimeError("Missing required note column '{}' in {}".format(c, OP_NOTES_CSV))

    print("Starting refine over expander-labeled cohort size:", len(exp0_ids))

    for chunk in iter_csv_latin1(OP_NOTES_CSV, usecols=usecols, chunksize=CHUNKSIZE):
        chunk[COL_PAT] = chunk[COL_PAT].fillna("").astype(str)
        chunk = chunk[chunk[COL_PAT].isin(exp0_ids)].copy()
        if chunk.empty:
            continue

        # event date
        chunk["dos_dt"] = to_dt(chunk[COL_DOS])
        chunk["op_dt"] = to_dt(chunk[COL_OPD])
        chunk["event_dt"] = chunk["dos_dt"].fillna(chunk["op_dt"])

        # filter to neighborhood: abs(event - stage1) <= WINDOW_DAYS
        chunk["stage1_dt"] = chunk[COL_PAT].map(stage1_map)
        chunk["delta"] = (chunk["event_dt"] - chunk["stage1_dt"]).dt.days
        chunk = chunk[chunk["delta"].notnull()].copy()
        chunk = chunk[chunk["delta"].abs() <= WINDOW_DAYS].copy()
        if chunk.empty:
            continue

        # text
        chunk["t"] = chunk[COL_TXT].apply(norm_text)

        # row-level evidence
        for pid, txt in zip(chunk[COL_PAT].tolist(), chunk["t"].tolist()):
            has_dti = bool(RX["DTI"].search(txt))
            has_te_place = bool(RX["TE_PLACEMENT"].search(txt))
            has_te_token = bool(RX["EXPANDER_TOKEN"].search(txt))
            has_implant = bool(RX["IMPLANT_TOKEN"].search(txt))

            if has_dti:
                ev_dti.add(pid)

            if has_te_place:
                ev_te_place.add(pid)

            # “implant but no expander mention” (DTI-ish signal) — only mark if implant present and no TE token
            if has_implant and (not has_te_token):
                ev_implant_no_te.add(pid)

    # assign refined buckets
    refined = exp0[[COL_PID_STG, COL_HAS_EXP, COL_STAGE1]].copy()
    refined = refined.rename(columns={COL_PID_STG: "patient_id"})

    def bucket(pid):
        # Priority: staged evidence wins, then DTI evidence, then implant-no-TE, else uncertain
        if pid in ev_te_place:
            return "STAGED_EVIDENCE"
        if pid in ev_dti:
            return "DTI_EVIDENCE"
        if pid in ev_implant_no_te:
            return "IMPLANT_NO_TE"
        return "UNCERTAIN"

    refined["expander_bucket"] = refined["patient_id"].apply(bucket)
    refined["has_expander_refined"] = refined["expander_bucket"].apply(lambda x: True if x == "STAGED_EVIDENCE" else False)

    refined.to_csv(OUT_REFINED, index=False, encoding="utf-8")

    # summary
    counts = refined["expander_bucket"].value_counts()
    lines = []
    lines.append("=== Refined expander cohort (based on OP note neighborhood ±{}d) ===".format(WINDOW_DAYS))
    lines.append("Start cohort (has_expander==True): {}".format(len(refined)))
    lines.append("")
    for k, v in counts.items():
        lines.append("{:<18} {}".format(k + ":", int(v)))
    lines.append("")
    lines.append("Refined expander True (STAGED_EVIDENCE): {}".format(int((refined["has_expander_refined"] == True).sum())))
    lines.append("Wrote: {}".format(OUT_REFINED))

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
