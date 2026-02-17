from __future__ import print_function
import re
import pandas as pd

PATIENT_STAGING_CSV = "patient_recon_staging.csv"
OP_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"

COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"
COL_NOTE_TYPE = "NOTE_TYPE"

def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        if "chunksize" not in kwargs:
            try: f.close()
            except: pass

def iter_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        for ch in pd.read_csv(f, engine="python", **kwargs):
            yield ch
    finally:
        try: f.close()
        except: pass

def to_bool(x):
    s = str(x).strip().lower()
    return s in ["true","1","yes","y"]

def norm(x):
    s = "" if x is None else str(x)
    s = s.replace("\r"," ").replace("\n"," ")
    try: s = s.replace(u"\xa0"," ")
    except: pass
    return re.sub(r"\s+"," ",s).strip()

RX_EXP = re.compile(r"\b(tissue\s*expander|expander|expandr|\bte\b)\b", re.I)
RX_IMPL = re.compile(r"\b(implant|permanent\s+implant)\b", re.I)
RX_DTI  = re.compile(r"\b(direct\s*to\s*implant|direct-to-implant|DTI)\b", re.I)
RX_TE_PLACE = re.compile(r"\b(place|placed|insert|inserted)\b.{0,120}\b(tissue\s*expander|expander|\bte\b)\b", re.I)

def main():
    stg = read_csv_safe(PATIENT_STAGING_CSV)
    stg["patient_id"] = stg["patient_id"].fillna("").astype(str)
    stg["has_expander_bool"] = stg["has_expander"].apply(to_bool)
    exp = stg[stg["has_expander_bool"]].copy()
    exp_ids = set(exp["patient_id"].tolist())

    exp["stage1_dt"] = pd.to_datetime(exp["stage1_date"], errors="coerce")
    stage1_map = dict(zip(exp["patient_id"], exp["stage1_dt"]))

    # Counters
    n = len(exp_ids)
    stage1_missing = int(exp["stage1_dt"].isna().sum())
    dti_mentions = set()
    implant_no_expander = set()
    expander_placement = set()

    usecols = [COL_PATIENT, COL_NOTE_TEXT, COL_NOTE_DOS, COL_OP_DATE, COL_NOTE_TYPE]

    for ch in iter_csv_safe(OP_NOTES_CSV, usecols=usecols, chunksize=120000):
        ch[COL_PATIENT] = ch[COL_PATIENT].fillna("").astype(str)
        ch = ch[ch[COL_PATIENT].isin(exp_ids)].copy()
        if ch.empty:
            continue

        # event date
        ch["dos"] = pd.to_datetime(ch[COL_NOTE_DOS], errors="coerce")
        ch["opd"] = pd.to_datetime(ch[COL_OP_DATE], errors="coerce")
        ch["event_dt"] = ch["dos"].fillna(ch["opd"])

        # Keep notes within +/- 14 days of stage1_dt (to approximate Stage1 op note neighborhood)
        ch["stage1_dt"] = ch[COL_PATIENT].map(stage1_map)
        ch["delta"] = (ch["event_dt"] - ch["stage1_dt"]).dt.days
        ch = ch[ch["delta"].between(-14, 14, inclusive=True)].copy()
        if ch.empty:
            continue

        for pid, txt in zip(ch[COL_PATIENT].tolist(), ch[COL_NOTE_TEXT].tolist()):
            t = norm(txt)
            has_exp = bool(RX_EXP.search(t))
            has_impl = bool(RX_IMPL.search(t))
            if RX_DTI.search(t):
                dti_mentions.add(pid)
            if has_impl and (not has_exp):
                implant_no_expander.add(pid)
            if RX_TE_PLACE.search(t):
                expander_placement.add(pid)

    print("Expander cohort size (has_expander==True):", n)
    print("Stage1 date missing:", stage1_missing)
    print("Near-Stage1 op-note neighborhood checks (+/-14d):")
    print("  Patients with explicit DTI mention:", len(dti_mentions))
    print("  Patients with implant but NO expander mention:", len(implant_no_expander))
    print("  Patients with explicit expander placement phrasing:", len(expander_placement))

if __name__ == "__main__":
    main()
