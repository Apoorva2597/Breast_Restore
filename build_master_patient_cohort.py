# build_master_patient_cohort.py
# Python 3.6.8 compatible
#
# Builds an "ALL patients" cohort spine from encounter files (patient_id = ENCRYPTED_PAT_ID),
# then attaches demographics + recon signals + (optional) Phase2 patient-level fields.
#
# Output:
#   patient_level_master_all.csv

from __future__ import print_function
import os
import re
import pandas as pd
import numpy as np

# -------------------------
# CONFIG (EDIT PATHS)
# -------------------------
CLINIC_ENC_FILE = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Encounters.csv"
OP_ENC_FILE     = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv"
IP_ENC_FILE     = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Encounters.csv"

# If you already created this using make_patient_level.py, set it here:
PHASE2_PATIENT_LEVEL_FIELDS = "patient_level_fields.csv"   # optional
# Or leave blank if you don't want it merged yet:
MERGE_PHASE2_FIELDS = True

OUT_FILE = "patient_level_master_all.csv"

# -------------------------
# Helpers
# -------------------------
BLANK_TOKENS = set(["", "nan", "none", "null", "na", "n/a", ".", "-", "--", "unknown"])

def read_csv_safe(path, nrows=None):
    # robust for EPIC exports
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            f = open(path, "r", encoding=enc, errors="replace")
            try:
                return pd.read_csv(f, engine="python", dtype=object, nrows=nrows)
            finally:
                try:
                    f.close()
                except Exception:
                    pass
        except Exception:
            continue
    # last resort
    return pd.read_csv(path, engine="python", dtype=object, nrows=nrows)

def norm_str(x):
    if x is None:
        return ""
    s = str(x)
    try:
        s = s.replace("\xa0", " ")
    except Exception:
        pass
    s = s.strip()
    if s.lower() in BLANK_TOKENS:
        return ""
    return s

def pick_first_nonempty(values):
    for v in values:
        s = norm_str(v)
        if s != "":
            return s
    return ""

def safe_to_datetime(series):
    # Handles "YYYY-MM-DD 0:00" etc.
    return pd.to_datetime(series, errors="coerce")

def dt_to_ymd(series):
    dt = safe_to_datetime(series)
    return dt.dt.strftime("%Y-%m-%d").fillna("")

def has_any_keyword(text, keywords):
    t = (text or "").lower()
    for k in keywords:
        if k in t:
            return True
    return False

def infer_recon_flags(proc_text, cpt_code):
    """
    Returns boolean flags for major recon categories based on procedure text + CPT.
    Conservative: only true when clear signal exists.
    """
    proc = (proc_text or "")
    cpt = (cpt_code or "")
    s = proc.lower()
    c = str(cpt).lower()

    # CPT patterns (string match)
    # Common breast recon-related CPTs appear in your profiling: 19357 (tissue expander), 19340, 19342, 19364, etc.
    is_expander = ("19357" in c) or ("tissue expand" in s) or ("expander" in s)
    is_implant  = ("19340" in c) or ("19342" in c) or ("implant" in s)
    is_flap     = ("diep" in s) or ("free flap" in s) or ("flap" in s) or ("latissimus" in s)

    # Some procedure lines include mastectomy CPTs etc; we keep them separate if you want later.
    is_mastectomy = ("mastectomy" in s)

    return {
        "Recon_has_expander": int(bool(is_expander)),
        "Recon_has_implant": int(bool(is_implant)),
        "Recon_has_flap": int(bool(is_flap)),
        "Recon_has_mastectomy": int(bool(is_mastectomy)),
    }

def choose_recon_type(row_flags):
    # Preference order (more specific)
    if row_flags.get("Recon_has_flap", 0) == 1:
        return "autologous_flap"
    if row_flags.get("Recon_has_expander", 0) == 1:
        return "tissue_expander"
    if row_flags.get("Recon_has_implant", 0) == 1:
        return "implant"
    return ""

# -------------------------
# Build spine (ALL patients)
# -------------------------
print("\n=== Build ALL-patient cohort spine from encounters ===")

clinic = read_csv_safe(CLINIC_ENC_FILE)
op     = read_csv_safe(OP_ENC_FILE)
ip     = read_csv_safe(IP_ENC_FILE)

def get_pid_col(df):
    # prefer ENCRYPTED_PAT_ID
    for c in df.columns:
        if str(c).strip().upper() == "ENCRYPTED_PAT_ID":
            return c
    # fallback contains
    for c in df.columns:
        if "ENCRYPTED" in str(c).upper() and "PAT" in str(c).upper() and "ID" in str(c).upper():
            return c
    return None

pid_c = get_pid_col(clinic)
pid_o = get_pid_col(op)
pid_i = get_pid_col(ip)

if not pid_c or not pid_o or not pid_i:
    raise RuntimeError("Could not detect ENCRYPTED_PAT_ID in one of the encounter files.")

clinic["patient_id"] = clinic[pid_c].map(norm_str)
op["patient_id"]     = op[pid_o].map(norm_str)
ip["patient_id"]     = ip[pid_i].map(norm_str)

clinic_pids = set([p for p in clinic["patient_id"].tolist() if p != ""])
op_pids     = set([p for p in op["patient_id"].tolist() if p != ""])
ip_pids     = set([p for p in ip["patient_id"].tolist() if p != ""])

all_pids = sorted(list(clinic_pids.union(op_pids).union(ip_pids)))

spine = pd.DataFrame({"patient_id": all_pids})
print("Unique patients:")
print("  Clinic encounters:", len(clinic_pids))
print("  Operation encounters:", len(op_pids))
print("  Inpatient encounters:", len(ip_pids))
print("  ALL unique (union):", len(all_pids))

# -------------------------
# Demographics (Race/Ethnicity)
# -------------------------
print("\n=== Attach demographics (Race/Ethnicity) from clinic encounters ===")
need_demo = ["patient_id", "RACE", "ETHNICITY"]
for c in ["RACE", "ETHNICITY"]:
    if c not in clinic.columns:
        raise RuntimeError("Missing {} in clinic encounters.".format(c))

demo_slim = clinic[["patient_id", "RACE", "ETHNICITY"]].copy()
demo_slim["RACE"] = demo_slim["RACE"].map(norm_str)
demo_slim["ETHNICITY"] = demo_slim["ETHNICITY"].map(norm_str)

demo_rows = []
for pid, g in demo_slim.groupby("patient_id", sort=False):
    if pid == "":
        continue
    race = pick_first_nonempty(g["RACE"].tolist())
    eth  = pick_first_nonempty(g["ETHNICITY"].tolist())
    demo_rows.append({"patient_id": pid, "Race": race, "Ethnicity": eth})

demo = pd.DataFrame(demo_rows)

# -------------------------
# Recon signals from Operation Encounters
# -------------------------
print("\n=== Attach reconstruction signals from operation encounters ===")

# Required columns per your profiling output
for c in ["PROCEDURE", "CPT_CODE"]:
    if c not in op.columns:
        raise RuntimeError("Missing {} in operation encounters.".format(c))

op_work = op[["patient_id", "PROCEDURE", "CPT_CODE"]].copy()

# Prefer OPERATION_DATE; else RECONSTRUCTION_DATE; else blank
date_col = None
for c in ["OPERATION_DATE", "RECONSTRUCTION_DATE", "OP_DATE", "DATE"]:
    if c in op.columns:
        date_col = c
        break
if date_col:
    op_work["op_date_raw"] = op[date_col]
else:
    op_work["op_date_raw"] = ""

op_work["op_date_ymd"] = dt_to_ymd(op_work["op_date_raw"])

# derive row flags
flag_rows = []
for idx, r in op_work.iterrows():
    flags = infer_recon_flags(r.get("PROCEDURE", ""), r.get("CPT_CODE", ""))
    flags["patient_id"] = r.get("patient_id", "")
    flags["op_date_ymd"] = r.get("op_date_ymd", "")
    flags["CPT_CODE_norm"] = norm_str(r.get("CPT_CODE", ""))
    flags["PROCEDURE_norm"] = norm_str(r.get("PROCEDURE", ""))
    flag_rows.append(flags)
op_flags = pd.DataFrame(flag_rows)

# aggregate to patient level
recon_rows = []
for pid, g in op_flags.groupby("patient_id", sort=False):
    if pid == "":
        continue

    # did we see any recon keyword/cpt signal at all?
    any_recon_signal = int(
        (g["Recon_has_expander"].sum() > 0) or
        (g["Recon_has_implant"].sum() > 0) or
        (g["Recon_has_flap"].sum() > 0)
    )

    # choose a "best type" based on any signal in patient history
    patient_flags = {
        "Recon_has_flap": int(g["Recon_has_flap"].sum() > 0),
        "Recon_has_expander": int(g["Recon_has_expander"].sum() > 0),
        "Recon_has_implant": int(g["Recon_has_implant"].sum() > 0),
        "Recon_has_mastectomy": int(g["Recon_has_mastectomy"].sum() > 0),
    }
    recon_type = choose_recon_type(patient_flags)

    # date range (from available op_date_ymd)
    dates = [d for d in g["op_date_ymd"].tolist() if norm_str(d) != ""]
    first_dt = min(dates) if dates else ""
    last_dt  = max(dates) if dates else ""

    # top CPT + top procedure (helpful for profiling)
    cpts = [c for c in g["CPT_CODE_norm"].tolist() if c != ""]
    procs = [p for p in g["PROCEDURE_norm"].tolist() if p != ""]
    top_cpt = ""
    top_proc = ""
    if cpts:
        top_cpt = pd.Series(cpts).value_counts().index[0]
    if procs:
        top_proc = pd.Series(procs).value_counts().index[0]

    recon_rows.append({
        "patient_id": pid,
        "Recon_any_signal_op_enc": any_recon_signal,
        "Recon_Type_op_enc": recon_type,
        "Recon_first_date_op_enc": first_dt,
        "Recon_last_date_op_enc": last_dt,
        "Recon_has_expander_op_enc": patient_flags["Recon_has_expander"],
        "Recon_has_implant_op_enc": patient_flags["Recon_has_implant"],
        "Recon_has_flap_op_enc": patient_flags["Recon_has_flap"],
        "Recon_has_mastectomy_op_enc": patient_flags["Recon_has_mastectomy"],
        "Recon_top_cpt_op_enc": top_cpt,
        "Recon_top_procedure_op_enc": top_proc[:140] if top_proc else "",
        "Recon_Source": "operation_encounters",
    })

recon = pd.DataFrame(recon_rows)

# -------------------------
# Merge into master
# -------------------------
print("\n=== Merge spine + demographics + recon ===")
master = spine.merge(demo, on="patient_id", how="left").merge(recon, on="patient_id", how="left")

# Fill obvious NaNs for indicator columns
for c in [
    "Recon_any_signal_op_enc",
    "Recon_has_expander_op_enc",
    "Recon_has_implant_op_enc",
    "Recon_has_flap_op_enc",
    "Recon_has_mastectomy_op_enc",
]:
    if c in master.columns:
        master[c] = master[c].fillna(0).astype(int)

# -------------------------
# Optional: merge Phase2 patient-level fields
# -------------------------
if MERGE_PHASE2_FIELDS:
    if os.path.exists(PHASE2_PATIENT_LEVEL_FIELDS):
        print("\n=== Merge Phase2 patient_level_fields.csv ===")
        p2 = read_csv_safe(PHASE2_PATIENT_LEVEL_FIELDS)
        if "patient_id" not in p2.columns:
            raise RuntimeError("patient_level_fields.csv is missing 'patient_id' column.")
        master = master.merge(p2, on="patient_id", how="left")
        print("Merged Phase2 fields. Columns now:", len(master.columns))
    else:
        print("\nNOTE: Phase2 fields file not found at '{}'; skipping.".format(PHASE2_PATIENT_LEVEL_FIELDS))

# -------------------------
# Write + quick summary
# -------------------------
master.to_csv(OUT_FILE, index=False, encoding="utf-8")
print("\nWrote:", OUT_FILE)
print("Rows (patients):", master.shape[0])
print("Race non-empty:", int((master["Race"].fillna("").map(norm_str) != "").sum()) if "Race" in master.columns else 0)
print("Ethnicity non-empty:", int((master["Ethnicity"].fillna("").map(norm_str) != "").sum()) if "Ethnicity" in master.columns else 0)
print("Recon_any_signal_op_enc=1:", int(master["Recon_any_signal_op_enc"].sum()) if "Recon_any_signal_op_enc" in master.columns else 0)

print("\nDone.\n")
