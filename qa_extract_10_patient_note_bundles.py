# qa_pull_2C_2NONE_all_notes_v2.py
# Python 3.6+ (pandas required)
#
# Fixes: avoids "I/O operation on closed file" by NOT wrapping open() around chunk iterators.

from __future__ import print_function

import os
import re
import sys
import pandas as pd

# -------------------------
# CONFIG: edit paths
# -------------------------
PATIENT_LEVEL_CSV = "stage2_from_notes_patient_level.csv"

NOTES_FILES = [
    ("operation_notes", "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"),
    ("clinic_notes",    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"),
    ("inpatient_notes", "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Notes.csv"),
]

OUT_ALL_NOTES_CSV = "qa_4_patients_all_notes.csv"
OUT_TXT_DIR = "qa_patient_notes"

N_TIER_C = 2
N_NONE = 2
CHUNKSIZE = 150000

COL_PAT = "ENCRYPTED_PAT_ID"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_ID = "NOTE_ID"
COL_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"

REQUIRED_NOTE_COLS = [COL_PAT, COL_NOTE_TYPE, COL_NOTE_TEXT, COL_NOTE_ID, COL_DOS, COL_OP_DATE]

# -------------------------
# Helpers
# -------------------------
def read_csv_try_encodings(path, **kwargs):
    """
    Read CSV by trying common encodings.
    For chunked reads, returns the TextFileReader iterator.
    """
    encodings = ["utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def ensure_cols_exist(df_cols, required):
    missing = [c for c in required if c not in df_cols]
    if missing:
        raise RuntimeError("Missing columns: {}".format(missing))

def clean_text(x):
    if x is None:
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = s.replace("\u00a0", " ")  # NBSP if present
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_dt(x):
    return pd.to_datetime(x, errors="coerce")

def pick_patients(pl, tier_value, n_needed):
    sub = pl[pl["stage2_tier_best_norm"] == tier_value].copy()
    sub = sub.sort_values(by=["patient_id"], ascending=[True])
    return sub["patient_id"].head(n_needed).tolist()

# -------------------------
# Main
# -------------------------
def main():
    # 1) Load patient-level file and pick patients
    pl = read_csv_try_encodings(PATIENT_LEVEL_CSV)

    for col in ["patient_id", "stage2_tier_best"]:
        if col not in pl.columns:
            raise RuntimeError("Expected column '{}' in {}".format(col, PATIENT_LEVEL_CSV))

    pl["patient_id"] = pl["patient_id"].fillna("").astype(str)
    pl["stage2_tier_best_norm"] = pl["stage2_tier_best"].fillna("NONE").astype(str).str.strip().str.upper()

    chosen_C = pick_patients(pl, "C", N_TIER_C)
    chosen_NONE = pick_patients(pl, "NONE", N_NONE)

    # Fallback logic if not enough
    if len(chosen_C) < N_TIER_C:
        print("WARN: Only found {} Tier C; filling remainder from Tier B.".format(len(chosen_C)))
        need = N_TIER_C - len(chosen_C)
        chosen_C.extend(pick_patients(pl, "B", need))

    if len(chosen_NONE) < N_NONE:
        print("WARN: Only found {} NONE; filling remainder from Tier A.".format(len(chosen_NONE)))
        need = N_NONE - len(chosen_NONE)
        chosen_NONE.extend(pick_patients(pl, "A", need))

    chosen = chosen_C + chosen_NONE
    chosen = list(dict.fromkeys(chosen))  # de-dupe while keeping order
    chosen_set = set(chosen)

    print("Chosen patients (n={}):".format(len(chosen)))
    for pid in chosen:
        tier = pl.loc[pl["patient_id"] == pid, "stage2_tier_best_norm"].iloc[0]
        print("  {}  tier={}".format(pid, tier))

    if len(chosen) == 0:
        raise RuntimeError("No patients selected. Check stage2_from_notes_patient_level.csv.")

    # 2) Stream each notes file and collect all notes for chosen patients
    os.makedirs(OUT_TXT_DIR, exist_ok=True)

    all_rows = []
    for file_tag, path in NOTES_FILES:
        print("\nScanning:", file_tag, path)

        head = read_csv_try_encodings(path, nrows=5)
        ensure_cols_exist(head.columns, REQUIRED_NOTE_COLS)

        # IMPORTANT: chunksize iterator from PATH (not open handle)
        chunk_iter = read_csv_try_encodings(path, usecols=REQUIRED_NOTE_COLS, chunksize=CHUNKSIZE)

        kept = 0
        for chunk in chunk_iter:
            chunk[COL_PAT] = chunk[COL_PAT].fillna("").astype(str)
            chunk = chunk[chunk[COL_PAT].isin(chosen_set)].copy()
            if chunk.empty:
                continue

            kept += len(chunk)

            chunk["file_tag"] = file_tag
            chunk["NOTE_TEXT_CLEAN"] = chunk[COL_NOTE_TEXT].apply(clean_text)
            chunk["NOTE_DATE_OF_SERVICE_DT"] = to_dt(chunk[COL_DOS])
            chunk["OPERATION_DATE_DT"] = to_dt(chunk[COL_OP_DATE])

            out_cols = [
                "file_tag", COL_PAT, COL_NOTE_TYPE, COL_NOTE_ID,
                COL_DOS, COL_OP_DATE, "NOTE_DATE_OF_SERVICE_DT", "OPERATION_DATE_DT",
                "NOTE_TEXT_CLEAN"
            ]
            all_rows.append(chunk[out_cols])

        print("  Kept rows:", kept)

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=[
            "file_tag", COL_PAT, COL_NOTE_TYPE, COL_NOTE_ID,
            COL_DOS, COL_OP_DATE, "NOTE_DATE_OF_SERVICE_DT", "OPERATION_DATE_DT",
            "NOTE_TEXT_CLEAN"
        ])

    # 3) Sort and write master CSV
    out = out.sort_values(
        by=[COL_PAT, "NOTE_DATE_OF_SERVICE_DT", "OPERATION_DATE_DT", "file_tag"],
        ascending=[True, True, True, True]
    )
    out.to_csv(OUT_ALL_NOTES_CSV, index=False)
    print("\nWrote:", OUT_ALL_NOTES_CSV, "rows=", len(out))

    # 4) Write one text file per patient
    for pid in chosen:
        sub = out[out[COL_PAT] == pid].copy()
        txt_path = os.path.join(OUT_TXT_DIR, "{}.txt".format(pid))
        tier = pl.loc[pl["patient_id"] == pid, "stage2_tier_best_norm"].iloc[0]

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("PATIENT: {}\n".format(pid))
            f.write("TIER (patient-level best): {}\n".format(tier))
            f.write("TOTAL NOTES ROWS: {}\n\n".format(len(sub)))

            for _, r in sub.iterrows():
                f.write("=" * 80 + "\n")
                f.write("file_tag: {}\n".format(r.get("file_tag", "")))
                f.write("NOTE_TYPE: {}\n".format(r.get(COL_NOTE_TYPE, "")))
                f.write("NOTE_ID: {}\n".format(r.get(COL_NOTE_ID, "")))
                f.write("NOTE_DOS: {}\n".format(r.get(COL_DOS, "")))
                f.write("OP_DATE: {}\n".format(r.get(COL_OP_DATE, "")))
                f.write("\n")
                f.write(r.get("NOTE_TEXT_CLEAN", ""))
                f.write("\n\n")

        print("Wrote:", txt_path)

    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
