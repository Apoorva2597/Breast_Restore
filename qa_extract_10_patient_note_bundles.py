# qa_extract_10_patient_note_bundles.py
# Python 3.6+ (pandas required)
#
# Purpose:
#   Pick ~10 expander patients and extract NOTE_TYPE + NOTE_TEXT from
#   Operation / Clinic / Inpatient Notes CSVs for manual review + rule refinement.
#
# Outputs:
#   1) qa_10patients_all_notes_minimal.csv
#   2) qa_10patients_note_bundles/ENCRYPTED_PAT_ID_<id>.txt   (one file per patient)
#
# Key design:
#   - Reads NOTES csvs with encoding='latin-1' to avoid utf-8 decode crashes (0xA0/NBSP).
#   - Keeps columns minimal: patient_id, note_type, note_text (+ a few IDs if present).
#
# Edit only the CONFIG section.

import os
import re
import sys
import random
import pandas as pd


# -------------------------
# CONFIG
# -------------------------
PATIENT_STAGING_CSV = "patient_recon_staging.csv"
# Optional (if present): used to stratify by best tier A/B/C/NONE
STAGE2_FROM_NOTES_PATIENT_LEVEL_CSV = "stage2_from_notes_patient_level.csv"

# Notes files (edit paths)
OP_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"
CLINIC_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"
INPATIENT_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Notes.csv"

# How many patients to sample per tier (total ~10)
# If STAGE2_FROM_NOTES_PATIENT_LEVEL_CSV is missing, we sample randomly from expander cohort.
N_PER_TIER = {
    "A": 3,
    "B": 3,
    "C": 3,
    "NONE": 1,
}
RANDOM_SEED = 42

# Output
OUT_CSV = "qa_10patients_all_notes_minimal.csv"
OUT_DIR = "qa_10patients_note_bundles"

# Column expectations (if some are missing in a file, script will fill blanks)
COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_ID = "NOTE_ID"
COL_NOTE_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"


# -------------------------
# Helpers
# -------------------------
def read_csv_safe(path, usecols=None, chunksize=None, is_notes=False):
    """
    Safe CSV reader.
    - For notes files: force encoding='latin-1' to avoid utf-8 decode errors (e.g., 0xA0).
    - For non-notes: try utf-8 then cp1252.
    """
    if is_notes:
        # latin-1 will never throw UnicodeDecodeError; bytes 0-255 map directly.
        return pd.read_csv(path, encoding="latin-1", engine="python", usecols=usecols, chunksize=chunksize)
    else:
        try:
            return pd.read_csv(path, encoding="utf-8", engine="python", usecols=usecols, chunksize=chunksize)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp1252", engine="python", usecols=usecols, chunksize=chunksize)


def norm_text_minimal(x):
    """Light normalization for review; keep content readable."""
    if x is None:
        return ""
    s = str(x)
    # Replace NBSP-like artifacts that can sneak in as weird spacing
    s = s.replace(u"\xa0", " ")
    s = s.replace("\r", "\n")
    # Collapse excessive whitespace but keep newlines for readability
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def to_bool(x):
    s = str(x).strip().lower()
    return s in ["true", "1", "yes", "y", "t"]


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def pick_patients(expander_ids, stage2_patient_level_path):
    """
    Returns a list of patient_ids (strings).
    If stage2 patient-level exists, stratify by stage2_tier_best into A/B/C/NONE.
    Else random sample from expander_ids.
    """
    random.seed(RANDOM_SEED)
    expander_ids = [str(x) for x in expander_ids]

    if stage2_patient_level_path and os.path.exists(stage2_patient_level_path):
        df = read_csv_safe(stage2_patient_level_path, is_notes=False)
        # tolerate different column name variants
        tier_col = None
        for c in ["stage2_tier_best", "tier", "best_tier", "stage2_tier"]:
            if c in df.columns:
                tier_col = c
                break
        if tier_col is None:
            print("WARN: {} found but no tier column recognized; falling back to random.".format(stage2_patient_level_path))
            return random.sample(expander_ids, min(10, len(expander_ids)))

        df["patient_id"] = df["patient_id"].astype(str) if "patient_id" in df.columns else df[COL_PATIENT].astype(str)
        df[tier_col] = df[tier_col].fillna("NONE").astype(str)

        # keep only expanders
        df = df[df["patient_id"].isin(set(expander_ids))].copy()

        chosen = []
        for tier, n in N_PER_TIER.items():
            pool = df[df[tier_col] == tier]["patient_id"].tolist()
            random.shuffle(pool)
            chosen.extend(pool[:n])

        # If we still have fewer than target, top up from remaining expanders
        target_n = sum(N_PER_TIER.values())
        if len(chosen) < target_n:
            remaining = [pid for pid in expander_ids if pid not in set(chosen)]
            random.shuffle(remaining)
            chosen.extend(remaining[: (target_n - len(chosen))])

        # Deduplicate while preserving order
        seen = set()
        out = []
        for pid in chosen:
            if pid not in seen:
                out.append(pid)
                seen.add(pid)
        return out

    # fallback: random sample
    target_n = sum(N_PER_TIER.values())
    return random.sample(expander_ids, min(target_n, len(expander_ids)))


def extract_notes_for_patients(file_tag, notes_csv_path, patient_ids_set, chunksize=200000):
    """
    Extract minimal note fields for selected patients from one notes CSV.
    Returns a dataframe.
    """
    if not os.path.exists(notes_csv_path):
        print("WARN: missing file, skipping:", notes_csv_path)
        return pd.DataFrame(columns=[
            "file_tag", COL_PATIENT, COL_NOTE_TYPE, COL_NOTE_ID, COL_NOTE_DOS, COL_OP_DATE, COL_NOTE_TEXT
        ])

    # Read header to see which columns exist
    head = read_csv_safe(notes_csv_path, is_notes=True, chunksize=None)
    cols_present = set(head.columns.tolist())

    wanted = [COL_PATIENT, COL_NOTE_TYPE, COL_NOTE_TEXT, COL_NOTE_ID, COL_NOTE_DOS, COL_OP_DATE]
    usecols = [c for c in wanted if c in cols_present]

    # Must have patient + text
    if COL_PATIENT not in cols_present or COL_NOTE_TEXT not in cols_present:
        raise RuntimeError("Notes file {} missing required columns: {} and/or {}".format(
            notes_csv_path, COL_PATIENT, COL_NOTE_TEXT
        ))

    out_chunks = []

    reader = read_csv_safe(notes_csv_path, usecols=usecols, chunksize=chunksize, is_notes=True)
    for chunk in reader:
        chunk[COL_PATIENT] = chunk[COL_PATIENT].fillna("").astype(str)
        chunk = chunk[chunk[COL_PATIENT].isin(patient_ids_set)].copy()
        if chunk.empty:
            continue

        # add missing columns as blanks for uniform output
        for c in wanted:
            if c not in chunk.columns:
                chunk[c] = ""

        chunk[COL_NOTE_TYPE] = chunk[COL_NOTE_TYPE].fillna("").astype(str)
        chunk[COL_NOTE_ID] = chunk[COL_NOTE_ID].fillna("").astype(str)
        chunk[COL_NOTE_DOS] = chunk[COL_NOTE_DOS].fillna("").astype(str)
        chunk[COL_OP_DATE] = chunk[COL_OP_DATE].fillna("").astype(str)

        # minimal normalization on text for readability
        chunk[COL_NOTE_TEXT] = chunk[COL_NOTE_TEXT].apply(norm_text_minimal)

        chunk.insert(0, "file_tag", file_tag)
        out_chunks.append(chunk[["file_tag"] + wanted])

    if out_chunks:
        return pd.concat(out_chunks, ignore_index=True)
    return pd.DataFrame(columns=["file_tag"] + wanted)


def write_patient_bundle_txt(df_all, patient_id, out_dir):
    """
    Write one human-readable text file for a patient with notes grouped by file_tag and note_type.
    """
    sub = df_all[df_all[COL_PATIENT].astype(str) == str(patient_id)].copy()
    if sub.empty:
        return

    # sort for stable review: file_tag, note_type, note_dos/op_date, note_id
    sub["sort_dt"] = sub[COL_NOTE_DOS].where(sub[COL_NOTE_DOS].astype(str).str.len() > 0, sub[COL_OP_DATE])
    sub = sub.sort_values(by=["file_tag", COL_NOTE_TYPE, "sort_dt", COL_NOTE_ID], ascending=True)

    p = os.path.join(out_dir, "ENCRYPTED_PAT_ID_{}.txt".format(patient_id))
    with open(p, "w") as f:
        f.write("PATIENT: {}\n".format(patient_id))
        f.write("=" * 80 + "\n\n")

        for _, r in sub.iterrows():
            f.write("[{}] NOTE_TYPE: {}\n".format(r["file_tag"], r[COL_NOTE_TYPE]))
            if str(r[COL_NOTE_DOS]).strip():
                f.write("NOTE_DATE_OF_SERVICE: {}\n".format(r[COL_NOTE_DOS]))
            if str(r[COL_OP_DATE]).strip():
                f.write("OPERATION_DATE: {}\n".format(r[COL_OP_DATE]))
            if str(r[COL_NOTE_ID]).strip():
                f.write("NOTE_ID: {}\n".format(r[COL_NOTE_ID]))
            f.write("-" * 80 + "\n")
            f.write(r[COL_NOTE_TEXT] + "\n")
            f.write("\n" + "=" * 80 + "\n\n")


def main():
    random.seed(RANDOM_SEED)

    # -------------------------
    # Load expander cohort (848)
    # -------------------------
    stg = read_csv_safe(PATIENT_STAGING_CSV, is_notes=False)
    if "patient_id" not in stg.columns or "has_expander" not in stg.columns:
        raise RuntimeError("patient_recon_staging.csv must contain columns: patient_id, has_expander")

    stg["patient_id"] = stg["patient_id"].astype(str)
    stg["has_expander_bool"] = stg["has_expander"].apply(to_bool)

    exp = stg[stg["has_expander_bool"]].copy()
    expander_ids = exp["patient_id"].tolist()

    print("Expander patients:", len(expander_ids))
    if len(expander_ids) == 0:
        raise RuntimeError("No expander patients found (has_expander==True).")

    # -------------------------
    # Choose 10 patients (stratified if stage2 patient-level exists)
    # -------------------------
    chosen = pick_patients(expander_ids, STAGE2_FROM_NOTES_PATIENT_LEVEL_CSV)
    chosen = [str(x) for x in chosen]
    chosen_set = set(chosen)

    print("Chosen patients (n={}):".format(len(chosen)))
    for pid in chosen:
        print("  -", pid)

    # -------------------------
    # Extract notes from each notes file
    # -------------------------
    all_parts = []

    all_parts.append(extract_notes_for_patients("operation_notes", OP_NOTES_CSV, chosen_set))
    all_parts.append(extract_notes_for_patients("clinic_notes", CLINIC_NOTES_CSV, chosen_set))
    all_parts.append(extract_notes_for_patients("inpatient_notes", INPATIENT_NOTES_CSV, chosen_set))

    df_all = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    if df_all.empty:
        print("No notes found for chosen patients across provided notes files.")
        df_all.to_csv(OUT_CSV, index=False)
        return

    # -------------------------
    # Write combined CSV
    # -------------------------
    df_all.to_csv(OUT_CSV, index=False)

    # -------------------------
    # Write per-patient bundles
    # -------------------------
    ensure_dir(OUT_DIR)
    for pid in chosen:
        write_patient_bundle_txt(df_all, pid, OUT_DIR)

    print("\nWrote:")
    print("  -", OUT_CSV)
    print("  -", OUT_DIR + "/ENCRYPTED_PAT_ID_<id>.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
