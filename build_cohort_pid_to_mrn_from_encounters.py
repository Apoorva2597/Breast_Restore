# build_cohort_pid_to_mrn_from_encounters.py
# Python 3.6.8 compatible
#
# Goal:
#   Build a COMPLETE mapping for ALL encounter-derived patients:
#     ENCRYPTED_PAT_ID (your cohort patient_id) -> MRN
#
# Output:
#   cohort_pid_to_mrn_from_encounters.csv

from __future__ import print_function
import os
import pandas as pd

FILES = [
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Encounters.csv",
]

OUT_FILE = "cohort_pid_to_mrn_from_encounters.csv"

BLANK_TOKENS = set(["", "nan", "none", "null", "na", "n/a", ".", "-", "--"])

PID_CANDIDATES = ["ENCRYPTED_PAT_ID", "ENCRYPTED PAT ID", "PATIENT_ID", "PAT_ID", "PATID", "patient_id"]
MRN_CANDIDATES = ["MRN", "PAT_MRN_ID", "PAT_MRN", "MRN_ID"]

def read_csv_safe(path):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", dtype=object)
    finally:
        try:
            f.close()
        except Exception:
            pass

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

def pick_col(cols, candidates):
    # exact match first (case-insensitive)
    cols_norm = [(c, str(c).strip().upper()) for c in cols]
    for want in candidates:
        w = str(want).strip().upper()
        for c, cu in cols_norm:
            if cu == w:
                return c
    # contains match fallback
    for want in candidates:
        w = str(want).strip().upper()
        for c, cu in cols_norm:
            if w in cu:
                return c
    return None

def main():
    print("\n=== Build cohort-wide patient_id -> MRN mapping from encounters ===")
    print("Files:", len(FILES))

    all_pairs = []  # list of dicts: {patient_id, MRN, source_file}

    for path in FILES:
        print("\n--------------------------------------------")
        print("FILE:", path)
        if not os.path.exists(path):
            print("  MISSING FILE, skipping.")
            continue

        df = read_csv_safe(path)
        print("  Rows:", len(df))
        print("  Cols:", len(df.columns))

        pid_col = pick_col(df.columns, PID_CANDIDATES)
        mrn_col = pick_col(df.columns, MRN_CANDIDATES)

        print("  Detected PID col:", pid_col if pid_col else "(none)")
        print("  Detected MRN col:", mrn_col if mrn_col else "(none)")

        if not pid_col or not mrn_col:
            print("  WARNING: missing PID or MRN column; skipping file.")
            continue

        slim = df[[pid_col, mrn_col]].copy()
        slim["patient_id"] = slim[pid_col].map(norm_str)
        slim["MRN"] = slim[mrn_col].map(norm_str)
        slim["source_file"] = os.path.basename(path)

        slim = slim[(slim["patient_id"] != "") & (slim["MRN"] != "")].copy()

        print("  Nonblank PID+MRN rows:", len(slim))
        all_pairs.append(slim[["patient_id", "MRN", "source_file"]])

    if not all_pairs:
        raise RuntimeError("No usable PID+MRN pairs found in any encounter file.")

    pairs = pd.concat(all_pairs, axis=0, ignore_index=True)

    # Ambiguity checks
    pid_to_mrn_counts = pairs.groupby("patient_id")["MRN"].nunique().sort_values(ascending=False)
    mrn_to_pid_counts = pairs.groupby("MRN")["patient_id"].nunique().sort_values(ascending=False)

    n_pid_multi_mrn = int((pid_to_mrn_counts > 1).sum())
    n_mrn_multi_pid = int((mrn_to_pid_counts > 1).sum())

    print("\n=== Ambiguity checks ===")
    print("patient_id -> multiple MRNs:", n_pid_multi_mrn)
    print("MRN -> multiple patient_ids:", n_mrn_multi_pid)

    if n_pid_multi_mrn > 0:
        print("\nTop patient_id with >1 MRN (first 10):")
        print(pid_to_mrn_counts[pid_to_mrn_counts > 1].head(10))

    if n_mrn_multi_pid > 0:
        print("\nTop MRN with >1 patient_id (first 10):")
        print(mrn_to_pid_counts[mrn_to_pid_counts > 1].head(10))

    # Resolve to ONE MRN per patient_id:
    # If there were conflicts, we pick the most frequent MRN for that patient_id.
    counts = pairs.groupby(["patient_id", "MRN"]).size().reset_index(name="n")
    counts = counts.sort_values(["patient_id", "n"], ascending=[True, False])

    resolved = counts.drop_duplicates(subset=["patient_id"], keep="first").copy()
    resolved = resolved[["patient_id", "MRN", "n"]].rename(columns={"n": "supporting_rows"})

    resolved.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print("\nWrote:", OUT_FILE)
    print("Rows (unique patient_id):", len(resolved))
    print("Unique MRNs:", int(resolved["MRN"].nunique()))
    print("Done.\n")

if __name__ == "__main__":
    main()
