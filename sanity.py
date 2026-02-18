# sanity.py
# Python 3.6.8+
from __future__ import print_function

import sys
import pandas as pd

PATH = "patient_recon_staging_refined.csv"  # change if needed

def main():
    df = pd.read_csv(PATH, engine="python")

    cols = list(df.columns)
    print("Loaded:", PATH)
    print("Columns ({}):".format(len(cols)))
    for c in cols:
        print("  -", c)

    # Look for plausible Stage2 columns
    cand = []
    for c in cols:
        cl = str(c).strip().lower()
        if ("stage2" in cl) or ("stage_2" in cl) or ("2nd" in cl) or ("second" in cl):
            cand.append(c)

    print("\nCandidate Stage2-related columns:", cand if cand else "NONE FOUND")

    # If a stage2_date-like column exists, compute non-null count
    preferred = ["stage2_date", "stage2_dt", "stage2", "stage_2_date", "stage2_dos", "stage2_date_of_service"]
    found = None
    lower_map = {str(c).strip().lower(): c for c in cols}
    for p in preferred:
        if p in lower_map:
            found = lower_map[p]
            break

    if found is None:
        print("\nNo stage2 date column found in this file, so there is nothing to count here.")
        print("If you want structured Stage2, run this script on patient_recon_structured.csv (or whichever file has recon fields).")
        print("If you want note-derived Stage2, run counts on stage2_from_notes_patient_level.csv (column: stage2_event_dt_best).")
        return

    n = int(df[found].notnull().sum())
    print("\nStructured Stage2 non-null count using column '{}': {}".format(found, n))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
