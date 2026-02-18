# sanity_stage2_notes.py
# Python 3.6.8+
from __future__ import print_function
import sys
import pandas as pd

PATH = "stage2_from_notes_patient_level.csv"

def main():
    df = pd.read_csv(PATH, engine="python")
    print("Loaded:", PATH)
    print("Rows:", len(df))

    required = ["patient_id", "stage2_tier_best", "stage2_event_dt_best", "stage2_after_index"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Missing required columns: {}".format(missing))

    total = len(df)
    any_hit = int(df["stage2_tier_best"].notnull().sum())
    after_index = int(df["stage2_after_index"].fillna(False).astype(bool).sum())

    print("Patients with ANY Stage2 hit:", any_hit, "({:.1f}%)".format(100.0 * any_hit / total if total else 0.0))
    print("Patients with AFTER-index best hit:", after_index, "({:.1f}%)".format(100.0 * after_index / total if total else 0.0))

    print("\nTier counts:")
    print(df["stage2_tier_best"].fillna("NONE").value_counts())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
