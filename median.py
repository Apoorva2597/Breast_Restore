import pandas as pd

EVIDENCE_FILE = "evidence_log_phase1_p50.csv"

FIELDS = [
    "Stage1_Reoperation",
    "Stage1_Rehospitalization",
    "Stage1_Failure"
]


def main():
    print("Loading evidence log...")
    df = pd.read_csv(EVIDENCE_FILE)

    df = df[df["field"].isin(FIELDS)]

    if df.empty:
        print("No Stage1 outcome rows found in evidence log.")
        return

    print("\nTotal evidence rows (Stage1 outcomes):", df.shape[0])

    for field in FIELDS:
        sub = df[df["field"] == field]

        print("\n==============================")
        print("FIELD:", field)
        print("==============================")

        if sub.empty:
            print("No rows.")
            continue

        # Status distribution
        print("\nStatus distribution:")
        print(sub["status"].value_counts())

        # Count by note type
        print("\nNote type distribution:")
        print(sub["note_type"].value_counts())

        # Show sample evidence lines
        print("\n--- Sample evidence (first 5) ---")
        samples = sub.head(5)
        for i, row in samples.iterrows():
            print("\nPatient:", row["patient_id"])
            print("Status:", row["status"])
            print("Note type:", row["note_type"])
            print("Section:", row["section"])
            print("Evidence:")
            print(row["evidence"])
            print("-" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()
