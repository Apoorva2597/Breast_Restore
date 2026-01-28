import sys
import pandas as pd


INFILE = "patient_level_phase1_p50.csv"


def main():
    try:
        df = pd.read_csv(INFILE)
    except Exception as e:
        print("ERROR: Could not read {}: {}".format(INFILE, e))
        sys.exit(1)

    required = ["patient_id", "LymphNodeMgmt_Performed", "LymphNodeMgmt_Type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERROR: Missing required columns in {}: {}".format(INFILE, ", ".join(missing)))
        print("Columns found:", ", ".join(list(df.columns)))
        sys.exit(1)

    n = int(df.shape[0])
    print("File:", INFILE)
    print("Patients (rows):", n)

    # Coverage
    perf_nonnull = int(df["LymphNodeMgmt_Performed"].notnull().sum())
    type_nonnull = int(df["LymphNodeMgmt_Type"].notnull().sum())

    def pct(x):
        return round((float(x) / float(n)) * 100.0, 1) if n else 0.0

    print("\n--- Coverage ---")
    print("LymphNodeMgmt_Performed non-null: {} / {} ({}%)".format(perf_nonnull, n, pct(perf_nonnull)))
    print("LymphNodeMgmt_Type      non-null: {} / {} ({}%)".format(type_nonnull, n, pct(type_nonnull)))

    # Distribution: performed
    print("\n--- Performed distribution (including blanks) ---")
    # keep blanks visible
    perf_series = df["LymphNodeMgmt_Performed"].copy()
    # normalize to string labels for stable printing
    perf_labels = perf_series.apply(lambda x: "BLANK" if pd.isnull(x) else str(bool(x)))
    counts = perf_labels.value_counts(dropna=False)
    for k in ["True", "False", "BLANK"]:
        if k in counts:
            print("  {:<5s} {}".format(k, int(counts[k])))

    # Type value counts
    print("\n--- Type distribution (top values) ---")
    type_counts = df["LymphNodeMgmt_Type"].dropna().astype(str).value_counts()
    if len(type_counts) == 0:
        print("  (no non-null type values)")
    else:
        for val, ct in type_counts.head(10).items():
            print("  {:<20s} {}".format(val, int(ct)))

    # Consistency checks
    print("\n--- Consistency checks ---")
    # Type present but performed missing
    bad1 = df[df["LymphNodeMgmt_Type"].notnull() & df["LymphNodeMgmt_Performed"].isnull()]
    # Type present but performed is False
    bad2 = df[df["LymphNodeMgmt_Type"].notnull() & (df["LymphNodeMgmt_Performed"] == False)]

    print("Type present but Performed is BLANK:", int(bad1.shape[0]))
    print("Type present but Performed is False:", int(bad2.shape[0]))

    if bad1.shape[0] > 0:
        print("\nExamples: Type present but Performed BLANK (first 5)")
        print(bad1[["patient_id", "LymphNodeMgmt_Performed", "LymphNodeMgmt_Type"]].head(5).to_string(index=False))

    if bad2.shape[0] > 0:
        print("\nExamples: Type present but Performed False (first 5)")
        print(bad2[["patient_id", "LymphNodeMgmt_Performed", "LymphNodeMgmt_Type"]].head(5).to_string(index=False))

    # Quick “looks plausible” hint (non-binding)
    print("\n--- Plausibility hint ---")
    # Just a gentle check: in breast surgery cohorts, nodal management is common.
    # We only warn if it's extremely low.
    if perf_nonnull < max(1, int(0.05 * n)):
        print("WARNING: Very low LN performed coverage (<5%). If unexpected, check extractor patterns or note_type precedence.")
    else:
        print("OK: LN fields appear populated at a non-trivial rate.")

    print("\nDone.")


if __name__ == "__main__":
    main()
