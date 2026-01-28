import pandas as pd

FILE = "evidence_log_phase1_p50.csv"

def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")

def main():
    df = read_csv_safely(FILE)
    df["evidence"] = df["evidence"].fillna("").astype(str)
    df["section"] = df["section"].fillna("").astype(str)

    treat = df[df["field"].isin(["Chemo", "Radiation"])].copy()
    print("Total treatment evidence rows:", len(treat))

    # (1) Quick sample for human review
    print("\n=== SAMPLE (10 Chemo + 10 Radiation) ===")
    for f in ["Chemo", "Radiation"]:
        sub = treat[treat["field"] == f].head(10)
        print("\n--", f, "sample rows:", len(sub))
        for i, row in sub.iterrows():
            ev = row["evidence"].replace("\n", " ")
            print("section=", row.get("section", ""), "| note_type=", row.get("note_type", ""), "|", ev[:200])

    # (2) Endocrine exclusion check
    tam = treat[treat["evidence"].str.lower().str.contains("tamoxifen")]
    print("\nTamoxifen present in treatment-labeled evidence rows:", len(tam))
    if len(tam) > 0:
        print("WARNING: tamoxifen appears inside Chemo/Radiation-labeled evidence. Review below:")
        print(tam[["field","status","section","note_type","evidence"]].head(5).to_string(index=False))

    # (3) Section suppression check
    bad_sections = treat[treat["section"].str.upper().isin(["FAMILY HISTORY", "ALLERGIES"])]
    print("\nTreatment hits from FAMILY HISTORY / ALLERGIES:", len(bad_sections))
    if len(bad_sections) > 0:
        print("WARNING: treatment extracted from suppressed sections. Review below:")
        print(bad_sections[["field","status","section","note_type","evidence"]].head(5).to_string(index=False))

    print("\nDone.")

if __name__ == "__main__":
    main()
