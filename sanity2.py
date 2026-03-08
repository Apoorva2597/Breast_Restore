import pandas as pd

GOLD = "gold_cleaned_for_cedar.csv"
PRED = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
EVID = "_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv"

gold = pd.read_csv(GOLD, dtype=str)
pred = pd.read_csv(PRED, dtype=str)
evid = pd.read_csv(EVID, dtype=str)

gold["MRN"] = gold["MRN"].astype(str).str.strip()
pred["MRN"] = pred["MRN"].astype(str).str.strip()
evid["MRN"] = evid["MRN"].astype(str).str.strip()

# Only true BMI extraction rows
bmi_evid = evid[evid["FIELD"].astype(str).str.strip() == "BMI"].copy()

# Clean up
for c in ["VALUE", "NOTE_DATE", "NOTE_TYPE", "SECTION", "EVIDENCE", "STATUS", "CONFIDENCE"]:
    if c in bmi_evid.columns:
        bmi_evid[c] = bmi_evid[c].astype(str).str.strip()

# Merge gold + pred first
merged = gold.merge(pred, on="MRN", how="inner", suffixes=("_gold", "_pred"))

# Then attach ALL BMI evidence rows
out = merged.merge(
    bmi_evid[["MRN", "VALUE", "NOTE_DATE", "NOTE_TYPE", "SECTION", "STATUS", "CONFIDENCE", "EVIDENCE"]],
    on="MRN",
    how="left"
)

# Keep only rows where there was some BMI extraction evidence
out_nonnull = out[out["VALUE"].notna()].copy()

print("\nBMI EXTRACTION QA\n")
if len(out_nonnull) == 0:
    print("No BMI extraction rows found in evidence file.")
else:
    print(
        out_nonnull[
            ["MRN", "BMI_gold", "BMI_pred", "VALUE", "NOTE_DATE", "NOTE_TYPE", "SECTION", "STATUS", "CONFIDENCE", "EVIDENCE"]
        ].head(100).to_string(index=False)
    )

out_nonnull[
    ["MRN", "BMI_gold", "BMI_pred", "VALUE", "NOTE_DATE", "NOTE_TYPE", "SECTION", "STATUS", "CONFIDENCE", "EVIDENCE"]
].to_csv("_outputs/bmi_extraction_qa.csv", index=False)

print("\nSaved:")
print("_outputs/bmi_extraction_qa.csv")
