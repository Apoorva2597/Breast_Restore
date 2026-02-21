import pandas as pd

gold = pd.read_csv("gold_cleaned_for_cedar.csv", dtype=str)
pred = pd.read_csv("pred_spine_stage1_stage2.csv", dtype=str)

# pick the join key you actually want to use
# usually PatientID or MRN, depending on what pred_spine has
print("Gold columns:", [c for c in gold.columns if c in ["MRN","PatientID","patient_id"]])
print("Pred columns:", [c for c in pred.columns if c in ["MRN","PatientID","patient_id"]])

key = "PatientID"  # change if needed
gold_ids = set(gold[key].dropna().str.strip())
pred_ids  = set(pred[key].dropna().str.strip())

print("Gold n:", len(gold_ids))
print("Pred n:", len(pred_ids))
print("Intersection:", len(gold_ids & pred_ids))
print("Gold-only:", len(gold_ids - pred_ids))
print("Pred-only:", len(pred_ids - gold_ids))

# optional: list a few missing IDs
print("Example gold-only:", list(gold_ids - pred_ids)[:10])
print("Example pred-only:", list(pred_ids - gold_ids)[:10])
