import pandas as pd

master = pd.read_csv("MASTER__STAGING_PATHWAY__vNEW.csv", dtype=str)
qa = pd.read_csv("qa_bundle_summary.csv", dtype=str)

qa_pids = set(qa["patient_id"])
sub = master[master["patient_id"].isin(qa_pids)]

print("QA Stage2 count:", sub["has_stage2_definitive"].astype(bool).sum())
print(sub[["patient_id","has_stage2_definitive","stage2_date","counts_19364"]].head())
