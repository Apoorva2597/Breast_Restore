import pandas as pd

st = pd.read_csv("/home/apokol/Breast_Restore/MASTER__STAGING_PATHWAY__vNEW.csv", dtype=str)
qa = pd.read_csv("/home/apokol/Breast_Restore/qa_bundle_summary.csv", dtype=str)  # or your qa patient list csv
qa_pids = set(qa["patient_id"].astype(str).str.strip())

sub = st[st["patient_id"].astype(str).str.strip().isin(qa_pids)].copy()

# Normalize booleans that might be strings
def as_bool(x):
    s = str(x).strip().lower()
    return s in ("true","1","t","yes","y")

sub["has_stage2_definitive_bool"] = sub["has_stage2_definitive"].apply(as_bool)

print("QA patients:", len(sub))
print("Stage2 True:", int(sub["has_stage2_definitive_bool"].sum()))

# Key: confirm counts_19364 is zero across QA
if "counts_19364" in sub.columns:
    sub["counts_19364_num"] = pd.to_numeric(sub["counts_19364"], errors="coerce").fillna(0).astype(int)
    print("Any counts_19364 > 0:", int((sub["counts_19364_num"] > 0).sum()))
else:
    print("WARNING: counts_19364 missing from staging file.")
