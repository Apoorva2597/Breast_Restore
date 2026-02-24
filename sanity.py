
import pandas as pd

path="MASTER__STAGING_PATHWAY__vNEW.csv"
df=pd.read_csv(path,dtype=str,engine="python")

def b(s): 
    return s.fillna("").str.strip().str.lower().isin(["true","t","1","yes","y"])

df["EXP"]=b(df["has_expander"])
df["REV"]=b(df["revision_only_flag"])
df["S2"]=b(df["has_stage2_definitive"])

bucket=df[df["EXP"] & df["REV"] & ~df["S2"]][["patient_id"]].drop_duplicates()

bucket.to_csv("qa_bucket_REV_EXP_noS2.csv",index=False)

sample=bucket.sample(n=min(50,len(bucket)),random_state=42)
sample.to_csv("qa_sample_50_REV_EXP_noS2.csv",index=False)

print("WROTE:")
print("  qa_bucket_REV_EXP_noS2.csv rows =",len(bucket))
print("  qa_sample_50_REV_EXP_noS2.csv rows =",len(sample))
