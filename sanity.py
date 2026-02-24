
import pandas as pd
j="/home/apokol/Breast_Restore/VALIDATION_STAGE12/joined_gold_crosswalk_pipeline.csv"
df=pd.read_csv(j, dtype=str, engine="python")
# normalize
df["gold_stage2"] = df["Stage2_Applicable"].fillna("").astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])
df["pipe_stage2"] = df.get("has_stage2_definitive","").fillna("").astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])
fn = df[(df["gold_stage2"]==True) & (df["pipe_stage2"]==False)].copy()
print("Stage2 FN rows:", len(fn))
# show any gold fields that hint at type (adjust column names if present)
for c in ["Recon_Type","Recon_Timing","Recon_Classification","Stage2_Applicable"]:
    if c in fn.columns:
        print("\nTop values for", c)
        print(fn[c].fillna("").value_counts().head(15))
