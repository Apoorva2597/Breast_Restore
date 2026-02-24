import pandas as pd
j="/home/apokol/Breast_Restore/VALIDATION_STAGE12/joined_gold_crosswalk_pipeline.csv"
df=pd.read_csv(j, dtype=str, engine="python")
def b(s):
    s=str(s).strip().lower()
    return s in ["1","true","t","yes","y"]
df["gold_stage2"]=df["Stage2_Applicable"].apply(b)

for col in ["counts_19364","counts_19380","counts_19350","counts_19357"]:
    if col in df.columns:
        df[col]=pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

g=df[df["gold_stage2"]==True]
print("Gold Stage2 True:", len(g))
for col in ["counts_19364","counts_19380","counts_19350","counts_19357"]:
    if col in g.columns:
        print(col, " >0 :", int((g[col]>0).sum()))
