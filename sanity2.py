import pandas as pd
def norm_token(x):
    s=str(x).strip().lower()
    if s in ["", "nan", "none", "null", "na"]: return ""
    if s in ["white","white or caucasian","caucasian"]: return "White"
    if s in ["black","black or african american","african american"]: return "Black or African American"
    if s in ["asian","filipino","other asian","asian indian","chinese","japanese","korean","vietnamese"]: return "Asian"
    if s=="american indian or alaska native": return "American Indian or Alaska Native"
    if s in ["native hawaiian","pacific islander","native hawaiian or other pacific islander"]: return "Native Hawaiian or Other Pacific Islander"
    if s=="other": return "Other"
    if s in ["unknown","declined","refused","patient refused","choose not to disclose","unknown / declined / not reported"]: return "Unknown / Declined / Not Reported"
    return str(x).strip()
def collapse(v):
    raw=str(v).replace(";",",").split(",")
    real=[]; unk=False
    for p in raw:
        n=norm_token(p)
        if not n: continue
        if n=="Unknown / Declined / Not Reported":
            unk=True
            continue
        if n not in real:
            real.append(n)
    if len(real)==0:
        return "Unknown / Declined / Not Reported" if unk else pd.NA
    if len(real)==1:
        return real[0]
    return "Multiracial"
m=pd.read_csv("_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv",dtype=str)
g=pd.read_csv("gold_cleaned_for_cedar.csv",dtype=str)
m["MRN"]=m["MRN"].astype(str).str.strip(); g["MRN"]=g["MRN"].astype(str).str.strip()
x=m.merge(g,on="MRN",suffixes=("_pred","_gold"))
x["Race_pred_n"]=x["Race_pred"].apply(collapse)
x["Race_gold_n"]=x["Race_gold"].apply(collapse)
print(x[(x["Race_gold"].notna())&(x["Race_gold"].astype(str).str.strip()!="")&(x["Race_pred_n"]!=x["Race_gold_n"])][["MRN","Race_gold","Race_pred","Race_gold_n","Race_pred_n"]])
