import pandas as pd

master=pd.read_csv("_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv",dtype=str)
gold=pd.read_csv("gold_cleaned_for_cedar.csv",dtype=str)

master["MRN"]=master["MRN"].astype(str).str.strip()
gold["MRN"]=gold["MRN"].astype(str).str.strip()

df=master.merge(gold,on="MRN",suffixes=("_pred","_gold"))

pred=pd.to_numeric(df["Age_pred"],errors="coerce")
gold_age=pd.to_numeric(df["Age_gold"],errors="coerce")

mask=gold_age.notna()

matches=((gold_age==pred)|(gold_age==pred-1)|(gold_age==pred+1))

matches=matches[mask]

print("Total compared:",mask.sum())
print("Matches with floor/round rule:",matches.sum())
print("Accuracy:",matches.sum()/mask.sum())
