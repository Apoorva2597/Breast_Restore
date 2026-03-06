import pandas as pd
m=pd.read_csv("_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv",dtype=str); g=pd.read_csv("gold_cleaned_for_cedar.csv",dtype=str)
x=m.merge(g,on="MRN",suffixes=("_pred","_gold")); print(x[(x.Race_gold.notna())&(x.Race_gold.str.strip()!="")&(x.Race_pred.str.strip()!=x.Race_gold.str.strip())][["MRN","Race_gold","Race_pred"]])
