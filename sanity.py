import pandas as pd
gold = pd.read_csv("gold_cleaned_for_cedar.csv")
print(gold[(gold["Stage2_MinorComp"]==1) & 
     (gold["Stage2_Reoperation"]==0) & 
     (gold["Stage2_Rehospitalization"]==0)].head(10))
