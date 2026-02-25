import pandas as pd 

print(gold[(gold["Stage2_MinorComp"]==1) & 
     (gold["Stage2_Reoperation"]==0) & 
     (gold["Stage2_Rehospitalization"]==0)][
     ["Stage2_MinorComp",
      "Stage2_Reoperation",
      "Stage2_Rehospitalization",
      "Stage2_MajorComp",
      "Stage2_Failure",
      "Stage2_Revision"]
].value_counts())
