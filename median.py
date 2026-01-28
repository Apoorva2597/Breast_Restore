import pandas as pd
df=pd.read_csv('patient_level_phase1_p50.csv')
print(df[['LymphNodeMgmt_Performed','LymphNodeMgmt_Type']].head())
print('Performed non-null:', df['LymphNodeMgmt_Performed'].notnull().sum()); print('Type non-null:', df['LymphNodeMgmt_Type'].notnull().sum())"
