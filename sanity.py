import pandas as pd
df=pd.read_csv('/home/apokol/Breast_Restore/MASTER__STAGING_PATHWAY__vNEW.csv',dtype=str,engine='python'); 
b=lambda s: s.fillna('').astype(str).str.strip().str.lower().isin(['true','t','1','yes','y']); 
df['EXP']=b(df['has_expander']); df['S2']=b(df['has_stage2_definitive']); df['REV']=b(df['revision_only_flag']); 
print(df.groupby(['S2','REV','EXP']).size().reset_index(name='patients').sort_values('patients',ascending=False).to_string(index=False))
