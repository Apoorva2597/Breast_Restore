import pandas as pd
df = pd.read_csv('_outputs/complications_patch_evidence.csv', dtype=str)
s2 = df[df['FIELD'] == 'Stage2_Rehospitalization']
print('Stage2_Rehospitalization rows:', len(s2))
print(s2[['NOTE_DATE','STATUS','VALUE','EVIDENCE']].to_string())
