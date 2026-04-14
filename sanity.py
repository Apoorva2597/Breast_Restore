import pandas as pd
df = pd.read_csv('_outputs/complications_patch_evidence.csv', dtype=str)
s2 = df[df['STAGE_ASSIGNED'] == 'STAGE2']
print('Total STAGE2 evidence rows:', len(s2))
print(s2['FIELD'].value_counts().head(20))
