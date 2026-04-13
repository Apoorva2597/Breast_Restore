import pandas as pd
df = pd.read_csv('_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv', dtype=str)
cols = ['PBS_Breast Reduction','PBS_Mastopexy','PBS_Other']
for c in cols:
    if c in df.columns:
        vc = df[c].value_counts(dropna=False)
        print(c, dict(vc))
    else:
        print(c, 'COLUMN MISSING')
