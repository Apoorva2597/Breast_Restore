import pandas as pd
df = pd.read_csv('_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv', dtype=str)
print('Total rows:', len(df))
cols = ['Age','BMI','Race','Ethnicity','SmokingStatus','Diabetes','Obesity',
        'Chemo_Before','Radiation_After','Recon_Classification','Recon_Timing','Recon_Laterality']
for c in cols:
    print(c, df[c].value_counts(dropna=False).head(6).to_dict())
