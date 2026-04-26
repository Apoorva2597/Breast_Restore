import pandas as pd
df = pd.read_csv('_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv', dtype=str)
print('Radiation_After:', df['Radiation_After'].value_counts(dropna=False).to_dict())
print('Radiation_Before:', df['Radiation_Before'].value_counts(dropna=False).to_dict())
print('Radiation:', df['Radiation'].value_counts(dropna=False).to_dict())
