
import pandas as pd
df = pd.read_csv('_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv', dtype=str)
print('Unique MRNs:', df['MRN'].nunique())
