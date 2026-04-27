import pandas as pd
df = pd.read_csv('_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv', dtype=str)
age = pd.to_numeric(df['Age'], errors='coerce')
bmi = pd.to_numeric(df['BMI'], errors='coerce')
print('Age:', round(age.mean(),1), round(age.std(),1))
print('BMI:', round(bmi.mean(),1), round(bmi.std(),1))
