import pandas as pd

df=pd.read_csv('stage1_complications_row_hits.csv')
print(df[df['complication'].eq('Other (systemic)')]['snippet'].head(20).tolist())
