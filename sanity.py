import pandas as pd

df=pd.read_csv('stage1_complications_row_hits.csv')
print(df['complication'].value_counts().head(20))
