
import pandas as pd
df = pd.read_csv("_outputs/validation_mismatches.csv", dtype=str, encoding="latin1")
print("nrows:", len(df))
print("columns:", list(df.columns))
print(df.head(3).to_string(index=False))
