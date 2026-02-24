
import pandas as pd

df = pd.read_csv("encounters_timeline.csv", dtype=str, engine="python")

# normalize date column
date_col = [c for c in df.columns if "event" in c.lower() or "date" in c.lower()][0]
cpt_col = [c for c in df.columns if "cpt" in c.lower()][0]

anchor_date = "6/28/2021"

subset = df[df[date_col].str.startswith(anchor_date)]

print("Rows on anchor date:")
print(subset[[date_col, cpt_col, "PROCEDURE"]])
