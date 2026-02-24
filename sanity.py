import pandas as pd

path = "encounters_timeline.csv"
df = pd.read_csv(path, dtype=str, engine="python")

cands = [c for c in df.columns if c.strip().upper() in ("CPT_CODE","CPT","CPTCODE")]
if not cands:
    cands = [c for c in df.columns if "cpt" in c.lower()]

print("Detected CPT column(s):", cands)
cpt_col = cands[0]

s = df[cpt_col].fillna("").astype(str).str.strip()
s = s[s != ""]

print("\nTop 10 CPT codes:")
top = s.value_counts().head(10)
for code, n in top.items():
    print(f"{n:>5}  {code}")

