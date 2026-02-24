
import pandas as pd

path = "encounters_timeline.csv"
df = pd.read_csv(path, dtype=str, engine="python")

# Try common header names. If your file uses a slightly different one, we’ll detect it.
candidates = [c for c in df.columns if c.strip().upper() in ("CPT_CODE","CPT","CPTCODE")]
if not candidates:
    # fallback: anything containing 'cpt'
    candidates = [c for c in df.columns if "cpt" in c.lower()]

print("Detected CPT column(s):", candidates)
cpt_col = candidates[0]

# Count top CPTs
counts = (df[cpt_col].fillna("").str.strip())
counts = counts[counts != ""]
top = counts.value_counts().head(10)

print("\nTop 10 CPT codes:")
for cpt, n in top.items():
    print(f"{n:>5}  {cpt}")
