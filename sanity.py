
import pandas as pd
from pathlib import Path

base = Path("PATIENT_BUNDLES")

all_cpts = []

for patient_dir in base.iterdir():
    f = patient_dir / "encounters_timeline.csv"
    if f.exists():
        df = pd.read_csv(f, dtype=str, engine="python")
        cpt_col = [c for c in df.columns if "cpt" in c.lower()][0]
        all_cpts.extend(df[cpt_col].dropna().unique())

unique_cpts = sorted(set([c.strip() for c in all_cpts if c.strip()]))

print("Unique CPT codes across cohort:")
for c in unique_cpts:
    print(c)
