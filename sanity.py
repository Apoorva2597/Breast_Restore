
import pandas as pd

path = "cohort_all_patient_level_final.csv"  # <-- change if your exact name differs
df = pd.read_csv(path, dtype=str, engine="python")
pid_col = None
for c in df.columns:
    if c.strip().upper() in ("ENCRYPTED_PAT_ID","ENCRYPTED_PATIENT_ID","PAT_ID","PATIENT_ID"):
        pid_col = c
        break
if pid_col is None:
    # heuristic fallback
    for c in df.columns:
        lc = c.lower()
        if ("pat" in lc or "patient" in lc) and "id" in lc:
            pid_col = c
            break

if pid_col is None:
    raise SystemExit("Could not detect patient id column. Columns:\n" + "\n".join(df.columns))

n_rows = len(df)
n_unique = df[pid_col].dropna().astype(str).str.strip()
n_unique = (n_unique[n_unique != ""]).nunique()

print("FILE:", path)
print("PID_COL:", pid_col)
print("ROWS:", n_rows)
print("UNIQUE_PATIENTS:", n_unique)
