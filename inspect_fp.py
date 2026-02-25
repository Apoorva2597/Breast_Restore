
import pandas as pd

path = "_outputs/validation_mismatches.csv"
df = pd.read_csv(path, dtype=str, encoding="latin1")

# normalize just in case there are blanks/spaces
for c in ["PRED_HAS_STAGE2","GOLD_HAS_STAGE2"]:
    df[c] = df[c].fillna("").str.strip()

fp = df[(df["PRED_HAS_STAGE2"]=="1") & (df["GOLD_HAS_STAGE2"]=="0")]
fn = df[(df["PRED_HAS_STAGE2"]=="0") & (df["GOLD_HAS_STAGE2"]=="1")]

print("Total mismatches:", len(df))
print("FP (pred=1, gold=0):", len(fp))
print("FN (pred=0, gold=1):", len(fn))

print("\nFP reason counts (quick heuristic):")
# Example heuristic buckets based on note type + pattern availability
fp_reason = []
for _,r in fp.iterrows():
    note_type = (r.get("STAGE2_NOTE_TYPE","") or "").upper()
    pat = (r.get("STAGE2_MATCH_PATTERN","") or "")
    if "OP" not in note_type and "BRIEF" not in note_type:
        fp_reason.append("non_op_note")
    elif ("permanent" in pat.lower() or "silicone" in pat.lower()) and ("exchange" not in pat.lower() and "remove" not in pat.lower()):
        fp_reason.append("implant_placement_only")
    else:
        fp_reason.append("other")

from collections import Counter
print(Counter(fp_reason))

print("\nSample FP rows (first 10):")
cols = ["MRN","ENCRYPTED_PAT_ID","STAGE2_DATE","STAGE2_NOTE_TYPE","STAGE2_NOTE_ID","STAGE2_MATCH_PATTERN","STAGE2_HITS"]
print(fp[cols].head(10).to_string(index=False))
