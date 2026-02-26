# =========================
# Stage2 Anchor – Refined Logic + Audit
# (FIX: auto-map NOTE_DATE/NOTEID/TEXT column name variants)
# =========================

import pandas as pd
import re
import os
import glob

BASE_DIR = "/home/apokol/Breast_Restore"
OUTPUT_DIR = f"{BASE_DIR}/outputs"
VALIDATION_FILE = f"{BASE_DIR}/stage2_gold_standard.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resolve_one(globs_list):
    hits = []
    for g in globs_list:
        hits.extend(glob.glob(g))
    hits = sorted(set(hits))
    return hits[0] if hits else None

PATIENT_FILE = resolve_one([f"{BASE_DIR}/*Patient*Notes*.csv", f"{BASE_DIR}/*patient*notes*.csv"])
OP_FILE      = resolve_one([f"{BASE_DIR}/*OP*Notes*.csv", f"{BASE_DIR}/*Operative*Notes*.csv", f"{BASE_DIR}/*op*notes*.csv"])
CLINIC_FILE  = resolve_one([f"{BASE_DIR}/*Clinic*Notes*.csv", f"{BASE_DIR}/*clinic*notes*.csv"])

df_list = []
for path, note_type in [(PATIENT_FILE, "PATIENT"), (OP_FILE, "OP NOTE"), (CLINIC_FILE, "CLINIC NOTE")]:
    if path and os.path.exists(path):
        tmp = pd.read_csv(path)
        tmp["NOTE_TYPE"] = note_type
        df_list.append(tmp)

if not df_list:
    raise SystemExit("ERROR: No note files found in /home/apokol/Breast_Restore")

notes = pd.concat(df_list, ignore_index=True)

# -------- COLUMN NORMALIZATION / MAPPING --------
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

notes.columns = [c.strip() for c in notes.columns]
norm_map = {norm(c): c for c in notes.columns}

CANDIDATES = {
    "ENCRYPTED_PAT_ID": ["encryptedpatid", "encpatid", "patientid", "patid", "encrypted_patient_id", "encrypted_pat_id"],
    "EVENT_DATE":       ["eventdate", "notedate", "servicedate", "documentdate", "date", "encounterdate"],
    "NOTE_ID":          ["noteid", "note_id", "noteidentifier", "noteidentifierid", "note_key", "notekey", "documentid", "docid"],
    "NOTE_TEXT":        ["notetext", "note_text", "text", "note", "documenttext", "content", "notecontent", "body"],
}

def pick_col(std_name):
    for cand in CANDIDATES[std_name]:
        if cand in norm_map:
            return norm_map[cand]
    return None

rename_dict = {}
for std in ["ENCRYPTED_PAT_ID", "EVENT_DATE", "NOTE_ID", "NOTE_TEXT"]:
    src = pick_col(std)
    if src is not None and src != std:
        rename_dict[src] = std

notes = notes.rename(columns=rename_dict)

missing = [c for c in ["ENCRYPTED_PAT_ID", "EVENT_DATE", "NOTE_ID", "NOTE_TEXT"] if c not in notes.columns]
if missing:
    cols_preview = ", ".join(list(notes.columns)[:80])
    raise SystemExit(f"ERROR: Missing columns after mapping: {missing}\nFOUND COLUMNS (first 80): {cols_preview}")

notes["NOTE_TEXT"] = notes["NOTE_TEXT"].astype(str).str.lower()

# -------- PATTERNS --------
PATTERNS = {
    "EXCHANGE_TIGHT": [
        r"\bimplant exchange\b",
        r"\bexchange of (the )?implant\b",
        r"\breplace(d|ment)? (with )?(a )?(new )?implant\b",
    ],
    "EXPANDER_TO_IMPLANT": [
        r"\bexchange (of )?tissue expander\b.*\bimplant\b",
        r"\bexpander (was )?removed\b.*\bimplant (was )?placed\b",
        r"\bexpander to implant\b",
    ],
    "OPONLY_IMPLANT_PLACEMENT_RECON_TIGHT": [
        r"\bplacement of (a )?(permanent )?implant\b",
        r"\bimplant (was )?placed\b",
    ],
}

NEGATION = r"\b(no|not|without|denies|deferred|declined)\b"
FUTURE   = r"\b(will|plan|scheduled|planning|consider)\b"

def classify(text):
    operative_context = bool(re.search(r"\b(procedure|operation|intraop|incision|anesthesia|drain)\b", text))
    for bucket, patterns in PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                if re.search(NEGATION + r".{0,40}" + pat, text):
                    continue
                if re.search(FUTURE + r".{0,40}" + pat, text) and not operative_context:
                    continue
                return bucket, pat, operative_context
    return None, None, False

rows = []
for _, r in notes.iterrows():
    bucket, pattern, operative = classify(r["NOTE_TEXT"])
    rows.append({
        "ENCRYPTED_PAT_ID": r["ENCRYPTED_PAT_ID"],
        "EVENT_DATE": r["EVENT_DATE"],
        "NOTE_ID": r["NOTE_ID"],
        "NOTE_TYPE": r["NOTE_TYPE"],
        "DETECTION_BUCKET": bucket,
        "PATTERN_NAME": pattern,
        "IS_OPERATIVE_CONTEXT": int(operative),
        "EVIDENCE_SNIPPET": r["NOTE_TEXT"][:400],
    })

pred_df = pd.DataFrame(rows)
pred_df["PRED_STAGE2"] = pred_df["DETECTION_BUCKET"].notnull().astype(int)

# -------- VALIDATION --------
gold = pd.read_csv(VALIDATION_FILE)
if "NOTE_ID" not in gold.columns or "GOLD_STAGE2" not in gold.columns:
    raise SystemExit("ERROR: stage2_gold_standard.csv must contain NOTE_ID and GOLD_STAGE2.")

merged = gold.merge(pred_df[["NOTE_ID", "PRED_STAGE2"]], on="NOTE_ID", how="left")
merged["PRED_STAGE2"] = merged["PRED_STAGE2"].fillna(0).astype(int)

TP = ((merged["GOLD_STAGE2"] == 1) & (merged["PRED_STAGE2"] == 1)).sum()
FP = ((merged["GOLD_STAGE2"] == 0) & (merged["PRED_STAGE2"] == 1)).sum()
FN = ((merged["GOLD_STAGE2"] == 1) & (merged["PRED_STAGE2"] == 0)).sum()
TN = ((merged["GOLD_STAGE2"] == 0) & (merged["PRED_STAGE2"] == 0)).sum()

precision = TP / (TP + FP) if (TP + FP) else 0
recall    = TP / (TP + FN) if (TP + FN) else 0
f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print("Validation complete.")
print("Stage2 Anchor:")
print(f"TP={TP} FP={FP} FN={FN} TN={TN}")
print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

# -------- AUDIT OUTPUTS --------
pred_df.to_csv(f"{OUTPUT_DIR}/audit_fp_events_sample.csv", index=False)

fp = merged[(merged["GOLD_STAGE2"] == 0) & (merged["PRED_STAGE2"] == 1)]
fn = merged[(merged["GOLD_STAGE2"] == 1) & (merged["PRED_STAGE2"] == 0)]
fp.to_csv(f"{OUTPUT_DIR}/audit_fp.csv", index=False)
fn.to_csv(f"{OUTPUT_DIR}/audit_fn.csv", index=False)

bucket_summary = (
    pred_df[pred_df["PRED_STAGE2"] == 1]
    .groupby("DETECTION_BUCKET")
    .size()
    .reset_index(name="Total_predictions")
)
bucket_summary.to_csv(f"{OUTPUT_DIR}/audit_bucket_summary.csv", index=False)

bucket_note_breakdown = (
    pred_df[pred_df["PRED_STAGE2"] == 1]
    .groupby(["DETECTION_BUCKET", "NOTE_TYPE"])
    .size()
    .reset_index(name="count")
)
bucket_note_breakdown.to_csv(f"{OUTPUT_DIR}/audit_bucket_noteType_breakdown.csv", index=False)
