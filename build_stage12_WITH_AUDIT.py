# =========================
# Stage2 Anchor – Refined Logic + Audit
# =========================

import pandas as pd
import re
import os

# -------- PATHS --------
BASE_DIR = "/home/apokol/Breast_Restore"
OUTPUT_DIR = f"{BASE_DIR}/outputs"

PATIENT_FILE = f"{BASE_DIR}/HPI11526 Patient Notes.csv"
OP_FILE = f"{BASE_DIR}/HPI11526 OP Notes.csv"
CLINIC_FILE = f"{BASE_DIR}/HPI11526 Clinic Notes.csv"

VALIDATION_FILE = f"{BASE_DIR}/stage2_gold_standard.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- LOAD NOTES --------
df_list = []

for path, note_type in [
    (PATIENT_FILE, "PATIENT"),
    (OP_FILE, "OP NOTE"),
    (CLINIC_FILE, "CLINIC NOTE"),
]:
    if os.path.exists(path):
        tmp = pd.read_csv(path)
        tmp["NOTE_TYPE"] = note_type
        df_list.append(tmp)

notes = pd.concat(df_list, ignore_index=True)

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
FUTURE = r"\b(will|plan|scheduled|planning|consider)\b"

# -------- CLASSIFIER --------
def classify(text):
    operative_context = bool(
        re.search(r"\b(procedure|operation|intraop|incision|anesthesia|drain)\b", text)
    )

    for bucket, patterns in PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                # exclude clear negation
                if re.search(NEGATION + r".{0,40}" + pat, text):
                    continue
                # exclude future/planning unless operative context
                if re.search(FUTURE + r".{0,40}" + pat, text) and not operative_context:
                    continue
                return bucket, pat, operative_context

    return None, None, False


# -------- APPLY --------
results = []
for _, row in notes.iterrows():
    bucket, pattern, operative = classify(row["NOTE_TEXT"])
    results.append(
        {
            "ENCRYPTED_PAT_ID": row["ENCRYPTED_PAT_ID"],
            "EVENT_DATE": row["EVENT_DATE"],
            "NOTE_ID": row["NOTE_ID"],
            "NOTE_TYPE": row["NOTE_TYPE"],
            "DETECTION_BUCKET": bucket,
            "PATTERN_NAME": pattern,
            "IS_OPERATIVE_CONTEXT": int(operative),
            "EVIDENCE_SNIPPET": row["NOTE_TEXT"][:400],
        }
    )

pred_df = pd.DataFrame(results)
pred_df["PRED_STAGE2"] = pred_df["DETECTION_BUCKET"].notnull().astype(int)

# -------- VALIDATION --------
gold = pd.read_csv(VALIDATION_FILE)

merged = gold.merge(
    pred_df[["NOTE_ID", "PRED_STAGE2"]],
    on="NOTE_ID",
    how="left",
)

merged["PRED_STAGE2"] = merged["PRED_STAGE2"].fillna(0)

TP = ((merged["GOLD_STAGE2"] == 1) & (merged["PRED_STAGE2"] == 1)).sum()
FP = ((merged["GOLD_STAGE2"] == 0) & (merged["PRED_STAGE2"] == 1)).sum()
FN = ((merged["GOLD_STAGE2"] == 1) & (merged["PRED_STAGE2"] == 0)).sum()
TN = ((merged["GOLD_STAGE2"] == 0) & (merged["PRED_STAGE2"] == 0)).sum()

precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print("Validation complete.")
print("Stage2 Anchor:")
print(f"TP={TP} FP={FP} FN={FN} TN={TN}")
print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

# -------- AUDIT FILES --------
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
bucket_note_breakdown.to_csv(
    f"{OUTPUT_DIR}/audit_bucket_noteType_breakdown.csv", index=False
)
