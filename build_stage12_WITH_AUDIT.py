#!/usr/bin/env python3
import os
import sys
import pandas as pd

# =========================
# FIXED PATHS (MATCH YOUR SETUP)
# =========================
OUT_DIR   = "/home/apokol/Breast_Restore/_outputs"
GOLD_PATH = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"

EVENT_CANDIDATES = [
    os.path.join(OUT_DIR, "stage_event_level_FINAL_FINAL.csv"),
    os.path.join(OUT_DIR, "stage_event_level.csv"),
    os.path.join(OUT_DIR, "stage2_event_level_FINAL_FINAL.csv"),
    os.path.join(OUT_DIR, "stage2_event_level.csv"),
]

# =========================
# HELPERS
# =========================
def die(msg: str, code: int = 1):
    print(msg)
    sys.exit(code)

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df

def pick(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def as01(x):
    return pd.to_numeric(x.astype(str).str.strip().replace({"True":"1","False":"0"}), errors="coerce").fillna(0).astype(int)

def sanitize(x):
    if pd.isna(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").strip()

# =========================
# INPUT RESOLUTION
# =========================
if not os.path.exists(OUT_DIR):
    die(f"ERROR: Missing OUT_DIR: {OUT_DIR}")
if not os.path.exists(GOLD_PATH):
    die(f"ERROR: Missing GOLD file: {GOLD_PATH}")

EVENT_PATH = first_existing(EVENT_CANDIDATES)
if EVENT_PATH is None:
    die("ERROR: Missing event-level file in _outputs (expected stage_event_level[_FINAL_FINAL].csv or stage2_event_level[_FINAL_FINAL].csv)")

# =========================
# LOAD
# =========================
gold  = norm_cols(pd.read_csv(GOLD_PATH, dtype=str))
ev    = norm_cols(pd.read_csv(EVENT_PATH, dtype=str))

# =========================
# MAP REQUIRED COLUMNS (NO EXTERNAL NOTES FILE NEEDED)
# =========================
# Patient id
g_pid = pick(gold, ["ENCRYPTED_PAT_ID", "PatientID", "PATIENT_ID", "patient_id"])
e_pid = pick(ev,   ["ENCRYPTED_PAT_ID", "PatientID", "PATIENT_ID", "patient_id"])
if g_pid is None: die("ERROR: gold missing ENCRYPTED_PAT_ID/PatientID")
if e_pid is None: die("ERROR: event file missing ENCRYPTED_PAT_ID/PatientID")
gold = gold.rename(columns={g_pid: "ENCRYPTED_PAT_ID"})
ev   = ev.rename(columns={e_pid: "ENCRYPTED_PAT_ID"})

# Gold label
g_lab = pick(gold, ["GOLD_HAS_STAGE2", "GOLD_STAGE2", "gold_has_stage2", "gold_stage2"])
if g_lab is None: die("ERROR: gold missing GOLD_HAS_STAGE2 (or GOLD_STAGE2)")
gold["GOLD_HAS_STAGE2"] = as01(gold[g_lab])

# Pred label (event-level)
e_lab = pick(ev, ["PRED_HAS_STAGE2", "PRED_STAGE2", "pred_has_stage2", "pred_stage2", "HAS_STAGE2_PRED", "STAGE2_PRED"])
if e_lab is None:
    die("ERROR: event file missing predicted label column (expected PRED_HAS_STAGE2 / PRED_STAGE2 / HAS_STAGE2_PRED / STAGE2_PRED)")
ev["PRED_HAS_STAGE2"] = as01(ev[e_lab])

# Note id / type (optional but used)
e_note_id   = pick(ev, ["NOTE_ID", "note_id", "NoteID", "NOTEID"])
e_note_type = pick(ev, ["NOTE_TYPE", "note_type"])
if e_note_id is not None and e_note_id != "NOTE_ID":
    ev = ev.rename(columns={e_note_id: "NOTE_ID"})
if e_note_type is not None and e_note_type != "NOTE_TYPE":
    ev = ev.rename(columns={e_note_type: "NOTE_TYPE"})
if "NOTE_ID" not in ev.columns:   ev["NOTE_ID"] = ""
if "NOTE_TYPE" not in ev.columns: ev["NOTE_TYPE"] = ""

# Event date (best available)
e_date = pick(ev, ["EVENT_DATE", "NOTE_DATE_OF_SERVICE", "OPERATION_DATE", "SURGERY_DATE", "DATE_OF_SERVICE", "DOS"])
if e_date is not None and e_date != "EVENT_DATE":
    ev = ev.rename(columns={e_date: "EVENT_DATE"})
if "EVENT_DATE" in ev.columns:
    ev["EVENT_DATE"] = to_dt(ev["EVENT_DATE"])
else:
    ev["EVENT_DATE"] = pd.NaT

# Bucket / pattern / context / evidence / note text (all optional, but keep for audit)
b_col = pick(ev, ["DETECTION_BUCKET", "bucket", "BUCKET"])
p_col = pick(ev, ["PATTERN_NAME", "pattern_name", "RULE_NAME"])
c_col = pick(ev, ["IS_OPERATIVE_CONTEXT", "is_operative_context", "OPERATIVE_CONTEXT"])
s_col = pick(ev, ["EVIDENCE_SNIPPET", "evidence_snippet", "SNIPPET"])
t_col = pick(ev, ["NOTE_TEXT_DEID", "NOTE_TEXT", "note_text_deid", "note_text"])

if b_col and b_col != "DETECTION_BUCKET": ev = ev.rename(columns={b_col: "DETECTION_BUCKET"})
if p_col and p_col != "PATTERN_NAME":     ev = ev.rename(columns={p_col: "PATTERN_NAME"})
if c_col and c_col != "IS_OPERATIVE_CONTEXT": ev = ev.rename(columns={c_col: "IS_OPERATIVE_CONTEXT"})
if s_col and s_col != "EVIDENCE_SNIPPET": ev = ev.rename(columns={s_col: "EVIDENCE_SNIPPET"})
if t_col and t_col != "NOTE_TEXT":        ev = ev.rename(columns={t_col: "NOTE_TEXT"})

for col, default in [
    ("DETECTION_BUCKET", "UNKNOWN"),
    ("PATTERN_NAME", ""),
    ("IS_OPERATIVE_CONTEXT", ""),
    ("EVIDENCE_SNIPPET", ""),
    ("NOTE_TEXT", ""),
]:
    if col not in ev.columns:
        ev[col] = default

ev["NOTE_TEXT_EXCERPT"] = ev["NOTE_TEXT"].map(sanitize).str.slice(0, 500)
ev["EVIDENCE_SNIPPET"]  = ev["EVIDENCE_SNIPPET"].map(sanitize)

# =========================
# PATIENT-LEVEL CONFUSION (ANY predicted-positive event => patient positive)
# =========================
pred_pt = ev.groupby("ENCRYPTED_PAT_ID", as_index=False)["PRED_HAS_STAGE2"].max()
gold_pt = gold.groupby("ENCRYPTED_PAT_ID", as_index=False)["GOLD_HAS_STAGE2"].max()

pt = pred_pt.merge(gold_pt, on="ENCRYPTED_PAT_ID", how="outer").fillna(0)
pt["PRED_HAS_STAGE2"] = pt["PRED_HAS_STAGE2"].astype(int)
pt["GOLD_HAS_STAGE2"] = pt["GOLD_HAS_STAGE2"].astype(int)

TP = int(((pt.PRED_HAS_STAGE2 == 1) & (pt.GOLD_HAS_STAGE2 == 1)).sum())
FP = int(((pt.PRED_HAS_STAGE2 == 1) & (pt.GOLD_HAS_STAGE2 == 0)).sum())
FN = int(((pt.PRED_HAS_STAGE2 == 0) & (pt.GOLD_HAS_STAGE2 == 1)).sum())
TN = int(((pt.PRED_HAS_STAGE2 == 0) & (pt.GOLD_HAS_STAGE2 == 0)).sum())

precision = TP / (TP + FP) if (TP + FP) else 0.0
recall    = TP / (TP + FN) if (TP + FN) else 0.0
f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

print("Validation complete.")
print("Stage2 Anchor:")
print(f"  TP={TP} FP={FP} FN={FN} TN={TN}")
print(f"  Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

# =========================
# AUDIT OUTPUTS (FP EVENTS)
# =========================
# Merge gold label onto events (patient-level gold)
ev2 = ev.merge(gold_pt, on="ENCRYPTED_PAT_ID", how="left")
ev2["GOLD_HAS_STAGE2"] = ev2["GOLD_HAS_STAGE2"].fillna(0).astype(int)

fp_events = ev2[(ev2["PRED_HAS_STAGE2"] == 1) & (ev2["GOLD_HAS_STAGE2"] == 0)].copy()

# 1) bucket summary (all predicted-positive events)
bucket_summary = (
    ev2[ev2["PRED_HAS_STAGE2"] == 1]
    .groupby("DETECTION_BUCKET", as_index=False)
    .size()
    .rename(columns={"size": "Total_predictions"})
    .sort_values("Total_predictions", ascending=False)
)
bucket_summary.to_csv(os.path.join(OUT_DIR, "audit_bucket_summary.csv"), index=False)

# 2) FP by bucket
fp_by_bucket = (
    fp_events.groupby("DETECTION_BUCKET", as_index=False)
    .size()
    .rename(columns={"size": "FP_count"})
    .sort_values("FP_count", ascending=False)
)
fp_by_bucket.to_csv(os.path.join(OUT_DIR, "audit_fp_by_bucket.csv"), index=False)

# 3) bucket x note_type breakdown for FP
bucket_noteType = (
    fp_events.groupby(["DETECTION_BUCKET", "NOTE_TYPE"], as_index=False)
    .size()
    .rename(columns={"size": "Count"})
    .sort_values(["DETECTION_BUCKET", "Count"], ascending=[True, False])
)
bucket_noteType.to_csv(os.path.join(OUT_DIR, "audit_bucket_noteType_breakdown.csv"), index=False)

# 4) FP event samples per bucket
SAMPLES_PER_BUCKET = 25
chunks = []
for b, g in fp_events.groupby("DETECTION_BUCKET", dropna=False):
    gg = g.copy()
    gg["IS_OPERATIVE_CONTEXT"] = gg["IS_OPERATIVE_CONTEXT"].astype(str)
    gg = gg.sort_values(["IS_OPERATIVE_CONTEXT", "EVENT_DATE"], ascending=[False, False])
    chunks.append(gg.head(SAMPLES_PER_BUCKET))
fp_sample = pd.concat(chunks, ignore_index=True) if chunks else fp_events.head(0)

out_cols = [
    "ENCRYPTED_PAT_ID","EVENT_DATE","NOTE_ID","NOTE_TYPE","DETECTION_BUCKET",
    "PATTERN_NAME","IS_OPERATIVE_CONTEXT","EVIDENCE_SNIPPET","NOTE_TEXT_EXCERPT"
]
for c in out_cols:
    if c not in fp_sample.columns:
        fp_sample[c] = ""
fp_sample[out_cols].to_csv(os.path.join(OUT_DIR, "audit_fp_events_sample.csv"), index=False)

# 5) FN patients (patient-level)
fn_pts = pt[(pt["PRED_HAS_STAGE2"] == 0) & (pt["GOLD_HAS_STAGE2"] == 1)][["ENCRYPTED_PAT_ID"]].copy()
fn_pts.to_csv(os.path.join(OUT_DIR, "audit_fn_patients.csv"), index=False)
