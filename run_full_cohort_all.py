# run_full_cohort_all.py
# Python 3.6.8 compatible
#
# Goal:
#   Run your existing pipeline on ALL patients (1410 spine),
#   then merge outputs into ONE patient-level cohort.
#
# Assumptions (based on your folder listing):
#   - patient_level_master_all.csv exists (1410 patients; spine + demo + recon)
#   - your pipeline scripts write patient-level outputs (stage1/stage2 CSVs etc.)
#   - patient_level_fields.csv is produced by make_patient_level.py (Phase2->patient wide)
#   - gold_cleaned_for_cedar.csv exists (259 patients; gold labels)
#   - gold_variables.csv exists (optional; used to align/ordering to gold variable list)
#
# You may edit the SCRIPT_RUN_ORDER list to match the exact scripts you want to run.

from __future__ import print_function
import os
import sys
import subprocess
import pandas as pd


# -------------------------
# CONFIG (edit if needed)
# -------------------------

SPINE_CSV = "patient_level_master_all.csv"     # 1410 patient spine you already produced
GOLD_CSV  = "gold_cleaned_for_cedar.csv"       # gold
GOLD_VARS_CSV = "gold_variables.csv"           # optional (if present; used for column order)

OUT_FINAL = "cohort_all_patient_level_final.csv"
OUT_FINAL_GOLD_ORDER = "cohort_all_patient_level_final_gold_order.csv"

# If your scripts require being run from this folder, keep as "."
WORKDIR = "."

# Run order: put your “driver” scripts here (preferred) OR the key stage scripts.
# Keep it simple: only include scripts that are correct/current.
# Comment out anything you don’t want to run today.
SCRIPT_RUN_ORDER = [
    # If you have a single driver that runs phase1/stage1/stage2 already, use that.
    # "run_phase1.py",
    # "run_phase2.py",  # you said older; keep off unless needed

    # Most projects: Stage1 + Stage2 drivers / finalizers:
    "stage1_abstract_complications.py",
    "stage1_make_outcomes_from_complications.py",
    "stage2_make_master.py",
    "stage2_make_outcomes_full_ab.py",
    "finalize_stage2_ab.py",

    # If you are using Phase2 note-level extraction -> patient wide:
    "make_patient_level.py",  # produces patient_level_fields.csv from all_phase2_final.csv (if present)

    # Optional validation scripts (only run if you want)
    # "validate_against_gold.py",
]

# Patient ID column name (spine uses patient_id)
PATIENT_ID_COL = "patient_id"


# -------------------------
# Helpers
# -------------------------

def _exists(path):
    return path is not None and os.path.exists(path)

def run_script(pyfile):
    path = os.path.join(WORKDIR, pyfile)
    if not _exists(path):
        print("SKIP (not found):", pyfile)
        return True

    print("\n==================================================")
    print("RUN:", pyfile)
    print("==================================================")
    cmd = [sys.executable, pyfile]
    p = subprocess.call(cmd, cwd=WORKDIR)
    if p != 0:
        print("ERROR: script failed with exit code", p, "->", pyfile)
        return False
    return True

def read_csv_safe(path):
    # CEDAR exports often need latin1 fallback
    try:
        return pd.read_csv(path, dtype=object, encoding="utf-8", engine="python")
    except Exception:
        return pd.read_csv(path, dtype=object, encoding="latin1", engine="python")

def coalesce_cols(df, candidates, new_name):
    """
    If any candidate column exists, coalesce into new_name (leftmost non-empty).
    """
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return df
    if new_name not in df.columns:
        df[new_name] = ""
    for c in cols:
        s = df[c].fillna("").astype(str)
        mask = (df[new_name].fillna("").astype(str).str.strip() == "") & (s.str.strip() != "")
        df.loc[mask, new_name] = s[mask]
    return df

def ensure_patient_id(df):
    # try common fallbacks
    if PATIENT_ID_COL in df.columns:
        return df
    for c in ["ENCRYPTED_PAT_ID", "encrypted_pat_id", "PAT_ID", "pat_id", "patientid", "PatientID"]:
        if c in df.columns:
            df = df.rename(columns={c: PATIENT_ID_COL})
            return df
    return df

def left_merge(base, add, label):
    if add is None or add.empty:
        print("NOTE: merge skipped (empty):", label)
        return base
    add = ensure_patient_id(add)
    if PATIENT_ID_COL not in add.columns:
        print("NOTE: merge skipped (no patient_id):", label)
        return base

    # avoid duplicate columns explosion
    dup = [c for c in add.columns if c in base.columns and c != PATIENT_ID_COL]
    if dup:
        add = add.drop(columns=dup)

    out = base.merge(add, on=PATIENT_ID_COL, how="left")
    print("Merged:", label, "-> columns now:", len(out.columns))
    return out

def load_optional(path):
    if _exists(path):
        return read_csv_safe(path)
    return None


# -------------------------
# Main
# -------------------------

def main():
    print("\n=== RUN FULL COHORT (ALL PATIENTS) ===")

    if not _exists(SPINE_CSV):
        raise RuntimeError("Missing spine file: {}. You already generated this; confirm location.".format(SPINE_CSV))

    spine = read_csv_safe(SPINE_CSV)
    spine = ensure_patient_id(spine)

    if PATIENT_ID_COL not in spine.columns:
        raise RuntimeError("Spine missing patient_id. Found columns: {}".format(list(spine.columns)[:30]))

    # Normalize patient_id
    spine[PATIENT_ID_COL] = spine[PATIENT_ID_COL].fillna("").astype(str).str.strip()
    spine = spine[spine[PATIENT_ID_COL] != ""].copy()

    print("Spine rows (patients):", spine.shape[0])
    print("Spine cols:", spine.shape[1])

    # 1) Run pipeline scripts (your logic lives here)
    for py in SCRIPT_RUN_ORDER:
        ok = run_script(py)
        if not ok:
            raise RuntimeError("Pipeline stopped because a script failed: {}".format(py))

    # 2) Collect expected outputs (adjust these to match what your scripts write)
    #    Based on your directory listing, these exist commonly:
    candidates = [
        ("patient_level_fields.csv", "Phase2 patient-wide fields"),
        ("stage1_complications_patient_level.csv", "Stage1 complications patient-level"),
        ("stage1_outcomes_patient_level.csv", "Stage1 outcomes patient-level"),
        ("stage2_final_ab_patient_level.csv", "Stage2 AB final patient-level"),
        ("stage2_ab_outcomes_patient_level.csv", "Stage2 AB outcomes patient-level"),
        ("patient_recon_structured.csv", "Structured recon from op encounters (if you still use it)"),
        ("patient_demographics.csv", "Demographics from clinic encounters (if you still use it)"),
    ]

    merged = spine.copy()

    for path, label in candidates:
        df_add = load_optional(path)
        if df_add is None:
            continue
        df_add = ensure_patient_id(df_add)
        merged = left_merge(merged, df_add, label)

    # 3) Light column harmonization (optional but helpful)
    #    Example: if some outputs use MRN instead of patient_id — keep them separate.
    #    We DO NOT attempt to map MRN <-> patient_id here (PHI risk + can be wrong).
    #
    #    If you have Race/Ethnicity variants:
    merged = coalesce_cols(merged, ["Race", "RACE"], "Race")
    merged = coalesce_cols(merged, ["Ethnicity", "ETHNICITY"], "Ethnicity")

    # 4) Write the raw merged cohort
    merged.to_csv(OUT_FINAL, index=False, encoding="utf-8")
    print("\nWrote:", OUT_FINAL)
    print("Final rows:", merged.shape[0])
    print("Final cols:", merged.shape[1])

    # 5) (Optional) Create a gold-ordered version if gold_variables.csv exists
    if _exists(GOLD_VARS_CSV):
        gv = read_csv_safe(GOLD_VARS_CSV)

        # Try to find a column that contains the gold variable names
        gold_var_col = None
        for c in gv.columns:
            cl = str(c).strip().lower()
            if cl in ["variable", "var", "field", "gold_variable", "name"]:
                gold_var_col = c
                break
        if gold_var_col is None:
            # fallback: use first column
            gold_var_col = gv.columns[0]

        gold_vars = (
            gv[gold_var_col]
            .fillna("")
            .astype(str)
            .map(lambda x: x.strip())
        )
        gold_vars = [v for v in gold_vars.tolist() if v != ""]

        # Ensure patient_id is first
        ordered = [PATIENT_ID_COL]

        # Add gold vars that exist in merged
        for v in gold_vars:
            if v in merged.columns and v not in ordered:
                ordered.append(v)

        # Add remaining columns at end
        for c in merged.columns:
            if c not in ordered:
                ordered.append(c)

        merged2 = merged.loc[:, ordered].copy()
        merged2.to_csv(OUT_FINAL_GOLD_ORDER, index=False, encoding="utf-8")
        print("Wrote:", OUT_FINAL_GOLD_ORDER, "(gold-friendly column order)")
    else:
        print("\nNOTE: gold_variables.csv not found; skipping gold-ordered output.")

    # 6) Quick overlap summary with GOLD (if present)
    if _exists(GOLD_CSV):
        gold = read_csv_safe(GOLD_CSV)

        # GOLD uses MRN; we do NOT merge by MRN here.
        # We only report counts, and you can later validate using your encounter-bridge method.
        print("\n--- Quick counts ---")
        print("GOLD rows:", gold.shape[0])
        print("ALL cohort rows:", merged.shape[0])
        print("Done.\n")


if __name__ == "__main__":
    main()
