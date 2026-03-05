#!/usr/bin/env python3
# build_master_rule_FINAL.py
#
# RULE-BASED ONLY master abstraction builder (no BART).
# - Loads your existing gold/master CSV (gold_cleaned_for_cedar.csv)
# - Loads ORIGINAL HPI11526 note CSVs (Clinic/Inpatient/Operation Notes)
# - Reconstructs full note text by NOTE_ID + LINE ordering
# - Lightweight sectionizer (captures FAMILY HISTORY / ALLERGIES / ROS etc.)
# - Runs rule-based extractors from ./extractors
# - Writes:
#     1) /home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL.csv
#     2) /home/apokol/Breast_Restore/_outputs/rule_hit_evidence_FINAL.csv
#     3) /home/apokol/Breast_Restore/_outputs/rule_validation_summary_FINAL.txt  (when gold has labels)

import os
import re
from glob import glob
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

# -----------------------
# CONFIG (NO USER INPUTS)
# -----------------------
BASE_DIR = "/home/apokol/Breast_Restore"

GOLD_PATH = f"{BASE_DIR}/gold_cleaned_for_cedar.csv"

NOTE_GLOBS = [
    f"{BASE_DIR}/**/HPI11526*Clinic Notes.csv",
    f"{BASE_DIR}/**/HPI11526*Inpatient Notes.csv",
    f"{BASE_DIR}/**/HPI11526*Operation Notes.csv",
    f"{BASE_DIR}/**/HPI11526*clinic notes.csv",
    f"{BASE_DIR}/**/HPI11526*inpatient notes.csv",
    f"{BASE_DIR}/**/HPI11526*operation notes.csv",
]

OUTPUT_MASTER = f"{BASE_DIR}/_outputs/master_abstraction_rule_FINAL.csv"
OUTPUT_EVID   = f"{BASE_DIR}/_outputs/rule_hit_evidence_FINAL.csv"
OUTPUT_SUMMARY = f"{BASE_DIR}/_outputs/rule_validation_summary_FINAL.txt"

MERGE_KEY = "MRN"

# -----------------------
# Imports from your repo
# -----------------------
# These must exist in your repo as you showed.
from models import SectionedNote, Candidate  # noqa: E402

from extractors.age import extract_age  # noqa: E402
from extractors.bmi import extract_bmi  # noqa: E402
from extractors.smoking import extract_smoking  # noqa: E402
from extractors.comorbidities import extract_comorbidities  # noqa: E402
from extractors.pbs import extract_pbs  # noqa: E402
from extractors.mastectomy import extract_mastectomy  # noqa: E402
from extractors.cancer_treatment import extract_cancer_treatment  # noqa: E402


# -----------------------
# Robust CSV read
# -----------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        # older pandas fallback
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df: pd.DataFrame) -> pd.DataFrame:
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MERGE_KEY:
                df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError(f"MRN column not found. Columns seen: {list(df.columns)[:40]}")
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df


# -----------------------
# Note schema helpers
# -----------------------
def pick_col(df: pd.DataFrame, options: List[str], required: bool = True) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"Required column missing. Tried={options}. Seen={list(df.columns)[:50]}")
    return None


def to_int_safe(x: Any) -> Optional[int]:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


# -----------------------
# Lightweight sectionizer
# -----------------------
# Matches headers like:
#   FAMILY HISTORY:
#   REVIEW OF SYSTEMS:
#   HPI:
HEADER_RX = re.compile(r"^\s*([A-Z][A-Z0-9 /&\-]{2,60})\s*:\s*$")

def sectionize(text: str) -> Dict[str, str]:
    """
    Simple, reliable sectionizer:
      - Splits on ALL-CAPS headers ending with ':'
      - Keeps a default "FULL" if no headers found
    """
    if not text:
        return {"FULL": ""}

    lines = text.splitlines()
    sections: Dict[str, List[str]] = {}
    current = "FULL"
    sections[current] = []

    for line in lines:
        m = HEADER_RX.match(line)
        if m:
            hdr = m.group(1).strip().upper()
            current = hdr
            if current not in sections:
                sections[current] = []
            continue
        sections[current].append(line)

    # join
    out = {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}
    if not out:
        return {"FULL": text}
    return out


def build_sectioned_note(
    note_text: str,
    note_type: str,
    note_id: str,
    note_date: str
) -> SectionedNote:
    sections = sectionize(note_text)
    # Your SectionedNote likely expects these attributes; this matches your extractor usage.
    return SectionedNote(
        sections=sections,
        note_type=note_type or "",
        note_id=note_id or "",
        note_date=note_date or ""
    )


# -----------------------
# Candidate aggregation logic
# -----------------------
def cand_score(c: Candidate) -> float:
    """
    Higher is better: prioritize confidence, then op notes, then newer dates (roughly).
    """
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    nt = str(getattr(c, "note_type", "") or "").lower()
    op_bonus = 0.05 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    # date bonus: prefer having a date; not strict parsing
    date_bonus = 0.01 if (getattr(c, "note_date", "") or "").strip() else 0.0
    return conf + op_bonus + date_bonus


def choose_best(existing: Optional[Candidate], new: Candidate) -> Candidate:
    if existing is None:
        return new
    return new if cand_score(new) > cand_score(existing) else existing


def merge_boolean(existing: Optional[Candidate], new: Candidate) -> Candidate:
    """
    For boolean fields:
      - any True wins over False
      - otherwise use best score
    """
    if existing is None:
        return new
    try:
        exv = bool(existing.value)
        nwv = bool(new.value)
    except Exception:
        return choose_best(existing, new)
    if nwv and not exv:
        return new
    if exv and not nwv:
        return existing
    return choose_best(existing, new)


# -----------------------
# Column mapping (gold/master may vary)
# -----------------------
# We try these in order; if none exist, we CREATE a clean column name.
COL_ALIASES: Dict[str, List[str]] = {
    "Age": ["4. Age", "Age", "Age_DOS", "Age_AT_Surgery"],
    "BMI": ["5. BMI", "BMI"],
    "SmokingStatus": ["7. SmokingStatus", "SmokingStatus"],
    "Diabetes": ["8. Diabetes", "Diabetes", "DiabetesMellitus"],
    "Hypertension": ["10. Hypertension", "Hypertension", "HTN"],
    "CardiacDisease": ["11. CardiacDisease", "CardiacDisease"],
    "VenousThromboembolism": ["12. VenousThromboembolism", "VenousThromboembolism", "VTE"],
    "Steroid": ["13. Steroid", "Steroid", "SteroidUse"],
    "PastBreastSurgery": ["14. PastBreastSurgery", "PastBreastSurgery"],
    "PBS_Lumpectomy": ["15. PBS_Lumpectomy", "PBS_Lumpectomy"],
    "PBS_Breast Reduction": ["16. PBS_Breast Reduction", "PBS_Breast Reduction", "PBS_Reduction"],
    "PBS_Mastopexy": ["17. PBS_Mastopexy", "PBS_Mastopexy"],
    "PBS_Augmentation": ["18. PBS_Augmentation", "PBS_Augmentation"],
    "PBS_Other": ["19. PBS_Other", "PBS_Other"],
    "Mastectomy_Laterality": ["20. Mastectomy_Laterality", "Mastectomy_Laterality"],
    "Radiation": ["24. Radiation", "Radiation"],
    "Chemo": ["27. Chemo", "Chemo", "Chemotherapy"],
}

# Candidate.field -> target logical name in this script
FIELD_MAP: Dict[str, str] = {
    "Age": "Age",
    "Age_DOS": "Age",
    "BMI": "BMI",
    "SmokingStatus": "SmokingStatus",

    "Diabetes": "Diabetes",
    "DiabetesMellitus": "Diabetes",
    "Hypertension": "Hypertension",
    "CardiacDisease": "CardiacDisease",
    "VenousThromboembolism": "VenousThromboembolism",
    "VTE": "VenousThromboembolism",
    "Steroid": "Steroid",
    "SteroidUse": "Steroid",

    "PastBreastSurgery": "PastBreastSurgery",
    "PBS_Lumpectomy": "PBS_Lumpectomy",
    "PBS_Breast Reduction": "PBS_Breast Reduction",
    "PBS_Mastopexy": "PBS_Mastopexy",
    "PBS_Augmentation": "PBS_Augmentation",
    "PBS_Other": "PBS_Other",

    "Mastectomy_Laterality": "Mastectomy_Laterality",
    "Radiation": "Radiation",
    "Chemo": "Chemo",
}


BOOLEAN_FIELDS = {
    "Diabetes", "Hypertension", "CardiacDisease", "VenousThromboembolism", "Steroid",
    "PastBreastSurgery", "PBS_Lumpectomy", "PBS_Breast Reduction", "PBS_Mastopexy",
    "PBS_Augmentation", "PBS_Other", "Radiation", "Chemo"
}


def resolve_output_col(df: pd.DataFrame, logical: str) -> str:
    """
    Find best matching existing column; else create a clean one.
    """
    for c in COL_ALIASES.get(logical, []):
        if c in df.columns:
            return c
    # create if none found
    clean = logical
    if clean not in df.columns:
        df[clean] = pd.NA
    return clean


# -----------------------
# Validation helpers
# -----------------------
def to_binary_series(s: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return pd.NA
        t = str(x).strip().lower()
        if t in {"1", "true", "yes", "y"}:
            return 1
        if t in {"0", "false", "no", "n"}:
            return 0
        # sometimes booleans stored as python True/False string
        if t == "nan":
            return pd.NA
        return pd.NA
    return s.apply(conv)


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> Tuple[int, int, int, int]:
    """
    Returns TP, FP, FN, TN using rows where both are non-missing.
    """
    mask = (~y_true.isna()) & (~y_pred.isna())
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)

    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    return tp, fp, fn, tn


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
    spec = safe_div(tn, tn + fp)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "specificity": spec}


# -----------------------
# MAIN
# -----------------------
def main():
    print("Loading gold/master...")
    gold = clean_cols(read_csv_robust(GOLD_PATH))
    gold = normalize_mrn(gold)

    # Ensure output columns exist (or are created cleanly)
    for logical in COL_ALIASES.keys():
        resolve_output_col(gold, logical)

    print("Finding note files...")
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))
    if not note_files:
        raise FileNotFoundError("No HPI11526 * Notes.csv files found via NOTE_GLOBS.")

    print(f"Found {len(note_files)} note files.")
    print("Loading and reconstructing notes by NOTE_ID...")

    all_notes_rows = []
    for fp in note_files:
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)

        note_text_col = pick_col(df, ["NOTE_TEXT", "NOTE TEXT", "NOTE_TEXT_FULL", "TEXT", "NOTE"])
        note_id_col   = pick_col(df, ["NOTE_ID", "NOTE ID"])
        line_col      = pick_col(df, ["LINE"], required=False)
        note_type_col = pick_col(df, ["NOTE_TYPE", "NOTE TYPE"], required=False)
        date_col      = pick_col(df, ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE", "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"], required=False)

        df[note_text_col] = df[note_text_col].fillna("").astype(str)
        df[note_id_col] = df[note_id_col].fillna("").astype(str)
        if line_col:
            df[line_col] = df[line_col].fillna("").astype(str)
        if note_type_col:
            df[note_type_col] = df[note_type_col].fillna("").astype(str)
        if date_col:
            df[date_col] = df[date_col].fillna("").astype(str)

        df["_SOURCE_FILE_"] = os.path.basename(fp)

        all_notes_rows.append(df[[MERGE_KEY, note_id_col, note_text_col] +
                                 ([line_col] if line_col else []) +
                                 ([note_type_col] if note_type_col else []) +
                                 ([date_col] if date_col else []) +
                                 ["_SOURCE_FILE_"]].rename(columns={
                                     note_id_col: "NOTE_ID",
                                     note_text_col: "NOTE_TEXT",
                                     (line_col if line_col else "NOTE_ID"): (line_col if line_col else "NOTE_ID"),
                                     (note_type_col if note_type_col else "NOTE_ID"): (note_type_col if note_type_col else "NOTE_ID"),
                                     (date_col if date_col else "NOTE_ID"): (date_col if date_col else "NOTE_ID"),
                                 }))

    notes_raw = pd.concat(all_notes_rows, ignore_index=True)

    # Normalize helper cols
    if "LINE" not in notes_raw.columns:
        notes_raw["LINE"] = ""
    if "NOTE_TYPE" not in notes_raw.columns:
        notes_raw["NOTE_TYPE"] = ""
    if "NOTE_DATE_OF_SERVICE" not in notes_raw.columns:
        # pick any likely date col we may have renamed weirdly above
        date_candidates = [c for c in notes_raw.columns if "DATE" in c.upper() or "TIME" in c.upper()]
        notes_raw["NOTE_DATE_OF_SERVICE"] = notes_raw[date_candidates[0]] if date_candidates else ""

    # Reconstruct full note text per NOTE_ID
    def join_note(group: pd.DataFrame) -> str:
        if "LINE" in group.columns:
            # sort by LINE if numeric-ish
            tmp = group.copy()
            tmp["_LINE_NUM_"] = tmp["LINE"].apply(to_int_safe)
            tmp = tmp.sort_values(by=["_LINE_NUM_"], na_position="last")
            return "\n".join(tmp["NOTE_TEXT"].tolist()).strip()
        return "\n".join(group["NOTE_TEXT"].tolist()).strip()

    grouped = notes_raw.groupby([MERGE_KEY, "NOTE_ID"], dropna=False)

    reconstructed = []
    for (mrn, nid), g in grouped:
        if not str(nid).strip():
            continue
        full_text = join_note(g)
        if not full_text:
            continue
        note_type = ""
        if "NOTE_TYPE" in g.columns and g["NOTE_TYPE"].astype(str).str.strip().any():
            note_type = g["NOTE_TYPE"].astype(str).iloc[0]
        else:
            note_type = g["_SOURCE_FILE_"].astype(str).iloc[0]
        note_date = ""
        if "NOTE_DATE_OF_SERVICE" in g.columns and g["NOTE_DATE_OF_SERVICE"].astype(str).str.strip().any():
            note_date = g["NOTE_DATE_OF_SERVICE"].astype(str).iloc[0]

        reconstructed.append({
            MERGE_KEY: mrn,
            "NOTE_ID": str(nid),
            "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date,
            "SOURCE_FILE": g["_SOURCE_FILE_"].astype(str).iloc[0],
            "NOTE_TEXT": full_text
        })

    notes_df = pd.DataFrame(reconstructed)
    print(f"Reconstructed {len(notes_df)} notes.")

    # Run extractors and aggregate best per MRN per field
    print("Running rule-based extractors...")
    best_by_mrn: Dict[str, Dict[str, Candidate]] = {}
    evidence_rows: List[Dict[str, Any]] = []

    extractor_fns = [
        extract_age,
        extract_bmi,
        extract_smoking,
        extract_comorbidities,
        extract_pbs,
        extract_mastectomy,
        extract_cancer_treatment,
    ]

    for _, row in notes_df.iterrows():
        mrn = str(row[MERGE_KEY]).strip()
        note_id = row["NOTE_ID"]
        note_type = row["NOTE_TYPE"]
        note_date = row["NOTE_DATE"]
        text = row["NOTE_TEXT"]

        snote = build_sectioned_note(
            note_text=text,
            note_type=note_type,
            note_id=note_id,
            note_date=note_date
        )

        all_cands: List[Candidate] = []
        for fn in extractor_fns:
            try:
                all_cands.extend(fn(snote))
            except Exception as e:
                # don’t crash the whole run because 1 extractor hit an edge case
                evidence_rows.append({
                    MERGE_KEY: mrn,
                    "NOTE_ID": note_id,
                    "NOTE_DATE": note_date,
                    "NOTE_TYPE": note_type,
                    "FIELD": "EXTRACTOR_ERROR",
                    "VALUE": "",
                    "STATUS": "",
                    "CONFIDENCE": "",
                    "SECTION": "",
                    "EVIDENCE": f"{fn.__name__} failed: {repr(e)}"
                })

        if not all_cands:
            continue

        if mrn not in best_by_mrn:
            best_by_mrn[mrn] = {}

        for c in all_cands:
            logical = FIELD_MAP.get(str(c.field), None)
            if not logical:
                continue

            # record evidence row for QA
            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": getattr(c, "note_id", note_id),
                "NOTE_DATE": getattr(c, "note_date", note_date),
                "NOTE_TYPE": getattr(c, "note_type", note_type),
                "FIELD": logical,
                "VALUE": getattr(c, "value", ""),
                "STATUS": getattr(c, "status", ""),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "EVIDENCE": getattr(c, "evidence", "")
            })

            existing = best_by_mrn[mrn].get(logical, None)
            if logical in BOOLEAN_FIELDS:
                best_by_mrn[mrn][logical] = merge_boolean(existing, c)
            else:
                best_by_mrn[mrn][logical] = choose_best(existing, c)

    print(f"Aggregated best candidates for {len(best_by_mrn)} MRNs.")

    # Write predictions into gold/master
    print("Writing extracted values into master dataframe...")
    for mrn, field_dict in best_by_mrn.items():
        mask = (gold[MERGE_KEY].astype(str).str.strip() == str(mrn).strip())
        if not mask.any():
            continue

        for logical, cand in field_dict.items():
            out_col = resolve_output_col(gold, logical)
            val = getattr(cand, "value", pd.NA)

            # normalize booleans as 0/1 (presentation-friendly)
            if logical in BOOLEAN_FIELDS:
                try:
                    val = 1 if bool(val) else 0
                except Exception:
                    val = pd.NA

            gold.loc[mask, out_col] = val

    # Write outputs
    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    gold.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    # Validation (only where gold already has a true label column with non-missing data)
    print("Attempting validation against existing gold labels (where available)...")
    lines = []
    lines.append("RULE-BASED VALIDATION SUMMARY (where gold labels exist)\n")

    # We validate boolean fields only (most reliable)
    for logical in sorted(BOOLEAN_FIELDS):
        # find a gold label column candidate that is NOT the same as our output col, and has non-null values
        # If your gold is both “master + labels”, this will still work as long as that column has 0/1 or yes/no.
        out_col = resolve_output_col(gold, logical)

        # If out_col is newly created, it won’t have label values; skip
        if out_col == logical and gold[out_col].isna().all():
            continue

        # If out_col existed but looks empty, skip
        if gold[out_col].isna().all():
            continue

        # Treat out_col as “label” AND “pred” is also in out_col => can’t validate.
        # So: only validate if you also have a separate prediction column.
        # BUT in this script we wrote predictions into out_col.
        # Workaround: if your gold had labels already in out_col, predictions overwrote them.
        # So we only validate when there is a clearly separate gold label column.
        #
        # If you want strict validation, keep labels in a separate file or duplicate columns.
        #
        # Here: we search for a separate label column name like "GOLD_<logical>".
        label_candidates = [f"GOLD_{logical}", f"Gold_{logical}", f"{logical}_GOLD", f"{logical}_gold"]
        label_col = None
        for lc in label_candidates:
            if lc in gold.columns:
                label_col = lc
                break
        if not label_col:
            continue

        y_true = to_binary_series(gold[label_col])
        y_pred = to_binary_series(gold[out_col])

        tp, fp, fn, tn = confusion_counts(y_true, y_pred)
        met = metrics_from_counts(tp, fp, fn, tn)

        lines.append(f"\n[{logical}]  label_col={label_col}  pred_col={out_col}")
        lines.append(f"TP={tp} FP={fp} FN={fn} TN={tn}")
        lines.append(
            f"Accuracy={met['accuracy']:.4f}  Precision={met['precision']:.4f}  "
            f"Recall={met['recall']:.4f}  F1={met['f1']:.4f}  Specificity={met['specificity']:.4f}"
        )

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write("\n".join(lines))

    print("\nDONE.")
    print(f"- Master output:   {OUTPUT_MASTER}")
    print(f"- Evidence output: {OUTPUT_EVID}")
    print(f"- Validation txt:  {OUTPUT_SUMMARY}")
    print("\nRun this:")
    print("  python build_master_rule_FINAL.py")


if __name__ == "__main__":
    main()
