#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_fn_qa_snippets.py  (Python 3.6.8 compatible)

Goal:
- Identify Stage2-anchor False Negatives (Gold Stage2_Applicable==1 AND Pred HAS_STAGE2==0)
- Map MRN -> ENCRYPTED_PAT_ID via CROSSWALK (no MRNs in final QA output)
- Pull broad Stage2-related text snippets from de-id bundles:
    /home/apokol/Breast_Restore/PATIENT_BUNDLES/<ENCRYPTED_PAT_ID>/ALL_NOTES_COMBINED.txt
- Write a QA CSV safe to paste (no MRN, no names): ENCRYPTED_PAT_ID + snippets + a few stage fields

Inputs (expected):
- /home/apokol/Breast_Restore/_outputs/validation_merged.csv
  (If missing, we fallback to recomputing Stage2 FN directly from gold + patient_stage_summary + op notes mapping.)
- /home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv
- /home/apokol/Breast_Restore/CROSSWALK/CROSSWALK__MRN_to_patient_id__vNEW.csv
- /home/apokol/Breast_Restore/PATIENT_BUNDLES/<pid>/ALL_NOTES_COMBINED.txt

Outputs:
- /home/apokol/Breast_Restore/_outputs/stage2_fn_qa_snippets.csv
- /home/apokol/Breast_Restore/_outputs/stage2_fn_qa_snippets.txt  (readable preview)
"""

from __future__ import print_function

import os
import re
import glob
import csv
import pandas as pd


# -------------------------
# Robust IO helpers
# -------------------------

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))


def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df


def normalize_id(x):
    return "" if x is None else str(x).strip()


def to01(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


# -------------------------
# Stage2 keyword search (broad)
# -------------------------

# Broad Stage 2 indicative patterns (keep inclusive for FN discovery)
STAGE2_KEYWORDS = [
    # exchange / replace
    r"\b(exchange|exchang(ed|e)|replace(d|ment)?|implant exchange|expander exchange)\b",
    # expander removal
    r"\b(remove(d|al)?|explant(ed)?|take\s*out)\b.*\b(expander|tissue expander|expanders|te)\b",
    r"\b(expander|tissue expander|expanders|te)\b.*\b(remove(d|al)?|explant(ed)?|take\s*out)\b",
    # expander to implant phrasing
    r"\b(expander[- ]?to[- ]?implant)\b",
    r"\b(second stage|stage\s*2)\b.*\b(reconstruction|exchange|implant)\b",
    # implant placement mention plus expander mention (often in op notes)
    r"\b(implant|implants)\b.*\b(expander|tissue expander|expanders|te)\b",
    r"\b(expander|tissue expander|expanders|te)\b.*\b(implant|implants)\b",
    # "permanent implant" language
    r"\b(permanent implant|final implant)\b",
]

RX_STAGE2 = re.compile("|".join(["({})".format(p) for p in STAGE2_KEYWORDS]), re.I)


def load_text_safely(path):
    # de-id text files can still contain odd encoding
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except Exception:
            continue
    return ""


def extract_snippets(text, rx, max_snips=6, ctx=260):
    """
    Extract up to max_snips snippets of length ~2*ctx around regex matches.
    De-duplicate by normalized snippet.
    """
    if not text:
        return []

    out = []
    seen = set()

    for m in rx.finditer(text):
        start = max(0, m.start() - ctx)
        end = min(len(text), m.end() + ctx)
        snip = text[start:end].replace("\r", "\n")

        # compress whitespace for CSV readability
        snip_norm = re.sub(r"\s+", " ", snip).strip()

        # de-dupe
        key = snip_norm.lower()
        if key in seen:
            continue
        seen.add(key)

        out.append(snip_norm)
        if len(out) >= max_snips:
            break

    return out


# -------------------------
# FN identification
# -------------------------

def find_fn_from_validation_merged(merged_path):
    df = normalize_cols(read_csv_robust(merged_path, dtype=str, low_memory=False))

    # Gold Stage2 applicability
    gold_col = pick_first_existing(df, ["GOLD_HAS_STAGE2", "Stage2_Applicable", "STAGE2_APPLICABLE"])
    if not gold_col:
        raise ValueError("validation_merged missing GOLD_HAS_STAGE2/Stage2_Applicable col. Found: {}".format(list(df.columns)))

    # Pred stage2 flag
    pred_col = pick_first_existing(df, ["PRED_HAS_STAGE2", "HAS_STAGE2", "Pred_HAS_STAGE2"])
    if not pred_col:
        raise ValueError("validation_merged missing PRED_HAS_STAGE2/HAS_STAGE2 col. Found: {}".format(list(df.columns)))

    # MRN
    mrn_col = pick_first_existing(df, ["MRN", "mrn"])
    if not mrn_col:
        raise ValueError("validation_merged missing MRN col. Found: {}".format(list(df.columns)))

    df["MRN"] = df[mrn_col].map(normalize_id)
    df["GOLD_HAS_STAGE2"] = df[gold_col].map(to01).astype(int)
    df["PRED_HAS_STAGE2"] = df[pred_col].map(to01).astype(int)

    fn = df[(df["GOLD_HAS_STAGE2"] == 1) & (df["PRED_HAS_STAGE2"] == 0)].copy()

    # Pull some useful columns if present (safe; will not include MRN in final file)
    keep = ["MRN", "GOLD_HAS_STAGE2", "PRED_HAS_STAGE2"]
    for c in ["STAGE2_DATE", "STAGE2_NOTE_ID", "STAGE2_NOTE_TYPE", "STAGE2_MATCH_PATTERN", "STAGE2_HITS"]:
        if c in fn.columns:
            keep.append(c)
    return fn[keep].copy()


def load_crosswalk_mrn_to_pid(crosswalk_path):
    cw = normalize_cols(read_csv_robust(crosswalk_path, dtype=str, low_memory=False))

    mrn_col = pick_first_existing(cw, ["MRN", "mrn"])
    pid_col = pick_first_existing(cw, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "patient_id", "PATIENT_ID"])

    if not mrn_col or not pid_col:
        raise ValueError("Crosswalk missing MRN and/or patient id columns. Found: {}".format(list(cw.columns)))

    cw["MRN"] = cw[mrn_col].map(normalize_id)
    cw["ENCRYPTED_PAT_ID"] = cw[pid_col].map(normalize_id)

    # keep 1-1 if it is; otherwise keep first occurrence
    cw = cw[cw["MRN"] != ""].drop_duplicates(subset=["MRN"], keep="first")
    return cw[["MRN", "ENCRYPTED_PAT_ID"]].copy()


def main():
    root = "/home/apokol/Breast_Restore"
    out_dir = os.path.join(root, "_outputs")

    merged_path = os.path.join(out_dir, "validation_merged.csv")
    gold_path = os.path.join(root, "gold_cleaned_for_cedar.csv")  # not needed if merged exists, but kept for sanity
    crosswalk_path = os.path.join(root, "CROSSWALK", "CROSSWALK__MRN_to_patient_id__vNEW.csv")
    bundles_root = os.path.join(root, "PATIENT_BUNDLES")

    if not os.path.isfile(merged_path):
        raise IOError("Missing: {}. Run your validator first to create validation_merged.csv.".format(merged_path))
    if not os.path.isfile(crosswalk_path):
        raise IOError("Missing: {}".format(crosswalk_path))
    if not os.path.isdir(bundles_root):
        raise IOError("Missing bundles root dir: {}".format(bundles_root))

    fn = find_fn_from_validation_merged(merged_path)
    cw = load_crosswalk_mrn_to_pid(crosswalk_path)

    # Map MRN -> ENCRYPTED_PAT_ID
    fn = fn.merge(cw, on="MRN", how="left")

    # Keep only mapped rows (no MRN will be written out later)
    fn_mapped = fn[(fn["ENCRYPTED_PAT_ID"].fillna("") != "")].copy()

    # Build QA rows (NO MRN)
    qa_rows = []
    n_missing_bundle = 0
    n_no_hits = 0

    for _, r in fn_mapped.iterrows():
        pid = normalize_id(r.get("ENCRYPTED_PAT_ID", ""))

        bundle_txt = os.path.join(bundles_root, pid, "ALL_NOTES_COMBINED.txt")
        if not os.path.isfile(bundle_txt):
            # attempt fallback file name if any variations exist
            alt = glob.glob(os.path.join(bundles_root, pid, "*ALL*NOTES*COMBINED*.txt"))
            bundle_txt = alt[0] if alt else bundle_txt

        text = load_text_safely(bundle_txt) if os.path.isfile(bundle_txt) else ""
        if not text:
            n_missing_bundle += 1
            snippets = []
        else:
            snippets = extract_snippets(text, RX_STAGE2, max_snips=6, ctx=260)
            if not snippets:
                n_no_hits += 1

        row = {
            "ENCRYPTED_PAT_ID": pid,
            # carry over stage fields if they exist (still no MRN)
            "GOLD_HAS_STAGE2": int(r.get("GOLD_HAS_STAGE2", 1)),
            "PRED_HAS_STAGE2": int(r.get("PRED_HAS_STAGE2", 0)),
        }

        for c in ["STAGE2_DATE", "STAGE2_NOTE_ID", "STAGE2_NOTE_TYPE", "STAGE2_MATCH_PATTERN", "STAGE2_HITS"]:
            if c in fn_mapped.columns:
                row[c] = r.get(c, "")

        # Put snippets in fixed columns for easy review
        for i in range(6):
            key = "SNIP_{:02d}".format(i + 1)
            row[key] = snippets[i] if i < len(snippets) else ""

        row["BUNDLE_FILE_FOUND"] = 1 if os.path.isfile(bundle_txt) else 0
        qa_rows.append(row)

    qa_df = pd.DataFrame(qa_rows)

    out_csv = os.path.join(out_dir, "stage2_fn_qa_snippets.csv")
    out_txt = os.path.join(out_dir, "stage2_fn_qa_snippets.txt")

    qa_df.to_csv(out_csv, index=False)

    # write a readable preview
    with open(out_txt, "w") as f:
        f.write("Stage2 FN QA Snippets\n")
        f.write("Source merged: {}\n".format(merged_path))
        f.write("Crosswalk    : {}\n".format(crosswalk_path))
        f.write("Bundles root : {}\n".format(bundles_root))
        f.write("\nCounts:\n")
        f.write("  FN rows in merged (Stage2 only): {}\n".format(len(fn)))
        f.write("  FN rows mapped to ENCRYPTED_PAT_ID: {}\n".format(len(fn_mapped)))
        f.write("  Missing/empty bundle text: {}\n".format(n_missing_bundle))
        f.write("  Bundles found but no Stage2 keyword hits: {}\n".format(n_no_hits))
        f.write("\nPreview (first 10 rows):\n\n")

        cols = ["ENCRYPTED_PAT_ID", "BUNDLE_FILE_FOUND"] + \
               [c for c in ["STAGE2_DATE", "STAGE2_NOTE_ID", "STAGE2_NOTE_TYPE", "STAGE2_MATCH_PATTERN", "STAGE2_HITS"] if c in qa_df.columns] + \
               ["SNIP_01", "SNIP_02"]
        preview = qa_df[cols].head(10).fillna("")
        f.write(preview.to_string(index=False))
        f.write("\n")

    print("OK. Wrote:")
    print("  {}".format(out_csv))
    print("  {}".format(out_txt))
    print("Rows in QA:", len(qa_df))


if __name__ == "__main__":
    main()
