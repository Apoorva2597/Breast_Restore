#!/usr/bin/env python3
"""
Breast_Restore: Stage 1 / Stage 2 detector from Operation Notes CSV (line-split notes)
Python 3.6.8 compatible (NO typing.Pattern, NO f-strings required).

USAGE
-----
# Run from Breast_Restore/ and read the CSV in the same folder:
python build_s12_FINAL.py

# Or point to the CSV explicitly:
python build_s12_FINAL.py --in "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"

# Put outputs into Breast_Restore/outputs/BR_Final (no need for new repo):
python build_s12_FINAL.py --outdir "outputs/BR_Final"
"""

from __future__ import print_function

import argparse
import os
import re
import sys

import pandas as pd


# -----------------------------
# Regex helpers (Py3.6-safe)
# -----------------------------
def _rx(pattern):
    return re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)


def _extract_snippets(text, patterns, window=80, max_snips=6):
    snips = []
    if not text:
        return snips
    for rx in patterns:
        for m in rx.finditer(text):
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            snippet = text[start:end].replace("\n", " ").strip()
            snippet = re.sub(r"\s+", " ", snippet)
            snips.append(snippet)
            if len(snips) >= max_snips:
                return snips
    return snips


# -----------------------------
# Stage rules
# -----------------------------
# Stage 1 = expander placement / immediate reconstruction
RX_TE_STRONG = [
    _rx(r"\btissue\s+expanders?\b"),
    _rx(r"\bexpander(s)?\b"),
]
RX_STAGE1_CONTEXT = [
    _rx(r"\b(place(d)?|insert(ed)?|placement|implant(ed)?|put|position(ed)?)\b"),
    _rx(r"\bimmediate\s+reconstruction\b"),
    _rx(r"\b(first|1st)\s+stage\b"),
    _rx(r"\breconstruction\b"),
]

# Stage 2 = expander removal/exchange + implant placement/permanent implant
RX_REMOVAL = [
    _rx(r"\b(remove(d)?|removal|explant(ed|ation)?|take(n)?\s+out|exchange(d)?)\b"),
    _rx(r"\bexpander\s+(removal|exchange|explant)\b"),
    _rx(r"\bexchange\s+of\s+(the\s+)?(tissue\s+)?expanders?\b"),
]
RX_IMPLANT = [
    _rx(r"\bpermanent\s+(silicone|saline)\s+implants?\b"),
    _rx(r"\b(permanent|final)\s+implants?\b"),
    _rx(r"\bsilicone\s+gel\s+implants?\b"),
    _rx(r"\bimplant(s)?\s+(placed|placement|inserted|insertion)\b"),
    _rx(r"\b(exchange|exchanged?)\s+with\s+permanent\s+implants?\b"),
    _rx(r"\bimplant\s+exchange\b"),
]
RX_NEGATIONS = [
    _rx(r"\bno\s+implant(s)?\b"),
    _rx(r"\bnot\s+performed\b"),
    _rx(r"\bdefer(red|ring)\b"),
    _rx(r"\babort(ed)?\b"),
]


def detect_stage1(full_text):
    """
    Stage 1 heuristic:
      - Must mention tissue expander(s)/expander AND
      - Must include placement/reconstruction context.
    Returns: (hit_bool, snippets_list)
    """
    if not full_text:
        return False, []
    te_hit = any(rx.search(full_text) for rx in RX_TE_STRONG)
    ctx_hit = any(rx.search(full_text) for rx in RX_STAGE1_CONTEXT)
    hit = bool(te_hit and ctx_hit)
    patterns = (RX_TE_STRONG + RX_STAGE1_CONTEXT) if hit else []
    return hit, _extract_snippets(full_text, patterns)


def detect_stage2(full_text):
    """
    Stage 2 heuristic:
      - Must include removal/exchange language AND implant placement/permanent implant language
        within the same note.
    Returns:
      hit_bool, snippets_list, removal_found_bool, implant_found_bool, neg_found_bool
    """
    if not full_text:
        return False, [], False, False, False

    removal_found = any(rx.search(full_text) for rx in RX_REMOVAL)
    implant_found = any(rx.search(full_text) for rx in RX_IMPLANT)
    neg_found = any(rx.search(full_text) for rx in RX_NEGATIONS)

    hit = bool(removal_found and implant_found)
    patterns = (RX_REMOVAL + RX_IMPLANT + RX_NEGATIONS) if hit else []
    snippets = _extract_snippets(full_text, patterns)
    return hit, snippets, bool(removal_found), bool(implant_found), bool(neg_found)


# -----------------------------
# IO + pipeline
# -----------------------------
REQUIRED_COLS = ["ENCRYPTED_PAT_ID", "NOTE_ID", "LINE", "NOTE_TEXT", "OPERATION_DATE"]
FALLBACK_DATE_COLS = ["NOTE_DATE_OF_SERVICE"]


def read_and_validate(csv_path):
    if not os.path.exists(csv_path):
        raise IOError("Input CSV not found: {0}".format(csv_path))

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {0}\nFound: {1}".format(missing, list(df.columns)))

    df["NOTE_TEXT"] = df["NOTE_TEXT"].fillna("").astype(str)
    df["LINE"] = pd.to_numeric(df["LINE"], errors="coerce").fillna(-1).astype(int)

    df["OPERATION_DATE"] = pd.to_datetime(df["OPERATION_DATE"], errors="coerce")
    if df["OPERATION_DATE"].isna().all():
        fallback = None
        for c in FALLBACK_DATE_COLS:
            if c in df.columns:
                fallback = c
                break
        if fallback is None:
            raise ValueError("OPERATION_DATE invalid/missing and no NOTE_DATE_OF_SERVICE fallback exists.")
        df["SURGERY_DATE"] = pd.to_datetime(df[fallback], errors="coerce")
    else:
        df["SURGERY_DATE"] = df["OPERATION_DATE"]

    # Optional columns (keep pipeline stable)
    for col in ["NOTE_TYPE", "ENCRYPTED_CSN", "MRN"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def reconstruct_notes(df):
    group_cols = ["ENCRYPTED_PAT_ID", "NOTE_ID"]
    if "ENCRYPTED_CSN" in df.columns and df["ENCRYPTED_CSN"].notna().any():
        group_cols = ["ENCRYPTED_PAT_ID", "NOTE_ID", "ENCRYPTED_CSN"]

    df_sorted = df.sort_values(group_cols + ["LINE"], kind="mergesort")

    note_df = (
        df_sorted
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(
            SURGERY_DATE=("SURGERY_DATE", "min"),
            NOTE_TYPE=("NOTE_TYPE", "first"),
            MRN=("MRN", "first"),
            FULL_NOTE_TEXT=("NOTE_TEXT", lambda s: "\n".join(s.tolist())),
        )
    )
    return note_df


def score_notes(note_df):
    stage1_hit = []
    stage1_snips = []

    stage2_hit = []
    stage2_snips = []
    stage2_removal = []
    stage2_implant = []
    stage2_neg = []

    texts = note_df["FULL_NOTE_TEXT"].fillna("").astype(str).tolist()
    for txt in texts:
        s1_hit, s1_sn = detect_stage1(txt)
        s2_hit, s2_sn, rm, im, ng = detect_stage2(txt)

        stage1_hit.append(bool(s1_hit))
        stage1_snips.append(" | ".join(s1_sn))

        stage2_hit.append(bool(s2_hit))
        stage2_snips.append(" | ".join(s2_sn))
        stage2_removal.append(bool(rm))
        stage2_implant.append(bool(im))
        stage2_neg.append(bool(ng))

    out = note_df.copy()
    out["stage1_hit"] = stage1_hit
    out["stage1_evidence_snippets"] = stage1_snips

    out["stage2_hit"] = stage2_hit
    out["stage2_evidence_snippets"] = stage2_snips
    out["stage2_removal_term_found"] = stage2_removal
    out["stage2_implant_term_found"] = stage2_implant
    out["stage2_negation_found"] = stage2_neg

    return out


def patient_rollup(scored_notes):
    scored_notes = scored_notes.copy()
    scored_notes["SURGERY_DATE"] = pd.to_datetime(scored_notes["SURGERY_DATE"], errors="coerce")

    rows = []
    for pid, g in scored_notes.groupby("ENCRYPTED_PAT_ID", dropna=False):
        gg = g.sort_values("SURGERY_DATE", kind="mergesort")

        s1 = gg[(gg["stage1_hit"] == True) & (gg["SURGERY_DATE"].notna())]
        if len(s1):
            idx1 = s1["SURGERY_DATE"].idxmin()
            stage1_date = s1.loc[idx1, "SURGERY_DATE"]
            stage1_note_id = s1.loc[idx1, "NOTE_ID"]
        else:
            stage1_date = pd.NaT
            stage1_note_id = pd.NA

        s2_all = gg[(gg["stage2_hit"] == True) & (gg["SURGERY_DATE"].notna())]
        missing_stage1 = pd.isna(stage1_date)

        if missing_stage1:
            if len(s2_all):
                idx2 = s2_all["SURGERY_DATE"].idxmin()
                stage2_date = s2_all.loc[idx2, "SURGERY_DATE"]
                stage2_note_id = s2_all.loc[idx2, "NOTE_ID"]
            else:
                stage2_date = pd.NaT
                stage2_note_id = pd.NA
        else:
            s2 = s2_all[s2_all["SURGERY_DATE"] > stage1_date]
            if len(s2):
                idx2 = s2["SURGERY_DATE"].idxmin()
                stage2_date = s2.loc[idx2, "SURGERY_DATE"]
                stage2_note_id = s2.loc[idx2, "NOTE_ID"]
            else:
                stage2_date = pd.NaT
                stage2_note_id = pd.NA

        stage2_conf = "none"
        if pd.notna(stage2_date):
            chosen = gg[(gg["NOTE_ID"] == stage2_note_id) & (gg["SURGERY_DATE"] == stage2_date)]
            if len(chosen):
                r = chosen.iloc[0]
                if bool(r["stage2_removal_term_found"]) and bool(r["stage2_implant_term_found"]):
                    stage2_conf = "high"
                else:
                    stage2_conf = "medium"

        rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "stage1_date": stage1_date,
            "stage1_note_id": stage1_note_id,
            "stage2_date": stage2_date,
            "stage2_note_id": stage2_note_id,
            "stage2_confidence": stage2_conf,
            "missing_stage1_flag": bool(missing_stage1),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="in_path",
        default="HPI11526 Operation Notes.csv",
        help="Path to Operation Notes CSV (default: ./HPI11526 Operation Notes.csv)",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir",
        default="outputs",
        help="Output directory inside your repo (default: outputs)",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        default="final_final_attempt",
        help="Run tag subfolder under outdir (default: final_final_attempt)",
    )
    args = parser.parse_args()

    in_path = os.path.expanduser(args.in_path)
    outdir = os.path.expanduser(args.outdir)
    tag = args.tag.strip().replace(" ", "_")

    run_dir = os.path.join(outdir, tag)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    out_note = os.path.join(run_dir, "note_level_hits.csv")
    out_patient = os.path.join(run_dir, "patient_level_stage_labels.csv")

    df = read_and_validate(in_path)
    note_df = reconstruct_notes(df)
    scored = score_notes(note_df)
    scored.to_csv(out_note, index=False)

    patients = patient_rollup(scored)
    patients.to_csv(out_patient, index=False)

    print("OK: input: {0}".format(in_path))
    print("OK: wrote {0}".format(out_note))
    print("OK: wrote {0}".format(out_patient))
    print("Notes processed: {0}".format(len(scored)))
    try:
        n_pat = patients["ENCRYPTED_PAT_ID"].nunique()
    except Exception:
        n_pat = len(patients)
    print("Patients processed: {0}".format(n_pat))


if __name__ == "__main__":
    main()
