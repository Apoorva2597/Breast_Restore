#!/usr/bin/env python3
"""
Breast_Restore: Stage 1 / Stage 2 detector from Operation Notes CSV (line-split notes)

- Designed to run from ANY working directory (including your git repo folder).
- By default, it reads the Operation Notes CSV in the CURRENT DIRECTORY.
- You can also pass an explicit input path, and/or set an output subfolder.

USAGE EXAMPLES
--------------
# 1) Run from Breast_Restore/ and read the CSV in the same folder:
python stage1_stage2_detect.py

# 2) Run from anywhere, point to the CSV explicitly:
python stage1_stage2_detect.py --in "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"

# 3) Put outputs into Breast_Restore/outputs/BR_Final (no need for a new repo):
python stage1_stage2_detect.py --outdir "outputs/BR_Final"
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict

import pandas as pd


# -----------------------------
# Regex helpers
# -----------------------------
def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)


@dataclass
class Evidence:
    hit: bool
    snippets: List[str]


def _extract_snippets(text: str, patterns: List[re.Pattern], window: int = 80, max_snips: int = 6) -> List[str]:
    snips: List[str] = []
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


def detect_stage1(full_text: str) -> Evidence:
    """
    Stage 1 heuristic:
    - Must mention tissue expander(s)/expander AND
    - Must include a placement/reconstruction context.
    """
    if not full_text:
        return Evidence(False, [])
    te_hit = any(rx.search(full_text) for rx in RX_TE_STRONG)
    ctx_hit = any(rx.search(full_text) for rx in RX_STAGE1_CONTEXT)
    hit = te_hit and ctx_hit
    patterns = RX_TE_STRONG + RX_STAGE1_CONTEXT if hit else []
    return Evidence(hit, _extract_snippets(full_text, patterns))


def detect_stage2(full_text: str) -> Tuple[Evidence, Dict[str, bool]]:
    """
    Stage 2 heuristic:
    - Must include removal/exchange language AND implant placement/permanent implant language
      within the same note.
    """
    if not full_text:
        return Evidence(False, []), {"removal_term_found": False, "implant_term_found": False, "negation_found": False}

    removal_found = any(rx.search(full_text) for rx in RX_REMOVAL)
    implant_found = any(rx.search(full_text) for rx in RX_IMPLANT)
    neg_found = any(rx.search(full_text) for rx in RX_NEGATIONS)

    hit = removal_found and implant_found
    patterns = RX_REMOVAL + RX_IMPLANT + RX_NEGATIONS if hit else []
    return Evidence(hit, _extract_snippets(full_text, patterns)), {
        "removal_term_found": removal_found,
        "implant_term_found": implant_found,
        "negation_found": neg_found,
    }


# -----------------------------
# IO + pipeline
# -----------------------------
REQUIRED_COLS = ["ENCRYPTED_PAT_ID", "NOTE_ID", "LINE", "NOTE_TEXT", "OPERATION_DATE"]
FALLBACK_DATE_COLS = ["NOTE_DATE_OF_SERVICE"]


def read_and_validate(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    df["NOTE_TEXT"] = df["NOTE_TEXT"].fillna("").astype(str)
    df["LINE"] = pd.to_numeric(df["LINE"], errors="coerce").fillna(-1).astype(int)

    df["OPERATION_DATE"] = pd.to_datetime(df["OPERATION_DATE"], errors="coerce")
    if df["OPERATION_DATE"].isna().all():
        fallback = next((c for c in FALLBACK_DATE_COLS if c in df.columns), None)
        if fallback is None:
            raise ValueError("OPERATION_DATE is invalid/missing and no NOTE_DATE_OF_SERVICE fallback exists.")
        df["SURGERY_DATE"] = pd.to_datetime(df[fallback], errors="coerce")
    else:
        df["SURGERY_DATE"] = df["OPERATION_DATE"]

    for col in ["NOTE_TYPE", "ENCRYPTED_CSN", "MRN"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def reconstruct_notes(df: pd.DataFrame) -> pd.DataFrame:
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


def score_notes(note_df: pd.DataFrame) -> pd.DataFrame:
    stage1_hits, stage1_snips = [], []
    stage2_hits, stage2_snips = [], []
    stage2_removal, stage2_implant, stage2_neg = [], [], []

    for txt in note_df["FULL_NOTE_TEXT"].fillna("").astype(str).tolist():
        ev1 = detect_stage1(txt)
        ev2, comps = detect_stage2(txt)

        stage1_hits.append(ev1.hit)
        stage1_snips.append(" | ".join(ev1.snippets))

        stage2_hits.append(ev2.hit)
        stage2_snips.append(" | ".join(ev2.snippets))
        stage2_removal.append(bool(comps["removal_term_found"]))
        stage2_implant.append(bool(comps["implant_term_found"]))
        stage2_neg.append(bool(comps["negation_found"]))

    out = note_df.copy()
    out["stage1_hit"] = stage1_hits
    out["stage1_evidence_snippets"] = stage1_snips

    out["stage2_hit"] = stage2_hits
    out["stage2_evidence_snippets"] = stage2_snips
    out["stage2_removal_term_found"] = stage2_removal
    out["stage2_implant_term_found"] = stage2_implant
    out["stage2_negation_found"] = stage2_neg
    return out


def patient_rollup(scored_notes: pd.DataFrame) -> pd.DataFrame:
    scored_notes = scored_notes.copy()
    scored_notes["SURGERY_DATE"] = pd.to_datetime(scored_notes["SURGERY_DATE"], errors="coerce")

    rows = []
    for pid, g in scored_notes.groupby("ENCRYPTED_PAT_ID", dropna=False):
        gg = g.sort_values("SURGERY_DATE", kind="mergesort")

        s1 = gg[gg["stage1_hit"] & gg["SURGERY_DATE"].notna()]
        stage1_date = s1["SURGERY_DATE"].min() if len(s1) else pd.NaT
        stage1_note_id = s1.loc[s1["SURGERY_DATE"].idxmin(), "NOTE_ID"] if len(s1) else pd.NA

        s2_all = gg[gg["stage2_hit"] & gg["SURGERY_DATE"].notna()]
        missing_stage1 = pd.isna(stage1_date)

        if missing_stage1:
            stage2_date = s2_all["SURGERY_DATE"].min() if len(s2_all) else pd.NaT
            stage2_note_id = s2_all.loc[s2_all["SURGERY_DATE"].idxmin(), "NOTE_ID"] if len(s2_all) else pd.NA
        else:
            s2 = s2_all[s2_all["SURGERY_DATE"] > stage1_date]
            stage2_date = s2["SURGERY_DATE"].min() if len(s2) else pd.NaT
            stage2_note_id = s2.loc[s2["SURGERY_DATE"].idxmin(), "NOTE_ID"] if len(s2) else pd.NA

        stage2_conf = "none"
        if pd.notna(stage2_date):
            chosen = gg[(gg["NOTE_ID"] == stage2_note_id) & (gg["SURGERY_DATE"] == stage2_date)]
            if len(chosen):
                r = chosen.iloc[0]
                stage2_conf = "high" if bool(r["stage2_removal_term_found"]) and bool(r["stage2_implant_term_found"]) else "medium"

        rows.append(
            {
                "ENCRYPTED_PAT_ID": pid,
                "stage1_date": stage1_date,
                "stage1_note_id": stage1_note_id,
                "stage2_date": stage2_date,
                "stage2_note_id": stage2_note_id,
                "stage2_confidence": stage2_conf,
                "missing_stage1_flag": bool(missing_stage1),
            }
        )

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
        default="run1",
        help="Run tag to create a subfolder under outdir (default: run1)",
    )
    args = parser.parse_args()

    in_path = os.path.expanduser(args.in_path)
    outdir = os.path.expanduser(args.outdir)
    tag = args.tag.strip().replace(" ", "_")

    run_dir = os.path.join(outdir, tag)
    os.makedirs(run_dir, exist_ok=True)

    out_note = os.path.join(run_dir, "note_level_hits.csv")
    out_patient = os.path.join(run_dir, "patient_level_stage_labels.csv")

    df = read_and_validate(in_path)
    note_df = reconstruct_notes(df)
    scored = score_notes(note_df)
    scored.to_csv(out_note, index=False)

    patients = patient_rollup(scored)
    patients.to_csv(out_patient, index=False)

    print(f"OK: input: {in_path}")
    print(f"OK: wrote {out_note}")
    print(f"OK: wrote {out_patient}")
    print(f"Notes processed: {len(scored):,}")
    print(f"Patients processed: {patients['ENCRYPTED_PAT_ID'].nunique():,}")


if __name__ == "__main__":
    main()
