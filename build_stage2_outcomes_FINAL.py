#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_complications_from_notes.py  (Python 3.6.8 compatible)

Run from:  ~/Breast_Restore
Inputs:
  - Stage2 anchor (prefers frozen pack if present):
      ./_frozen_stage2/*/stage2_patient_clean.csv
    else:
      ./_outputs/patient_stage_summary.csv
  - Notes (auto-found under ~/my_data_Breast/**/):
      HPI11526 Clinic Notes.csv
      HPI11526 Inpatient Notes.csv
      HPI11526 Operation Notes.csv

Output:
  ./_outputs/stage2_outcomes_pred.csv

What it does (high level):
- Takes your Stage 2 date per patient as the anchor
- Defines a 1-year window post Stage 2 (inclusive start; exclusive end)
- Scans NOTE_TEXT in Clinic/Inpatient/Operation NOTES for complication/treatment signals
- Produces patient-level outcome flags + “best evidence” (date/source/note_id/snippet/pattern)

Notes:
- This is intentionally conservative: it does NOT infer complications from generic encounter reasons.
- It only fires when it sees explicit complication terms in note text, plus reop/failure/revision cues.
"""

from __future__ import print_function

import os
import glob
import re
import csv
from datetime import datetime, timedelta

import pandas as pd


# -------------------------
# Robust IO / normalization
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


def normalize_text(s):
    if s is None:
        return ""
    s = str(s).replace("\r", "\n").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_date_any(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%y %H:%M",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass

    m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", s)
    if m:
        token = m.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(token, fmt).date()
            except Exception:
                pass

    m = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", s)
    if m:
        token = m.group(1)
        try:
            return datetime.strptime(token, "%Y-%m-%d").date()
        except Exception:
            pass

    return None


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


# -------------------------
# File discovery
# -------------------------

def find_latest_frozen_stage2_patient_clean(root):
    base = os.path.join(root, "_frozen_stage2")
    if not os.path.isdir(base):
        return None
    candidates = sorted(glob.glob(os.path.join(base, "*", "stage2_patient_clean.csv")))
    if not candidates:
        return None
    # folders are timestamped; last is newest
    return os.path.abspath(candidates[-1])


def find_patient_stage_summary(root):
    p = os.path.join(root, "_outputs", "patient_stage_summary.csv")
    return os.path.abspath(p) if os.path.isfile(p) else None


def find_notes_files():
    """
    Auto-find your HPI11526 notes under ~/my_data_Breast/**.
    """
    home = os.path.expanduser("~")
    base = os.path.join(home, "my_data_Breast")
    patterns = [
        "**/HPI11526 Clinic Notes.csv",
        "**/HPI11526 Inpatient Notes.csv",
        "**/HPI11526 Operation Notes.csv",
    ]
    found = []
    for pat in patterns:
        found += glob.glob(os.path.join(base, pat), recursive=True)
    # keep only existing files
    found = [os.path.abspath(p) for p in found if os.path.isfile(p)]
    # de-dupe preserving order
    seen = set()
    out = []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# -------------------------
# Signal dictionaries (regex)
# -------------------------

# Specific complication categories (for future expansion; currently used as evidence for Minor/Major etc.)
COMPLICATION_PATTERNS = [
    ("hematoma", r"\bhematoma\b"),
    ("seroma", r"\bseroma\b"),
    ("infection", r"\b(infect(ion|ed)?|cellulit(is)?|abscess|purulen(t|ce)|sepsis)\b"),
    ("dehiscence", r"\b(dehisc(ence|ed)?|wound (open|opened)|incision (open|opened))\b"),
    ("necrosis", r"\b(necros(is|ed)?|eschar|skin flap necros)\b"),
    ("capsular_contracture", r"\b(capsular contracture|baker (i|ii|iii|iv)|capsule contracture)\b"),
    ("malposition", r"\b(implant malposition|malposition|bottom(ing)? out|symmastia)\b"),
    ("rupture_deflation", r"\b(rupture|leak(age)?|deflat(ion|ed)?|collapse)\b"),
    ("extrusion", r"\b(extrusion|exposed implant|implant exposure)\b"),
    ("thromboembolism", r"\b(pulmonary embol(ism)?|\bpe\b|dvt|thromboembol(ism)?)\b"),
]

# Treatment signals
REOP_PATTERNS = [
    ("return_to_or", r"\b(return(ed)? to (the )?or|take\s*back|takeback)\b"),
    ("washout", r"\b(wash\s*out|irrigat(e|ion))\b"),
    ("debridement", r"\bdebrid(e|ement)\b"),
    ("i_and_d", r"\b(i\s*&\s*d|incision and drainage)\b"),
    ("explants", r"\b(explant|remove(d)? implant|implant removal|expander removal)\b"),
    ("capsulectomy", r"\b(capsulectomy|capsulotomy)\b"),
]

NONOP_PATTERNS = [
    ("antibiotics", r"\b(antibiotic(s)?|keflex|cephalexin|clinda(mycin)?|doxy(cycline)?|augmentin|bactrim|vancomycin|zosyn|cefazolin)\b"),
    ("drainage", r"\b(aspirat(e|ion)|percutaneous drainage|needle drainage|ir drain|drain placement)\b"),
    ("wound_care", r"\b(dressing changes?|wound care|packing|wet to dry|local wound care)\b"),
]

FAILURE_PATTERNS = [
    ("implant_removed", r"\b(explant|implant removal|remove(d)? implant)\b"),
    ("flap_loss", r"\b(flap loss|failed flap|takeback.*flap)\b"),
    ("extrusion_failure", r"\b(extrusion|implant exposure)\b.*\b(explant|remov)\b"),
]

REVISION_PATTERNS = [
    ("revision", r"\b(revision|scar revision|dog[- ]?ear|contour deformit(y|ies)|asymmetr(y|ies)|fat graft(ing)?|lipofill(ing)?|capsulorrhaphy)\b"),
    ("cpt_19380_hint", r"\b19380\b"),  # if CPT appears in note text
]

# fast combined regexes
COMP_RX = re.compile("|".join(["({})".format(p[1]) for p in COMPLICATION_PATTERNS]))
REOP_RX = re.compile("|".join(["({})".format(p[1]) for p in REOP_PATTERNS]))
NONOP_RX = re.compile("|".join(["({})".format(p[1]) for p in NONOP_PATTERNS]))
FAIL_RX = re.compile("|".join(["({})".format(p[1]) for p in FAILURE_PATTERNS]))
REV_RX = re.compile("|".join(["({})".format(p[1]) for p in REVISION_PATTERNS]))


def best_match(pattern_list, text):
    for label, pat in pattern_list:
        if re.search(pat, text):
            return label, pat
    return "", ""


def make_snippet(text, width=220):
    t = normalize_text(text)
    if len(t) <= width:
        return t
    return t[:width] + "..."


# -------------------------
# Notes ingestion / aggregation
# -------------------------

def load_notes_aggregated(path):
    """
    Aggregates multi-line notes into one row per NOTE_ID (when NOTE_ID exists).
    If NOTE_ID is missing, falls back to row-level scanning.
    Returns a DataFrame with:
      ENCRYPTED_PAT_ID, NOTE_DATE, NOTE_ID, NOTE_TYPE, NOTE_TEXT, SOURCE_FILE
    """
    df = normalize_cols(read_csv_robust(path, dtype=str, low_memory=False))
    src = os.path.basename(path)

    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not enc_col:
        raise ValueError("Notes file missing ENCRYPTED_PAT_ID: {}".format(path))

    note_id_col = pick_first_existing(df, ["NOTE_ID", "Note_ID", "note_id"])
    note_type_col = pick_first_existing(df, ["NOTE_TYPE", "Note_Type", "note_type"])
    text_col = pick_first_existing(df, ["NOTE_TEXT", "Note_Text", "note_text", "NOTE_TEXT_DEID"])
    dos_col = pick_first_existing(df, ["NOTE_DATE_OF_SERVICE", "NOTE_DATE", "DATE_OF_SERVICE", "SERVICE_DATE"])
    op_date_col = pick_first_existing(df, ["OPERATION_DATE", "OPER_DATE"])

    # Normalize minimal columns
    df[enc_col] = df[enc_col].map(normalize_id)
    if note_id_col:
        df[note_id_col] = df[note_id_col].map(normalize_id)
    if note_type_col:
        df[note_type_col] = df[note_type_col].fillna("").map(str)
    if text_col:
        df[text_col] = df[text_col].fillna("").map(str)

    # Build NOTE_DATE
    def row_note_date(r):
        d = None
        if op_date_col and r.get(op_date_col):
            d = parse_date_any(r.get(op_date_col))
        if not d and dos_col and r.get(dos_col):
            d = parse_date_any(r.get(dos_col))
        return d

    if note_id_col and text_col:
        # aggregate by NOTE_ID
        grp_cols = [enc_col, note_id_col]
        keep_cols = [enc_col, note_id_col]
        if note_type_col:
            keep_cols.append(note_type_col)

        # sort by LINE if present for better concatenation
        line_col = pick_first_existing(df, ["LINE", "Line", "line"])
        if line_col:
            try:
                df[line_col] = pd.to_numeric(df[line_col], errors="coerce")
                df = df.sort_values([enc_col, note_id_col, line_col])
            except Exception:
                pass

        agg = df.groupby(grp_cols, as_index=False).agg({
            (note_type_col if note_type_col else enc_col): "first",
            text_col: lambda x: "\n".join([str(v) for v in x if str(v).strip() != ""])
        })

        # rebuild columns cleanly
        out = pd.DataFrame()
        out["ENCRYPTED_PAT_ID"] = agg[enc_col].map(normalize_id)
        out["NOTE_ID"] = agg[note_id_col].map(normalize_id)
        out["NOTE_TYPE"] = agg[note_type_col] if note_type_col else ""
        out["NOTE_TEXT"] = agg[text_col].fillna("").map(str)

        # compute NOTE_DATE using earliest date among component rows
        # (do a lightweight join for date computation)
        tmp = df[[enc_col, note_id_col]].copy()
        tmp["NOTE_DATE"] = df.apply(row_note_date, axis=1)
        tmp = tmp.dropna(subset=["NOTE_DATE"])
        if len(tmp) > 0:
            tmp2 = tmp.groupby([enc_col, note_id_col], as_index=False)["NOTE_DATE"].min()
            tmp2.columns = [enc_col, note_id_col, "NOTE_DATE"]
            out = out.merge(tmp2, left_on=["ENCRYPTED_PAT_ID", "NOTE_ID"], right_on=[enc_col, note_id_col], how="left")
            out = out.drop([enc_col, note_id_col], axis=1)
        else:
            out["NOTE_DATE"] = None

        out["SOURCE_FILE"] = src
        return out

    # fallback: row-level
    out = pd.DataFrame()
    out["ENCRYPTED_PAT_ID"] = df[enc_col].map(normalize_id)
    out["NOTE_ID"] = df[note_id_col].map(normalize_id) if note_id_col else ""
    out["NOTE_TYPE"] = df[note_type_col] if note_type_col else ""
    out["NOTE_TEXT"] = df[text_col].fillna("").map(str) if text_col else ""
    out["NOTE_DATE"] = df.apply(row_note_date, axis=1)
    out["SOURCE_FILE"] = src
    return out


# -------------------------
# Stage2 anchor loading
# -------------------------

def load_stage2_anchors(root):
    frozen = find_latest_frozen_stage2_patient_clean(root)
    if frozen:
        df = normalize_cols(read_csv_robust(frozen, dtype=str, low_memory=False))
        # expect ENCRYPTED_PAT_ID + STAGE2_DATE
        if "ENCRYPTED_PAT_ID" not in df.columns:
            raise ValueError("Frozen stage2_patient_clean.csv missing ENCRYPTED_PAT_ID: {}".format(frozen))
        if "STAGE2_DATE" not in df.columns:
            raise ValueError("Frozen stage2_patient_clean.csv missing STAGE2_DATE: {}".format(frozen))
        df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)
        df["STAGE2_DATE"] = df["STAGE2_DATE"].map(lambda x: parse_date_any(x))
        df = df.dropna(subset=["STAGE2_DATE"])
        print("Using Stage2 anchors (frozen):", frozen)
        return df[["ENCRYPTED_PAT_ID", "STAGE2_DATE"]].drop_duplicates()

    summ = find_patient_stage_summary(root)
    if not summ:
        raise IOError("Could not find Stage2 anchor: frozen stage2_patient_clean.csv OR _outputs/patient_stage_summary.csv")

    df = normalize_cols(read_csv_robust(summ, dtype=str, low_memory=False))
    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not enc_col:
        raise ValueError("patient_stage_summary missing ENCRYPTED_PAT_ID: {}".format(summ))
    if "STAGE2_DATE" not in df.columns:
        raise ValueError("patient_stage_summary missing STAGE2_DATE: {}".format(summ))

    df = df.rename(columns={enc_col: "ENCRYPTED_PAT_ID"})
    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)
    df["STAGE2_DATE"] = df["STAGE2_DATE"].map(lambda x: parse_date_any(x))
    # filter to stage2 patients (HAS_STAGE2==1 if present, else STAGE2_DATE not null)
    if "HAS_STAGE2" in df.columns:
        df = df[df["HAS_STAGE2"].map(to01) == 1]
    df = df.dropna(subset=["STAGE2_DATE"])
    print("Using Stage2 anchors (outputs):", summ)
    return df[["ENCRYPTED_PAT_ID", "STAGE2_DATE"]].drop_duplicates()


# -------------------------
# Evidence selection
# -------------------------

def update_best(best, candidate):
    """
    best: dict or None
    candidate: dict with evidence_date (date or None), priority (int), etc.
    Preference:
      1) earlier evidence_date (if exists)
      2) higher priority (lower int)
      3) keep first otherwise
    """
    if best is None:
        return candidate
    bd = best.get("evidence_date")
    cd = candidate.get("evidence_date")
    if cd and bd:
        if cd < bd:
            return candidate
        if cd > bd:
            return best
    elif cd and not bd:
        return candidate
    elif bd and not cd:
        return best

    # same date/none: compare priority
    bp = best.get("priority", 999)
    cp = candidate.get("priority", 999)
    if cp < bp:
        return candidate
    return best


# -------------------------
# Main scanning logic
# -------------------------

def main():
    root = os.path.abspath(".")
    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    anchors = load_stage2_anchors(root)
    anchors = anchors.dropna(subset=["ENCRYPTED_PAT_ID", "STAGE2_DATE"])
    anchors["ENCRYPTED_PAT_ID"] = anchors["ENCRYPTED_PAT_ID"].map(normalize_id)

    stage2_pids = set(anchors["ENCRYPTED_PAT_ID"].tolist())
    print("Stage2 patients (anchors):", len(stage2_pids))

    # Map pid -> stage2_date
    stage2_date_map = {}
    for _, r in anchors.iterrows():
        pid = r["ENCRYPTED_PAT_ID"]
        d = r["STAGE2_DATE"]
        if pid and d:
            # keep earliest stage2 date if duplicates
            if (pid not in stage2_date_map) or (d < stage2_date_map[pid]):
                stage2_date_map[pid] = d

    note_files = find_notes_files()
    if not note_files:
        raise IOError("Could not find HPI11526 *Notes.csv under ~/my_data_Breast/**")

    print("Notes files found:")
    for p in note_files:
        print("  -", p)

    # Load + concatenate aggregated notes
    all_notes = []
    for nf in note_files:
        df = load_notes_aggregated(nf)
        # keep only stage2 patients early to reduce memory
        df = df[df["ENCRYPTED_PAT_ID"].isin(stage2_pids)]
        all_notes.append(df)

    notes = pd.concat(all_notes, ignore_index=True) if all_notes else pd.DataFrame()
    if len(notes) == 0:
        raise IOError("No notes matched Stage2 patients after filtering.")

    # Normalize and compute window membership
    notes["NOTE_TEXT_NORM"] = notes["NOTE_TEXT"].map(normalize_text)
    # Some NOTE_DATE may be NaN/None
    notes["NOTE_DATE"] = notes["NOTE_DATE"]

    # Prepare per-patient outputs
    results = {}
    for pid in stage2_pids:
        results[pid] = {
            "ENCRYPTED_PAT_ID": pid,
            "STAGE2_DATE": stage2_date_map.get(pid),
            "WINDOW_START": stage2_date_map.get(pid),
            "WINDOW_END": (stage2_date_map.get(pid) + timedelta(days=365)) if stage2_date_map.get(pid) else None,

            "Stage2_MinorComp_pred": 0,
            "Stage2_Reoperation_pred": 0,
            "Stage2_Rehospitalization_pred": 0,
            "Stage2_MajorComp_pred": 0,
            "Stage2_Failure_pred": 0,
            "Stage2_Revision_pred": 0,

            # evidence blocks
            "minor_evidence_date": "",
            "minor_evidence_source": "",
            "minor_evidence_note_id": "",
            "minor_evidence_pattern": "",
            "minor_evidence_snippet": "",

            "reop_evidence_date": "",
            "reop_evidence_source": "",
            "reop_evidence_note_id": "",
            "reop_evidence_pattern": "",
            "reop_evidence_snippet": "",

            "rehosp_evidence_date": "",
            "rehosp_evidence_source": "",
            "rehosp_evidence_note_id": "",
            "rehosp_evidence_pattern": "",
            "rehosp_evidence_snippet": "",

            "failure_evidence_date": "",
            "failure_evidence_source": "",
            "failure_evidence_note_id": "",
            "failure_evidence_pattern": "",
            "failure_evidence_snippet": "",

            "revision_evidence_date": "",
            "revision_evidence_source": "",
            "revision_evidence_note_id": "",
            "revision_evidence_pattern": "",
            "revision_evidence_snippet": "",
        }

    # Best evidence trackers
    best_minor = {}
    best_reop = {}
    best_rehosp = {}
    best_failure = {}
    best_revision = {}

    # Scan row-by-row (already aggregated per NOTE_ID where possible)
    for idx, r in notes.iterrows():
        pid = normalize_id(r.get("ENCRYPTED_PAT_ID"))
        if not pid or pid not in stage2_date_map:
            continue

        s2 = stage2_date_map[pid]
        win_start = s2
        win_end = s2 + timedelta(days=365)

        nd = r.get("NOTE_DATE")
        # nd may be float NaN; handle safely
        if isinstance(nd, float):
            nd = None

        # If date exists, enforce window. If date missing, still allow *only* for strong signals (reop/failure/revision)
        in_window = False
        if nd:
            try:
                in_window = (nd >= win_start) and (nd < win_end)
            except Exception:
                in_window = False

        t = r.get("NOTE_TEXT_NORM", "")
        if not t:
            continue

        src = str(r.get("SOURCE_FILE", "") or "")
        note_id = normalize_id(r.get("NOTE_ID", ""))
        note_type = str(r.get("NOTE_TYPE", "") or "")

        # Complication present?
        has_comp = bool(COMP_RX.search(t))

        # -------- Minor complication (complication + nonop/no-op; no reop/rehosp implied)
        # We flag minor if: in-window AND complication AND (nonop OR explicit "managed conservatively") AND NOT failure terms
        if in_window and has_comp:
            nonop = bool(NONOP_RX.search(t)) or ("conservative" in t) or ("managed with" in t and "antibi" in t)
            fail = bool(FAIL_RX.search(t))
            reop = bool(REOP_RX.search(t))
            if nonop and (not reop) and (not fail):
                lbl, pat = best_match(COMPLICATION_PATTERNS, t)
                cand = {
                    "evidence_date": nd,
                    "priority": 10,
                    "source": src,
                    "note_id": note_id,
                    "pattern": pat if pat else lbl,
                    "snippet": make_snippet(r.get("NOTE_TEXT", "")),
                }
                best_minor[pid] = update_best(best_minor.get(pid), cand)

        # -------- Reoperation (complication + reop terms) OR explicit implant/expander removal in OR context
        # For date-missing notes, we require BOTH comp + reop/failure keyword to avoid false positives.
        if (in_window and has_comp and REOP_RX.search(t)) or ((not nd) and has_comp and (REOP_RX.search(t) or FAIL_RX.search(t))):
            lbl, pat = best_match(REOP_PATTERNS, t)
            if not pat:
                # fallback to complication label if reop label missing
                lbl2, pat2 = best_match(COMPLICATION_PATTERNS, t)
                pat = pat2 or lbl2
            cand = {
                "evidence_date": nd,
                "priority": 1,  # strong
                "source": src,
                "note_id": note_id,
                "pattern": pat if pat else lbl,
                "snippet": make_snippet(r.get("NOTE_TEXT", "")),
            }
            best_reop[pid] = update_best(best_reop.get(pid), cand)

        # -------- Failure (implant/flap removal due to complication)
        if (in_window and FAIL_RX.search(t)) or ((not nd) and FAIL_RX.search(t) and has_comp):
            lbl, pat = best_match(FAILURE_PATTERNS, t)
            cand = {
                "evidence_date": nd,
                "priority": 0,  # strongest
                "source": src,
                "note_id": note_id,
                "pattern": pat if pat else lbl,
                "snippet": make_snippet(r.get("NOTE_TEXT", "")),
            }
            best_failure[pid] = update_best(best_failure.get(pid), cand)

        # -------- Revision (revision terms; separate from reop-for-complication)
        if in_window and REV_RX.search(t):
            lbl, pat = best_match(REVISION_PATTERNS, t)
            cand = {
                "evidence_date": nd,
                "priority": 5,
                "source": src,
                "note_id": note_id,
                "pattern": pat if pat else lbl,
                "snippet": make_snippet(r.get("NOTE_TEXT", "")),
            }
            best_revision[pid] = update_best(best_revision.get(pid), cand)

        # -------- Rehospitalization (use inpatient note source + complication)
        # Conservative: requires in-window + complication + inpatient source file
        if in_window and has_comp:
            if ("Inpatient Notes".lower() in src.lower()) or ("discharge" in note_type.lower()) or ("h&p" in note_type.lower()):
                lbl, pat = best_match(COMPLICATION_PATTERNS, t)
                cand = {
                    "evidence_date": nd,
                    "priority": 2,
                    "source": src,
                    "note_id": note_id,
                    "pattern": pat if pat else lbl,
                    "snippet": make_snippet(r.get("NOTE_TEXT", "")),
                }
                best_rehosp[pid] = update_best(best_rehosp.get(pid), cand)

    # Finalize per-patient flags + evidence
    def fmt_date(d):
        if not d:
            return ""
        try:
            return d.strftime("%Y-%m-%d")
        except Exception:
            return ""

    for pid in results.keys():
        if pid in best_minor:
            results[pid]["Stage2_MinorComp_pred"] = 1
            results[pid]["minor_evidence_date"] = fmt_date(best_minor[pid].get("evidence_date"))
            results[pid]["minor_evidence_source"] = best_minor[pid].get("source", "")
            results[pid]["minor_evidence_note_id"] = best_minor[pid].get("note_id", "")
            results[pid]["minor_evidence_pattern"] = best_minor[pid].get("pattern", "")
            results[pid]["minor_evidence_snippet"] = best_minor[pid].get("snippet", "")

        if pid in best_reop:
            results[pid]["Stage2_Reoperation_pred"] = 1
            results[pid]["reop_evidence_date"] = fmt_date(best_reop[pid].get("evidence_date"))
            results[pid]["reop_evidence_source"] = best_reop[pid].get("source", "")
            results[pid]["reop_evidence_note_id"] = best_reop[pid].get("note_id", "")
            results[pid]["reop_evidence_pattern"] = best_reop[pid].get("pattern", "")
            results[pid]["reop_evidence_snippet"] = best_reop[pid].get("snippet", "")

        if pid in best_rehosp:
            results[pid]["Stage2_Rehospitalization_pred"] = 1
            results[pid]["rehosp_evidence_date"] = fmt_date(best_rehosp[pid].get("evidence_date"))
            results[pid]["rehosp_evidence_source"] = best_rehosp[pid].get("source", "")
            results[pid]["rehosp_evidence_note_id"] = best_rehosp[pid].get("note_id", "")
            results[pid]["rehosp_evidence_pattern"] = best_rehosp[pid].get("pattern", "")
            results[pid]["rehosp_evidence_snippet"] = best_rehosp[pid].get("snippet", "")

        if pid in best_failure:
            results[pid]["Stage2_Failure_pred"] = 1
            results[pid]["failure_evidence_date"] = fmt_date(best_failure[pid].get("evidence_date"))
            results[pid]["failure_evidence_source"] = best_failure[pid].get("source", "")
            results[pid]["failure_evidence_note_id"] = best_failure[pid].get("note_id", "")
            results[pid]["failure_evidence_pattern"] = best_failure[pid].get("pattern", "")
            results[pid]["failure_evidence_snippet"] = best_failure[pid].get("snippet", "")

        if pid in best_revision:
            results[pid]["Stage2_Revision_pred"] = 1
            results[pid]["revision_evidence_date"] = fmt_date(best_revision[pid].get("evidence_date"))
            results[pid]["revision_evidence_source"] = best_revision[pid].get("source", "")
            results[pid]["revision_evidence_note_id"] = best_revision[pid].get("note_id", "")
            results[pid]["revision_evidence_pattern"] = best_revision[pid].get("pattern", "")
            results[pid]["revision_evidence_snippet"] = best_revision[pid].get("snippet", "")

        # MajorComp is derived: reop OR rehosp
        if results[pid]["Stage2_Reoperation_pred"] == 1 or results[pid]["Stage2_Rehospitalization_pred"] == 1:
            results[pid]["Stage2_MajorComp_pred"] = 1

        # Ensure dates present in output
        results[pid]["STAGE2_DATE"] = fmt_date(results[pid]["STAGE2_DATE"])
        results[pid]["WINDOW_START"] = fmt_date(results[pid]["WINDOW_START"])
        results[pid]["WINDOW_END"] = fmt_date(results[pid]["WINDOW_END"])

    # Write output
    out_path = os.path.join(out_dir, "stage2_outcomes_pred.csv")

    # Stable column order
    cols = [
        "ENCRYPTED_PAT_ID",
        "STAGE2_DATE", "WINDOW_START", "WINDOW_END",
        "Stage2_MinorComp_pred",
        "Stage2_Reoperation_pred",
        "Stage2_Rehospitalization_pred",
        "Stage2_MajorComp_pred",
        "Stage2_Failure_pred",
        "Stage2_Revision_pred",

        "minor_evidence_date", "minor_evidence_source", "minor_evidence_note_id", "minor_evidence_pattern", "minor_evidence_snippet",
        "reop_evidence_date", "reop_evidence_source", "reop_evidence_note_id", "reop_evidence_pattern", "reop_evidence_snippet",
        "rehosp_evidence_date", "rehosp_evidence_source", "rehosp_evidence_note_id", "rehosp_evidence_pattern", "rehosp_evidence_snippet",
        "failure_evidence_date", "failure_evidence_source", "failure_evidence_note_id", "failure_evidence_pattern", "failure_evidence_snippet",
        "revision_evidence_date", "revision_evidence_source", "revision_evidence_note_id", "revision_evidence_pattern", "revision_evidence_snippet",
    ]

    out_df = pd.DataFrame(list(results.values()))
    # Add any missing cols (safety)
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = ""
    out_df = out_df[cols]

    out_df.to_csv(out_path, index=False)

    # Console summary
    def cnt(c):
        return int(pd.to_numeric(out_df[c], errors="coerce").fillna(0).astype(int).sum())

    print("")
    print("OK: wrote", out_path)
    print("Stage2 patients:", len(out_df))
    print("  Stage2_MinorComp_pred =", cnt("Stage2_MinorComp_pred"))
    print("  Stage2_Reoperation_pred =", cnt("Stage2_Reoperation_pred"))
    print("  Stage2_Rehospitalization_pred =", cnt("Stage2_Rehospitalization_pred"))
    print("  Stage2_MajorComp_pred =", cnt("Stage2_MajorComp_pred"))
    print("  Stage2_Failure_pred =", cnt("Stage2_Failure_pred"))
    print("  Stage2_Revision_pred =", cnt("Stage2_Revision_pred"))


if __name__ == "__main__":
    main()
