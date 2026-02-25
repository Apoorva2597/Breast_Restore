#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage2_outcomes_FINAL.py  (Python 3.6.8 compatible)

Run from: ~/Breast_Restore

Inputs (auto-detected):
  1) Stage2 anchor (required):
       ./_outputs/patient_stage_summary.csv
     Uses:
       ENCRYPTED_PAT_ID, STAGE2_DATE, HAS_STAGE2

  2) Encounter sources (auto-detected; at least 1 recommended):
       HPI11526 Clinic Encounters.csv
       HPI11526 Inpatient Encounters.csv
       HPI11526 Operation Encounters.csv

Outputs:
  ./_outputs/stage2_outcomes_pred.csv

What it does (proper, definition-aligned):
  - Anchors 1-year window AFTER Stage2 date per patient (window_start = stage2_date, window_end = stage2_date + 365d)
  - Detects *complication-driven* reoperation / rehospitalization (NOT just any reconstructive CPT)
  - Detects failure (implant/flap removal due to complication)
  - Detects revision (elective contour/asymmetry/scar correction; can be CPT/keyword-based)
  - Detects minor complication (complication evidence within window WITHOUT reop/rehosp)
  - Derives major complication = reop OR rehosp
"""

from __future__ import print_function
import os
import glob
import re
from datetime import datetime, timedelta

import pandas as pd


# -----------------------------
# Robust IO / normalization
# -----------------------------

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
    except:
        return 0


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


def parse_date_any(s):
    """
    Returns a python date (datetime.date) or None.
    Handles common formats and timestamp-like strings.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() == "nan":
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
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass

    # fallback: extract date token
    m = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
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

    return None


def norm_text(*parts):
    s = " ".join([("" if p is None else str(p)) for p in parts])
    s = s.replace("\r", "\n").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Auto-detect input files
# -----------------------------

def find_stage_summary(root):
    preferred = os.path.join(root, "_outputs", "patient_stage_summary.csv")
    if os.path.isfile(preferred):
        return os.path.abspath(preferred)

    cands = []
    cands += glob.glob(os.path.join(root, "_outputs", "*patient*stage*summary*.csv"))
    cands += glob.glob(os.path.join(root, "**", "*patient*stage*summary*.csv"), recursive=True)
    cands = [os.path.abspath(p) for p in cands if os.path.isfile(p)]
    if not cands:
        return None
    cands.sort(key=lambda x: len(x))
    return cands[0]


def _search_data_roots(root):
    """
    Common places you keep the HPI files.
    We search:
      - current repo
      - ~/my_data_Breast/
      - ../my_data_Breast/
    """
    homes = []
    try:
        homes.append(os.path.expanduser("~"))
    except Exception:
        pass

    roots = [root]
    for h in homes:
        roots.append(os.path.join(h, "my_data_Breast"))
    roots.append(os.path.abspath(os.path.join(root, "..", "my_data_Breast")))
    # de-dup, keep existing
    out = []
    seen = set()
    for r in roots:
        ar = os.path.abspath(r)
        if ar not in seen and os.path.exists(ar):
            out.append(ar)
            seen.add(ar)
    return out


def find_encounter_file(root, kind):
    """
    kind in {"Clinic", "Inpatient", "Operation"}
    """
    targets = []
    for base in _search_data_roots(root):
        # allow either exact file or anywhere under base
        targets += glob.glob(os.path.join(base, "**", "*{} Encounters.csv".format(kind)), recursive=True)
        targets += glob.glob(os.path.join(base, "**", "*{} Encounter*.csv".format(kind)), recursive=True)

    targets = [os.path.abspath(p) for p in targets if os.path.isfile(p)]
    if not targets:
        return None
    # prefer shortest path (usually the intended dataset location)
    targets.sort(key=lambda x: len(x))
    return targets[0]


# -----------------------------
# Patterns (definition-aligned)
# -----------------------------
# Complications list is from your spec:
# Hematoma, Wound dehiscence, Wound infection, Mastectomy skin flap necrosis,
# Seroma, Capsular contracture, Implant malposition, Implant rupture/leak/deflation,
# Implant/expander extrusion, Other (systemic etc.)
COMP_PATTERNS = [
    ("hematoma", r"\bhematoma\b|\bevacuati(on|e)\b.*\bhematoma\b"),
    ("seroma", r"\bseroma\b|\baspirat(e|ion)\b.*\bseroma\b"),
    ("infection", r"\binfect(ion|ed|ious)\b|\bcellulit(is|ic)\b|\babscess\b|\bpus\b|\bpurulen(t|ce)\b"),
    ("dehiscence", r"\bdehiscen(ce|t)\b|\bwound\b.*\b(open|breakdown|separat)\w*\b"),
    ("skin_flap_necrosis", r"\b(necros(is|e)|eschar)\b|\bskin\s*flap\b.*\b(necros|ischemi)\w*\b"),
    ("capsular_contracture", r"\bcapsular contracture\b|\bbaker\b.*\b(i|ii|iii|iv|1|2|3|4)\b"),
    ("malposition", r"\bmalposition\b|\bbottom(ing)?\s*out\b|\bsymmastia\b|\blateral(iz|is)ation\b"),
    ("rupture_leak_deflation", r"\bruptur(e|ed)\b|\bleak(age)?\b|\bdeflat(e|ion)\b"),
    ("extrusion", r"\bextrusion\b|\bexpos(ed|ure)\b.*\b(implant|expander)\b"),
    ("thromboembolism", r"\b(pulmonary embol(ism)?|pe\b|dvt\b|deep venous thromb)\b"),
    ("systemic", r"\b(sepsis|bacteremi(a|c)|pneumonia|aki\b|acute kidney)\b"),
]

# Surgical intervention / “return to OR” language (complication-driven)
INTERVENTION_PATTERNS = [
    ("washout", r"\bwashout\b|\birrigat(e|ion)\b.*\bdebrid\w*\b"),
    ("debridement", r"\bdebrid(e|ement|ed)\b"),
    ("drainage", r"\bdrain(age|ed)\b|\bevacuati(on|e)\b|\bincision and drainage\b|\bi\s*&\s*d\b"),
    ("reoperation_generic", r"\breturn(ed)?\s+to\s+or\b|\btake\s*back\b|\bre-?operat(e|ion)\b"),
    ("implant_exchange", r"\bimplant\b.*\bexchange|exchang(e|ed)\b.*\bimplant\b"),
    ("implant_removal", r"\b(explant|remove(d)?|removal)\b.*\b(implant|expander)\b"),
]

# Failure (implant/flap removal due to complication) – must be complication-driven OR explicit failure wording
FAILURE_PATTERNS = [
    ("implant_removed", r"\b(explant|remove(d)?|removal)\b.*\b(implant|expander)\b"),
    ("flap_loss", r"\bflap\b.*\b(loss|fail(ure|ed)|necros)\w*\b"),
    ("device_extrusion", r"\bextrusion\b|\bexpos(ed|ure)\b.*\b(implant|expander)\b"),
]

# Revision (elective correction) – gold defines revision surgery after stage2 for contour/asymmetry/scar
REVISION_PATTERNS = [
    ("revision_word", r"\brevision\b|\brevise\b"),
    ("scar_revision", r"\bscar\b.*\brevision\b|\bscar revision\b"),
    ("fat_grafting", r"\bfat graft\w*\b|\blipofill(ing)?\b|\blipo-?injection\b|\bautologous fat\b"),
    ("contour_asymmetry", r"\bcontour\b|\basymmetr(y|ies)\b|\bdeformit(y|ies)\b"),
    ("capsulorrhaphy", r"\bcapsulorrhaphy\b|\bpocket\b.*\b(revision|repair)\b"),
    ("implant_reposition", r"\breposition\b.*\bimplant\b|\bimplant\b.*\breposition\b"),
]

# CPT “hints”
# IMPORTANT: we DO NOT mark reoperation solely on reconstruction CPTs (19340/19357/etc).
# We only use these CPTs for revision/failure support or as additional evidence when comp keywords exist.
CPT_FAILURE = set(["19328"])   # removal of implant material (common)
CPT_REVISION = set(["19380"])  # revision of reconstructed breast


def match_first(patterns, text):
    """
    patterns: list of (name, regex)
    returns (name, regex) of first match else ("","")
    """
    for name, rx in patterns:
        if re.search(rx, text):
            return name, rx
    return "", ""


def has_any(patterns, text):
    for _, rx in patterns:
        if re.search(rx, text):
            return True
    return False


# -----------------------------
# Evidence selection helpers
# -----------------------------

def within_window(d, start, end):
    if d is None or start is None or end is None:
        return False
    return (d >= start) and (d <= end)


def take_earliest(e1, e2):
    """
    Each evidence is dict or None. Choose one with earlier date (or keep existing if one missing date).
    """
    if e1 is None:
        return e2
    if e2 is None:
        return e1
    d1 = e1.get("date")
    d2 = e2.get("date")
    if d1 and d2:
        return e1 if d1 <= d2 else e2
    if d1 and not d2:
        return e1
    if d2 and not d1:
        return e2
    return e1


def ev(date_obj, source, detail):
    # detail = dict with keys: kind, pattern_name, pattern_rx, snippet(optional)
    return {
        "date": date_obj,
        "source": source,
        "pattern": detail.get("pattern_name", ""),
        "pattern_rx": detail.get("pattern_rx", ""),
        "kind": detail.get("kind", ""),
    }


# -----------------------------
# Core extraction
# -----------------------------

def load_stage2_anchor(stage_path):
    df = normalize_cols(read_csv_robust(stage_path, dtype=str, low_memory=False))
    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    s2date_col = pick_first_existing(df, ["STAGE2_DATE", "Stage2_Date", "stage2_date"])
    has_s2_col = pick_first_existing(df, ["HAS_STAGE2", "has_stage2", "PRED_HAS_STAGE2"])

    if not enc_col or not s2date_col:
        raise ValueError("Stage summary must have ENCRYPTED_PAT_ID and STAGE2_DATE. Found: {}".format(list(df.columns)))

    df = df.rename(columns={enc_col: "ENCRYPTED_PAT_ID"})
    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)
    df["STAGE2_DATE"] = df[s2date_col].map(lambda x: parse_date_any(x))
    if has_s2_col:
        df["HAS_STAGE2"] = df[has_s2_col].map(to01).astype(int)
    else:
        df["HAS_STAGE2"] = df["STAGE2_DATE"].notna().astype(int)

    # keep only stage2 patients with a usable date
    df = df[(df["HAS_STAGE2"] == 1) & (df["STAGE2_DATE"].notna())].copy()
    # window
    df["WINDOW_START"] = df["STAGE2_DATE"]
    df["WINDOW_END"] = df["STAGE2_DATE"].map(lambda d: d + timedelta(days=365))
    return df[["ENCRYPTED_PAT_ID", "STAGE2_DATE", "WINDOW_START", "WINDOW_END"]]


def load_encounters(path, kind):
    """
    Standardize a minimal schema:
      ENCRYPTED_PAT_ID, EVENT_DATE, REASON_FOR_VISIT, CPT_CODE, PROCEDURE
    """
    if not path or not os.path.isfile(path):
        return None

    df = normalize_cols(read_csv_robust(path, dtype=str, low_memory=False))
    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not enc_col:
        # if missing, we cannot join reliably
        return None
    df = df.rename(columns={enc_col: "ENCRYPTED_PAT_ID"})
    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)

    # date column differs by file
    date_col = None
    if kind == "Clinic":
        date_col = pick_first_existing(df, ["ADMIT_DATE", "VISIT_DATE", "ENCOUNTER_DATE", "DATE"])
    elif kind == "Inpatient":
        date_col = pick_first_existing(df, ["HOSP_ADMSN_TM", "ADMIT_DATE", "ADMISSION_DATE", "DATE"])
    elif kind == "Operation":
        date_col = pick_first_existing(df, ["OPERATION_DATE", "SURGERY_DATE", "DATE"])

    if not date_col:
        # last resort: any column with "DATE" in name
        for c in df.columns:
            if "DATE" in c.upper():
                date_col = c
                break

    df["EVENT_DATE"] = df[date_col].map(parse_date_any) if date_col else None

    # optional text cols
    rfv = pick_first_existing(df, ["REASON_FOR_VISIT", "REASON", "VISIT_REASON"])
    cpt = pick_first_existing(df, ["CPT_CODE", "CPT", "CPT_CD"])
    proc = pick_first_existing(df, ["PROCEDURE", "PROC", "PROCEDURE_NAME"])

    df["REASON_FOR_VISIT"] = df[rfv] if rfv else ""
    df["CPT_CODE"] = df[cpt] if cpt else ""
    df["PROCEDURE"] = df[proc] if proc else ""

    df["SOURCE_FILE"] = os.path.basename(path)
    return df[["ENCRYPTED_PAT_ID", "EVENT_DATE", "REASON_FOR_VISIT", "CPT_CODE", "PROCEDURE", "SOURCE_FILE"]]


def classify_row(text_blob, cpt_code):
    """
    Returns flags + matched patterns.
    """
    comp_name, comp_rx = match_first(COMP_PATTERNS, text_blob)
    int_name, int_rx = match_first(INTERVENTION_PATTERNS, text_blob)
    fail_name, fail_rx = match_first(FAILURE_PATTERNS, text_blob)
    rev_name, rev_rx = match_first(REVISION_PATTERNS, text_blob)

    cpt = ("" if cpt_code is None else str(cpt_code)).strip()
    cpt_short = re.sub(r"\D", "", cpt)  # keep digits

    flags = {
        "has_comp": bool(comp_name),
        "has_intervention": bool(int_name),
        "has_failure_kw": bool(fail_name),
        "has_revision_kw": bool(rev_name),
        "cpt_is_failure": (cpt_short in CPT_FAILURE) if cpt_short else False,
        "cpt_is_revision": (cpt_short in CPT_REVISION) if cpt_short else False,
        "comp_name": comp_name,
        "comp_rx": comp_rx,
        "int_name": int_name,
        "int_rx": int_rx,
        "fail_name": fail_name,
        "fail_rx": fail_rx,
        "rev_name": rev_name,
        "rev_rx": rev_rx,
        "cpt_short": cpt_short,
    }
    return flags


def build_outcomes(stage2_anchor, clinic_df, inpatient_df, op_df):
    """
    Produces patient-level outcomes with evidence.
    """
    # index encounters by patient for speed
    by_pid = {}

    def add_df(df):
        if df is None or df.empty:
            return
        for _, r in df.iterrows():
            pid = normalize_id(r.get("ENCRYPTED_PAT_ID"))
            if not pid:
                continue
            by_pid.setdefault(pid, []).append(r)

    add_df(clinic_df)
    add_df(inpatient_df)
    add_df(op_df)

    rows_out = []
    for _, a in stage2_anchor.iterrows():
        pid = a["ENCRYPTED_PAT_ID"]
        s2 = a["STAGE2_DATE"]
        ws = a["WINDOW_START"]
        we = a["WINDOW_END"]

        # evidence containers
        reop_ev = None
        rehosp_ev = None
        failure_ev = None
        revision_ev = None
        minor_comp_ev = None

        events = by_pid.get(pid, [])
        for r in events:
            d = r.get("EVENT_DATE")
            if not within_window(d, ws, we):
                continue

            src = r.get("SOURCE_FILE", "")
            kind = "UNKNOWN"
            if src.lower().find("clinic") >= 0:
                kind = "CLINIC"
            elif src.lower().find("inpatient") >= 0:
                kind = "INPATIENT"
            elif src.lower().find("operation") >= 0:
                kind = "OPERATION"

            text_blob = norm_text(r.get("REASON_FOR_VISIT", ""), r.get("PROCEDURE", ""), r.get("CPT_CODE", ""))
            flags = classify_row(text_blob, r.get("CPT_CODE", ""))

            # ---- Rehospitalization (must be inpatient + complication evidence)
            if kind == "INPATIENT" and flags["has_comp"]:
                rehosp_ev = take_earliest(
                    rehosp_ev,
                    ev(d, src, {"kind": "rehosp", "pattern_name": flags["comp_name"], "pattern_rx": flags["comp_rx"]})
                )

            # ---- Reoperation (must be complication-driven + intervention OR explicit OR takeback language)
            # Accept OPERATION encounters directly; inpatient can also represent OR return if intervention language exists.
            if flags["has_comp"] and flags["has_intervention"] and (kind in ["OPERATION", "INPATIENT"]):
                reop_ev = take_earliest(
                    reop_ev,
                    ev(d, src, {"kind": "reop", "pattern_name": flags["int_name"] + "|" + flags["comp_name"],
                               "pattern_rx": flags["int_rx"] + " || " + flags["comp_rx"]})
                )

            # ---- Failure (implant/flap removal) – require:
            #  (A) explicit removal/extrusion/flap loss wording, OR CPT 19328
            #  AND preferably complication evidence; however explicit failure phrases can stand alone.
            failure_signal = False
            fail_pat = ""
            fail_rx = ""
            if flags["has_failure_kw"]:
                failure_signal = True
                fail_pat = flags["fail_name"]
                fail_rx = flags["fail_rx"]
            elif flags["cpt_is_failure"]:
                # CPT 19328 alone is strong device removal; but could be elective exchange context.
                # Require either complication keyword OR removal wording.
                if flags["has_comp"] or re.search(r"\b(remove(d)?|removal|explant|take out)\b", text_blob):
                    failure_signal = True
                    fail_pat = "CPT_19328"
                    fail_rx = "CPT_19328"

            if failure_signal:
                # If comp exists, embed it into the evidence pattern for traceability.
                pat_name = fail_pat if not flags["has_comp"] else (fail_pat + "|" + flags["comp_name"])
                pat_rx = fail_rx if not flags["has_comp"] else (fail_rx + " || " + flags["comp_rx"])
                failure_ev = take_earliest(
                    failure_ev,
                    ev(d, src, {"kind": "failure", "pattern_name": pat_name, "pattern_rx": pat_rx})
                )

            # ---- Revision (elective) – allow keyword OR CPT 19380, within window (no comp required)
            rev_signal = False
            rev_pat = ""
            rev_rx = ""
            if flags["has_revision_kw"]:
                rev_signal = True
                rev_pat = flags["rev_name"]
                rev_rx = flags["rev_rx"]
            elif flags["cpt_is_revision"]:
                rev_signal = True
                rev_pat = "CPT_19380"
                rev_rx = "CPT_19380"

            if rev_signal:
                revision_ev = take_earliest(
                    revision_ev,
                    ev(d, src, {"kind": "revision", "pattern_name": rev_pat, "pattern_rx": rev_rx})
                )

            # ---- Minor complication evidence (clinic-focused; comp keyword + non-OR management hints)
            # Minor comp = comp evidence in window but no reop/rehosp ultimately.
            # Here we store candidate evidence; final minor flag applied after we know reop/rehosp.
            if flags["has_comp"]:
                # Non-operative hints (antibiotics, wound care, dressing changes, aspiration etc.)
                nonop_hint = bool(re.search(r"\b(antibiotic|abx|augmentin|keflex|clinda|doxy|iv antibiotics|oral antibiotics)\b", text_blob)) \
                             or bool(re.search(r"\b(dressing|wound care|packing|clinic visit|local care|aspirat(e|ion)|office)\b", text_blob)) \
                             or bool(re.search(r"\b(draina(ge|ged)|aspirat(e|ion))\b", text_blob))
                # prefer clinic sources for minor evidence
                if kind == "CLINIC" or nonop_hint:
                    minor_comp_ev = take_earliest(
                        minor_comp_ev,
                        ev(d, src, {"kind": "minor_comp", "pattern_name": flags["comp_name"], "pattern_rx": flags["comp_rx"]})
                    )

        # Final binary outputs
        reop = 1 if reop_ev is not None else 0
        rehosp = 1 if rehosp_ev is not None else 0
        major = 1 if (reop or rehosp) else 0
        failure = 1 if failure_ev is not None else 0
        revision = 1 if revision_ev is not None else 0

        # Minor comp: must have comp evidence, but NOT major
        minor = 0
        if (minor_comp_ev is not None) and (major == 0):
            minor = 1

        rows_out.append({
            "ENCRYPTED_PAT_ID": pid,
            "STAGE2_DATE": s2.strftime("%Y-%m-%d") if s2 else "",
            "WINDOW_START": ws.strftime("%Y-%m-%d") if ws else "",
            "WINDOW_END": we.strftime("%Y-%m-%d") if we else "",

            "Stage2_MinorComp_pred": minor,
            "Stage2_Reoperation_pred": reop,
            "Stage2_Rehospitalization_pred": rehosp,
            "Stage2_MajorComp_pred": major,
            "Stage2_Failure_pred": failure,
            "Stage2_Revision_pred": revision,

            # Evidence columns (dates/sources/patterns)
            "reop_evidence_date": reop_ev["date"].strftime("%Y-%m-%d") if reop_ev and reop_ev.get("date") else "",
            "reop_evidence_source": reop_ev.get("source", "") if reop_ev else "",
            "reop_evidence_pattern": reop_ev.get("pattern", "") if reop_ev else "",

            "rehosp_evidence_date": rehosp_ev["date"].strftime("%Y-%m-%d") if rehosp_ev and rehosp_ev.get("date") else "",
            "rehosp_evidence_source": rehosp_ev.get("source", "") if rehosp_ev else "",
            "rehosp_evidence_pattern": rehosp_ev.get("pattern", "") if rehosp_ev else "",

            "failure_evidence_date": failure_ev["date"].strftime("%Y-%m-%d") if failure_ev and failure_ev.get("date") else "",
            "failure_evidence_source": failure_ev.get("source", "") if failure_ev else "",
            "failure_evidence_pattern": failure_ev.get("pattern", "") if failure_ev else "",

            "revision_evidence_date": revision_ev["date"].strftime("%Y-%m-%d") if revision_ev and revision_ev.get("date") else "",
            "revision_evidence_source": revision_ev.get("source", "") if revision_ev else "",
            "revision_evidence_pattern": revision_ev.get("pattern", "") if revision_ev else "",

            "minorcomp_evidence_date": minor_comp_ev["date"].strftime("%Y-%m-%d") if minor_comp_ev and minor_comp_ev.get("date") else "",
            "minorcomp_evidence_source": minor_comp_ev.get("source", "") if minor_comp_ev else "",
            "minorcomp_evidence_pattern": minor_comp_ev.get("pattern", "") if minor_comp_ev else "",
        })

    return pd.DataFrame(rows_out)


def main():
    root = os.path.abspath(".")
    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    stage_path = find_stage_summary(root)
    if not stage_path:
        raise IOError("Could not find _outputs/patient_stage_summary.csv (Stage2 anchor required).")

    clinic_path = find_encounter_file(root, "Clinic")
    inpatient_path = find_encounter_file(root, "Inpatient")
    openc_path = find_encounter_file(root, "Operation")

    print("Using:")
    print("  Stage summary:", stage_path)
    print("  Clinic enc   :", clinic_path if clinic_path else "(not found)")
    print("  Inpatient enc:", inpatient_path if inpatient_path else "(not found)")
    print("  Operation enc:", openc_path if openc_path else "(not found)")
    print("")

    stage2_anchor = load_stage2_anchor(stage_path)

    clinic_df = load_encounters(clinic_path, "Clinic") if clinic_path else None
    inpatient_df = load_encounters(inpatient_path, "Inpatient") if inpatient_path else None
    op_df = load_encounters(openc_path, "Operation") if openc_path else None

    if (clinic_df is None) and (inpatient_df is None) and (op_df is None):
        raise IOError("No encounter files found. Put the HPI11526 *Encounters.csv files under ~/my_data_Breast or within the repo tree.")

    pred = build_outcomes(stage2_anchor, clinic_df, inpatient_df, op_df)

    out_path = os.path.join(out_dir, "stage2_outcomes_pred.csv")
    pred.to_csv(out_path, index=False)

    # quick console sanity
    n = int(pred.shape[0])
    print("OK: wrote", out_path)
    print("Stage2 patients:", n)
    for col in ["Stage2_MinorComp_pred", "Stage2_Reoperation_pred", "Stage2_Rehospitalization_pred",
                "Stage2_MajorComp_pred", "Stage2_Failure_pred", "Stage2_Revision_pred"]:
        if col in pred.columns:
            print("  {} = {}".format(col, int(pred[col].sum())))


if __name__ == "__main__":
    main()
