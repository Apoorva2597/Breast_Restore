#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage2_outcomes_FINAL.py  (Python 3.6.8 compatible)

Revision v3:
- Reoperation: reduce planned/index-op false positives by requiring strong unplanned/takeback cues OR
  explant/exchange + complication driver + "unplanned/urgent/for infection/exposure/necrosis" context.
- Failure: expand to include "abandoned reconstruction", "conversion to autologous", "left without implant/expander".
- Dates: if NOTE_DATE missing, try in-note operative date; if still missing, allow ONLY for Operation Notes with strong op-header,
  and treat evidence_date as Stage2+0 (used only for window inclusion heuristic) but keep evidence_date blank in output.
"""

from __future__ import print_function
import os, glob, re
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

def ensure_encpat_col(df, file_label=""):
    df = normalize_cols(df)
    enc_col = pick_first_existing(df, [
        "ENCRYPTED_PAT_ID","ENCRYPTED_PATID","ENCRYPTED_PATIENT_ID",
        "Encrypted_Pat_ID","Encrypted_Patient_ID",
        "encrypted_pat_id","encrypted_patient_id",
    ])
    if not enc_col:
        raise ValueError("Missing encrypted patient id column in {}. Found: {}".format(file_label, list(df.columns)))
    if enc_col != "ENCRYPTED_PAT_ID":
        df = df.rename(columns={enc_col: "ENCRYPTED_PAT_ID"})
    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)
    return df

def ensure_note_text_col(df, file_label=""):
    df = normalize_cols(df)
    text_col = pick_first_existing(df, ["NOTE_TEXT", "Note_Text", "note_text", "NOTE_TEXT_DEID"])
    if not text_col:
        raise ValueError("Missing NOTE_TEXT column in {}. Found: {}".format(file_label, list(df.columns)))
    if text_col != "NOTE_TEXT":
        df = df.rename(columns={text_col: "NOTE_TEXT"})
    df["NOTE_TEXT"] = df["NOTE_TEXT"].fillna("").map(str)
    return df

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
    return os.path.abspath(candidates[-1])

def find_patient_stage_summary(root):
    p = os.path.join(root, "_outputs", "patient_stage_summary.csv")
    return os.path.abspath(p) if os.path.isfile(p) else None

def find_notes_files():
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
    found = [os.path.abspath(p) for p in found if os.path.isfile(p)]
    seen = set()
    out = []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

# -------------------------
# Context / cueing
# -------------------------

NEGATION_CUES = ["no ", "not ", "without ", "denies", "deny", "negative for", "neg for", "ruled out", "rule out", "r/o ", "does not", "did not"]
HISTORY_CUES  = ["history of", "hx of", "prior", "previous", "remote", "years ago", "last year", "in the past"]
PLAN_CUES     = ["plan", "planned", "scheduled", "will ", "may ", "might ", "consider", "discussed", "if "]

BREAST_CONTEXT_RX = re.compile(r"\b(breast|recon(struction)?|implant|expander|tissue expander|mastectomy|capsule|capsul|adm|acellular)\b")

OP_HEADER_RX = re.compile(
    r"\b(operative report|brief op note|procedure performed|procedures?:|pre[- ]?op diagnosis|post[- ]?op diagnosis|surgeon:|anesthesia|date of operation|date of surgery)\b"
)

INNOTE_DATE_RX = re.compile(
    r"\b(date of (surgery|operation)|surgery date|date of procedure|date of operation)\s*:\s*([a-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\b"
)

MONTH_MAP = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,
             "jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
             "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}

def parse_innote_date(token):
    token = (token or "").strip()
    if not token:
        return None
    d = parse_date_any(token)
    if d:
        return d
    m = re.match(r"([a-z]{3,9})\s+(\d{1,2}),\s+(\d{4})", token.lower())
    if m:
        mon = MONTH_MAP.get(m.group(1))
        day = int(m.group(2))
        yr = int(m.group(3))
        if mon:
            try:
                return datetime(yr, mon, day).date()
            except Exception:
                return None
    return None

def has_nearby_cue(text_norm, span, cue_list, window_chars=80):
    start, end = span
    lo = max(0, start - window_chars)
    hi = min(len(text_norm), end + window_chars)
    ctx = text_norm[lo:hi]
    for cue in cue_list:
        if cue in ctx:
            return True
    return False

def guarded_search(rx, text_norm):
    m = rx.search(text_norm)
    if not m:
        return None
    span = m.span()
    if has_nearby_cue(text_norm, span, NEGATION_CUES):
        return None
    if has_nearby_cue(text_norm, span, HISTORY_CUES):
        return None
    if has_nearby_cue(text_norm, span, PLAN_CUES) and (not OP_HEADER_RX.search(text_norm)):
        return None
    return m

def is_operation_source(src):
    return "operation notes" in (src or "").lower()

def is_inpatient_source(src):
    return "inpatient notes" in (src or "").lower()

# -------------------------
# Patterns
# -------------------------

COMPLICATION_PATTERNS = [
    ("hematoma", r"\bhematoma\b"),
    ("seroma", r"\bseroma\b"),
    ("infection", r"\b(infect(ion|ed)?|cellulit(is)?|abscess|purulen(t|ce)|sepsis)\b"),
    ("dehiscence", r"\b(dehisc(ence|ed)?|wound (open|opened)|incision (open|opened))\b"),
    ("necrosis", r"\b(necros(is|ed)?|eschar|skin flap necros)\b"),
    ("capsular_contracture", r"\b(capsular contracture|baker (i|ii|iii|iv)|capsule contracture)\b"),
    ("malposition", r"\b(implant malposition|bottom(ing)? out|symmastia)\b"),
    ("rupture_deflation", r"\b(rupture|leak(age)?|deflat(ion|ed)?|collapse)\b"),
    ("extrusion", r"\b(extrusion|exposed implant|implant exposure)\b"),
]

COMP_RX = re.compile("|".join(["({})".format(p[1]) for p in COMPLICATION_PATTERNS]))

NONOP_RX = re.compile(r"\b(antibiotic(s)?|keflex|cephalexin|clinda(mycin)?|doxy(cycline)?|augmentin|bactrim|vancomycin|zosyn|cefazolin|aspirat(e|ion)|percutaneous drainage|needle drainage|wound care|dressing changes?)\b")

REOP_STRONG_RX = re.compile(r"\b(return(ed)? to (the )?or|take\s*back|takeback|unplanned return|wash\s*out|debrid(e|ement)|i\s*&\s*d|incision and drainage)\b")

EXPLANT_RX = re.compile(r"\b(explant|explantation|implant removal|remove(d)? implant|expander removal|exchange(d)?|replacement)\b")

# drivers that make explant/exchange “complication driven”
FAIL_DRIVER_RX = re.compile(r"\b(infect(ion|ed)?|cellulit(is)?|abscess|purulen(t|ce)|sepsis|implant exposure|exposed implant|extrusion|necros(is|ed)?|eschar|hematoma|seroma|dehisc)\b")

UNPLANNED_CUE_RX = re.compile(r"\b(unplanned|urgent|emergent|due to|secondary to|because of|for (infection|exposure|necrosis|hematoma|seroma|wound))\b")

# failure expansions
FAIL_EXTRA_RX = re.compile(
    r"\b(abandon(ed)? (reconstruction|recon)|conversion to autologous|converted to (diep|tr(a)?m|latissimus|flap)|left without (implant|expander)|no (implant|expander) placed|removed and not replaced)\b"
)

REV_RX = re.compile(r"\b(revision|scar revision|dog[- ]?ear|contour deformit(y|ies)|asymmetr(y|ies)|fat graft(ing)?|lipofill(ing)?|capsulorrhaphy|19380)\b")

ADMISSION_RX = re.compile(r"\b(readmit(ted|sion)?|readmission|admitted (for|with|to)|was admitted|hospitali(z|s)ed|present(ed)? to (the )?(ed|er)|seen in (the )?(ed|er))\b")
REHOSP_EXCLUDE_RX = re.compile(r"\b(admit type:\s*op|op scheduled|outpatient scheduled|scheduled or date|planned procedure|pre[- ]?operative history and physical|preop|pre[- ]?op|surgical consent|video visit)\b")
REHOSP_EXCLUDE_FIRST_N_DAYS = 2

def best_match(pattern_list, text):
    for label, pat in pattern_list:
        if re.search(pat, text):
            return label, pat
    return "", ""

def make_snippet(text, width=220):
    t = normalize_text(text)
    return t if len(t) <= width else t[:width] + "..."

# -------------------------
# Notes ingestion
# -------------------------

def load_notes_aggregated(path):
    df = read_csv_robust(path, dtype=str, low_memory=False)
    df = normalize_cols(df)
    src = os.path.basename(path)

    df = ensure_encpat_col(df, file_label=src)
    df = ensure_note_text_col(df, file_label=src)

    note_id_col = pick_first_existing(df, ["NOTE_ID", "Note_ID", "note_id"])
    note_type_col = pick_first_existing(df, ["NOTE_TYPE", "Note_Type", "note_type"])
    dos_col = pick_first_existing(df, ["NOTE_DATE_OF_SERVICE", "NOTE_DATE", "DATE_OF_SERVICE", "SERVICE_DATE"])
    op_date_col = pick_first_existing(df, ["OPERATION_DATE", "OPER_DATE"])

    if note_id_col:
        df[note_id_col] = df[note_id_col].map(normalize_id)
    if note_type_col:
        df[note_type_col] = df[note_type_col].fillna("").map(str)

    def row_note_date(r):
        d = None
        if op_date_col and r.get(op_date_col):
            d = parse_date_any(r.get(op_date_col))
        if not d and dos_col and r.get(dos_col):
            d = parse_date_any(r.get(dos_col))
        return d

    if note_id_col:
        line_col = pick_first_existing(df, ["LINE", "Line", "line"])
        if line_col:
            try:
                df[line_col] = pd.to_numeric(df[line_col], errors="coerce")
                df = df.sort_values(["ENCRYPTED_PAT_ID", note_id_col, line_col])
            except Exception:
                pass

        agg = df.groupby(["ENCRYPTED_PAT_ID", note_id_col], as_index=False).agg({
            (note_type_col if note_type_col else "ENCRYPTED_PAT_ID"): "first",
            "NOTE_TEXT": lambda x: "\n".join([str(v) for v in x if str(v).strip() != ""])
        })

        out = pd.DataFrame()
        out["ENCRYPTED_PAT_ID"] = agg["ENCRYPTED_PAT_ID"].map(normalize_id)
        out["NOTE_ID"] = agg[note_id_col].map(normalize_id)
        out["NOTE_TYPE"] = agg[note_type_col] if note_type_col else ""
        out["NOTE_TEXT"] = agg["NOTE_TEXT"].fillna("").map(str)

        tmp = df[["ENCRYPTED_PAT_ID", note_id_col]].copy()
        tmp["NOTE_DATE"] = df.apply(row_note_date, axis=1)
        tmp = tmp.dropna(subset=["NOTE_DATE"])
        if len(tmp) > 0:
            tmp2 = tmp.groupby(["ENCRYPTED_PAT_ID", note_id_col], as_index=False)["NOTE_DATE"].min()
            tmp2.columns = ["ENCRYPTED_PAT_ID", note_id_col, "NOTE_DATE"]
            out = out.merge(tmp2, on=["ENCRYPTED_PAT_ID", "NOTE_ID"], how="left")
        else:
            out["NOTE_DATE"] = None

        out["SOURCE_FILE"] = src
        return out

    out = pd.DataFrame()
    out["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)
    out["NOTE_ID"] = ""
    out["NOTE_TYPE"] = df[note_type_col] if note_type_col else ""
    out["NOTE_TEXT"] = df["NOTE_TEXT"].fillna("").map(str)
    out["NOTE_DATE"] = df.apply(row_note_date, axis=1)
    out["SOURCE_FILE"] = src
    return out

# -------------------------
# Stage2 anchor loading
# -------------------------

def load_stage2_anchors(root):
    frozen = find_latest_frozen_stage2_patient_clean(root)
    if frozen:
        df = read_csv_robust(frozen, dtype=str, low_memory=False)
        df = normalize_cols(df)
        df = ensure_encpat_col(df, file_label=os.path.basename(frozen))
        if "STAGE2_DATE" not in df.columns:
            raise ValueError("Frozen stage2_patient_clean.csv missing STAGE2_DATE: {}".format(frozen))
        df["STAGE2_DATE"] = df["STAGE2_DATE"].map(parse_date_any)
        df = df.dropna(subset=["STAGE2_DATE"])
        print("Using Stage2 anchors (frozen):", frozen)
        return df[["ENCRYPTED_PAT_ID", "STAGE2_DATE"]].drop_duplicates()

    summ = find_patient_stage_summary(root)
    if not summ:
        raise IOError("Could not find Stage2 anchor")
    df = read_csv_robust(summ, dtype=str, low_memory=False)
    df = normalize_cols(df)
    df = ensure_encpat_col(df, file_label=os.path.basename(summ))
    if "STAGE2_DATE" not in df.columns:
        raise ValueError("patient_stage_summary missing STAGE2_DATE: {}".format(summ))
    df["STAGE2_DATE"] = df["STAGE2_DATE"].map(parse_date_any)
    if "HAS_STAGE2" in df.columns:
        df = df[df["HAS_STAGE2"].map(to01) == 1]
    df = df.dropna(subset=["STAGE2_DATE"])
    print("Using Stage2 anchors (outputs):", summ)
    return df[["ENCRYPTED_PAT_ID", "STAGE2_DATE"]].drop_duplicates()

# -------------------------
# Evidence selection
# -------------------------

def update_best(best, cand):
    if best is None:
        return cand
    bd = best.get("evidence_date")
    cd = cand.get("evidence_date")
    if cd and bd:
        if cd < bd:
            return cand
        if cd > bd:
            return best
    elif cd and not bd:
        return cand
    elif bd and not cd:
        return best
    if cand.get("priority", 999) < best.get("priority", 999):
        return cand
    return best

# -------------------------
# Main
# -------------------------

def main():
    root = os.path.abspath(".")
    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    anchors = load_stage2_anchors(root)
    anchors = anchors.dropna(subset=["ENCRYPTED_PAT_ID", "STAGE2_DATE"])
    anchors["ENCRYPTED_PAT_ID"] = anchors["ENCRYPTED_PAT_ID"].map(normalize_id)

    stage2_date_map = {}
    for _, r in anchors.iterrows():
        pid = r["ENCRYPTED_PAT_ID"]
        d = r["STAGE2_DATE"]
        if pid and d:
            if (pid not in stage2_date_map) or (d < stage2_date_map[pid]):
                stage2_date_map[pid] = d

    stage2_pids = set(stage2_date_map.keys())
    print("Stage2 patients (anchors):", len(stage2_pids))

    note_files = find_notes_files()
    if not note_files:
        raise IOError("Could not find HPI11526 *Notes.csv under ~/my_data_Breast/**")

    all_notes = []
    for nf in note_files:
        df = load_notes_aggregated(nf)
        df = df[df["ENCRYPTED_PAT_ID"].isin(stage2_pids)]
        all_notes.append(df)

    notes = pd.concat(all_notes, ignore_index=True)
    if len(notes) == 0:
        raise IOError("No notes matched Stage2 patients after filtering.")

    notes["NOTE_TEXT_NORM"] = notes["NOTE_TEXT"].map(normalize_text)

    results = {}
    for pid in stage2_pids:
        s2 = stage2_date_map.get(pid)
        results[pid] = {
            "ENCRYPTED_PAT_ID": pid,
            "STAGE2_DATE": s2,
            "WINDOW_START": s2,
            "WINDOW_END": (s2 + timedelta(days=365)) if s2 else None,

            "Stage2_MinorComp_pred": 0,
            "Stage2_Reoperation_pred": 0,
            "Stage2_Rehospitalization_pred": 0,
            "Stage2_MajorComp_pred": 0,
            "Stage2_Failure_pred": 0,
            "Stage2_Revision_pred": 0,

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

    best_minor, best_reop, best_rehosp, best_failure, best_revision = {}, {}, {}, {}, {}

    for _, r in notes.iterrows():
        pid = normalize_id(r.get("ENCRYPTED_PAT_ID"))
        if pid not in stage2_date_map:
            continue

        s2 = stage2_date_map[pid]
        win_start = s2
        win_end = s2 + timedelta(days=365)

        src = str(r.get("SOURCE_FILE", "") or "")
        note_id = normalize_id(r.get("NOTE_ID", ""))
        t = r.get("NOTE_TEXT_NORM", "")
        if not t or (not BREAST_CONTEXT_RX.search(t)):
            continue

        # date handling
        nd = r.get("NOTE_DATE")
        if isinstance(nd, float):
            nd = None

        if not nd:
            m = INNOTE_DATE_RX.search(t)
            if m:
                nd = parse_innote_date(m.group(3))

        # If still missing date: allow ONLY for Operation Notes that look like real op reports.
        # We'll treat them as "eligible" but evidence_date will be blank later.
        allow_undated_op = False
        if (not nd) and is_operation_source(src) and OP_HEADER_RX.search(t):
            allow_undated_op = True

        in_window = False
        if nd:
            try:
                in_window = (nd >= win_start) and (nd < win_end)
            except Exception:
                in_window = False
        elif allow_undated_op:
            # conservative heuristic: assume op note pertains to a surgical event and allow it
            # but still require strong operative+complication cues below.
            in_window = True

        if not in_window:
            continue

        # complication presence
        comp_m = guarded_search(COMP_RX, t)
        has_comp = bool(comp_m)

        # ---------------- MinorComp ----------------
        if has_comp:
            nonop = bool(guarded_search(NONOP_RX, t)) or ("managed with" in t and "antibi" in t) or ("conservative" in t)
            reop_strong = bool(REOP_STRONG_RX.search(t))
            explant = bool(EXPLANT_RX.search(t))
            driver = bool(FAIL_DRIVER_RX.search(t))
            if nonop and (not reop_strong) and (not (explant and driver)):
                lbl, pat = best_match(COMPLICATION_PATTERNS, t)
                best_minor[pid] = update_best(best_minor.get(pid), {
                    "evidence_date": nd,
                    "priority": 10,
                    "source": src,
                    "note_id": note_id,
                    "pattern": pat if pat else lbl,
                    "snippet": make_snippet(r.get("NOTE_TEXT", "")),
                })

        # ---------------- Reoperation (tighter) ----------------
        op_context = is_operation_source(src) or is_inpatient_source(src) or OP_HEADER_RX.search(t)
        reop_strong = bool(REOP_STRONG_RX.search(t)) and op_context

        explant = bool(EXPLANT_RX.search(t))
        driver = bool(FAIL_DRIVER_RX.search(t))
        unplannedish = bool(UNPLANNED_CUE_RX.search(t))

        # key: explant/exchange counts for reop ONLY if complication-driven AND has unplanned-ish cue
        reop_ok = False
        if reop_strong:
            reop_ok = True
        elif explant and driver and op_context and unplannedish:
            reop_ok = True

        if reop_ok:
            pat = "REOP_STRONG" if reop_strong else "EXPLANT_DRIVER_UNPLANNED"
            best_reop[pid] = update_best(best_reop.get(pid), {
                "evidence_date": nd,
                "priority": 1,
                "source": src,
                "note_id": note_id,
                "pattern": pat,
                "snippet": make_snippet(r.get("NOTE_TEXT", "")),
            })

        # ---------------- Failure (expanded) ----------------
        # counts if: (explant/exchange + driver + op_context) OR explicit abandonment/conversion language
        fail_extra = bool(FAIL_EXTRA_RX.search(t))
        failure_ok = False
        if fail_extra and op_context:
            failure_ok = True
        elif explant and driver and op_context:
            failure_ok = True

        if failure_ok:
            pat = "FAIL_EXTRA" if fail_extra else "EXPLANT_DRIVER"
            best_failure[pid] = update_best(best_failure.get(pid), {
                "evidence_date": nd,
                "priority": 0,
                "source": src,
                "note_id": note_id,
                "pattern": pat,
                "snippet": make_snippet(r.get("NOTE_TEXT", "")),
            })

        # ---------------- Revision (slightly loosened) ----------------
        if REV_RX.search(t) and op_context:
            best_revision[pid] = update_best(best_revision.get(pid), {
                "evidence_date": nd,
                "priority": 5,
                "source": src,
                "note_id": note_id,
                "pattern": "REV_OP",
                "snippet": make_snippet(r.get("NOTE_TEXT", "")),
            })

        # ---------------- Rehospitalization ----------------
        if has_comp and ADMISSION_RX.search(t) and (not REHOSP_EXCLUDE_RX.search(t)):
            if nd and nd >= (s2 + timedelta(days=REHOSP_EXCLUDE_FIRST_N_DAYS)):
                lbl, pat = best_match(COMPLICATION_PATTERNS, t)
                best_rehosp[pid] = update_best(best_rehosp.get(pid), {
                    "evidence_date": nd,
                    "priority": 2,
                    "source": src,
                    "note_id": note_id,
                    "pattern": pat if pat else lbl,
                    "snippet": make_snippet(r.get("NOTE_TEXT", "")),
                })

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
            results[pid]["minor_evidence_source"] = best_minor[pid].get("source","")
            results[pid]["minor_evidence_note_id"] = best_minor[pid].get("note_id","")
            results[pid]["minor_evidence_pattern"] = best_minor[pid].get("pattern","")
            results[pid]["minor_evidence_snippet"] = best_minor[pid].get("snippet","")

        if pid in best_reop:
            results[pid]["Stage2_Reoperation_pred"] = 1
            results[pid]["reop_evidence_date"] = fmt_date(best_reop[pid].get("evidence_date"))
            results[pid]["reop_evidence_source"] = best_reop[pid].get("source","")
            results[pid]["reop_evidence_note_id"] = best_reop[pid].get("note_id","")
            results[pid]["reop_evidence_pattern"] = best_reop[pid].get("pattern","")
            results[pid]["reop_evidence_snippet"] = best_reop[pid].get("snippet","")

        if pid in best_rehosp:
            results[pid]["Stage2_Rehospitalization_pred"] = 1
            results[pid]["rehosp_evidence_date"] = fmt_date(best_rehosp[pid].get("evidence_date"))
            results[pid]["rehosp_evidence_source"] = best_rehosp[pid].get("source","")
            results[pid]["rehosp_evidence_note_id"] = best_rehosp[pid].get("note_id","")
            results[pid]["rehosp_evidence_pattern"] = best_rehosp[pid].get("pattern","")
            results[pid]["rehosp_evidence_snippet"] = best_rehosp[pid].get("snippet","")

        if pid in best_failure:
            results[pid]["Stage2_Failure_pred"] = 1
            results[pid]["failure_evidence_date"] = fmt_date(best_failure[pid].get("evidence_date"))
            results[pid]["failure_evidence_source"] = best_failure[pid].get("source","")
            results[pid]["failure_evidence_note_id"] = best_failure[pid].get("note_id","")
            results[pid]["failure_evidence_pattern"] = best_failure[pid].get("pattern","")
            results[pid]["failure_evidence_snippet"] = best_failure[pid].get("snippet","")

        if pid in best_revision:
            results[pid]["Stage2_Revision_pred"] = 1
            results[pid]["revision_evidence_date"] = fmt_date(best_revision[pid].get("evidence_date"))
            results[pid]["revision_evidence_source"] = best_revision[pid].get("source","")
            results[pid]["revision_evidence_note_id"] = best_revision[pid].get("note_id","")
            results[pid]["revision_evidence_pattern"] = best_revision[pid].get("pattern","")
            results[pid]["revision_evidence_snippet"] = best_revision[pid].get("snippet","")

        if results[pid]["Stage2_Reoperation_pred"] == 1 or results[pid]["Stage2_Rehospitalization_pred"] == 1:
            results[pid]["Stage2_MajorComp_pred"] = 1

        results[pid]["STAGE2_DATE"] = fmt_date(results[pid]["STAGE2_DATE"])
        results[pid]["WINDOW_START"] = fmt_date(results[pid]["WINDOW_START"])
        results[pid]["WINDOW_END"] = fmt_date(results[pid]["WINDOW_END"])

    out_path = os.path.join(out_dir, "stage2_outcomes_pred.csv")

    cols = [
        "ENCRYPTED_PAT_ID",
        "STAGE2_DATE","WINDOW_START","WINDOW_END",
        "Stage2_MinorComp_pred",
        "Stage2_Reoperation_pred",
        "Stage2_Rehospitalization_pred",
        "Stage2_MajorComp_pred",
        "Stage2_Failure_pred",
        "Stage2_Revision_pred",
        "minor_evidence_date","minor_evidence_source","minor_evidence_note_id","minor_evidence_pattern","minor_evidence_snippet",
        "reop_evidence_date","reop_evidence_source","reop_evidence_note_id","reop_evidence_pattern","reop_evidence_snippet",
        "rehosp_evidence_date","rehosp_evidence_source","rehosp_evidence_note_id","rehosp_evidence_pattern","rehosp_evidence_snippet",
        "failure_evidence_date","failure_evidence_source","failure_evidence_note_id","failure_evidence_pattern","failure_evidence_snippet",
        "revision_evidence_date","revision_evidence_source","revision_evidence_note_id","revision_evidence_pattern","revision_evidence_snippet",
    ]

    out_df = pd.DataFrame(list(results.values()))
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = ""
    out_df = out_df[cols]
    out_df.to_csv(out_path, index=False)

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
