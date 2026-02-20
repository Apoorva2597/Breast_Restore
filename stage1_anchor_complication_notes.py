# stage1_anchor_complication_notes.py
# Purpose: Anchor note rows to Stage 1 date (stage1_date) within 0-365 days post-op.

from __future__ import print_function

import sys
import re
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
STAGING_CSV = "patient_recon_staging_refined.csv"
NOTE_ROWS_CSV = "patient_note_index.csv"   # <-- change if your note index file has a different name

OUT_ANCHOR_ROWS = "stage1_complication_anchor_rows.csv"
OUT_SUMMARY = "stage1_complication_anchor_summary.txt"


# -------------------------
# Robust CSV read (Py3.6 safe)
# -------------------------
def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def to_dt(series):
    return pd.to_datetime(series, errors="coerce")


def norm_colname(c):
    return str(c).strip().lower().replace(" ", "_")


def pick_first_present(cols, candidates):
    cset = set([c.lower() for c in cols])
    for cand in candidates:
        if cand.lower() in cset:
            for c in cols:
                if c.lower() == cand.lower():
                    return c
    return None


def auto_detect_event_dt_col(df):
    cols = list(df.columns)
    norm = {c: norm_colname(c) for c in cols}

    # Prefer EVENT_DT if present
    for c, n in norm.items():
        if n in ["event_dt", "event_date"]:
            return c

    # common alternatives
    for target in ["note_date_of_service", "operation_date", "admit_date", "hosp_admsn_time"]:
        for c, n in norm.items():
            if n == target:
                return c

    # fallback: any date-like col not delta/days/bin
    bad = ["delta", "days", "diff", "bin"]
    dateish = []
    for c, n in norm.items():
        if ("date" in n or "dt" in n or "time" in n) and (not any(b in n for b in bad)):
            dateish.append(c)

    return dateish[0] if dateish else None


def auto_detect_text_col(df):
    # optional, but nice to preserve if available
    cands = ["NOTE_SNIPPET", "NOTE_TEXT", "snippet", "SNIPPET", "note_text", "text"]
    return pick_first_present(df.columns.tolist(), cands)


def main():
    lines = []
    lines.append("=== Stage 1 Anchor Notes (0-365d) ===")
    lines.append("Staging file: {}".format(STAGING_CSV))
    lines.append("Note rows file: {}".format(NOTE_ROWS_CSV))
    lines.append("")

    # -------- load staging --------
    st = read_csv_safe(STAGING_CSV, dtype=object)
    if st is None or st.empty:
        raise RuntimeError("Could not read staging file or it is empty: {}".format(STAGING_CSV))

    if "patient_id" not in st.columns:
        raise RuntimeError("Staging file missing patient_id: {}".format(STAGING_CSV))

    if "stage1_date" not in st.columns:
        raise RuntimeError("Staging file missing stage1_date: {}".format(STAGING_CSV))

    st["patient_id"] = st["patient_id"].fillna("").astype(str).str.strip()
    st["STAGE1_DT"] = to_dt(st["stage1_date"])

    st = st[(st["patient_id"] != "") & (st["STAGE1_DT"].notnull())].copy()
    if st.empty:
        raise RuntimeError("No rows with non-null stage1_date found in {}".format(STAGING_CSV))

    stage1_map = dict(zip(st["patient_id"].tolist(), st["STAGE1_DT"].tolist()))
    stage1_patients = set(stage1_map.keys())

    lines.append("Patients with Stage1 date: {}".format(len(stage1_patients)))

    # -------- load note rows --------
    nr = read_csv_safe(NOTE_ROWS_CSV, dtype=object)
    if nr is None or nr.empty:
        raise RuntimeError("Could not read note rows file or it is empty: {}".format(NOTE_ROWS_CSV))

    # patient_id support
    if "patient_id" not in nr.columns:
        if "ENCRYPTED_PAT_ID" in nr.columns:
            nr = nr.rename(columns={"ENCRYPTED_PAT_ID": "patient_id"})
        else:
            raise RuntimeError("Note rows file must have patient_id or ENCRYPTED_PAT_ID: {}".format(NOTE_ROWS_CSV))

    event_col = auto_detect_event_dt_col(nr)
    if event_col is None:
        raise RuntimeError("Could not detect an EVENT_DT-like column in {}".format(NOTE_ROWS_CSV))

    text_col = auto_detect_text_col(nr)  # optional
    note_type_col = pick_first_present(nr.columns.tolist(), ["NOTE_TYPE", "note_type"])
    note_id_col = pick_first_present(nr.columns.tolist(), ["NOTE_ID", "note_id"])
    file_tag_col = pick_first_present(nr.columns.tolist(), ["file_tag", "FILE_TAG"])

    lines.append("Detected columns:")
    lines.append("  Event date column: {}".format(event_col))
    if text_col:
        lines.append("  Text column: {}".format(text_col))
    if note_type_col:
        lines.append("  NOTE_TYPE: {}".format(note_type_col))
    if note_id_col:
        lines.append("  NOTE_ID: {}".format(note_id_col))
    if file_tag_col:
        lines.append("  file_tag: {}".format(file_tag_col))
    lines.append("")

    pre_rows = len(nr)
    nr["patient_id"] = nr["patient_id"].fillna("").astype(str).str.strip()

    # keep only patients with stage1 date
    nr = nr[nr["patient_id"].isin(stage1_patients)].copy()
    kept_pat = len(nr)

    nr["STAGE1_DT"] = nr["patient_id"].map(stage1_map)
    nr["EVENT_DT"] = to_dt(nr[event_col])

    nr = nr[(nr["EVENT_DT"].notnull()) & (nr["STAGE1_DT"].notnull())].copy()
    nr["DELTA_DAYS_FROM_STAGE1"] = (nr["EVENT_DT"] - nr["STAGE1_DT"]).dt.days

    # enforce 0-365 window
    nr = nr[nr["DELTA_DAYS_FROM_STAGE1"].notnull()].copy()
    nr = nr[(nr["DELTA_DAYS_FROM_STAGE1"] >= 0) & (nr["DELTA_DAYS_FROM_STAGE1"] <= 365)].copy()
    kept_window = len(nr)

    # output columns
    out_cols = ["patient_id", "STAGE1_DT", "EVENT_DT", "DELTA_DAYS_FROM_STAGE1"]
    for c in [note_type_col, note_id_col, file_tag_col, text_col]:
        if c and c in nr.columns and c not in out_cols:
            out_cols.append(c)

    nr[out_cols].to_csv(OUT_ANCHOR_ROWS, index=False, encoding="utf-8")

    lines.append("Row counts:")
    lines.append("  Note rows scanned (all): {}".format(pre_rows))
    lines.append("  Rows after patient filter: {}".format(kept_pat))
    lines.append("  Rows kept in 0-365d window: {}".format(kept_window))
    lines.append("  Patients with >=1 kept row: {}".format(int(nr["patient_id"].nunique()) if kept_window else 0))
    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_ANCHOR_ROWS))
    lines.append("  - {}".format(OUT_SUMMARY))

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
