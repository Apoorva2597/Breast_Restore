# stage1_abstract_complications.py
# Purpose:
#   From Stage1-anchored rows (0-365d), produce:
#     1) Row-level complication hits (category + inferred treatment + minor/major)
#     2) Patient-level S1_Comp1..S1_Comp3 fields

from __future__ import print_function

import re
import sys
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
ANCHOR_ROWS_CSV = "stage1_anchor_rows_with_bins.csv"

OUT_ROW_HITS = "stage1_complications_row_hits.csv"
OUT_PATIENT_LEVEL = "stage1_complications_patient_level.csv"
OUT_SUMMARY = "stage1_complications_summary.txt"

COL_PATIENT = "patient_id"


def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def norm_text(x):
    if x is None:
        return ""
    s = str(x)
    try:
        s = s.replace(u"\xa0", " ")
    except Exception:
        pass
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def snippet(s, n=260):
    t = norm_text(s)
    return (t[:n] + "...") if len(t) > n else t


# -------------------------
# Complication category patterns (same set as your Stage 2 script)
# -------------------------
COMP_PATTERNS = [
    ("Hematoma", re.compile(r"\bhematoma\b", re.I)),
    ("Wound dehiscence", re.compile(r"\b(dehiscence|wound\s+dehisce|incision\s+dehisce|wound\s+separation)\b", re.I)),
    ("Wound infection", re.compile(r"\b(infection|infected|cellulitis|abscess|purulence|purulent|ssi|surgical\s+site\s+infection)\b", re.I)),
    ("Mastectomy skin flap necrosis", re.compile(r"\b(skin\s+flap\s+necrosis|mastectomy\s+flap\s+necrosis|msfn|flap\s+necrosis)\b", re.I)),
    ("Seroma", re.compile(r"\bseroma\b", re.I)),
    ("Capsular contracture", re.compile(r"\b(capsular\s+contracture)\b", re.I)),
    ("Implant malposition", re.compile(r"\b(implant\s+malposition|malposition)\b", re.I)),
    ("Implant rupture/leak/deflation", re.compile(r"\b(implant\s+(rupture|ruptured|leak|leakage|deflation|deflated))\b|\b(ruptured\s+implant|leaking\s+implant)\b", re.I)),
    ("Implant/expander extrusion", re.compile(r"\b(extrusion|exposed\s+(implant|expander)|implant\s+exposure|expander\s+exposure)\b", re.I)),
    ("Acute partial flap necrosis", re.compile(r"\b(partial\s+flap\s+necrosis)\b", re.I)),
    ("Total flap loss", re.compile(r"\b(total\s+flap\s+loss|flap\s+loss)\b", re.I)),
]

OTHER_SYSTEMIC_RE = re.compile(r"\b(pulmonary\s+embolism|pe\b|deep\s+vein\s+thrombosis|dvt\b|pneumonia|sepsis)\b", re.I)


# -------------------------
# Treatment inference (same as Stage 2)
# -------------------------
RX_REOP = re.compile(
    r"\b(return(ed)?\s+to\s+or|take\s*back|takeback|re-?operation|reop|washout|operative\s+debridement|"
    r"incision\s+and\s+drainage|i\s*&\s*d|explant|explanted|implant\s+removal|remove(d)?\s+implant|"
    r"expander\s+removal|remove(d)?\s+expander|capsulectomy|capsulotomy)\b",
    re.I
)

RX_REHOSP = re.compile(
    r"\b(readmit(ted)?|re-?admit(ted)?|hospitaliz(ed|ation)|inpatient|admitted|admission|"
    r"presented\s+to\s+ed|emergency\s+department)\b",
    re.I
)

RX_NONOP = re.compile(
    r"\b(oral\s+antibiotic|iv\s+antibiotic|antibiotic(s)?|augmentin|keflex|clindamycin|vancomycin|"
    r"drainage|aspiration|tap(ped)?|percutaneous|ir\s+drain|wound\s+care|dressing\s+changes|packing|"
    r"topical|clinic\s+follow-?up)\b",
    re.I
)

RX_NOTX = re.compile(
    r"\b(no\s+treatment|no\s+intervention|observe|observation|monitor|watchful\s+waiting)\b",
    re.I
)

def infer_treatment_bucket(text):
    t = norm_text(text)
    if RX_REOP.search(t):
        return "REOPERATION"
    if RX_REHOSP.search(t):
        return "REHOSPITALIZATION"
    if RX_NONOP.search(t):
        return "NON-OPERATIVE"
    if RX_NOTX.search(t):
        return "NO TREATMENT"
    return "UNKNOWN"

def major_minor_from_treatment(bucket):
    if bucket in ["REOPERATION", "REHOSPITALIZATION"]:
        return "MAJOR"
    if bucket in ["NON-OPERATIVE", "NO TREATMENT"]:
        return "MINOR"
    return "UNKNOWN"


# -------------------------
# Main
# -------------------------
def main():
    df = read_csv_safe(ANCHOR_ROWS_CSV, dtype=object)
    if df is None or df.empty:
        raise RuntimeError("Could not read or empty: {}".format(ANCHOR_ROWS_CSV))

    # Required columns
    need_cols = ["patient_id", "STAGE1_DT", "EVENT_DT", "DELTA_DAYS_FROM_STAGE1"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError("Missing required column '{}' in {}".format(c, ANCHOR_ROWS_CSV))

    # Optional columns (based on your Stage1 anchor output)
    text_col = None
    for c in ["note_text", "NOTE_TEXT", "NOTE_SNIPPET", "snippet", "SNIPPET", "TEXT"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise RuntimeError("Could not find note text column (expected note_text or similar) in {}".format(ANCHOR_ROWS_CSV))

    note_type_col = "note_type" if "note_type" in df.columns else ("NOTE_TYPE" if "NOTE_TYPE" in df.columns else None)
    note_id_col = "note_id" if "note_id" in df.columns else ("NOTE_ID" if "NOTE_ID" in df.columns else None)

    # Parse dates
    df["STAGE1_DT"] = to_dt(df["STAGE1_DT"])
    df["EVENT_DT"] = to_dt(df["EVENT_DT"])
    df["DELTA_DAYS_FROM_STAGE1"] = pd.to_numeric(df["DELTA_DAYS_FROM_STAGE1"], errors="coerce")

    # Safety filter (should already be true)
    df = df[df["DELTA_DAYS_FROM_STAGE1"].notnull()].copy()
    df = df[(df["DELTA_DAYS_FROM_STAGE1"] >= 0) & (df["DELTA_DAYS_FROM_STAGE1"] <= 365)].copy()

    total_rows = int(len(df))
    n_patients = int(df["patient_id"].nunique())

    # Normalize text once
    df["_TEXT_NORM"] = df[text_col].fillna("").apply(norm_text)

    hits = []
    patients_with_hit = set()

    for _, r in df.iterrows():
        pid = str(r.get("patient_id", "")).strip()
        if not pid:
            continue

        txt = r.get("_TEXT_NORM", "")
        if not txt:
            continue

        found_any = False

        # Category hits
        for comp_name, comp_re in COMP_PATTERNS:
            if comp_re.search(txt):
                found_any = True
                bucket = infer_treatment_bucket(txt)
                mm = major_minor_from_treatment(bucket)

                hits.append({
                    "patient_id": pid,
                    "STAGE1_DT": r.get("STAGE1_DT", None),
                    "EVENT_DT": r.get("EVENT_DT", None),
                    "DELTA_DAYS_FROM_STAGE1": r.get("DELTA_DAYS_FROM_STAGE1", None),
                    "TIME_BIN_STAGE1": r.get("TIME_BIN_STAGE1", ""),
                    "NOTE_TYPE": r.get(note_type_col, "") if note_type_col else "",
                    "NOTE_ID": r.get(note_id_col, "") if note_id_col else "",
                    "complication": comp_name,
                    "treatment_bucket": bucket,
                    "comp_classification": mm,
                    "snippet": snippet(r.get(text_col, ""), 260),
                })

        # Other systemic
        if OTHER_SYSTEMIC_RE.search(txt):
            found_any = True
            bucket = infer_treatment_bucket(txt)
            mm = major_minor_from_treatment(bucket)

            hits.append({
                "patient_id": pid,
                "STAGE1_DT": r.get("STAGE1_DT", None),
                "EVENT_DT": r.get("EVENT_DT", None),
                "DELTA_DAYS_FROM_STAGE1": r.get("DELTA_DAYS_FROM_STAGE1", None),
                "TIME_BIN_STAGE1": r.get("TIME_BIN_STAGE1", ""),
                "NOTE_TYPE": r.get(note_type_col, "") if note_type_col else "",
                "NOTE_ID": r.get(note_id_col, "") if note_id_col else "",
                "complication": "Other (systemic)",
                "treatment_bucket": bucket,
                "comp_classification": mm,
                "snippet": snippet(r.get(text_col, ""), 260),
            })

        if found_any:
            patients_with_hit.add(pid)

    # Write row hits
    if hits:
        hits_df = pd.DataFrame(hits)
        hits_df = hits_df.sort_values(by=["patient_id", "EVENT_DT", "complication"], ascending=[True, True, True])
    else:
        hits_df = pd.DataFrame(columns=[
            "patient_id","STAGE1_DT","EVENT_DT","DELTA_DAYS_FROM_STAGE1","TIME_BIN_STAGE1",
            "NOTE_TYPE","NOTE_ID","complication","treatment_bucket","comp_classification","snippet"
        ])

    hits_df.to_csv(OUT_ROW_HITS, index=False, encoding="utf-8")

    # Patient-level Comp1..3 (chronological, de-duped)
    # Build base patient list from df (not only hit patients)
    base = df[["patient_id"]].drop_duplicates().copy()
    base["stage1_dt"] = base["patient_id"].map(df.groupby("patient_id")["STAGE1_DT"].first().to_dict())

    patient_rows = []

    if not hits_df.empty:
        for pid, g in hits_df.groupby("patient_id"):
            g2 = g.sort_values(by=["EVENT_DT", "complication"], ascending=[True, True]).copy()
            g2["dedup_key"] = g2["EVENT_DT"].astype(str) + "||" + g2["complication"].astype(str)
            g2 = g2.drop_duplicates(subset=["dedup_key"], keep="first")

            comps = g2.head(3).to_dict("records")

            row = {"patient_id": pid}
            row["stage1_dt"] = base.loc[base["patient_id"] == pid, "stage1_dt"].iloc[0] if (base["patient_id"] == pid).any() else None

            for i in range(3):
                idx = i + 1
                if i < len(comps):
                    c = comps[i]
                    row["S1_Comp{}_Date".format(idx)] = c.get("EVENT_DT", None)
                    row["S1_Comp{}".format(idx)] = c.get("complication", "")
                    row["S1_Comp{}_Treatment".format(idx)] = c.get("treatment_bucket", "")
                    row["S1_Comp{}_Classification".format(idx)] = c.get("comp_classification", "")
                    row["S1_Comp{}_NoteType".format(idx)] = c.get("NOTE_TYPE", "")
                    row["S1_Comp{}_NoteID".format(idx)] = c.get("NOTE_ID", "")
                else:
                    row["S1_Comp{}_Date".format(idx)] = None
                    row["S1_Comp{}".format(idx)] = ""
                    row["S1_Comp{}_Treatment".format(idx)] = ""
                    row["S1_Comp{}_Classification".format(idx)] = ""
                    row["S1_Comp{}_NoteType".format(idx)] = ""
                    row["S1_Comp{}_NoteID".format(idx)] = ""

            patient_rows.append(row)

    pl = base.copy()

    if patient_rows:
        pl_hits = pd.DataFrame(patient_rows)
        pl = pl.merge(pl_hits, on=["patient_id", "stage1_dt"], how="left")
    else:
        # Create empty columns
        for idx in [1, 2, 3]:
            pl["S1_Comp{}_Date".format(idx)] = None
            pl["S1_Comp{}".format(idx)] = ""
            pl["S1_Comp{}_Treatment".format(idx)] = ""
            pl["S1_Comp{}_Classification".format(idx)] = ""
            pl["S1_Comp{}_NoteType".format(idx)] = ""
            pl["S1_Comp{}_NoteID".format(idx)] = ""

    pl = pl.sort_values(by=["patient_id"], ascending=[True])
    pl.to_csv(OUT_PATIENT_LEVEL, index=False, encoding="utf-8")

    # Summary
    n_hit_pat = len(patients_with_hit)
    n_hit_rows = int(len(hits_df)) if not hits_df.empty else 0

    comp_counts = hits_df["complication"].value_counts() if not hits_df.empty else pd.Series(dtype=int)
    treat_counts = hits_df["treatment_bucket"].value_counts() if not hits_df.empty else pd.Series(dtype=int)

    lines = []
    lines.append("=== Stage1 Complication Abstraction (anchored rows -> S1_Comp1..3) ===")
    lines.append("Input: {}".format(ANCHOR_ROWS_CSV))
    lines.append("")
    lines.append("Anchored rows (0-365d) scanned: {}".format(total_rows))
    lines.append("Unique patients in anchor rows: {}".format(n_patients))
    lines.append("Row-level complication hits: {}".format(n_hit_rows))
    lines.append("Patients with >=1 hit: {} ({:.1f}%)".format(n_hit_pat, (100.0 * n_hit_pat / n_patients) if n_patients else 0.0))
    lines.append("")
    if not hits_df.empty:
        lines.append("Top complication categories (top 15):")
        for k, v in comp_counts.head(15).items():
            lines.append("  {:>6}  {}".format(int(v), k))
        lines.append("")
        lines.append("Treatment buckets:")
        for k, v in treat_counts.items():
            lines.append("  {:>6}  {}".format(int(v), k))
    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_ROW_HITS))
    lines.append("  - {}".format(OUT_PATIENT_LEVEL))
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
