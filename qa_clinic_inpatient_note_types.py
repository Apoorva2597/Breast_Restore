# qa_inpatient_clinic_note_types.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Run the SAME NOTE_TYPE distribution + Stage2-like phrase-hit QA that you ran on
#   Operation Notes, but for:
#     - Clinic Notes.csv
#     - Inpatient Notes.csv
#
# Outputs (all UTF-8):
#   1) qa_clinic_note_types_summary.txt
#   2) qa_clinic_note_types_counts.csv
#   3) qa_clinic_stage2_hits_by_type.csv
#   4) qa_inpatient_note_types_summary.txt
#   5) qa_inpatient_note_types_counts.csv
#   6) qa_inpatient_stage2_hits_by_type.csv
#   7) qa_all_nonop_note_types_summary.txt   (combined rollup across clinic+inpatient)
#   8) qa_all_nonop_note_types_counts.csv
#   9) qa_all_nonop_stage2_hits_by_type.csv
#
# Notes:
#   - Reads with latin1(errors=replace) to avoid UnicodeDecodeError (e.g., 0xA0 NBSP)
#   - Designed to run on WVD after you git pull
#
# Fix for your error:
#   The earlier "str object has no attribute columns" happens when code tries to get
#   header columns by iterating over a DataFrame (iterating yields strings = column names).
#   Here we ALWAYS use read_csv_safe(..., nrows=5) for header probing.

from __future__ import print_function

import re
import sys
import pandas as pd

# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
CLINIC_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"
INPATIENT_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Notes.csv"

# Column names (must match the CSV headers)
COL_PAT = "ENCRYPTED_PAT_ID"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_ID = "NOTE_ID"
COL_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"  # may not exist in clinic/inpatient; script handles if missing

CHUNKSIZE = 120000

# -------------------------
# Robust readers (Python 3.6 safe)
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


def iter_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        reader = pd.read_csv(f, engine="python", **kwargs)
        for chunk in reader:
            yield chunk
    finally:
        try:
            f.close()
        except Exception:
            pass


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


def norm_type(x):
    if x is None:
        return "NA"
    s = str(x).strip()
    if not s:
        return "NA"
    return s.upper()


# -------------------------
# Stage2-like evidence regex (strict + context)
# -------------------------
RX_STAGE2 = re.compile(
    r"("
    r"\bexchange\b.{0,120}\b(tissue\s*expander|expander|\bte\b)\b.{0,260}\b(implant|permanent\s+implant|implnt)\b"
    r"|"
    r"\b(remove|removed|explant|explanted)\b.{0,220}\b(tissue\s*expander|expander|\bte\b)\b.{0,520}\b(place|placed|insert|inserted|insertion|implantation)\b.{0,180}\b(implant|permanent\s+implant|implnt)\b"
    r")",
    re.I
)

RX_CONTEXT = re.compile(
    r"\b(tissue\s*expander|expander|\bte\b)\b.*\b(implant|permanent\s+implant|implnt)\b",
    re.I
)


def analyze_file(in_csv, out_prefix):
    """
    Analyze one file and write:
      - <out_prefix>_summary.txt
      - <out_prefix>_note_types_counts.csv
      - <out_prefix>_stage2_hits_by_type.csv

    Returns dict with rollup artifacts for optional combination:
      {
        "note_type_counts": dict,
        "per_type": dict (with sets),
        "total_rows": int,
        "total_rows_with_type": int,
        "total_stage2_rows": int,
        "total_ctx_rows": int
      }
    """
    out_summary = "{}_summary.txt".format(out_prefix)
    out_counts = "{}_note_types_counts.csv".format(out_prefix)
    out_hits = "{}_stage2_hits_by_type.csv".format(out_prefix)

    # -------------------------
    # Header probe (SAFE)
    # -------------------------
    head = read_csv_safe(in_csv, nrows=5)
    if head is None or head.empty:
        raise RuntimeError("Could not read: {}".format(in_csv))

    # Validate required columns
    required = [COL_NOTE_TYPE, COL_NOTE_TEXT]
    for c in required:
        if c not in head.columns:
            raise RuntimeError("Missing required column '{}' in {}".format(c, in_csv))

    # Use only columns that exist
    usecols = []
    for c in [COL_PAT, COL_NOTE_TYPE, COL_NOTE_TEXT, COL_NOTE_ID, COL_DOS, COL_OP_DATE]:
        if c in head.columns:
            usecols.append(c)

    # -------------------------
    # Aggregators
    # -------------------------
    note_type_counts = {}
    per_type = {}

    total_rows = 0
    total_rows_with_type = 0
    total_stage2_rows = 0
    total_ctx_rows = 0

    # -------------------------
    # Stream file
    # -------------------------
    for chunk in iter_csv_safe(in_csv, usecols=usecols, chunksize=CHUNKSIZE):
        total_rows += len(chunk)

        types = chunk[COL_NOTE_TYPE] if COL_NOTE_TYPE in chunk.columns else pd.Series(["NA"] * len(chunk))
        types = types.fillna("NA").apply(norm_type)

        for t in types.tolist():
            note_type_counts[t] = note_type_counts.get(t, 0) + 1
        total_rows_with_type += int((types != "NA").sum())

        texts = chunk[COL_NOTE_TEXT] if COL_NOTE_TEXT in chunk.columns else pd.Series([""] * len(chunk))
        texts = texts.fillna("").apply(norm_text)

        pats = chunk[COL_PAT] if COL_PAT in chunk.columns else pd.Series([""] * len(chunk))
        pats = pats.fillna("").astype(str)

        stage2_hit = texts.str.contains(RX_STAGE2, regex=True)
        ctx_hit = texts.str.contains(RX_CONTEXT, regex=True)

        total_stage2_rows += int(stage2_hit.sum())
        total_ctx_rows += int(ctx_hit.sum())

        for i in range(len(chunk)):
            t = types.iat[i]
            pid = pats.iat[i]
            st2 = bool(stage2_hit.iat[i])
            ctx = bool(ctx_hit.iat[i])

            if t not in per_type:
                per_type[t] = {
                    "rows": 0,
                    "stage2_rows": 0,
                    "context_rows": 0,
                    "unique_patients_any": set(),
                    "unique_patients_stage2": set(),
                    "unique_patients_context": set(),
                }

            d = per_type[t]
            d["rows"] += 1

            if pid:
                d["unique_patients_any"].add(pid)

            if ctx:
                d["context_rows"] += 1
                if pid:
                    d["unique_patients_context"].add(pid)

            if st2:
                d["stage2_rows"] += 1
                if pid:
                    d["unique_patients_stage2"].add(pid)

    # -------------------------
    # Write outputs
    # -------------------------
    nt_df = pd.DataFrame([{"NOTE_TYPE": k, "ROWS": v} for k, v in note_type_counts.items()])
    if not nt_df.empty:
        nt_df = nt_df.sort_values(by="ROWS", ascending=False)
    nt_df.to_csv(out_counts, index=False, encoding="utf-8")

    rows = []
    for t, d in per_type.items():
        rows.append({
            "NOTE_TYPE": t,
            "ROWS": d["rows"],
            "STAGE2_ROWS_STRICT": d["stage2_rows"],
            "CONTEXT_ROWS_EXPANDER+IMPLANT": d["context_rows"],
            "UNIQUE_PATIENTS_ANY": len(d["unique_patients_any"]),
            "UNIQUE_PATIENTS_STAGE2_STRICT": len(d["unique_patients_stage2"]),
            "UNIQUE_PATIENTS_CONTEXT": len(d["unique_patients_context"]),
        })
    hits_df = pd.DataFrame(rows)
    if not hits_df.empty:
        hits_df = hits_df.sort_values(
            by=["UNIQUE_PATIENTS_STAGE2_STRICT", "STAGE2_ROWS_STRICT", "UNIQUE_PATIENTS_CONTEXT"],
            ascending=[False, False, False]
        )
    hits_df.to_csv(out_hits, index=False, encoding="utf-8")

    # Summary text
    lines = []
    lines.append("=== QA: NOTE_TYPE coverage + Stage2 phrase hits ===")
    lines.append("File: {}".format(in_csv))
    lines.append("Read encoding: latin1(errors=replace) | Python 3.6.8 compatible")
    lines.append("")
    lines.append("Total rows scanned: {}".format(total_rows))
    lines.append("Rows with non-empty NOTE_TYPE: {}".format(total_rows_with_type))
    lines.append("Total strict Stage2-hit rows (all NOTE_TYPEs): {}".format(total_stage2_rows))
    lines.append("Total context rows (expander+implant anywhere in note): {}".format(total_ctx_rows))
    lines.append("")

    lines.append("Top NOTE_TYPEs by row count (top 25):")
    top25 = nt_df.head(25) if not nt_df.empty else pd.DataFrame()
    if not top25.empty:
        for _, r in top25.iterrows():
            lines.append("  {:>7}  {}".format(int(r["ROWS"]), r["NOTE_TYPE"]))
    else:
        lines.append("  (no rows)")

    lines.append("")
    lines.append("Top NOTE_TYPEs by UNIQUE_PATIENTS_STAGE2_STRICT (top 25):")
    top_hits = hits_df.head(25) if not hits_df.empty else pd.DataFrame()
    if not top_hits.empty:
        for _, r in top_hits.iterrows():
            lines.append("  {:>6} patients | {:>6} rows | {}".format(
                int(r["UNIQUE_PATIENTS_STAGE2_STRICT"]),
                int(r["STAGE2_ROWS_STRICT"]),
                r["NOTE_TYPE"]
            ))
    else:
        lines.append("  (no stage2 hits)")

    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(out_summary))
    lines.append("  - {}".format(out_counts))
    lines.append("  - {}".format(out_hits))

    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))

    return {
        "note_type_counts": note_type_counts,
        "per_type": per_type,
        "total_rows": total_rows,
        "total_rows_with_type": total_rows_with_type,
        "total_stage2_rows": total_stage2_rows,
        "total_ctx_rows": total_ctx_rows,
        "out_summary": out_summary,
        "out_counts": out_counts,
        "out_hits": out_hits,
    }


def merge_rollups(r1, r2):
    """
    Merge two rollups from analyze_file into one combined rollup.
    """
    out = {
        "note_type_counts": {},
        "per_type": {},
        "total_rows": 0,
        "total_rows_with_type": 0,
        "total_stage2_rows": 0,
        "total_ctx_rows": 0,
    }

    for r in [r1, r2]:
        out["total_rows"] += int(r.get("total_rows", 0))
        out["total_rows_with_type"] += int(r.get("total_rows_with_type", 0))
        out["total_stage2_rows"] += int(r.get("total_stage2_rows", 0))
        out["total_ctx_rows"] += int(r.get("total_ctx_rows", 0))

        for t, c in r.get("note_type_counts", {}).items():
            out["note_type_counts"][t] = out["note_type_counts"].get(t, 0) + int(c)

        for t, d in r.get("per_type", {}).items():
            if t not in out["per_type"]:
                out["per_type"][t] = {
                    "rows": 0,
                    "stage2_rows": 0,
                    "context_rows": 0,
                    "unique_patients_any": set(),
                    "unique_patients_stage2": set(),
                    "unique_patients_context": set(),
                }
            od = out["per_type"][t]
            od["rows"] += int(d.get("rows", 0))
            od["stage2_rows"] += int(d.get("stage2_rows", 0))
            od["context_rows"] += int(d.get("context_rows", 0))
            od["unique_patients_any"].update(d.get("unique_patients_any", set()))
            od["unique_patients_stage2"].update(d.get("unique_patients_stage2", set()))
            od["unique_patients_context"].update(d.get("unique_patients_context", set()))

    return out


def write_combined_rollup(rollup, out_prefix, label):
    out_summary = "{}_summary.txt".format(out_prefix)
    out_counts = "{}_note_types_counts.csv".format(out_prefix)
    out_hits = "{}_stage2_hits_by_type.csv".format(out_prefix)

    nt_df = pd.DataFrame([{"NOTE_TYPE": k, "ROWS": v} for k, v in rollup["note_type_counts"].items()])
    if not nt_df.empty:
        nt_df = nt_df.sort_values(by="ROWS", ascending=False)
    nt_df.to_csv(out_counts, index=False, encoding="utf-8")

    rows = []
    for t, d in rollup["per_type"].items():
        rows.append({
            "NOTE_TYPE": t,
            "ROWS": d["rows"],
            "STAGE2_ROWS_STRICT": d["stage2_rows"],
            "CONTEXT_ROWS_EXPANDER+IMPLANT": d["context_rows"],
            "UNIQUE_PATIENTS_ANY": len(d["unique_patients_any"]),
            "UNIQUE_PATIENTS_STAGE2_STRICT": len(d["unique_patients_stage2"]),
            "UNIQUE_PATIENTS_CONTEXT": len(d["unique_patients_context"]),
        })
    hits_df = pd.DataFrame(rows)
    if not hits_df.empty:
        hits_df = hits_df.sort_values(
            by=["UNIQUE_PATIENTS_STAGE2_STRICT", "STAGE2_ROWS_STRICT", "UNIQUE_PATIENTS_CONTEXT"],
            ascending=[False, False, False]
        )
    hits_df.to_csv(out_hits, index=False, encoding="utf-8")

    lines = []
    lines.append("=== QA: NOTE_TYPE coverage + Stage2 phrase hits (COMBINED) ===")
    lines.append(label)
    lines.append("Read encoding: latin1(errors=replace) | Python 3.6.8 compatible")
    lines.append("")
    lines.append("Total rows scanned: {}".format(rollup["total_rows"]))
    lines.append("Rows with non-empty NOTE_TYPE: {}".format(rollup["total_rows_with_type"]))
    lines.append("Total strict Stage2-hit rows (all NOTE_TYPEs): {}".format(rollup["total_stage2_rows"]))
    lines.append("Total context rows (expander+implant anywhere in note): {}".format(rollup["total_ctx_rows"]))
    lines.append("")

    lines.append("Top NOTE_TYPEs by row count (top 25):")
    top25 = nt_df.head(25) if not nt_df.empty else pd.DataFrame()
    if not top25.empty:
        for _, r in top25.iterrows():
            lines.append("  {:>7}  {}".format(int(r["ROWS"]), r["NOTE_TYPE"]))
    else:
        lines.append("  (no rows)")

    lines.append("")
    lines.append("Top NOTE_TYPEs by UNIQUE_PATIENTS_STAGE2_STRICT (top 25):")
    top_hits = hits_df.head(25) if not hits_df.empty else pd.DataFrame()
    if not top_hits.empty:
        for _, r in top_hits.iterrows():
            lines.append("  {:>6} patients | {:>6} rows | {}".format(
                int(r["UNIQUE_PATIENTS_STAGE2_STRICT"]),
                int(r["STAGE2_ROWS_STRICT"]),
                r["NOTE_TYPE"]
            ))
    else:
        lines.append("  (no stage2 hits)")

    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(out_summary))
    lines.append("  - {}".format(out_counts))
    lines.append("  - {}".format(out_hits))

    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


def main():
    # Analyze clinic
    clinic = analyze_file(CLINIC_NOTES_CSV, out_prefix="qa_clinic_note_types")

    # Analyze inpatient
    inpatient = analyze_file(INPATIENT_NOTES_CSV, out_prefix="qa_inpatient_note_types")

    # Combined rollup (clinic + inpatient)
    combined = merge_rollups(clinic, inpatient)
    write_combined_rollup(
        combined,
        out_prefix="qa_all_nonop_note_types",
        label="Combined sources: Clinic Notes + Inpatient Notes"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
