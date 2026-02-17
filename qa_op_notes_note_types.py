# qa_op_notes_note_types.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   1) Show distribution of NOTE_TYPE in Operation Notes.csv
#   2) For each NOTE_TYPE, count Stage2-like evidence hits (TE->implant exchange patterns)
#   3) Output:
#       - qa_op_notes_note_types_summary.txt
#       - qa_op_notes_note_types_counts.csv
#       - qa_op_notes_stage2_hits_by_type.csv
#
# Notes:
#   - Reads with latin1(errors=replace) to avoid UnicodeDecodeError 0xA0
#   - Designed to run on WVD after you git pull
#
# Fix:
#   - DO NOT use iter_csv_safe(..., nrows=5) because nrows returns a DataFrame,
#     and iterating over it yields column names (strings). Use read_csv_safe for header probing.

from __future__ import print_function

import re
import sys
import pandas as pd

# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
OP_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"

OUT_SUMMARY_TXT = "qa_op_notes_note_types_summary.txt"
OUT_NOTE_TYPE_COUNTS_CSV = "qa_op_notes_note_types_counts.csv"
OUT_STAGE2_HITS_BY_TYPE_CSV = "qa_op_notes_stage2_hits_by_type.csv"

CHUNKSIZE = 120000

COL_PAT = "ENCRYPTED_PAT_ID"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_ID = "NOTE_ID"
COL_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"

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
# Stage2-like evidence regex (same spirit as your extractor)
# -------------------------
RX_STAGE2 = re.compile(
    r"("
    r"\bexchange\b.{0,120}\b(tissue\s*expander|expander|\bte\b)\b.{0,260}\b(implant|permanent\s+implant|implnt)\b"
    r"|"
    r"\b(remove|removed|explant|explanted)\b.{0,220}\b(tissue\s*expander|expander|\bte\b)\b.{0,520}\b(place|placed|insert|inserted|insertion|implantation)\b.{0,180}\b(implant|permanent\s+implant|implnt)\b"
    r")",
    re.I
)

# A looser “context” signal (expander + implant in same note)
RX_CONTEXT = re.compile(
    r"\b(tissue\s*expander|expander|\bte\b)\b.*\b(implant|permanent\s+implant|implnt)\b",
    re.I
)


def main():
    # -------------------------
    # Validate columns exist quickly (header probe)
    # -------------------------
    head = read_csv_safe(OP_NOTES_CSV, nrows=5)
    if head is None or head.empty:
        raise RuntimeError("Could not read: {}".format(OP_NOTES_CSV))

    required = [COL_NOTE_TYPE, COL_NOTE_TEXT]
    for c in required:
        if c not in head.columns:
            raise RuntimeError("Missing required column '{}' in {}".format(c, OP_NOTES_CSV))

    usecols = []
    for c in [COL_PAT, COL_NOTE_TYPE, COL_NOTE_TEXT, COL_NOTE_ID, COL_DOS, COL_OP_DATE]:
        if c in head.columns:
            usecols.append(c)

    # -------------------------
    # Aggregators
    # -------------------------
    note_type_counts = {}  # NOTE_TYPE -> rows
    per_type = {}          # NOTE_TYPE -> dict with counts + sets

    total_rows = 0
    total_rows_with_type = 0
    total_stage2_rows = 0
    total_ctx_rows = 0

    # -------------------------
    # Stream file
    # -------------------------
    for chunk in iter_csv_safe(OP_NOTES_CSV, usecols=usecols, chunksize=CHUNKSIZE):
        total_rows += len(chunk)

        types = chunk[COL_NOTE_TYPE] if COL_NOTE_TYPE in chunk.columns else pd.Series(["NA"] * len(chunk))
        types = types.fillna("NA").apply(norm_type)

        # count NOTE_TYPE distribution
        for t in types.tolist():
            note_type_counts[t] = note_type_counts.get(t, 0) + 1
        total_rows_with_type += int((types != "NA").sum())

        texts = chunk[COL_NOTE_TEXT] if COL_NOTE_TEXT in chunk.columns else pd.Series([""] * len(chunk))
        texts = texts.fillna("").apply(norm_text)

        pats = chunk[COL_PAT] if COL_PAT in chunk.columns else pd.Series([""] * len(chunk))
        pats = pats.fillna("").astype(str)

        # row-level hits
        stage2_hit = texts.str.contains(RX_STAGE2, regex=True)
        ctx_hit = texts.str.contains(RX_CONTEXT, regex=True)

        total_stage2_rows += int(stage2_hit.sum())
        total_ctx_rows += int(ctx_hit.sum())

        # aggregate by type
        # (loop is fine at this chunksize; keeps logic explicit for Python 3.6)
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
    # Write NOTE_TYPE counts
    # -------------------------
    nt_df = pd.DataFrame([{"NOTE_TYPE": k, "ROWS": v} for k, v in note_type_counts.items()])
    if not nt_df.empty:
        nt_df = nt_df.sort_values(by="ROWS", ascending=False)
    nt_df.to_csv(OUT_NOTE_TYPE_COUNTS_CSV, index=False, encoding="utf-8")

    # -------------------------
    # Write stage2 hits by type
    # -------------------------
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
    hits_df.to_csv(OUT_STAGE2_HITS_BY_TYPE_CSV, index=False, encoding="utf-8")

    # -------------------------
    # Summary text
    # -------------------------
    lines = []
    lines.append("=== QA: Operation Notes NOTE_TYPE coverage + Stage2 phrase hits ===")
    lines.append("File: {}".format(OP_NOTES_CSV))
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
    lines.append("  - {}".format(OUT_SUMMARY_TXT))
    lines.append("  - {}".format(OUT_NOTE_TYPE_COUNTS_CSV))
    lines.append("  - {}".format(OUT_STAGE2_HITS_BY_TYPE_CSV))

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
