# qa_systemic_impact.py
# Python 3.6.8+ (pandas required)
#
# Goal:
#   Measure how much patient-level signal is driven by "Other (systemic)"
#   for Stage 1 and Stage 2 (AB) complication row-hits files.
#
# Inputs:
#   - stage1_complications_row_hits.csv
#   - stage2_ab_complications_row_hits.csv
#
# Outputs:
#   - qa_systemic_impact_patient_level.csv
#   - qa_systemic_impact_summary.txt

from __future__ import print_function

import sys
import pandas as pd


STAGE1_ROWS = "stage1_complications_row_hits.csv"
STAGE2_ROWS = "stage2_ab_complications_row_hits.csv"

OUT_PATIENT_LEVEL = "qa_systemic_impact_patient_level.csv"
OUT_SUMMARY = "qa_systemic_impact_summary.txt"

COL_PATIENT = "patient_id"
COL_COMP = "complication"

SYSTEMIC_LABEL = "Other (systemic)"


def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def summarize_one(label, df, lines):
    # Basic sanity
    if df is None or df.empty:
        lines.append("=== {} ===".format(label))
        lines.append("File empty or not found rows.")
        lines.append("")
        return None

    if COL_PATIENT not in df.columns or COL_COMP not in df.columns:
        raise RuntimeError("Missing required columns in {}: need '{}' and '{}'".format(label, COL_PATIENT, COL_COMP))

    x = df[[COL_PATIENT, COL_COMP]].copy()
    x[COL_PATIENT] = x[COL_PATIENT].fillna("").astype(str)
    x[COL_COMP] = x[COL_COMP].fillna("").astype(str)

    # Drop blank patient ids
    x = x[x[COL_PATIENT] != ""].copy()

    # Patient-level flags
    g = x.groupby(COL_PATIENT)[COL_COMP].apply(list).reset_index(name="comp_list")

    def has_systemic(comp_list):
        return SYSTEMIC_LABEL in set(comp_list)

    def has_specific(comp_list):
        s = set(comp_list)
        if SYSTEMIC_LABEL in s:
            s.remove(SYSTEMIC_LABEL)
        return len(s) > 0

    g["has_systemic"] = g["comp_list"].apply(has_systemic).astype(int)
    g["has_specific"] = g["comp_list"].apply(has_specific).astype(int)

    g["systemic_only"] = ((g["has_systemic"] == 1) & (g["has_specific"] == 0)).astype(int)
    g["specific_only"] = ((g["has_systemic"] == 0) & (g["has_specific"] == 1)).astype(int)
    g["both"] = ((g["has_systemic"] == 1) & (g["has_specific"] == 1)).astype(int)

    # Counts
    n_pat_any = int(g.shape[0])
    n_sys_only = int(g["systemic_only"].sum())
    n_spec_only = int(g["specific_only"].sum())
    n_both = int(g["both"].sum())
    n_with_systemic = int(g["has_systemic"].sum())
    n_with_specific = int(g["has_specific"].sum())

    # Row-level counts too (context)
    total_rows = int(x.shape[0])
    rows_systemic = int((x[COL_COMP] == SYSTEMIC_LABEL).sum())
    rows_specific = int(total_rows - rows_systemic)

    # Top specific categories (row-level) excluding systemic
    top_specific = (
        x[x[COL_COMP] != SYSTEMIC_LABEL][COL_COMP]
        .value_counts()
        .head(15)
    )

    # Summary text
    lines.append("=== {} ===".format(label))
    lines.append("Row-level totals:")
    lines.append("  Total rows: {}".format(total_rows))
    lines.append("  Systemic rows ('{}'): {} ({:.1f}%)".format(
        SYSTEMIC_LABEL, rows_systemic, (100.0 * rows_systemic / total_rows) if total_rows else 0.0
    ))
    lines.append("  Specific rows (everything else): {} ({:.1f}%)".format(
        rows_specific, (100.0 * rows_specific / total_rows) if total_rows else 0.0
    ))
    lines.append("")
    lines.append("Patient-level totals (patients with >=1 row hit): {}".format(n_pat_any))
    lines.append("  Systemic-only patients: {} ({:.1f}%)".format(
        n_sys_only, (100.0 * n_sys_only / n_pat_any) if n_pat_any else 0.0
    ))
    lines.append("  Specific-only patients: {} ({:.1f}%)".format(
        n_spec_only, (100.0 * n_spec_only / n_pat_any) if n_pat_any else 0.0
    ))
    lines.append("  Both systemic + specific: {} ({:.1f}%)".format(
        n_both, (100.0 * n_both / n_pat_any) if n_pat_any else 0.0
    ))
    lines.append("")
    lines.append("If you removed ONLY systemic ('{}') rows:".format(SYSTEMIC_LABEL))
    lines.append("  Patients that would still have >=1 hit (specific present): {} ({:.1f}%)".format(
        n_with_specific, (100.0 * n_with_specific / n_pat_any) if n_pat_any else 0.0
    ))
    lines.append("  Patients that would drop to 0 hits (systemic-only): {} ({:.1f}%)".format(
        n_sys_only, (100.0 * n_sys_only / n_pat_any) if n_pat_any else 0.0
    ))
    lines.append("")
    lines.append("Top specific categories (row-level, excluding systemic) (top 15):")
    if top_specific.empty:
        lines.append("  (none)")
    else:
        for k, v in top_specific.items():
            lines.append("  {:>6}  {}".format(int(v), k))
    lines.append("")

    # Prepare a compact patient-level output for this label
    out = g[[COL_PATIENT, "has_systemic", "has_specific", "systemic_only", "specific_only", "both"]].copy()
    out.insert(0, "stage", label)

    return out


def main():
    lines = []
    lines.append("=== QA: Systemic Impact on Patient-Level Signals ===")
    lines.append("Python: 3.6.8 compatible | Read encoding: latin1(errors=replace) | Write: utf-8")
    lines.append("")

    # Load inputs
    s1 = read_csv_safe(STAGE1_ROWS)
    s2 = read_csv_safe(STAGE2_ROWS)

    out_frames = []

    out1 = summarize_one("STAGE1", s1, lines)
    if out1 is not None:
        out_frames.append(out1)

    out2 = summarize_one("STAGE2_AB", s2, lines)
    if out2 is not None:
        out_frames.append(out2)

    # Write patient-level file
    if out_frames:
        all_out = pd.concat(out_frames, ignore_index=True)
    else:
        all_out = pd.DataFrame(columns=["stage", COL_PATIENT, "has_systemic", "has_specific", "systemic_only", "specific_only", "both"])

    all_out.to_csv(OUT_PATIENT_LEVEL, index=False, encoding="utf-8")

    # Write summary
    lines.append("Wrote:")
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
