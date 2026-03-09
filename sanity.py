#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
qa_obesity_mismatch_evidence_tol_0_5.py

Shows ONLY the obesity mismatches that remain AFTER applying the
same ±0.5 BMI tolerance logic used in:

    Obesity_from_BMI_tol_0_5

So borderline cases like:
    gold 30.3 vs pred 29.9
will NOT appear here.

Outputs:
- MRN
- BMI_gold
- BMI_pred
- gold_obesity
- pred_obesity
- diff_abs
- note metadata
- BMI snippet/evidence

Python 3.6.8 compatible.
"""

import os
import pandas as pd

MASTER_FILE = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"
EVIDENCE_FILE = "_outputs/bmi_only_evidence.csv"
OUTPUT_FILE = "_outputs/qa_obesity_mismatches_tol_0_5_with_evidence.csv"

MRN = "MRN"
BMI_TOL = 0.5


def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8", dtype=str)
    except Exception:
        return pd.read_csv(path, encoding="latin1", dtype=str)


def clean_string_series(series):
    series = series.copy()
    series = series.astype(str)
    series = series.str.strip()

    series = series.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "none": pd.NA,
        "NA": pd.NA,
        "na": pd.NA,
        "null": pd.NA,
        "Null": pd.NA
    })

    return series


def normalize_numeric(series):
    series = clean_string_series(series)
    return pd.to_numeric(series, errors="coerce")


def choose_best_evidence_row(group):
    """
    Pick the most useful evidence row:
    1) measured over computed
    2) higher confidence
    3) earlier stage preference: day0, pm7, pm14
    """
    g = group.copy()

    def status_rank(x):
        s = str(x).strip().lower()
        if s == "measured":
            return 0
        if s == "computed":
            return 1
        return 9

    def stage_rank(x):
        s = str(x).strip().lower()
        if s == "day0":
            return 0
        if s == "pm7":
            return 1
        if s == "pm14":
            return 2
        return 9

    if "STATUS" in g.columns:
        g["_status_rank"] = g["STATUS"].apply(status_rank)
    else:
        g["_status_rank"] = 9

    if "STAGE_USED" in g.columns:
        g["_stage_rank"] = g["STAGE_USED"].apply(stage_rank)
    else:
        g["_stage_rank"] = 9

    if "CONFIDENCE" in g.columns:
        g["_conf_num"] = pd.to_numeric(g["CONFIDENCE"], errors="coerce").fillna(-1)
    else:
        g["_conf_num"] = -1

    g = g.sort_values(
        by=["_status_rank", "_stage_rank", "_conf_num"],
        ascending=[True, True, False]
    )

    return g.iloc[0]


def main():

    print("Loading files...")

    master = safe_read_csv(MASTER_FILE)
    gold = safe_read_csv(GOLD_FILE)
    evid = safe_read_csv(EVIDENCE_FILE)

    if MRN not in master.columns:
        print("ERROR: master missing MRN")
        return

    if MRN not in gold.columns:
        print("ERROR: gold missing MRN")
        return

    master[MRN] = master[MRN].astype(str).str.strip()
    gold[MRN] = gold[MRN].astype(str).str.strip()
    evid[MRN] = evid[MRN].astype(str).str.strip()

    master = master[master[MRN] != ""].copy()
    gold = gold[gold[MRN] != ""].copy()

    master = master.drop_duplicates(subset=[MRN])
    gold = gold.drop_duplicates(subset=[MRN])

    print("Merging master and gold on MRN...")
    merged = pd.merge(master, gold, on=MRN, how="inner", suffixes=("_pred", "_gold"))
    print("Merged rows:", len(merged))

    if "BMI_pred" not in merged.columns or "BMI_gold" not in merged.columns:
        print("ERROR: BMI columns missing after merge.")
        return

    merged["BMI_pred_num"] = normalize_numeric(merged["BMI_pred"])
    merged["BMI_gold_num"] = normalize_numeric(merged["BMI_gold"])

    # same denominator rule as validator: GOLD PRESENT ONLY
    mask = merged["BMI_gold_num"].notna()
    subset = merged.loc[mask].copy()

    subset["gold_obesity"] = (subset["BMI_gold_num"] >= 30).astype(int)

    # if pred missing, treat as non-match and unresolved
    subset["pred_obesity"] = (subset["BMI_pred_num"] >= 30).fillna(False).astype(int)

    subset["diff_abs"] = (subset["BMI_pred_num"] - subset["BMI_gold_num"]).abs()

    # strict obesity mismatch
    strict_mismatch = (subset["gold_obesity"] != subset["pred_obesity"])

    # close BMI within tolerance
    close_bmi = (subset["diff_abs"] <= BMI_TOL)

    # unresolved after tolerance:
    # still obesity mismatch AND not close enough on BMI
    unresolved = subset[strict_mismatch & (~close_bmi.fillna(False))].copy()

    print("Strict obesity mismatches:", int(strict_mismatch.sum()))
    print("Resolved by tolerance <= 0.5:", int((strict_mismatch & close_bmi.fillna(False)).sum()))
    print("Remaining obesity mismatches after tolerance:", len(unresolved))

    rows = []

    for _, row in unresolved.iterrows():

        mrn = str(row[MRN]).strip()
        pred_bmi = row["BMI_pred_num"]

        ev_sub = evid[evid[MRN] == mrn].copy()

        if len(ev_sub) > 0 and pd.notna(pred_bmi):
            if "VALUE" in ev_sub.columns:
                ev_sub["VALUE_num"] = pd.to_numeric(ev_sub["VALUE"], errors="coerce")
                ev_sub = ev_sub[ev_sub["VALUE_num"] == pred_bmi].copy()

        if len(ev_sub) > 0:
            best = choose_best_evidence_row(ev_sub)
            note_id = best.get("NOTE_ID", "")
            note_type = best.get("NOTE_TYPE", "")
            note_date = best.get("NOTE_DATE", "")
            anchor_date = best.get("ANCHOR_DATE", "")
            stage_used = best.get("STAGE_USED", "")
            status = best.get("STATUS", "")
            confidence = best.get("CONFIDENCE", "")
            section = best.get("SECTION", "")
            snippet = best.get("EVIDENCE", "")
        else:
            note_id = ""
            note_type = ""
            note_date = ""
            anchor_date = ""
            stage_used = ""
            status = ""
            confidence = ""
            section = ""
            snippet = "NO_MATCHING_BMI_EVIDENCE_FOUND"

        rows.append({
            "MRN": mrn,
            "BMI_gold": row["BMI_gold_num"],
            "BMI_pred": row["BMI_pred_num"],
            "gold_obesity": row["gold_obesity"],
            "pred_obesity": row["pred_obesity"],
            "diff_abs": row["diff_abs"],
            "NOTE_ID": note_id,
            "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date,
            "ANCHOR_DATE": anchor_date,
            "STAGE_USED": stage_used,
            "STATUS": status,
            "CONFIDENCE": confidence,
            "SECTION": section,
            "SNIPPET": snippet
        })

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(by=["diff_abs", "MRN"], ascending=[False, True])

    if not os.path.exists("_outputs"):
        os.makedirs("_outputs")

    out.to_csv(OUTPUT_FILE, index=False)

    print("")
    print("Wrote QA file:")
    print(OUTPUT_FILE)
    print("")

    if len(out) > 0:
        print("Preview:")
        print(out[[
            "MRN",
            "BMI_gold",
            "BMI_pred",
            "gold_obesity",
            "pred_obesity",
            "diff_abs",
            "NOTE_TYPE",
            "NOTE_DATE",
            "ANCHOR_DATE",
            "STAGE_USED"
        ]].to_string(index=False))
    else:
        print("No remaining obesity mismatches after tolerance.")


if __name__ == "__main__":
    main()
