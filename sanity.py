#!/usr/bin/env python3
# qa_smoking_mismatches_categorized.py
#
# Stronger smoking QA script:
# 1. Outputs ALL smoking mismatches vs gold
# 2. Keeps rows even if there is no evidence row
# 3. Adds automatic mismatch categorization to speed debugging
# 4. Flags likely failure modes:
#    - recent quit misread
#    - former template/history misread as current
#    - questionnaire/template issue
#    - no evidence row
#    - family history contamination
#    - current narrative missed
#    - never-vs-former confusion
#
# MRN is used internally but NOT written to output.
#
# Python 3.6.8 compatible

import re
import pandas as pd

MASTER_FILE = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"
EVID_FILE = "_outputs/bmi_smoking_only_evidence.csv"

MRN = "MRN"


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_smoking(x):
    s = clean(x).lower()

    if s in [
        "current", "current smoker", "smoker", "active smoker",
        "currently smoking", "currently smokes"
    ]:
        return "Current"

    if s in [
        "former", "former smoker", "ex-smoker", "quit smoking",
        "quit tobacco", "stopped smoking", "stopped tobacco"
    ]:
        return "Former"

    if s in [
        "never", "never smoker", "never smoked", "nonsmoker",
        "non-smoker", "lifetime nonsmoker"
    ]:
        return "Never"

    return clean(x)


def safe_float(x, default=-1.0):
    try:
        return float(str(x).strip())
    except Exception:
        return default


def contains_any(text, patterns):
    t = clean(text).lower()
    for p in patterns:
        if re.search(p, t, re.IGNORECASE):
            return True
    return False


def categorize_mismatch(gold, pred, evidence, value, note_type, section):
    txt = " ".join([
        clean(evidence),
        clean(value),
        clean(note_type),
        clean(section)
    ]).lower()

    if clean(evidence) == "NO EVIDENCE ROW FOUND":
        return "no_evidence_row"

    if contains_any(txt, [
        r"family history", r"\bmother\b", r"\bfather\b", r"\baunt\b",
        r"\buncle\b", r"\bsister\b", r"\bbrother\b",
        r"\bgrandmother\b", r"\bgrandfather\b"
    ]):
        return "family_history_contamination"

    if contains_any(txt, [
        r"resources?\s+to\s+help\s+quit\s+smoking",
        r"interested\s+in\s+resources?\s+to\s+help\s+quit\s+smoking",
        r"referral\s+to\s+mhealthy",
        r"referred\s+to\s+mhealthy",
        r"advised\s+by\s+provider\s+to\s+quit\s+smoking",
        r"is patient currently smoking\?\s*no",
        r"active tobacco use\?\s*no",
        r"active tobacco use\s*[:\-]?\s*no",
        r"current tobacco use\s*[:\-]?\s*no"
    ]):
        if gold == "Never" and pred == "Former":
            return "questionnaire_template_false_former"
        if gold == "Never" and pred == "Current":
            return "questionnaire_template_false_current"
        if gold == "Former" and pred == "Never":
            return "questionnaire_template_false_never"
        return "questionnaire_template_issue"

    if gold == "Current" and pred == "Former":
        if contains_any(txt, [
            r"since\s+(our|the)\s+last\s+visit.*quit",
            r"recently quit",
            r"quit date",
            r"years since quitting\s*[:\-]?\s*0\.",
            r"\b0\.[0-9]+\s*years since quitting\b",
            r"\bdown to\s+\d",
            r"\bcigs?\s+daily\b",
            r"\bcigarettes?\s+(a|per)\s+(day|week)\b",
            r"\bsmokes approximately\b",
            r"\bstill smoking\b",
            r"\bcontinues to smoke\b",
            r"\busing chantix\b"
        ]):
            return "recent_quit_or_current_narrative_missed"
        return "current_misread_as_former"

    if gold == "Former" and pred == "Current":
        if contains_any(txt, [
            r"former smoker",
            r"smoking status\s*[:\-]?\s*former",
            r"history smoking status\s*[:\-]?\s*former",
            r"quit date",
            r"years since quitting",
            r"quit .* years ago",
            r"stopped .* years ago"
        ]):
            return "former_template_misread_as_current"
        return "former_misread_as_current"

    if gold == "Never" and pred == "Former":
        if contains_any(txt, [
            r"no tobacco use",
            r"never smoker",
            r"never smoked",
            r"never used tobacco",
            r"denies tobacco",
            r"denies smoking"
        ]):
            return "never_template_misread_as_former"
        return "never_misread_as_former"

    if gold == "Former" and pred == "Never":
        if contains_any(txt, [
            r"former smoker",
            r"quit date",
            r"years since quitting",
            r"history smoking status\s*[:\-]?\s*former",
            r"smoking status\s*[:\-]?\s*former"
        ]):
            return "former_history_overridden_by_never"
        return "former_misread_as_never"

    if gold == "Never" and pred == "Current":
        return "never_misread_as_current"

    if gold == "Current" and pred == "Never":
        return "current_misread_as_never"

    return "other"


print("Loading files...")

master = pd.read_csv(MASTER_FILE, dtype=str)
gold = pd.read_csv(GOLD_FILE, dtype=str)
evid = pd.read_csv(EVID_FILE, dtype=str)

master[MRN] = master[MRN].astype(str).str.strip()
gold[MRN] = gold[MRN].astype(str).str.strip()
evid[MRN] = evid[MRN].astype(str).str.strip()

merged = pd.merge(master, gold, on=MRN, suffixes=("_pred", "_gold"))

merged["Smoking_pred"] = merged["SmokingStatus_pred"].apply(normalize_smoking)
merged["Smoking_gold"] = merged["SmokingStatus_gold"].apply(normalize_smoking)

mismatches = merged[merged["Smoking_pred"] != merged["Smoking_gold"]].copy()

print("\nSmoking mismatches:", len(mismatches))

rows = []

for _, r in mismatches.iterrows():
    mrn = r[MRN]

    ev = evid[
        (evid[MRN] == mrn) &
        (evid["FIELD"] == "SmokingStatus")
    ].copy()

    if len(ev) > 0:
        if "CONFIDENCE" not in ev.columns:
            ev["CONFIDENCE"] = ""

        if "NOTE_DATE" not in ev.columns:
            ev["NOTE_DATE"] = ""

        if "STAGE_USED" not in ev.columns:
            ev["STAGE_USED"] = ""

        stage_rank = {"day0": 1, "pm7": 2, "pm14": 3}
        ev["_stage_rank"] = ev["STAGE_USED"].astype(str).str.strip().map(
            lambda x: stage_rank.get(x, 9)
        )
        ev["_conf_num"] = ev["CONFIDENCE"].apply(safe_float)

        ev = ev.sort_values(
            by=["_stage_rank", "_conf_num"],
            ascending=[True, False]
        )

        e = ev.iloc[0]

        gold_val = clean(r["Smoking_gold"])
        pred_val = clean(r["Smoking_pred"])
        note_date = clean(e.get("NOTE_DATE"))
        note_type = clean(e.get("NOTE_TYPE"))
        section = clean(e.get("SECTION"))
        value = clean(e.get("VALUE"))
        evidence = clean(e.get("EVIDENCE"))

    else:
        gold_val = clean(r["Smoking_gold"])
        pred_val = clean(r["Smoking_pred"])
        note_date = ""
        note_type = ""
        section = ""
        value = ""
        evidence = "NO EVIDENCE ROW FOUND"

    category = categorize_mismatch(
        gold=gold_val,
        pred=pred_val,
        evidence=evidence,
        value=value,
        note_type=note_type,
        section=section
    )

    rows.append({
        "Mismatch_Category": category,
        "Gold": gold_val,
        "Pred": pred_val,
        "Note_Date": note_date,
        "Note_Type": note_type,
        "Section": section,
        "Value": value,
        "Evidence": evidence
    })

qa = pd.DataFrame(rows)

# sort for easier review
sort_order = {
    "recent_quit_or_current_narrative_missed": 1,
    "former_template_misread_as_current": 2,
    "never_template_misread_as_former": 3,
    "former_history_overridden_by_never": 4,
    "questionnaire_template_false_former": 5,
    "questionnaire_template_false_current": 6,
    "questionnaire_template_false_never": 7,
    "questionnaire_template_issue": 8,
    "family_history_contamination": 9,
    "no_evidence_row": 10,
    "current_misread_as_former": 11,
    "former_misread_as_current": 12,
    "never_misread_as_former": 13,
    "former_misread_as_never": 14,
    "never_misread_as_current": 15,
    "current_misread_as_never": 16,
    "other": 99
}

if len(qa) > 0:
    qa["_sort"] = qa["Mismatch_Category"].map(lambda x: sort_order.get(x, 99))
    qa = qa.sort_values(by=["_sort", "Mismatch_Category", "Gold", "Pred", "Note_Date"]).drop(columns=["_sort"])

out = "_outputs/qa_smoking_mismatches_categorized.csv"
qa.to_csv(out, index=False)

print("Saved:", out)
print("Rows written:", len(qa))

if len(qa) > 0:
    print("\nMismatch category counts:")
    print(qa["Mismatch_Category"].value_counts(dropna=False).to_string())
