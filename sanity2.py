#!/usr/bin/env python3
# qa_bmi_missing_with_snippets.py
#
# Purpose:
# - Focus only on gold MRNs where BMI_pred is missing
# - Pull nearby notes around reconstruction date
# - Show snippets containing:
#     * BMI / body mass index
#     * height / ht
#     * weight / wt
# - Helps quickly determine whether the miss is:
#     1) true missing documentation
#     2) regex miss
#     3) computable from height/weight
#
# Output:
#   /home/apokol/Breast_Restore/_outputs/qa_bmi_missing_snippets.csv
#
# Python 3.6.8 compatible

import os
import re
from glob import glob
from datetime import datetime
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
GOLD_FILE = "{0}/gold_cleaned_for_cedar.csv".format(BASE_DIR)
OUTPUT_FILE = "{0}/_outputs/qa_bmi_missing_snippets.csv".format(BASE_DIR)

MERGE_KEY = "MRN"
WINDOW_BEFORE = 45
WINDOW_AFTER = 14
MAX_NOTES_PER_MRN = 12

STRUCT_GLOBS = [
    "{0}/**/HPI11526*Clinic Encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation encounters.csv".format(BASE_DIR),
]

NOTE_GLOBS = [
    "{0}/**/HPI11526*Clinic Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation notes.csv".format(BASE_DIR),
]

BMI_RX = re.compile(r"\b(?:BMI|body\s+mass\s+index)\b", re.IGNORECASE)
HT_RX = re.compile(r"\b(?:Ht|Height)\b", re.IGNORECASE)
WT_RX = re.compile(r"\b(?:Wt|Weight)\b", re.IGNORECASE)

def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                error_bad_lines=False,
                warn_bad_lines=True
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                on_bad_lines="skip"
            )
        except TypeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )

def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def normalize_mrn(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MERGE_KEY:
                df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError("MRN column not found. Columns seen: {0}".format(list(df.columns)[:40]))
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df

def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError("Required column missing. Tried={0}. Seen={1}".format(
            options, list(df.columns)[:60]
        ))
    return None

def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s

def to_int_safe(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def parse_date_safe(x):
    s = clean_cell(x)
    if not s:
        return None
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%Y/%m/%d",
        "%d-%b-%Y",
        "%d-%b-%Y %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None

def days_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    return (dt1.date() - dt2.date()).days

def normalize_text(text):
    text = str(text or "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def window_around(text, start, end, width):
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].strip()

def load_structured_encounters():
    rows = []
    struct_files = []
    for g in STRUCT_GLOBS:
        struct_files.extend(glob(g, recursive=True))

    for fp in sorted(set(struct_files)):
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)
        source_name = os.path.basename(fp).lower()

        if "operation encounters" in source_name:
            encounter_source = "operation"
            priority = 1
        elif "clinic encounters" in source_name:
            encounter_source = "clinic"
            priority = 2
        elif "inpatient encounters" in source_name:
            encounter_source = "inpatient"
            priority = 3
        else:
            encounter_source = "other"
            priority = 9

        admit_col = pick_col(df, ["ADMIT_DATE", "Admit_Date"], required=False)
        recon_col = pick_col(df, ["RECONSTRUCTION_DATE", "RECONSTRUCTION DATE"], required=False)
        cpt_col = pick_col(df, ["CPT_CODE", "CPT CODE", "CPT"], required=False)
        proc_col = pick_col(df, ["PROCEDURE", "Procedure"], required=False)
        reason_col = pick_col(df, ["REASON_FOR_VISIT", "REASON FOR VISIT"], required=False)
        date_col = pick_col(df, ["OPERATION_DATE", "CHECKOUT_TIME", "DISCHARGE_DATE_DT"], required=False)

        out = pd.DataFrame()
        out[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
        out["STRUCT_SOURCE"] = encounter_source
        out["STRUCT_PRIORITY"] = priority
        out["STRUCT_DATE_RAW"] = df[date_col].astype(str) if date_col else ""
        out["ADMIT_DATE_STRUCT"] = df[admit_col].astype(str) if admit_col else ""
        out["RECONSTRUCTION_DATE_STRUCT"] = df[recon_col].astype(str) if recon_col else ""
        out["CPT_CODE_STRUCT"] = df[cpt_col].astype(str) if cpt_col else ""
        out["PROCEDURE_STRUCT"] = df[proc_col].astype(str) if proc_col else ""
        out["REASON_FOR_VISIT_STRUCT"] = df[reason_col].astype(str) if reason_col else ""
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            MERGE_KEY, "STRUCT_SOURCE", "STRUCT_PRIORITY", "STRUCT_DATE_RAW",
            "ADMIT_DATE_STRUCT", "RECONSTRUCTION_DATE_STRUCT",
            "CPT_CODE_STRUCT", "PROCEDURE_STRUCT", "REASON_FOR_VISIT_STRUCT"
        ])

    return pd.concat(rows, ignore_index=True)

def choose_best_bmi_anchor_rows(struct_df):
    bmi_best = {}
    if len(struct_df) == 0:
        return bmi_best

    source_priority = {
        "clinic": 1,
        "operation": 2,
        "inpatient": 3
    }

    preferred_cpts = set(["19357", "19340", "19342", "19361", "19364", "19367", "S2068"])
    primary_exclude_cpts = set(["19325", "19330"])
    fallback_allowed_cpts = set(["19350", "19380"])

    eligible_sources = struct_df[struct_df["STRUCT_SOURCE"].isin(["clinic", "operation", "inpatient"])].copy()
    if len(eligible_sources) == 0:
        return bmi_best

    has_preferred_cpt = {}
    for mrn, g in eligible_sources.groupby(MERGE_KEY):
        found = False
        for val in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist():
            cpt = clean_cell(val).upper()
            if cpt in preferred_cpts:
                found = True
                break
        has_preferred_cpt[mrn] = found

    for _, row in eligible_sources.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        if source not in source_priority:
            continue

        admit_date = parse_date_safe(row.get("ADMIT_DATE_STRUCT", ""))
        recon_date = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))
        cpt_code = clean_cell(row.get("CPT_CODE_STRUCT", "")).upper()
        procedure = clean_cell(row.get("PROCEDURE_STRUCT", "")).lower()

        if admit_date is None or recon_date is None:
            continue
        if cpt_code in primary_exclude_cpts:
            continue
        if has_preferred_cpt.get(mrn, False) and cpt_code in fallback_allowed_cpts:
            continue

        is_anchor = False
        if cpt_code in preferred_cpts:
            is_anchor = True
        if (not has_preferred_cpt.get(mrn, False)) and (cpt_code in fallback_allowed_cpts):
            is_anchor = True
        if not is_anchor:
            if (
                ("tissue expander" in procedure) or
                ("breast recon" in procedure) or
                ("implant on same day of mastectomy" in procedure) or
                ("insert or replcmnt breast implnt on sep day from mastectomy" in procedure) or
                ("latissimus" in procedure) or
                ("diep" in procedure) or
                ("tram" in procedure) or
                ("flap" in procedure)
            ):
                is_anchor = True

        if not is_anchor:
            continue

        score = (
            source_priority[source],
            recon_date,
            admit_date
        )

        current_best = bmi_best.get(mrn)
        if current_best is None or score < current_best["score"]:
            bmi_best[mrn] = {
                "admit_date": admit_date.strftime("%Y-%m-%d"),
                "recon_date": recon_date.strftime("%Y-%m-%d"),
                "score": score,
                "source": source,
                "cpt_code": cpt_code,
                "procedure": clean_cell(row.get("PROCEDURE_STRUCT", "")),
                "reason_for_visit": clean_cell(row.get("REASON_FOR_VISIT_STRUCT", ""))
            }

    return bmi_best

def load_and_reconstruct_notes():
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))

    if not note_files:
        raise FileNotFoundError("No HPI11526 * Notes.csv files found via NOTE_GLOBS.")

    all_notes_rows = []

    for fp in note_files:
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)

        note_text_col = pick_col(df, ["NOTE_TEXT", "NOTE TEXT", "NOTE_TEXT_FULL", "TEXT", "NOTE"])
        note_id_col = pick_col(df, ["NOTE_ID", "NOTE ID"])
        line_col = pick_col(df, ["LINE"], required=False)
        note_type_col = pick_col(df, ["NOTE_TYPE", "NOTE TYPE"], required=False)
        date_col = pick_col(
            df,
            ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE", "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"],
            required=False
        )

        df[note_text_col] = df[note_text_col].fillna("").astype(str)
        df[note_id_col] = df[note_id_col].fillna("").astype(str)
        if line_col:
            df[line_col] = df[line_col].fillna("").astype(str)
        if note_type_col:
            df[note_type_col] = df[note_type_col].fillna("").astype(str)
        if date_col:
            df[date_col] = df[date_col].fillna("").astype(str)

        df["_SOURCE_FILE_"] = os.path.basename(fp)

        keep_cols = [MERGE_KEY, note_id_col, note_text_col, "_SOURCE_FILE_"]
        if line_col:
            keep_cols.append(line_col)
        if note_type_col:
            keep_cols.append(note_type_col)
        if date_col:
            keep_cols.append(date_col)

        tmp = df[keep_cols].copy()
        tmp = tmp.rename(columns={
            note_id_col: "NOTE_ID",
            note_text_col: "NOTE_TEXT",
        })

        if line_col and line_col != "LINE":
            tmp = tmp.rename(columns={line_col: "LINE"})
        if note_type_col and note_type_col != "NOTE_TYPE":
            tmp = tmp.rename(columns={note_type_col: "NOTE_TYPE"})
        if date_col and date_col != "NOTE_DATE_OF_SERVICE":
            tmp = tmp.rename(columns={date_col: "NOTE_DATE_OF_SERVICE"})

        if "LINE" not in tmp.columns:
            tmp["LINE"] = ""
        if "NOTE_TYPE" not in tmp.columns:
            tmp["NOTE_TYPE"] = ""
        if "NOTE_DATE_OF_SERVICE" not in tmp.columns:
            tmp["NOTE_DATE_OF_SERVICE"] = ""

        all_notes_rows.append(tmp)

    notes_raw = pd.concat(all_notes_rows, ignore_index=True)

    def join_note(group):
        tmp = group.copy()
        tmp["_LINE_NUM_"] = tmp["LINE"].apply(to_int_safe)
        tmp = tmp.sort_values(by=["_LINE_NUM_"], na_position="last")
        return "\n".join(tmp["NOTE_TEXT"].tolist()).strip()

    reconstructed = []
    grouped = notes_raw.groupby([MERGE_KEY, "NOTE_ID"], dropna=False)

    for (mrn, nid), g in grouped:
        mrn = str(mrn).strip()
        nid = str(nid).strip()
        if not nid:
            continue

        full_text = join_note(g)
        if not full_text:
            continue

        if g["NOTE_TYPE"].astype(str).str.strip().any():
            note_type = g["NOTE_TYPE"].astype(str).iloc[0]
        else:
            note_type = g["_SOURCE_FILE_"].astype(str).iloc[0]

        if g["NOTE_DATE_OF_SERVICE"].astype(str).str.strip().any():
            note_date = g["NOTE_DATE_OF_SERVICE"].astype(str).iloc[0]
        else:
            note_date = ""

        reconstructed.append({
            MERGE_KEY: mrn,
            "NOTE_ID": nid,
            "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date,
            "SOURCE_FILE": g["_SOURCE_FILE_"].astype(str).iloc[0],
            "NOTE_TEXT": full_text
        })

    return pd.DataFrame(reconstructed)

def note_type_bucket(note_type, source_file):
    s = "{0} {1}".format(clean_cell(note_type).lower(), clean_cell(source_file).lower())

    if "brief op" in s:
        return "brief_op"
    if "operative" in s or "operation" in s or "op note" in s or "oper report" in s:
        return "operation"
    if "anesthesia" in s:
        return "anesthesia"
    if "pre-op" in s or "preop" in s:
        return "preop"
    if "h&p" in s or "history and physical" in s:
        return "hp"
    if "progress" in s:
        return "progress"
    if "clinic" in s or "office" in s:
        return "clinic"
    if "consult" in s:
        return "consult"
    return "other"

def rank_note(note_type, source_file, day_diff):
    bucket = note_type_bucket(note_type, source_file)
    abs_dd = abs(day_diff)

    if day_diff == 0 and bucket in ("brief_op", "operation"):
        return (0, abs_dd, bucket)
    if day_diff == 0 and bucket in ("progress", "clinic", "preop", "anesthesia", "hp"):
        return (1, abs_dd, bucket)
    if bucket in ("brief_op", "operation") and abs_dd <= 3:
        return (2, abs_dd, bucket)
    if day_diff < 0 and bucket in ("progress", "clinic", "preop", "anesthesia", "hp"):
        return (3, abs_dd, bucket)
    if day_diff > 0 and bucket in ("progress", "clinic", "preop", "anesthesia", "hp"):
        return (4, abs_dd, bucket)
    return (5, abs_dd, bucket)

def find_snippets(text):
    text_norm = normalize_text(text)
    snippets = []

    for rx in [BMI_RX, HT_RX, WT_RX]:
        for m in rx.finditer(text_norm):
            snippets.append(window_around(text_norm, m.start(), m.end(), 140))

    dedup = []
    seen = set()
    for s in snippets:
        if s not in seen:
            seen.add(s)
            dedup.append(s)

    return dedup

def main():
    print("Loading master...")
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)

    print("Loading gold...")
    gold = clean_cols(read_csv_robust(GOLD_FILE))
    gold = normalize_mrn(gold)

    print("Loading structured encounters...")
    struct_df = load_structured_encounters()
    anchor_map = choose_best_bmi_anchor_rows(struct_df)
    print("BMI anchor rows found: {0}".format(len(anchor_map)))

    print("Loading and reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    if "BMI" not in gold.columns:
        raise RuntimeError("Gold file missing BMI column.")

    if "BMI" not in master.columns:
        master["BMI"] = ""

    gold["BMI"] = gold["BMI"].astype(str).str.strip()
    gold_nonmissing = gold[
        (gold["BMI"] != "") &
        (~gold["BMI"].isin(["nan", "None", "NA", "null"]))
    ].copy()

    merged = pd.merge(
        gold_nonmissing[[MERGE_KEY, "BMI"]],
        master[[MERGE_KEY, "BMI"]],
        on=MERGE_KEY,
        how="left",
        suffixes=("_gold", "_pred")
    )

    missing_df = merged[
        merged["BMI_gold"].notna() &
        (
            merged["BMI_pred"].isna() |
            (merged["BMI_pred"].astype(str).str.strip() == "") |
            (merged["BMI_pred"].astype(str).str.lower().isin(["nan", "none", "null", "na"]))
        )
    ].copy()

    target_mrns = set(missing_df[MERGE_KEY].astype(str).str.strip().tolist())
    print("Gold BMI present but pred missing MRNs: {0}".format(len(target_mrns)))

    notes_df = notes_df[notes_df[MERGE_KEY].astype(str).isin(target_mrns)].copy()

    rows = []

    for mrn in sorted(target_mrns):
        gold_bmi = clean_cell(missing_df[missing_df[MERGE_KEY] == mrn]["BMI_gold"].iloc[0])

        anchor = anchor_map.get(mrn)
        if anchor is None:
            rows.append({
                "MRN": mrn,
                "BMI_gold": gold_bmi,
                "recon_date": "",
                "note_rank": "",
                "note_date": "",
                "days_from_recon": "",
                "note_type": "",
                "source_file": "",
                "note_bucket": "",
                "snippet_type": "NO_ANCHOR",
                "snippet": ""
            })
            continue

        recon_date_str = anchor.get("recon_date", "")
        recon_dt = parse_date_safe(recon_date_str)

        mrn_notes = notes_df[notes_df[MERGE_KEY].astype(str).str.strip() == mrn].copy()

        ranked_notes = []
        for _, note_row in mrn_notes.iterrows():
            note_dt = parse_date_safe(note_row.get("NOTE_DATE", ""))
            if note_dt is None or recon_dt is None:
                continue

            dd = days_between(note_dt, recon_dt)
            if dd is None:
                continue

            if dd < -WINDOW_BEFORE or dd > WINDOW_AFTER:
                continue

            nt = clean_cell(note_row.get("NOTE_TYPE", ""))
            sf = clean_cell(note_row.get("SOURCE_FILE", ""))
            bucket = note_type_bucket(nt, sf)

            ranked_notes.append({
                "rank_tuple": rank_note(nt, sf, dd),
                "note_date": note_dt.strftime("%Y-%m-%d"),
                "days_from_recon": dd,
                "note_type": nt,
                "source_file": sf,
                "note_bucket": bucket,
                "note_id": clean_cell(note_row.get("NOTE_ID", "")),
                "note_text": note_row.get("NOTE_TEXT", ""),
            })

        ranked_notes = sorted(
            ranked_notes,
            key=lambda x: (x["rank_tuple"], abs(x["days_from_recon"]), x["note_date"], x["note_id"])
        )

        if len(ranked_notes) == 0:
            rows.append({
                "MRN": mrn,
                "BMI_gold": gold_bmi,
                "recon_date": recon_date_str,
                "note_rank": "",
                "note_date": "",
                "days_from_recon": "",
                "note_type": "",
                "source_file": "",
                "note_bucket": "",
                "snippet_type": "NO_NOTE_IN_WINDOW",
                "snippet": ""
            })
            continue

        note_counter = 0
        for note_info in ranked_notes:
            note_counter += 1
            if note_counter > MAX_NOTES_PER_MRN:
                break

            snippets = find_snippets(note_info["note_text"])
            if not snippets:
                rows.append({
                    "MRN": mrn,
                    "BMI_gold": gold_bmi,
                    "recon_date": recon_date_str,
                    "note_rank": note_counter,
                    "note_date": note_info["note_date"],
                    "days_from_recon": note_info["days_from_recon"],
                    "note_type": note_info["note_type"],
                    "source_file": note_info["source_file"],
                    "note_bucket": note_info["note_bucket"],
                    "snippet_type": "NO_BMI_HT_WT_TERM",
                    "snippet": ""
                })
                continue

            for snip in snippets[:5]:
                snip_low = snip.lower()
                if "bmi" in snip_low or "body mass index" in snip_low:
                    stype = "BMI_TERM"
                elif "height" in snip_low or "ht" in snip_low:
                    stype = "HEIGHT_TERM"
                elif "weight" in snip_low or "wt" in snip_low:
                    stype = "WEIGHT_TERM"
                else:
                    stype = "OTHER_TERM"

                rows.append({
                    "MRN": mrn,
                    "BMI_gold": gold_bmi,
                    "recon_date": recon_date_str,
                    "note_rank": note_counter,
                    "note_date": note_info["note_date"],
                    "days_from_recon": note_info["days_from_recon"],
                    "note_type": note_info["note_type"],
                    "source_file": note_info["source_file"],
                    "note_bucket": note_info["note_bucket"],
                    "snippet_type": stype,
                    "snippet": snip
                })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print("")
    print("DONE.")
    print("Rows written: {0}".format(len(out_df)))
    print("Output: {0}".format(OUTPUT_FILE))
    print("")
    print("Useful filters:")
    print("1) snippet_type = BMI_TERM")
    print("2) snippet_type = HEIGHT_TERM or WEIGHT_TERM")
    print("3) snippet_type = NO_BMI_HT_WT_TERM")
    print("4) sort by MRN, then note_rank")

if __name__ == "__main__":
    main()
