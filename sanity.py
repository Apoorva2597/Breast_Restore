import os
from glob import glob
from datetime import datetime
import pandas as pd
import re

BASE_DIR = "/home/apokol/Breast_Restore"
MERGE_KEY = "MRN"

STRUCT_GLOBS = [
    BASE_DIR + "/**/HPI11526*Clinic Encounters.csv",
    BASE_DIR + "/**/HPI11526*clinic encounters.csv",
    BASE_DIR + "/**/HPI11526*Operation Encounters.csv",
    BASE_DIR + "/**/HPI11526*operation encounters.csv",
    BASE_DIR + "/**/HPI11526*Inpatient Encounters.csv",
    BASE_DIR + "/**/HPI11526*inpatient encounters.csv",
]

NOTE_GLOBS = [
    BASE_DIR + "/**/HPI11526*Operation Notes.csv",
    BASE_DIR + "/**/HPI11526*operation notes.csv",
]

BMI_SNIPPET_RX = re.compile(
    r".{0,80}\b(?:morbid\s+obesity|obesity|BMI|body\s+mass\s+index)\b.{0,120}",
    re.IGNORECASE
)


def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", error_bad_lines=False, warn_bad_lines=True)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", error_bad_lines=False, warn_bad_lines=True)


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
        raise RuntimeError("MRN column not found")
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df


def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError("Required column missing: {0}".format(options))
    return None


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


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


def to_date_str(x):
    dt = parse_date_safe(x)
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d")


def to_int_safe(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def choose_best_bmi_recon_rows(struct_df):
    out_best = {}

    if len(struct_df) == 0:
        return out_best

    source_priority = {
        "clinic": 1,
        "operation": 2,
        "inpatient": 3
    }

    preferred_cpts = set([
        "19357",
        "19340",
        "19342",
        "19361",
        "19364",
        "19367",
        "S2068"
    ])

    primary_exclude_cpts = set([
        "19325",
        "19330"
    ])

    fallback_allowed_cpts = set([
        "19350",
        "19380"
    ])

    eligible_sources = struct_df[struct_df["STRUCT_SOURCE"].isin(["clinic", "operation", "inpatient"])].copy()
    if len(eligible_sources) == 0:
        return out_best

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

        current_best = out_best.get(mrn)

        if current_best is None or score < current_best["score"]:
            out_best[mrn] = {
                "recon_date": recon_date.strftime("%Y-%m-%d"),
                "source": source,
                "cpt_code": cpt_code,
                "procedure": clean_cell(row.get("PROCEDURE_STRUCT", "")),
                "score": score
            }

    return out_best


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

        out = pd.DataFrame()
        out[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
        out["STRUCT_SOURCE"] = encounter_source
        out["STRUCT_PRIORITY"] = priority
        out["ADMIT_DATE_STRUCT"] = df[admit_col].astype(str) if admit_col else ""
        out["RECONSTRUCTION_DATE_STRUCT"] = df[recon_col].astype(str) if recon_col else ""
        out["CPT_CODE_STRUCT"] = df[cpt_col].astype(str) if cpt_col else ""
        out["PROCEDURE_STRUCT"] = df[proc_col].astype(str) if proc_col else ""
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            MERGE_KEY, "STRUCT_SOURCE", "STRUCT_PRIORITY",
            "ADMIT_DATE_STRUCT", "RECONSTRUCTION_DATE_STRUCT",
            "CPT_CODE_STRUCT", "PROCEDURE_STRUCT"
        ])

    return pd.concat(rows, ignore_index=True)


def load_and_reconstruct_op_notes():
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))

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

        keep_cols = [MERGE_KEY, note_id_col, note_text_col]
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

        note_type = g["NOTE_TYPE"].astype(str).iloc[0] if "NOTE_TYPE" in g.columns else ""
        note_date = g["NOTE_DATE_OF_SERVICE"].astype(str).iloc[0] if "NOTE_DATE_OF_SERVICE" in g.columns else ""

        reconstructed.append({
            "MRN": mrn,
            "NOTE_ID": nid,
            "NOTE_TYPE": str(note_type).strip(),
            "NOTE_DATE": to_date_str(note_date),
            "NOTE_TEXT": full_text
        })

    return pd.DataFrame(reconstructed)


struct_df = load_structured_encounters()
bmi_anchor_map = choose_best_bmi_recon_rows(struct_df)
notes_df = load_and_reconstruct_op_notes()

rows = []

for mrn, info in bmi_anchor_map.items():
    recon_date = info.get("recon_date", "")

    same_day = notes_df[
        (notes_df["MRN"] == mrn) &
        (notes_df["NOTE_DATE"] == recon_date) &
        (notes_df["NOTE_TYPE"].astype(str).str.upper().isin(["OP NOTE", "BRIEF OP NOTE", "BRIEF OP NOTES"]))
    ].copy()

    if len(same_day) == 0:
        rows.append({
            "MRN": mrn,
            "RECON_DATE": recon_date,
            "NOTE_ID": "",
            "NOTE_TYPE": "",
            "BMI_TEXT_FOUND": 0,
            "BMI_SNIPPET": ""
        })
        continue

    for _, row in same_day.iterrows():
        text = str(row["NOTE_TEXT"])
        m = BMI_SNIPPET_RX.search(text.replace("\n", " "))

        rows.append({
            "MRN": mrn,
            "RECON_DATE": recon_date,
            "NOTE_ID": row["NOTE_ID"],
            "NOTE_TYPE": row["NOTE_TYPE"],
            "BMI_TEXT_FOUND": 1 if m else 0,
            "BMI_SNIPPET": m.group(0).strip() if m else ""
        })

out = pd.DataFrame(rows)

print("\nRECON-DATE OP NOTE BMI TEXT AUDIT\n")
print(out.head(50).to_string(index=False))

print("\nSUMMARY")
print("Total rows audited:", len(out))
print("Rows with BMI text found:", int(out["BMI_TEXT_FOUND"].sum()))
print("Rows without BMI text found:", int((out["BMI_TEXT_FOUND"] == 0).sum()))

out.to_csv("_outputs/recon_date_opnote_bmi_text_audit.csv", index=False)
print("\nSaved: _outputs/recon_date_opnote_bmi_text_audit.csv")
