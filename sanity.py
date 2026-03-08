import os
from glob import glob
from datetime import datetime
import pandas as pd

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


def load_operation_notes():
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))

    all_rows = []

    for fp in note_files:
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)

        note_id_col = pick_col(df, ["NOTE_ID", "NOTE ID"])
        note_type_col = pick_col(df, ["NOTE_TYPE", "NOTE TYPE"], required=False)
        date_col = pick_col(
            df,
            ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE", "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"],
            required=False
        )

        tmp = pd.DataFrame()
        tmp[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
        tmp["NOTE_ID"] = df[note_id_col].astype(str).str.strip()
        tmp["NOTE_TYPE"] = df[note_type_col].astype(str).str.strip() if note_type_col else ""
        tmp["NOTE_DATE_RAW"] = df[date_col].astype(str).str.strip() if date_col else ""
        tmp["NOTE_DATE"] = tmp["NOTE_DATE_RAW"].apply(to_date_str)
        all_rows.append(tmp)

    if not all_rows:
        return pd.DataFrame(columns=[MERGE_KEY, "NOTE_ID", "NOTE_TYPE", "NOTE_DATE_RAW", "NOTE_DATE"])

    notes = pd.concat(all_rows, ignore_index=True)
    notes = notes.drop_duplicates(subset=[MERGE_KEY, "NOTE_ID", "NOTE_TYPE", "NOTE_DATE"])
    return notes


struct_df = load_structured_encounters()
bmi_anchor_map = choose_best_bmi_recon_rows(struct_df)
notes_df = load_operation_notes()

rows = []

for mrn, info in bmi_anchor_map.items():
    recon_date = info.get("recon_date", "")

    note_subset = notes_df[
        (notes_df[MERGE_KEY] == mrn) &
        (notes_df["NOTE_DATE"] == recon_date)
    ].copy()

    op_matches = note_subset[note_subset["NOTE_TYPE"].str.upper() == "OP NOTE"]
    brief_matches = note_subset[note_subset["NOTE_TYPE"].str.upper() == "BRIEF OP NOTE"]

    rows.append({
        "MRN": mrn,
        "RECON_DATE": recon_date,
        "OP_NOTE_DATE_MATCH": 1 if len(op_matches) > 0 else 0,
        "OP_NOTE_IDS": "; ".join(sorted(op_matches["NOTE_ID"].astype(str).unique().tolist())) if len(op_matches) > 0 else "",
        "BRIEF_OP_NOTE_DATE_MATCH": 1 if len(brief_matches) > 0 else 0,
        "BRIEF_OP_NOTE_IDS": "; ".join(sorted(brief_matches["NOTE_ID"].astype(str).unique().tolist())) if len(brief_matches) > 0 else "",
    })

out = pd.DataFrame(rows)

print("\nEXACT DATE MATCH AUDIT\n")
print(out.head(50).to_string(index=False))

print("\nSUMMARY")
print("Total MRNs with recon anchor:", len(out))
print("MRNs with exact-date OP NOTE:", int(out["OP_NOTE_DATE_MATCH"].sum()))
print("MRNs with exact-date BRIEF OP NOTE:", int(out["BRIEF_OP_NOTE_DATE_MATCH"].sum()))
print("MRNs with either exact-date OP NOTE or BRIEF OP NOTE:",
      int(((out["OP_NOTE_DATE_MATCH"] == 1) | (out["BRIEF_OP_NOTE_DATE_MATCH"] == 1)).sum()))

out.to_csv("_outputs/recon_date_exact_opnote_audit.csv", index=False)
print("\nSaved: _outputs/recon_date_exact_opnote_audit.csv")
