# qa_op_encounters_stage2_audit.py
# Python 3.6+ (pandas required)
#
# Purpose:
#   Audit OPERATION ENCOUNTERS to understand why Stage 2 isn't being detected.
#   This script produces 3 core audits:
#     (1) Multiplicity audit: do "expander patients" have >=2 distinct OP dates?
#     (2) Post-expander audit: what is the NEXT procedure/CPT after the first expander date?
#     (3) CPT normalization audit: what CPT values exist (raw + normalized), do they have .0, multi-codes, etc?
#
# Outputs (CSV):
#   - qa_op_proc_inventory.csv
#   - qa_op_cpt_inventory_raw.csv
#   - qa_op_cpt_inventory_norm.csv
#   - qa_op_patient_date_counts.csv
#   - qa_op_expander_post_index_next_event.csv
#   - qa_op_expander_next_event_counts.csv
#   - qa_op_expander_stage2_candidate_after_index.csv
#
# NOTE:
#   This script writes only aggregate outputs plus per-patient summaries WITHOUT note text.
#   It does include procedure strings, which are structured/billing descriptors (still treat as sensitive).

import re
import sys
import pandas as pd


# -------------------------
# CONFIG (update only if headers differ)
# -------------------------
OP_ENCOUNTERS_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv"

COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_MRN = "MRN"
COL_OP_DATE = "OPERATION_DATE"
COL_ALT_DATE = "DISCHARGE_DATE_DT"
COL_CPT = "CPT_CODE"
COL_PROC = "PROCEDURE"

# How many top rows to print to terminal
TOP_N_PRINT = 15


# -------------------------
# Patterns (high-recall, conservative enough for QA)
# -------------------------
# Expander-ish procedure descriptions seen in your inventory:
#   "PC-TISSUE EXPANDR PLACMNT IN BREAST RECONST INC SUBSEQ EXPANSIONS ..."
# We treat that as "expander index candidate" for the audit.
EXPANDER_ANY_RE = re.compile(r"\b(tissue\s*expand|tiss\s*expand|expander|expandr)\b", re.I)

# Stage2-ish language (what you *expected* to see):
STAGE2_TEXT_RE = re.compile(
    r"\bsep(arate)?\s+day\b.*\bimplant\b|"
    r"\bimplant\b.*\bsep(arate)?\s+day\b|"
    r"\bexchange\b.*\b(implant|expander)\b|"
    r"\b(implant|expander)\b.*\bexchange\b|"
    r"\bremove(d)?\b.*\b(tissue\s*expander|expander)\b.*\bimplant\b|"
    r"\bdelayed\b.*\bimplant\b",
    re.I
)

# CPTs commonly associated with TE->implant exchange / delayed insertion
STAGE2_CPT_HINTS = set(["11970", "19342"])


# -------------------------
# Helpers
# -------------------------
def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def norm_proc(x):
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("_", " ")
    return norm_ws(s)


def parse_date_series(df, primary_col, fallback_col=None):
    dt = pd.to_datetime(df[primary_col], errors="coerce") if primary_col in df.columns else pd.Series([pd.NaT] * len(df))
    if fallback_col and fallback_col in df.columns:
        fb = pd.to_datetime(df[fallback_col], errors="coerce")
        dt = dt.fillna(fb)
    return dt


def normalize_cpt_to_list(raw):
    """
    Return a list of normalized CPT tokens from a raw CPT cell.
    Handles:
      - 11970.0 -> 11970
      - " 19342 " -> 19342
      - "19342,19380" or "19342; 19380" -> ["19342","19380"]
      - non-numeric tokens like S2068 retained as-is (uppercased)
    """
    if raw is None:
        return []

    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return []

    # Split on common separators
    parts = re.split(r"[,\;/\|\s]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        # Remove trailing .0, .00 if purely numeric floats
        m = re.match(r"^(\d+)\.0+$", p)
        if m:
            p = m.group(1)

        # Uppercase alpha CPT-like codes (e.g., S2068)
        p = p.upper()

        # Keep only plausible tokens (alnum)
        p = re.sub(r"[^A-Z0-9]", "", p)
        if p:
            out.append(p)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def has_any_stage2_cpt(cpt_list):
    for c in cpt_list:
        if c in STAGE2_CPT_HINTS:
            return True
    return False


# -------------------------
# Main audits
# -------------------------
def main():
    df = read_csv_fallback(OP_ENCOUNTERS_CSV)

    # Validate
    for c in [COL_PATIENT, COL_PROC]:
        if c not in df.columns:
            raise RuntimeError("Missing required column in OP encounters file: {}".format(c))

    # Working columns
    df["patient_id"] = df[COL_PATIENT].fillna("").astype(str)
    df["mrn"] = df[COL_MRN].fillna("").astype(str) if COL_MRN in df.columns else ""
    df["proc_norm"] = df[COL_PROC].apply(norm_proc)
    df["op_date"] = parse_date_series(df, COL_OP_DATE, fallback_col=COL_ALT_DATE)
    df["cpt_raw"] = df[COL_CPT] if COL_CPT in df.columns else ""

    # Drop missing patient_id
    df = df[df["patient_id"].str.len() > 0].copy()

    # CPT normalization
    df["cpt_list"] = df["cpt_raw"].apply(normalize_cpt_to_list)
    df["has_stage2_cpt_hint"] = df["cpt_list"].apply(has_any_stage2_cpt)

    # Basic flags
    df["is_expander_any"] = df["proc_norm"].apply(lambda t: bool(EXPANDER_ANY_RE.search(t)))
    df["is_stage2_text"] = df["proc_norm"].apply(lambda t: bool(STAGE2_TEXT_RE.search(t)))
    df["has_any_date"] = df["op_date"].notnull()

    # =========================
    # (A) FULL PROC INVENTORY (not only top)
    # =========================
    proc_inv = (
        df.groupby("proc_norm")
          .agg(
              encounter_rows=("proc_norm", "size"),
              unique_patients=("patient_id", lambda x: int(x.nunique())),
              unique_cpt_codes=("cpt_list", lambda x: int(len(set([c for sub in x for c in sub])))),
              any_expander=("is_expander_any", "max"),
              any_stage2_text=("is_stage2_text", "max"),
              any_stage2_cpt_hint=("has_stage2_cpt_hint", "max"),
          )
          .reset_index()
          .sort_values(["encounter_rows", "unique_patients"], ascending=False)
    )
    proc_inv.to_csv("qa_op_proc_inventory.csv", index=False)

    # =========================
    # (B) CPT INVENTORY (raw + normalized)
    # =========================
    # Raw
    if COL_CPT in df.columns:
        cpt_raw_inv = (
            df.assign(cpt_raw_str=df["cpt_raw"].fillna("").astype(str))
              .groupby("cpt_raw_str")
              .agg(
                  encounter_rows=("cpt_raw_str", "size"),
                  unique_patients=("patient_id", lambda x: int(x.nunique())),
                  unique_procedures=("proc_norm", lambda x: int(x.nunique())),
              )
              .reset_index()
              .sort_values(["encounter_rows", "unique_patients"], ascending=False)
        )
    else:
        cpt_raw_inv = pd.DataFrame(columns=["cpt_raw_str", "encounter_rows", "unique_patients", "unique_procedures"])
    cpt_raw_inv.to_csv("qa_op_cpt_inventory_raw.csv", index=False)

    # Normalized: explode CPT list
    cpt_norm = df[["patient_id", "proc_norm", "cpt_list"]].copy()
    cpt_norm = cpt_norm.explode("cpt_list")
    cpt_norm = cpt_norm.rename(columns={"cpt_list": "cpt_norm"})
    cpt_norm = cpt_norm[cpt_norm["cpt_norm"].notnull() & (cpt_norm["cpt_norm"].astype(str).str.len() > 0)].copy()

    if not cpt_norm.empty:
        cpt_norm_inv = (
            cpt_norm.groupby("cpt_norm")
                    .agg(
                        encounter_rows=("cpt_norm", "size"),
                        unique_patients=("patient_id", lambda x: int(x.nunique())),
                        unique_procedures=("proc_norm", lambda x: int(x.nunique())),
                    )
                    .reset_index()
                    .sort_values(["encounter_rows", "unique_patients"], ascending=False)
        )
    else:
        cpt_norm_inv = pd.DataFrame(columns=["cpt_norm", "encounter_rows", "unique_patients", "unique_procedures"])
    cpt_norm_inv.to_csv("qa_op_cpt_inventory_norm.csv", index=False)

    # =========================
    # (C) MULTIPLICITY AUDIT (per-patient distinct op dates)
    # =========================
    # Only keep dated rows for date-count audit
    df_dated = df[df["op_date"].notnull()].copy()

    # Per patient: number of encounter rows, distinct dates, expander presence, stage2 text presence, stage2 cpt hint presence
    patient_dates = (
        df_dated.groupby("patient_id")
                .agg(
                    mrn=("mrn", "first"),
                    encounter_rows=("patient_id", "size"),
                    distinct_op_dates=("op_date", lambda x: int(x.dt.date.nunique())),
                    min_op_date=("op_date", "min"),
                    max_op_date=("op_date", "max"),
                    has_expander_any=("is_expander_any", "max"),
                    has_stage2_text=("is_stage2_text", "max"),
                    has_stage2_cpt_hint=("has_stage2_cpt_hint", "max"),
                )
                .reset_index()
    )
    patient_dates["date_span_days"] = (patient_dates["max_op_date"] - patient_dates["min_op_date"]).dt.days
    patient_dates.to_csv("qa_op_patient_date_counts.csv", index=False)

    # Define "expander patients" for subsequent audits
    expander_patients = set(patient_dates.loc[patient_dates["has_expander_any"] == True, "patient_id"].astype(str).tolist())

    # =========================
    # (D) POST-EXPANDER NEXT EVENT AUDIT
    #     For each expander patient:
    #       - find earliest expander row date (index-ish)
    #       - find the very next dated encounter AFTER that date (any procedure)
    #       - capture its procedure_norm + CPTs
    # =========================
    next_rows = []

    for pid, sub in df_dated[df_dated["patient_id"].isin(expander_patients)].groupby("patient_id", sort=True):
        sub = sub.sort_values("op_date").copy()

        exp_rows = sub[sub["is_expander_any"] == True]
        if exp_rows.empty:
            continue

        idx_row = exp_rows.iloc[0]
        idx_date = idx_row["op_date"]

        # next event strictly after index date
        after = sub[sub["op_date"] > idx_date]
        if after.empty:
            # still record patient with no next event
            next_rows.append({
                "patient_id": pid,
                "mrn": idx_row.get("mrn", ""),
                "index_expander_date": idx_date.strftime("%Y-%m-%d") if pd.notnull(idx_date) else None,
                "index_expander_proc": idx_row.get("proc_norm", ""),
                "next_date": None,
                "next_proc": None,
                "next_cpt_raw": None,
                "next_cpt_norm_list": None,
                "next_is_stage2_text": 0,
                "next_has_stage2_cpt_hint": 0,
            })
            continue

        nxt = after.iloc[0]
        nxt_date = nxt["op_date"]

        next_rows.append({
            "patient_id": pid,
            "mrn": nxt.get("mrn", ""),
            "index_expander_date": idx_date.strftime("%Y-%m-%d") if pd.notnull(idx_date) else None,
            "index_expander_proc": idx_row.get("proc_norm", ""),
            "next_date": nxt_date.strftime("%Y-%m-%d") if pd.notnull(nxt_date) else None,
            "next_proc": nxt.get("proc_norm", ""),
            "next_cpt_raw": str(nxt.get("cpt_raw", "")),
            "next_cpt_norm_list": "|".join(nxt.get("cpt_list", [])),
            "next_is_stage2_text": int(bool(nxt.get("is_stage2_text", False))),
            "next_has_stage2_cpt_hint": int(bool(nxt.get("has_stage2_cpt_hint", False))),
        })

    next_df = pd.DataFrame(next_rows)
    next_df.to_csv("qa_op_expander_post_index_next_event.csv", index=False)

    # Aggregate next-event types
    if not next_df.empty:
        next_counts = (
            next_df.groupby(["next_proc"])
                   .agg(
                       patients=("patient_id", lambda x: int(x.nunique())),
                       rows=("patient_id", "size"),
                       any_stage2_text=("next_is_stage2_text", "max"),
                       any_stage2_cpt_hint=("next_has_stage2_cpt_hint", "max"),
                   )
                   .reset_index()
                   .sort_values(["patients", "rows"], ascending=False)
        )
    else:
        next_counts = pd.DataFrame(columns=["next_proc", "patients", "rows", "any_stage2_text", "any_stage2_cpt_hint"])
    next_counts.to_csv("qa_op_expander_next_event_counts.csv", index=False)

    # =========================
    # (E) EXPLICIT STAGE2 CANDIDATES AFTER INDEX (expander patients)
    #     Find any rows after expander index with:
    #       - stage2 text OR stage2 CPT hint
    # =========================
    cand_rows = []
    for pid, sub in df_dated[df_dated["patient_id"].isin(expander_patients)].groupby("patient_id", sort=True):
        sub = sub.sort_values("op_date").copy()
        exp_rows = sub[sub["is_expander_any"] == True]
        if exp_rows.empty:
            continue
        idx_date = exp_rows.iloc[0]["op_date"]

        after = sub[sub["op_date"] > idx_date].copy()
        if after.empty:
            continue

        after["is_stage2_candidate"] = after.apply(
            lambda r: bool(r["is_stage2_text"]) or bool(r["has_stage2_cpt_hint"]),
            axis=1
        )
        after = after[after["is_stage2_candidate"] == True].copy()
        if after.empty:
            continue

        # Keep first candidate only per patient for summary (you can change if needed)
        r = after.iloc[0]
        reason = []
        if bool(r["is_stage2_text"]):
            reason.append("text")
        if bool(r["has_stage2_cpt_hint"]):
            reason.append("cpt_hint")

        cand_rows.append({
            "patient_id": pid,
            "mrn": r.get("mrn", ""),
            "index_expander_date": idx_date.strftime("%Y-%m-%d") if pd.notnull(idx_date) else None,
            "candidate_date": r["op_date"].strftime("%Y-%m-%d") if pd.notnull(r["op_date"]) else None,
            "candidate_proc": r.get("proc_norm", ""),
            "candidate_cpt_raw": str(r.get("cpt_raw", "")),
            "candidate_cpt_norm_list": "|".join(r.get("cpt_list", [])),
            "candidate_reason": "+".join(reason) if reason else "unknown",
        })

    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv("qa_op_expander_stage2_candidate_after_index.csv", index=False)

    # =========================
    # Terminal summary (concise)
    # =========================
    total_patients = int(df["patient_id"].nunique())
    total_patients_dated = int(df_dated["patient_id"].nunique())
    exp_patients_n = int(len(expander_patients))

    # multiplicity for expander patients
    exp_pd = patient_dates[patient_dates["patient_id"].isin(expander_patients)].copy()
    exp_two_dates = int((exp_pd["distinct_op_dates"] >= 2).sum())
    exp_three_dates = int((exp_pd["distinct_op_dates"] >= 3).sum())

    # stage2 candidates after index among expander patients
    exp_with_candidate = int(cand_df["patient_id"].nunique()) if not cand_df.empty else 0

    print("\n=== OP Encounters Stage2 Audit Summary ===")
    print("Total patients (any rows): {}".format(total_patients))
    print("Total patients (with >=1 parsed date): {}".format(total_patients_dated))
    print("Expander patients (any expander-ish proc text): {}".format(exp_patients_n))
    if exp_patients_n:
        print("  Expander patients with >=2 distinct op dates: {} ({:.1f}%)".format(
            exp_two_dates, 100.0 * exp_two_dates / exp_patients_n
        ))
        print("  Expander patients with >=3 distinct op dates: {} ({:.1f}%)".format(
            exp_three_dates, 100.0 * exp_three_dates / exp_patients_n
        ))
        print("  Expander patients with a Stage2 candidate AFTER index: {} ({:.1f}%)".format(
            exp_with_candidate, 100.0 * exp_with_candidate / exp_patients_n
        ))

    print("\nWrote CSV outputs:")
    outs = [
        "qa_op_proc_inventory.csv",
        "qa_op_cpt_inventory_raw.csv",
        "qa_op_cpt_inventory_norm.csv",
        "qa_op_patient_date_counts.csv",
        "qa_op_expander_post_index_next_event.csv",
        "qa_op_expander_next_event_counts.csv",
        "qa_op_expander_stage2_candidate_after_index.csv",
    ]
    for o in outs:
        print("  - {}".format(o))

    # Quick top previews
    print("\nTop procedures (full inventory):")
    print(proc_inv[["proc_norm", "encounter_rows", "unique_patients", "unique_cpt_codes"]].head(TOP_N_PRINT).to_string(index=False))

    print("\nTop normalized CPTs:")
    if not cpt_norm_inv.empty:
        print(cpt_norm_inv.head(TOP_N_PRINT).to_string(index=False))
    else:
        print("(No CPTs found after normalization)")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
