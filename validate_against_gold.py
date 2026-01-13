#!/usr/bin/env python3
"""
Validate NLP-extracted variables against the gold Excel sheet.

Inputs (all in current directory by default):
  - gold_variables.csv           (team's gold data; patient-level)
  - all_phase2_final.csv         (Phase 2 resolved candidates; note-level)
  - gold_extracted_links.csv     (link: PatientID <-> note_id from age/BMI)

Output:
  - validation_patient_level.csv (one row per PatientID with gold vs extracted)
And prints simple accuracy stats per variable.

Compatible with Python 3.6.x.
"""

import csv
from collections import defaultdict

# -------------------------------------------------------------------------
# File names (edit here if needed)
# -------------------------------------------------------------------------
GOLD_FILE = "gold_variables.csv"
PHASE2_FILE = "all_phase2_final.csv"
LINK_FILE = "gold_extracted_links.csv"
OUT_FILE = "validation_patient_level.csv"

# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------


def _clean_header_name(raw):
    """
    Convert '4. Age' -> 'Age', '12 . VenousThromboembolism' -> 'VenousThromboembolism'.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    # Strip leading numbering like '4.', '12 .'
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] in " ."):
        i += 1
    return s[i:].strip()


def _to_float(val):
    try:
        return float(str(val).strip())
    except Exception:
        return None


def _to_bool(val, status=None):
    """
    Convert a value + optional status to boolean.
    Returns True, False, or None (unknown/missing).
    """
    if status:
        s = str(status).strip().lower()
        if s == "denied":
            return False
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "y"):
        return True
    if v in ("0", "false", "no", "n", ""):
        return False
    # Sometimes we wrote Python bools which will be caught above, but just in case
    if v == "none":
        return None
    return None


def _norm_smoking(val):
    if val is None:
        return None
    v = str(val).strip().lower()
    if not v:
        return None
    if "never" in v:
        return "never"
    if "former" in v or "quit" in v or "ex-smoker" in v:
        return "former"
    if "current" in v or "smokes" in v or "daily" in v:
        return "current"
    return v  # fall back to raw normalized text


def _norm_str(val):
    if val is None:
        return None
    v = str(val).strip()
    return v if v else None


def _aggregate_for_patient(note_ids, extracted_by_note, field, field_type):
    """
    Aggregate note-level values into a single patient-level value.

    field_type: "numeric", "binary", or "categorical"
    """
    values = []

    for nid in note_ids:
        note_fields = extracted_by_note.get(nid)
        if not note_fields:
            continue
        row = note_fields.get(field)
        if not row:
            continue
        val = row.get("value")
        status = (row.get("status") or "").lower()

        if field_type == "binary":
            b = _to_bool(val, status)
            if b is not None:
                values.append(b)
        elif field_type == "numeric":
            f = _to_float(val)
            if f is not None:
                values.append(f)
        else:  # categorical
            v = _norm_str(val)
            if v is not None:
                values.append(v.lower())

    if not values:
        return None

    if field_type == "binary":
        # Any True → True; otherwise False
        return True if any(values) else False

    if field_type == "numeric":
        # Take the first; Age/BMI should be consistent across notes
        return values[0]

    # categorical
    uniq = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
    if len(uniq) == 1:
        return uniq[0]
    return "mixed"


# -------------------------------------------------------------------------
# 1. Load gold_variables.csv (patient-level)
# -------------------------------------------------------------------------


def load_gold():
    """
    Returns:
      gold_by_patient:  PatientID -> {canonical_col_name: value}
      gold_header:      list of canonical column names
    """
    with open(GOLD_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Find header row (the one containing 'PatientID')
    header_row_idx = None
    for i, row in enumerate(rows):
        joined = " ".join(row)
        if "PatientID" in joined:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise RuntimeError("Could not find header row with 'PatientID' in gold file.")

    raw_header = rows[header_row_idx]
    header = [_clean_header_name(c) for c in raw_header]

    # Build patient-level dict
    gold_by_patient = {}
    for row in rows[header_row_idx + 1 :]:
        if not row or all((c is None or str(c).strip() == "") for c in row):
            continue
        # Use first column (canonical 'PatientID') as key
        if len(row) < 1:
            continue
        patient_id = str(row[0]).strip()
        if not patient_id:
            continue
        rec = {}
        for col_name, val in zip(header, row):
            rec[col_name] = val
        gold_by_patient[patient_id] = rec

    return gold_by_patient, header


# -------------------------------------------------------------------------
# 2. Load all_phase2_final.csv (note-level candidates)
# -------------------------------------------------------------------------


def load_extracted():
    """
    Returns:
      extracted_by_note:  note_id -> {field_name: row_dict}
    """
    extracted_by_note = defaultdict(dict)
    with open(PHASE2_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            note_id = row.get("note_id")
            field = row.get("field")
            if not note_id or not field:
                continue
            extracted_by_note[note_id][field] = row
    return extracted_by_note


# -------------------------------------------------------------------------
# 3. Load gold_extracted_links.csv (PatientID <-> note_id mapping)
# -------------------------------------------------------------------------


def load_links():
    """
    Returns:
      patient_to_notes: PatientID -> set([note_id, ...])
    """
    patient_to_notes = defaultdict(set)

    with open(LINK_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Expect columns like: PatientID, note_id, note_type, note_date, Age_gold, BMI_gold, Age_DOS, BMI
        for row in reader:
            pid = row.get("PatientID")
            nid = row.get("note_id")
            if not pid or not nid:
                continue
            pid = str(pid).strip()
            nid = str(nid).strip()
            if pid and nid:
                patient_to_notes[pid].add(nid)

    return patient_to_notes


# -------------------------------------------------------------------------
# 4. Validation mappings
# -------------------------------------------------------------------------

# Each entry:
#   name           : label used in output
#   gold_col       : canonical column name in gold sheet
#   extracted_field: field name in Phase2 file
#   type           : "numeric" | "binary" | "categorical"
#   norm_gold      : optional custom normalizer for gold values
#   norm_extracted : optional custom normalizer for extracted values
FIELD_MAPPINGS = [
    # Demographics
    {
        "name": "Age",
        "gold_col": "Age",
        "extracted_field": "Age_DOS",
        "type": "numeric",
    },
    {
        "name": "BMI",
        "gold_col": "BMI",
        "extracted_field": "BMI",
        "type": "numeric",
    },
    # Smoking
    {
        "name": "SmokingStatus",
        "gold_col": "SmokingStatus",
        "extracted_field": "SmokingStatus",
        "type": "categorical",
        "norm_gold": _norm_smoking,
        "norm_extracted": _norm_smoking,
    },
    # Comorbidities (binary flags)
    {
        "name": "Diabetes",
        "gold_col": "Diabetes",
        "extracted_field": "DiabetesMellitus",
        "type": "binary",
    },
    {
        "name": "Hypertension",
        "gold_col": "Hypertension",
        "extracted_field": "Hypertension",
        "type": "binary",
    },
    {
        "name": "CardiacDisease",
        "gold_col": "CardiacDisease",
        "extracted_field": "CardiacDisease",
        "type": "binary",
    },
    {
        "name": "VTE",
        "gold_col": "VenousThromboembolism",
        "extracted_field": "VTE",
        "type": "binary",
    },
    {
        "name": "Steroid",
        "gold_col": "Steroid",
        "extracted_field": "SteroidUse",
        "type": "binary",
    },
    # PBS + mastectomy + lymph node
    {
        "name": "PBS_Lumpectomy",
        "gold_col": "PBS_Lumpectomy",
        "extracted_field": "PBS_Lumpectomy",
        "type": "binary",
    },
    {
        "name": "PBS_Other",
        "gold_col": "PBS_Other",
        "extracted_field": "PBS_Other",
        "type": "binary",
    },
    {
        "name": "Mastectomy_Laterality",
        "gold_col": "Mastectomy_Laterality",
        "extracted_field": "Mastectomy_Laterality",
        "type": "categorical",
        "norm_gold": _norm_str,
        "norm_extracted": _norm_str,
    },
    {
        "name": "LymphNode",
        "gold_col": "LymphNode",
        "extracted_field": "LymphNodeMgmt",
        "type": "categorical",
        "norm_gold": _norm_str,
        "norm_extracted": _norm_str,
    },
    # Reconstruction
    {
        "name": "ReconLaterality",
        "gold_col": "ReconLaterality",
        "extracted_field": "Recon_Laterality",
        "type": "categorical",
        "norm_gold": _norm_str,
        "norm_extracted": _norm_str,
    },
    {
        "name": "Recon_Type",
        "gold_col": "Recon_Type",
        "extracted_field": "Recon_Type",
        "type": "categorical",
        "norm_gold": _norm_str,
        "norm_extracted": _norm_str,
    },
    {
        "name": "Recon_Timing",
        "gold_col": "Recon_Timing",
        "extracted_field": "Recon_Timing",
        "type": "categorical",
        "norm_gold": _norm_str,
        "norm_extracted": _norm_str,
    },
]


# -------------------------------------------------------------------------
# 5. Main validation logic
# -------------------------------------------------------------------------


def main():
    # Load all inputs
    print("Loading gold sheet...")
    gold_by_patient, gold_header = load_gold()
    print("  Gold patients:", len(gold_by_patient))

    print("Loading extracted Phase 2 file...")
    extracted_by_note = load_extracted()
    print("  Notes with extractions:", len(extracted_by_note))

    print("Loading PatientID <-> note_id links...")
    patient_to_notes = load_links()
    print("  Patients with linked notes:", len(patient_to_notes))

    # Prepare stats
    stats = {}
    for fm in FIELD_MAPPINGS:
        stats[fm["name"]] = {"n": 0, "matches": 0}

    # Output rows
    out_rows = []

    all_patient_ids = sorted(patient_to_notes.keys())
    for pid in all_patient_ids:
        gold_row = gold_by_patient.get(pid)
        if not gold_row:
            # Linked, but not present in gold (should be rare)
            continue

        note_ids = sorted(patient_to_notes[pid])

        out_rec = {
            "PatientID": pid,
            "note_ids": ";".join(note_ids),
        }

        for fm in FIELD_MAPPINGS:
            name = fm["name"]
            gold_col = fm["gold_col"]
            field = fm["extracted_field"]
            ftype = fm["type"]
            norm_g = fm.get("norm_gold")
            norm_e = fm.get("norm_extracted")

            gold_val_raw = gold_row.get(gold_col)

            # Normalise gold
            if ftype == "numeric":
                gold_val = _to_float(gold_val_raw)
            elif ftype == "binary":
                gold_val = _to_bool(gold_val_raw)
            elif name == "SmokingStatus":
                gold_val = _norm_smoking(gold_val_raw)
            else:
                gold_val = _norm_str(gold_val_raw)

            if norm_g is not None:
                gold_val = norm_g(gold_val)

            # Aggregated extracted
            ext_val = _aggregate_for_patient(note_ids, extracted_by_note, field, ftype)
            if norm_e is not None:
                ext_val = norm_e(ext_val)

            # Store printable versions
            out_rec["gold_" + name] = "" if gold_val is None else gold_val
            out_rec["extracted_" + name] = "" if ext_val is None else ext_val

            # Compare if both present
            match = ""
            if gold_val is not None and ext_val is not None:
                stats[name]["n"] += 1
                if ftype == "numeric":
                    # within small tolerance
                    if abs(float(gold_val) - float(ext_val)) <= 0.2:
                        stats[name]["matches"] += 1
                        match = 1
                    else:
                        match = 0
                elif ftype == "binary":
                    if bool(gold_val) == bool(ext_val):
                        stats[name]["matches"] += 1
                        match = 1
                    else:
                        match = 0
                else:  # categorical
                    if str(gold_val).strip().lower() == str(ext_val).strip().lower():
                        stats[name]["matches"] += 1
                        match = 1
                    else:
                        match = 0
            out_rec["match_" + name] = match

        out_rows.append(out_rec)

    # Write output CSV
    if out_rows:
        # Build header
        base_cols = ["PatientID", "note_ids"]
        dyn_cols = []
        for fm in FIELD_MAPPINGS:
            name = fm["name"]
            dyn_cols.append("gold_" + name)
            dyn_cols.append("extracted_" + name)
            dyn_cols.append("match_" + name)
        fieldnames = base_cols + dyn_cols

        with open(OUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in out_rows:
                writer.writerow(r)

        print("Wrote {} patient rows → {}".format(len(out_rows), OUT_FILE))
    else:
        print("No overlapping patients between gold + links; nothing written.")

    # Print summary stats
    print("\n=== Validation summary ===")
    for fm in FIELD_MAPPINGS:
        name = fm["name"]
        n = stats[name]["n"]
        m = stats[name]["matches"]
        if n == 0:
            acc = 0.0
        else:
            acc = 100.0 * float(m) / float(n)
        print(
            "{:<20s}  n={:<4d}  matches={:<4d}  accuracy={:5.1f}%".format(
                name, n, m, acc
            )
        )


if __name__ == "__main__":
    main()
