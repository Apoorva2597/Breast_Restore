# match_age_bmi_gold_and_extracted.py
#
# Link gold chart-review rows to extracted rows by exact (Age,BMI) match.
# Now robust to headers like "4. Age", "5. BMI", "1. PatientID".
#
# Inputs:
#   - gold_variables.csv        (manual abstraction; per-patient)
#   - age_bmi_for_linking.csv   (from make_age_bmi_link_file.py; per-note)
#
# Output:
#   - gold_extracted_links.csv  (rows where Age+BMI match exactly)

import csv

GOLD_CSV = "gold_variables.csv"
EXTRACTED_CSV = "age_bmi_for_linking.csv"
OUTPUT_CSV = "gold_extracted_links.csv"


def _pick_col(fieldnames, keywords):
    """
    Choose the first column whose lowercase name contains *all* keywords.
    e.g. ["age"] will match "4. Age", ["patient","id"] will match "1. PatientID".
    """
    names = list(fieldnames)  # ensure indexable
    for name in names:
        norm = name.lower().replace(" ", "")
        ok = True
        for kw in keywords:
            if kw not in norm:
                ok = False
                break
        if ok:
            return name
    return None


def _safe_float(s):
    try:
        return float(str(s).strip())
    except Exception:
        return None


def _safe_int(s):
    try:
        return int(round(float(str(s).strip())))
    except Exception:
        return None


def _normalize_pair(age_val, bmi_val):
    """
    Normalize (age, bmi) for matching:
      - age as integer (years)
      - bmi rounded to 1 decimal
    Returns (age_int, bmi_1dp) or (None, None) if invalid.
    """
    age = _safe_int(age_val)
    bmi_f = _safe_float(bmi_val)
    if age is None or bmi_f is None:
        return None, None
    bmi = round(bmi_f, 1)
    return age, bmi


def main():
    # ------------------- 1) Read GOLD -------------------
    with open(GOLD_CSV, "r", newline="") as f:
        gold_reader = csv.DictReader(f)
        gold_fieldnames = gold_reader.fieldnames or []

        # Auto-detect columns
        gold_pid_col = _pick_col(gold_fieldnames, ["patient"])
        gold_age_col = _pick_col(gold_fieldnames, ["age"])
        gold_bmi_col = _pick_col(gold_fieldnames, ["bmi"])

        print("GOLD header columns:")
        print("  PatientID col:", gold_pid_col)
        print("  Age col      :", gold_age_col)
        print("  BMI col      :", gold_bmi_col)

        if not (gold_age_col and gold_bmi_col):
            print("ERROR: Could not find Age/BMI columns in gold file; "
                  "check header names.")
            return

        gold_rows = list(gold_reader)

    gold_pairs = {}  # (age,bmi) -> list of gold rows
    gold_usable = 0

    for row in gold_rows:
        age_raw = row.get(gold_age_col, "")
        bmi_raw = row.get(gold_bmi_col, "")
        age, bmi = _normalize_pair(age_raw, bmi_raw)
        if age is None or bmi is None:
            continue
        gold_usable += 1
        key = (age, bmi)
        gold_pairs.setdefault(key, []).append(row)

    print("GOLD: total rows = {}, usable Age+BMI = {}".format(
        len(gold_rows), gold_usable))
    print("GOLD: distinct (Age,BMI) pairs = {}".format(len(gold_pairs)))

    # ------------------- 2) Read EXTRACTED -------------------
    with open(EXTRACTED_CSV, "r", newline="") as f:
        ext_reader = csv.DictReader(f)
        ext_fieldnames = ext_reader.fieldnames or []

        # Expected columns from make_age_bmi_link_file.py
        note_id_col = "note_id"
        note_type_col = "note_type"
        note_date_col = "note_date"
        age_col = "Age_DOS"
        bmi_col = "BMI"

        extracted_rows = list(ext_reader)

    ext_pairs = {}  # (age,bmi) -> list of extracted rows
    ext_usable = 0

    for row in extracted_rows:
        age_raw = row.get(age_col, "")
        bmi_raw = row.get(bmi_col, "")
        age, bmi = _normalize_pair(age_raw, bmi_raw)
        if age is None or bmi is None:
            continue
        ext_usable += 1
        key = (age, bmi)
        ext_pairs.setdefault(key, []).append(row)

    print("EXTRACTED: total rows = {}, usable Age+BMI = {}".format(
        len(extracted_rows), ext_usable))
    print("EXTRACTED: distinct (Age,BMI) pairs = {}".format(len(ext_pairs)))

    # ------------------- 3) Find overlaps -------------------
    gold_keys = set(gold_pairs.keys())
    ext_keys = set(ext_pairs.keys())
    overlap_keys = sorted(gold_keys & ext_keys)

    print("Overlap: distinct (Age,BMI) pairs in common = {}".format(
        len(overlap_keys)))
    if not overlap_keys:
        print("WARNING: No shared (Age,BMI) pairs between gold and extracted.")
        print("         0 matches is expected with strict exact matching.")
        # Still write an empty file with header for convenience.

    # ------------------- 4) Write matched rows -------------------
    out_fieldnames = [
        # identifying info
        "PatientID_gold",
        "Age_gold",
        "BMI_gold",
        "note_id",
        "note_type",
        "note_date",
        "Age_extracted",
        "BMI_extracted",
        # Optional: keep the exact pair used for matching
        "Age_match",
        "BMI_match",
    ]

    written = 0
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()

        for key in overlap_keys:
            age, bmi = key
            g_list = gold_pairs.get(key, [])
            e_list = ext_pairs.get(key, [])

            for g in g_list:
                pid = g.get(gold_pid_col, "")
                g_age_raw = g.get(gold_age_col, "")
                g_bmi_raw = g.get(gold_bmi_col, "")
                for e in e_list:
                    row_out = {
                        "PatientID_gold": pid,
                        "Age_gold": g_age_raw,
                        "BMI_gold": g_bmi_raw,
                        "note_id": e.get(note_id_col, ""),
                        "note_type": e.get(note_type_col, ""),
                        "note_date": e.get(note_date_col, ""),
                        "Age_extracted": e.get("Age_DOS", ""),
                        "BMI_extracted": e.get("BMI", ""),
                        "Age_match": age,
                        "BMI_match": bmi,
                    }
                    writer.writerow(row_out)
                    written += 1

    print("Wrote {} matched rows → {}".format(written, OUTPUT_CSV))


if __name__ == "__main__":
    main()
