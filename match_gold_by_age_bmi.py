#!/usr/bin/env python3
"""
Match gold sheet patients to extracted notes using (Age, BMI) pairs.

- GOLD CSV:  gold_variables.csv  (with numbered headers like "4. Age", "5. BMI")
- EXTRACTED CSV: age_bmi_for_linking.csv (from make_age_bmi_for_linking.py)

Output: gold_extracted_links.csv

Compatible with Python 3.6.x.
"""

import csv
import re
from collections import defaultdict

GOLD_PATH = "gold_variables.csv"
EXTRACTED_PATH = "age_bmi_for_linking.csv"
OUTPUT_PATH = "gold_extracted_links.csv"


def _normalize_header_name(cell):
    """Strip leading numbering like '4. Age' -> 'Age', and trim spaces."""
    cell = cell.strip()
    cell = re.sub(r"^\s*\d+\.\s*", "", cell)
    return cell.strip()


def _find_gold_header_and_indices(rows):
    """
    Find the row that contains 'PatientID' and return:
      header_row_index, patient_col_index, age_col_index, bmi_col_index
    """
    header_idx = None
    for i, row in enumerate(rows):
        lowered = [c.lower() for c in row]
        if any("patientid" in c for c in lowered):
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError("Could not find a row containing 'PatientID' in gold file.")

    header = rows[header_idx]

    def find_col(target_name):
        target = target_name.lower()
        for j, cell in enumerate(header):
            clean = _normalize_header_name(cell).lower()
            if clean == target:
                return j
        return None

    pid_idx = find_col("PatientID")
    age_idx = find_col("Age")
    bmi_idx = find_col("BMI")

    print("GOLD header row index:", header_idx)
    print("GOLD raw header row:", header)
    print("Resolved columns:")
    print("  PatientID col:", header[pid_idx] if pid_idx is not None else None)
    print("  Age col      :", header[age_idx] if age_idx is not None else None)
    print("  BMI col      :", header[bmi_idx] if bmi_idx is not None else None)

    if pid_idx is None or age_idx is None or bmi_idx is None:
        raise RuntimeError("Could not resolve PatientID / Age / BMI columns in gold file.")

    return header_idx, pid_idx, age_idx, bmi_idx


def _safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_gold(path):
    """
    Return list of dicts:
      {PatientID, Age, BMI, age_norm, bmi_norm}
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header_idx, pid_idx, age_idx, bmi_idx = _find_gold_header_and_indices(rows)

    gold = []
    for row in rows[header_idx + 1 :]:
        if not row or all(not c.strip() for c in row):
            continue
        if len(row) <= max(pid_idx, age_idx, bmi_idx):
            continue

        pid = row[pid_idx].strip()
        age_str = row[age_idx].strip()
        bmi_str = row[bmi_idx].strip()

        if not pid or not age_str or not bmi_str:
            continue

        age = _safe_float(age_str)
        bmi = _safe_float(bmi_str)
        if age is None or bmi is None:
            continue

        # Normalise: age as integer, BMI to 1 decimal place
        age_norm = int(round(age))
        bmi_norm = round(bmi, 1)

        gold.append(
            {
                "PatientID": pid,
                "Age": age,
                "BMI": bmi,
                "age_norm": age_norm,
                "bmi_norm": bmi_norm,
            }
        )

    return gold


def load_extracted(path):
    """
    Read age_bmi_for_linking.csv produced by make_age_bmi_for_linking.py.

    Expected columns include:
      - note_id
      - note_type
      - note_date
      - Age_DOS (or similar)
      - BMI
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Find age and BMI columns by name (case-insensitive)
        age_col = None
        bmi_col = None
        for name in fieldnames:
            nlow = name.lower()
            if age_col is None and "age" in nlow:
                age_col = name
            if bmi_col is None and "bmi" in nlow:
                bmi_col = name

        print("EXTRACTED header columns:", fieldnames)
        print("  Using age column:", age_col)
        print("  Using BMI column:", bmi_col)

        if age_col is None or bmi_col is None:
            raise RuntimeError("Could not find Age/BMI columns in extracted file.")

        rows = []
        for row in reader:
            age_str = (row.get(age_col) or "").strip()
            bmi_str = (row.get(bmi_col) or "").strip()
            if not age_str or not bmi_str:
                continue

            age = _safe_float(age_str)
            bmi = _safe_float(bmi_str)
            if age is None or bmi is None:
                continue

            age_norm = int(round(age))
            bmi_norm = round(bmi, 1)

            rows.append(
                {
                    "note_id": row.get("note_id", ""),
                    "note_type": row.get("note_type", ""),
                    "note_date": row.get("note_date", ""),
                    "Age": age,
                    "BMI": bmi,
                    "age_norm": age_norm,
                    "bmi_norm": bmi_norm,
                }
            )

    return rows


def main():
    # ------------ Load data ------------
    gold = load_gold(GOLD_PATH)
    extracted = load_extracted(EXTRACTED_PATH)

    print("GOLD: total rows =", len(gold))
    print(
        "GOLD: distinct (age_norm, bmi_norm) pairs =",
        len({(g["age_norm"], g["bmi_norm"]) for g in gold}),
    )

    print("EXTRACTED: total rows =", len(extracted))
    print(
        "EXTRACTED: distinct (age_norm, bmi_norm) pairs =",
        len({(e["age_norm"], e["bmi_norm"]) for e in extracted}),
    )

    # ------------ Index by (age_norm, bmi_norm) ------------
    gold_index = defaultdict(list)
    for g in gold:
        gold_index[(g["age_norm"], g["bmi_norm"])].append(g)

    ext_index = defaultdict(list)
    for e in extracted:
        ext_index[(e["age_norm"], e["bmi_norm"])].append(e)

    shared_keys = sorted(set(gold_index.keys()) & set(ext_index.keys()))
    print("Overlap: shared (age_norm, bmi_norm) pairs =", len(shared_keys))

    if not shared_keys:
        print("WARNING: No shared (Age,BMI) pairs between gold and extracted after normalisation.")
        print("         This can happen if the cohorts do not overlap,")
        print("         or if the Age/BMI values are coded very differently.")

    # ------------ Build linked rows ------------
    out_rows = []
    for key in shared_keys:
        gold_list = gold_index[key]
        ext_list = ext_index[key]
        for g in gold_list:
            for e in ext_list:
                out_rows.append(
                    {
                        "PatientID": g["PatientID"],
                        "gold_Age": g["Age"],
                        "gold_BMI": g["BMI"],
                        "note_id": e["note_id"],
                        "note_type": e["note_type"],
                        "note_date": e["note_date"],
                        "extracted_Age": e["Age"],
                        "extracted_BMI": e["BMI"],
                        "age_norm": key[0],
                        "bmi_norm": key[1],
                    }
                )

    # ------------ Write output ------------
    fieldnames = [
        "PatientID",
        "gold_Age",
        "gold_BMI",
        "note_id",
        "note_type",
        "note_date",
        "extracted_Age",
        "extracted_BMI",
        "age_norm",
        "bmi_norm",
    ]

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print("Wrote {} matched rows -> {}".format(len(out_rows), OUTPUT_PATH))


if __name__ == "__main__":
    main()
