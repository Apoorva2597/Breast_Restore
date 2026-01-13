#!/usr/bin/env python3
# Python 3.6-compatible

import csv

INPUT = "validation_patient_level.csv"
OUTPUT = "extracted_patient_variables.csv"

PATIENT_COL = "PatientID"
EXTRACTED_PREFIX = "extracted_"


def main():
    with open(INPUT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []

        # All columns that come from the extractors
        extracted_cols = [c for c in fieldnames_in if c.startswith(EXTRACTED_PREFIX)]

        if PATIENT_COL not in fieldnames_in:
            raise SystemExit("ERROR: Could not find PatientID column in {}"
                             .format(INPUT))

        if not extracted_cols:
            raise SystemExit("ERROR: No 'extracted_*' columns found in {}; "
                             "did the validation script run?"
                             .format(INPUT))

        # Output header: PatientID + one column per extracted variable
        out_fields = [PATIENT_COL]
        for col in extracted_cols:
            # Strip the 'extracted_' prefix so names match the gold sheet style
            out_fields.append(col[len(EXTRACTED_PREFIX):])

        with open(OUTPUT, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                out_row = {PATIENT_COL: row.get(PATIENT_COL, "")}
                for col in extracted_cols:
                    short_name = col[len(EXTRACTED_PREFIX):]
                    out_row[short_name] = row.get(col, "")
                writer.writerow(out_row)

    print("Wrote extracted patient-level sheet → {}".format(OUTPUT))


if __name__ == "__main__":
    main()
