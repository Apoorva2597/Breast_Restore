#!/usr/bin/env bash

set -e

LOG_DIR="/home/apokol/Breast_Restore/QA_DEID_BUNDLES/logs"
OUT_FILE="${LOG_DIR}/ALL_OUT_COMBINED.txt"

cd "$LOG_DIR"

echo "Combining .out.txt files..."
echo "Created: $(date)" > "$OUT_FILE"
echo "=============================================" >> "$OUT_FILE"

for f in *.out.txt; do
    echo "" >> "$OUT_FILE"
    echo "=============================================" >> "$OUT_FILE"
    echo "FILE: $f" >> "$OUT_FILE"
    echo "=============================================" >> "$OUT_FILE"
    cat "$f" >> "$OUT_FILE"
done

echo ""
echo "Done."
echo "Combined file: $OUT_FILE"
