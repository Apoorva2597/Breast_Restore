#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# CONFIG (edit only if needed)
# ----------------------------
PROJECT_DIR="/home/apokol/Breast_Restore"
SCRIPT_NAME="build_stage12_WITH_AUDIT.py"

OUT_DIR="${PROJECT_DIR}/_outputs"
FROZEN_DIR="${PROJECT_DIR}/_frozen_rules"

# If your script writes to these fixed filenames, we will version-copy them after it runs:
BASE_SUMMARY="${OUT_DIR}/patient_stage_summary.csv"
BASE_HITS="${OUT_DIR}/stage2_event_hits.csv"

# ----------------------------
# VERSION TAG
# ----------------------------
TS="$(date +%Y%m%d_%H%M%S)"
VERSION="stage2_rules_${TS}"

mkdir -p "${FROZEN_DIR}" "${OUT_DIR}"

echo "==> Freezing ruleset as: ${VERSION}"

# ----------------------------
# Freeze the script (exact copy)
# ----------------------------
SRC_SCRIPT="${PROJECT_DIR}/${SCRIPT_NAME}"
FROZEN_SCRIPT="${FROZEN_DIR}/${VERSION}__${SCRIPT_NAME}"

if [[ ! -f "${SRC_SCRIPT}" ]]; then
  echo "ERROR: Cannot find ${SRC_SCRIPT}"
  exit 1
fi

cp -p "${SRC_SCRIPT}" "${FROZEN_SCRIPT}"
echo "==> Saved frozen script: ${FROZEN_SCRIPT}"

# ----------------------------
# Optional: capture git hash if repo
# ----------------------------
GIT_HASH="NA"
if command -v git >/dev/null 2>&1; then
  if git -C "${PROJECT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_HASH="$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo NA)"
  fi
fi

# ----------------------------
# Run the frozen script
# ----------------------------
echo "==> Running frozen script..."
python "${FROZEN_SCRIPT}"

# ----------------------------
# Version the outputs
# ----------------------------
V_SUMMARY="${OUT_DIR}/${VERSION}__patient_stage_summary.csv"
V_HITS="${OUT_DIR}/${VERSION}__stage2_event_hits.csv"

if [[ ! -f "${BASE_SUMMARY}" ]]; then
  echo "ERROR: Expected summary not found: ${BASE_SUMMARY}"
  exit 1
fi

cp -p "${BASE_SUMMARY}" "${V_SUMMARY}"
echo "==> Versioned summary: ${V_SUMMARY}"

if [[ -f "${BASE_HITS}" ]]; then
  cp -p "${BASE_HITS}" "${V_HITS}"
  echo "==> Versioned hits: ${V_HITS}"
else
  echo "WARN: Hits file not found (skipping): ${BASE_HITS}"
fi

# ----------------------------
# Write manifest
# ----------------------------
MANIFEST="${FROZEN_DIR}/${VERSION}__MANIFEST.txt"
cat > "${MANIFEST}" <<EOF
VERSION=${VERSION}
TIMESTAMP=${TS}
PROJECT_DIR=${PROJECT_DIR}
FROZEN_SCRIPT=${FROZEN_SCRIPT}
SUMMARY_VERSIONED=${V_SUMMARY}
HITS_VERSIONED=${V_HITS}
GIT_HASH=${GIT_HASH}
EOF

echo "==> Manifest: ${MANIFEST}"
echo "==> Done."
