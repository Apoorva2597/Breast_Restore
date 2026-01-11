import re
from typing import Dict, List

# -------------------------------------------------------------------
# Canonical sections: tuned for Breast RESTORE clinic + op notes
# -------------------------------------------------------------------
HEADER_CANON = {
    # Clinic / Epic headings
    "REASON FOR VISIT": "REASON FOR VISIT",
    "CHIEF COMPLAINT": "CHIEF COMPLAINT",
    "CC": "CHIEF COMPLAINT",

    "HPI": "HPI",
    "HISTORY OF PRESENT ILLNESS": "HPI",
    "HISTORY OF PRESENT ILLNESS (HPI)": "HPI",

    "DIAGNOSIS": "DIAGNOSIS",
    "IMAGING": "IMAGING",
    "DIAGNOSTIC IMAGING": "IMAGING",
    "PATHOLOGY": "PATHOLOGY",
    "PATHOLOGY REVIEW": "PATHOLOGY",

    "PAST MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "PAST SURGICAL HISTORY": "PAST SURGICAL HISTORY",
    "FAMILY HISTORY": "FAMILY HISTORY",

    "MEDICATIONS": "MEDICATIONS",
    "CURRENT MEDICATIONS": "MEDICATIONS",
    "OUTPATIENT MEDICATIONS PRIOR TO VISIT": "MEDICATIONS",

    "ALLERGIES": "ALLERGIES",
    "SOCIAL HISTORY": "SOCIAL HISTORY",
    "REVIEW OF SYSTEMS": "REVIEW OF SYSTEMS",

    "PHYSICAL EXAM": "PHYSICAL EXAM",
    "OBJECTIVE": "PHYSICAL EXAM",

    "ASSESSMENT": "ASSESSMENT/PLAN",
    "ASSESSMENT/PLAN": "ASSESSMENT/PLAN",
    "ASSESSMENT AND PLAN": "ASSESSMENT/PLAN",
    "PLAN": "ASSESSMENT/PLAN",
    "PLAN OF CARE": "ASSESSMENT/PLAN",
    "ONCOLOGY CARE MODEL DOCUMENTATION REQUIREMENT": "ASSESSMENT/PLAN",

    # Op note headings – keep as distinct sections
    "OP NOTE": "OP NOTE",
    "OPERATIVE REPORT": "OP NOTE",

    "PREOPERATIVE DIAGNOSIS": "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS": "POSTOPERATIVE DIAGNOSIS",
    "PROCEDURE": "PROCEDURE",
    "ATTENDING SURGEON": "ATTENDING SURGEON",
    "ASSISTANT": "ASSISTANT",
    "ANESTHESIA": "ANESTHESIA",
    "IV FLUIDS": "IV FLUIDS",
    "ESTIMATED BLOOD LOSS": "ESTIMATED BLOOD LOSS",
    "URINE OUTPUT": "URINE OUTPUT",
    "MICRO SURGICAL DETAILS": "MICRO SURGICAL DETAILS",
    "COMPLICATIONS": "COMPLICATIONS",
    "CONDITION AT THE END OF THE PROCEDURE": "CONDITION AT THE END OF THE PROCEDURE",
    "DISPOSITION": "DISPOSITION",
    "INDICATIONS FOR OPERATION": "INDICATIONS FOR OPERATION",
    "DETAILS OF OPERATION": "DETAILS OF OPERATION",
}

# Allow ALL-CAPS + colon headings only if on this strict list (prevents junk splits)
STRICT_ALLCAPS_COLON = {
    # Legacy op-note patterns from earlier samples (harmless to keep)
    "PREOP DIAGNOSES",
    "POSTOP DIAGNOSES",
    "PROCEDURES PERFORMED",
    "DETAILS OF THE PROCEDURE",
    "MICROSURGICAL DETAILS",
    "DISPOSITION",

    # Breast RESTORE op-note headings (all caps with colon)
    "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS",
    "PROCEDURE",
    "ATTENDING SURGEON",
    "ASSISTANT",
    "ANESTHESIA",
    "IV FLUIDS",
    "ESTIMATED BLOOD LOSS",
    "URINE OUTPUT",
    "MICRO SURGICAL DETAILS",
    "COMPLICATIONS",
    "CONDITION AT THE END OF THE PROCEDURE",
    "INDICATIONS FOR OPERATION",
    "DETAILS OF OPERATION",
}

# Epic sometimes uses title-case headings without colon; keep only safest ones
TITLECASE_ALLOW = {"REASON FOR VISIT", "OP NOTE"}

# -------------------------------------------------------------------
# Guardrails: prevent false headings from tables/lists
# -------------------------------------------------------------------
TABLEISH_RE = re.compile(r".+\|.+")                 # "A | B | C"
NUMBERED_ITEM_RE = re.compile(r"^\s*\d+\.\s+\w+")   # "1. ..."
BULLET_RE = re.compile(r"^\s*[•\-\*]\s+")           # bullet lines

# -------------------------------------------------------------------
# Disambiguation: Radiology "PHYSICAL EXAM:" sometimes appears inside IMAGING blocks
# -------------------------------------------------------------------
IMAGING_CUES_RE = re.compile(
    r"\b("
    r"MAMMOGRAPHIC\s+FINDINGS|ULTRASOUND\s+FINDINGS|BI-?RADS|"
    r"COMPUTER-?AIDED\s+DETECTION|TARGETED\s+ULTRASOUND|"
    r"COMPARISON\s*:|CLINICAL\s+INDICATION\s*:|PER\s+TECHNOLOGIST"
    r")\b",
    re.IGNORECASE,
)

# Clinic-style exam cues: if these appear, it's likely a real clinic exam section
CLINIC_EXAM_CUES_RE = re.compile(
    r"\b("
    r"\bBP\b|\bPulse\b|\bTemp\b|\bResp\b|\bHt\b|\bWt\b|\bBMI\b|"
    r"General\s+appearance|No\s+acute\s+distress|"
    r"Heart\b|Lungs?\b|Abdomen\b|Extremities\b|"
    r"Breasts?\b|Axillary\b|Lymph\s+nodes?\b"
    r")\b",
    re.IGNORECASE,
)


def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_trailing_colon(s: str) -> str:
    s = s.strip()
    return s[:-1].strip() if s.endswith(":") else s


def _key(line: str) -> str:
    return _clean_spaces(_strip_trailing_colon(line)).upper()


def _canon(line: str) -> str:
    """
    Convert a heading line into a canonical section key.
    Supports inline headings like "HPI: blah" / "CC: blah".
    """
    if ":" in line:
        left = _clean_spaces(line.split(":", 1)[0]).upper()
        if left in HEADER_CANON:
            return HEADER_CANON[left]
        if left in {"CC", "HPI"}:
            return HEADER_CANON.get(left, left)
    k = _key(line)
    return HEADER_CANON.get(k, _clean_spaces(_strip_trailing_colon(line)))


def _looks_like_heading(raw_line: str) -> bool:
    raw = raw_line.strip()
    if not raw:
        return False

    # Reject obvious non-headings early
    if TABLEISH_RE.match(raw):
        return False
    if NUMBERED_ITEM_RE.match(raw):
        return False
    if BULLET_RE.match(raw):
        return False

    k = _key(raw)

    # 1) Exact known headings always win
    if k in HEADER_CANON:
        return True

    # 2) "Heading:" where Heading is known
    if raw.endswith(":"):
        left = _clean_spaces(raw[:-1]).upper()
        if left in HEADER_CANON:
            return True

    # 3) Prefix style: "HPI: blah", "CC: blah"
    if ":" in raw:
        left = _clean_spaces(raw.split(":", 1)[0]).upper()
        if left in HEADER_CANON or left in {"CC", "HPI"}:
            return True

    # 4) Allow certain title-case headings without colon
    if k in TITLECASE_ALLOW:
        return True

    # 5) Strict ALLCAPS + colon (op-note style) only for a tight list
    if raw.endswith(":") and raw.isupper():
        left = _clean_spaces(raw[:-1]).upper()
        if left in STRICT_ALLCAPS_COLON:
            return True

    return False


def _disambiguate_heading(canon_heading: str, lookahead_text: str) -> str:
    """
    Fix known ambiguity:
      - Radiology blocks sometimes include "PHYSICAL EXAM:" inside IMAGING.
    If we see imaging cues soon after PHYSICAL EXAM, treat as IMAGING.
    """
    if canon_heading == "PHYSICAL EXAM":
        if IMAGING_CUES_RE.search(lookahead_text) and not CLINIC_EXAM_CUES_RE.search(lookahead_text):
            return "IMAGING"
    return canon_heading


def sectionize(text: str, lookahead_lines: int = 12) -> Dict[str, str]:
    """
    Split note text into sections.

    Output keys are canonical headings (see HEADER_CANON), plus "__PREAMBLE__".
    """
    lines = text.splitlines()

    sections = {}  # type: Dict[str, List[str]]
    current = "__PREAMBLE__"
    sections[current] = []

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip("\n")

        if _looks_like_heading(ln):
            canon = _canon(ln)

            # Lookahead window for ambiguity resolution
            j_end = min(len(lines), i + 1 + lookahead_lines)
            lookahead = "\n".join(lines[i + 1 : j_end])
            canon = _disambiguate_heading(canon, lookahead)

            current = canon
            sections.setdefault(current, [])

            # If line is "HPI: blah", keep "blah" as the first content line
            if ":" in ln:
                left, right = ln.split(":", 1)
                left_key = _clean_spaces(left).upper()
                if left_key in HEADER_CANON or left_key in {"CC", "HPI"}:
                    right = right.strip()
                    if right:
                        sections[current].append(right)

            i += 1
            continue

        if ln.strip():
            sections[current].append(ln)

        i += 1

    # Join and drop empties
    out = {}  # type: Dict[str, str]
    for k, chunk_lines in sections.items():
        body = "\n".join(chunk_lines).strip()
        if body:
            out[k] = body
    return out
