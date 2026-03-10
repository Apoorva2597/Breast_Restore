# extractors/smoking.py
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

# ----------------------------------------------
# Smoking extraction
#
# Dataset rule:
# Patients reported as having smoked in the past
# 3 months are considered Current smokers.
#
# Strategy:
# 1. Prefer structured former-status fields with quit date / years since quitting
# 2. Then explicit current-use statements
# 3. Then explicit never-use statements
# 4. Then lower-confidence screening/template answers
#
# Python 3.6.8 compatible
# ----------------------------------------------

CURRENT_PATTERNS = [
    re.compile(r"\bcurrent smoker\b", re.IGNORECASE),
    re.compile(r"\bactive smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking currently\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\b", re.IGNORECASE),
    re.compile(r"\bcurrently smokes\b", re.IGNORECASE),
    re.compile(r"\bsmokes\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+\d+", re.IGNORECASE),
    re.compile(r"\bsmokes\s+(?:a\s+)?(?:couple|few)\s+cig", re.IGNORECASE),
    re.compile(r"\btobacco use\s*[:\-]?\s*current\b", re.IGNORECASE),
]

FORMER_PATTERNS = [
    re.compile(r"\bformer smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking status\s*[:\-]?\s*former\b", re.IGNORECASE),
    re.compile(r"\bhistory smoking status\s*[:\-]?\s*former smoker\b", re.IGNORECASE),
    re.compile(r"\bprevious history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bhistory of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bprior tobacco use\b", re.IGNORECASE),
    re.compile(r"\bex[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\byears?\s+since\s+quitting\b", re.IGNORECASE),
    re.compile(r"\bquit date\s*[:\-]?\s*(?:19|20)\d{2}\b", re.IGNORECASE),
    re.compile(r"\bquit\s+(?:smoking|tobacco)[^\.]{0,60}?\d+\s*(?:year|years)\b", re.IGNORECASE),
    re.compile(r"\bquit\s+(?:smoking|tobacco)\s+about\s+\d+\s*(?:year|years)\s+ago\b", re.IGNORECASE),
    re.compile(r"\bquit\s+(?:smoking|tobacco)\s+\d+\s*(?:year|years)\s+ago\b", re.IGNORECASE),
    re.compile(r"\bstopped\s+(?:smoking|tobacco)\s+about\s+\d+\s*(?:year|years)\s+ago\b", re.IGNORECASE),
    re.compile(r"\bstopped\s+(?:smoking|tobacco)\s+\d+\s*(?:year|years)\s+ago\b", re.IGNORECASE),
]

NEVER_PATTERNS = [
    re.compile(r"\bnever smoked\b", re.IGNORECASE),
    re.compile(r"\bnever smoker\b", re.IGNORECASE),
    re.compile(r"\bnever tobacco user\b", re.IGNORECASE),
    re.compile(r"\blifetime nonsmoker\b", re.IGNORECASE),
    re.compile(r"\bnonsmoker\b", re.IGNORECASE),
    re.compile(r"\bnon[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\bdenies smoking\b", re.IGNORECASE),
    re.compile(r"\bdenies tobacco\b", re.IGNORECASE),
    re.compile(r"\bdenies tobacco use\b", re.IGNORECASE),
    re.compile(r"\bno history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bno tobacco use\b", re.IGNORECASE),
    re.compile(r"\bnever used tobacco\b", re.IGNORECASE),
    re.compile(r"\bsmokeless tobacco\s+(?:never used|none)\b", re.IGNORECASE),
]

# Quit time expressions
QUIT_TIME_PATTERN = re.compile(
    r"(quit|stopped)\s+(smoking|tobacco)[^\.]{0,60}?(\d+)\s*(day|days|week|weeks|month|months|year|years)",
    re.IGNORECASE
)

GENERIC_QUIT_PATTERN = re.compile(
    r"\b(quit smoking|quit tobacco|stopped smoking|stopped tobacco)\b",
    re.IGNORECASE
)

YEARS_SINCE_QUITTING_PATTERN = re.compile(
    r"\byears?\s+since\s+quitting\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE
)

QUIT_DATE_YEAR_PATTERN = re.compile(
    r"\bquit date\s*[:\-]?\s*((?:19|20)\d{2})\b",
    re.IGNORECASE
)

RECENT_QUIT_CONTEXT_PATTERN = re.compile(
    r"\b(?:since\s+(?:our|the)\s+last\s+visit[^\.]{0,80}?quit|recently\s+quit|trying\s+to\s+quit|plans\s+to\s+quit|encouraged\s+to\s+quit)\b",
    re.IGNORECASE
)

QUIT_YEARS_AGO_PATTERN = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.[0-9]+)?)\s+years?\s+ago\b",
    re.IGNORECASE
)

QUIT_MONTHS_AGO_PATTERN = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.[0-9]+)?)\s+months?\s+ago\b",
    re.IGNORECASE
)

QUIT_WEEKS_AGO_PATTERN = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.[0-9]+)?)\s+weeks?\s+ago\b",
    re.IGNORECASE
)

QUIT_DAYS_AGO_PATTERN = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.[0-9]+)?)\s+days?\s+ago\b",
    re.IGNORECASE
)

# Lower-confidence screening/template phrases
SCREENING_NEVER_PATTERNS = [
    re.compile(r"\bactive tobacco use\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bactive tobacco use\s*[:\-]?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrent tobacco use\s*[:\-]?\s*no\b", re.IGNORECASE),
]

FAMILY_HISTORY_PATTERN = re.compile(
    r"\bfamily history\b|\bmother\b|\bfather\b|\bgrandmother\b|\bgrandfather\b|\baunt\b|\buncle\b|\bsister\b|\bbrother\b",
    re.IGNORECASE
)

PREFERRED_SECTIONS = {
    "SOCIAL HISTORY",
    "HISTORY",
    "FULL"
}

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "ALLERGIES"
}


def _normalize_text(text):
    text = text or ""
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _quit_within_three_months(match):
    number = int(match.group(3))
    unit = match.group(4).lower()

    if unit.startswith("day"):
        return number <= 90
    if unit.startswith("week"):
        return number <= 12
    if unit.startswith("month"):
        return number <= 3
    if unit.startswith("year"):
        return False
    return False


def _candidate(note, section, value, text, start, end, confidence):
    ctx = window_around(text, start, end, 120)
    return Candidate(
        field="SmokingStatus",
        value=value,
        status="present",
        evidence=ctx,
        section=section,
        note_type=note.note_type,
        note_id=note.note_id,
        note_date=note.note_date,
        confidence=confidence,
    )


def _section_priority(section_name):
    s = str(section_name or "").upper().strip()
    if s == "SOCIAL HISTORY":
        return 0
    if s == "HISTORY":
        return 1
    if s == "FULL":
        return 2
    return 3


def _is_family_history_context(text, start, end):
    left = max(0, start - 160)
    right = min(len(text), end + 160)
    ctx = text[left:right]
    return FAMILY_HISTORY_PATTERN.search(ctx) is not None


def _find_best(patterns, text, note, section, value, confidence, suppress_family=True):
    best = None
    best_key = None

    for rx in patterns:
        for m in rx.finditer(text):
            if suppress_family and _is_family_history_context(text, m.start(), m.end()):
                continue

            cand = _candidate(note, section, value, text, m.start(), m.end(), confidence)
            key = (m.start(), -confidence)

            if best is None or key < best_key:
                best = cand
                best_key = key

    return best


def _find_quit_time_candidate(text, note, section):
    best = None
    best_key = None

    for m in QUIT_TIME_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        if _quit_within_three_months(m):
            value = "Current"
        else:
            value = "Former"

        cand = _candidate(note, section, value, text, m.start(), m.end(), 0.94)
        key = (m.start(),)

        if best is None or key < best_key:
            best = cand
            best_key = key

    return best


def _find_years_since_quit_candidate(text, note, section):
    for m in YEARS_SINCE_QUITTING_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        return _candidate(note, section, "Former", text, m.start(), m.end(), 0.94)
    return None


def _find_quit_date_year_candidate(text, note, section):
    for m in QUIT_DATE_YEAR_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        return _candidate(note, section, "Former", text, m.start(), m.end(), 0.94)
    return None


def _find_recent_quit_context_candidate(text, note, section):
    for m in RECENT_QUIT_CONTEXT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        return _candidate(note, section, "Current", text, m.start(), m.end(), 0.91)
    return None


def _find_generic_quit_candidate(text, note, section):
    for m in GENERIC_QUIT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        return _candidate(note, section, "Former", text, m.start(), m.end(), 0.89)
    return None


def _find_quit_years_ago_candidate(text, note, section):
    for m in QUIT_YEARS_AGO_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        return _candidate(note, section, "Former", text, m.start(), m.end(), 0.95)
    return None


def _find_quit_months_ago_candidate(text, note, section):
    for m in QUIT_MONTHS_AGO_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        try:
            months = float(m.group(1))
        except Exception:
            months = 999.0

        if months <= 3.0:
            value = "Current"
        else:
            value = "Former"

        return _candidate(note, section, value, text, m.start(), m.end(), 0.95)
    return None


def _find_quit_weeks_ago_candidate(text, note, section):
    for m in QUIT_WEEKS_AGO_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        try:
            weeks = float(m.group(1))
        except Exception:
            weeks = 999.0

        if weeks <= 12.0:
            value = "Current"
        else:
            value = "Former"

        return _candidate(note, section, value, text, m.start(), m.end(), 0.95)
    return None


def _find_quit_days_ago_candidate(text, note, section):
    for m in QUIT_DAYS_AGO_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        try:
            days = float(m.group(1))
        except Exception:
            days = 999.0

        if days <= 90.0:
            value = "Current"
        else:
            value = "Former"

        return _candidate(note, section, value, text, m.start(), m.end(), 0.95)
    return None


def extract_smoking(note: SectionedNote) -> List[Candidate]:
    section_order = []

    for s in note.sections.keys():
        if s in PREFERRED_SECTIONS and s not in SUPPRESS_SECTIONS:
            section_order.append(s)

    for s in note.sections.keys():
        if s not in section_order and s not in SUPPRESS_SECTIONS:
            section_order.append(s)

    all_candidates = []

    for section in section_order:
        raw_text = note.sections.get(section, "") or ""
        if not raw_text:
            continue

        text = _normalize_text(raw_text)

        # 1. Strong former indicators from structured tobacco history
        cand = _find_years_since_quit_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        cand = _find_quit_date_year_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        cand = _find_quit_years_ago_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        # 2. Explicit quit with recent/older duration
        cand = _find_quit_months_ago_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        cand = _find_quit_weeks_ago_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        cand = _find_quit_days_ago_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        cand = _find_quit_time_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        # 3. Strong structured former labels
        cand = _find_best(FORMER_PATTERNS, text, note, section, "Former", 0.92, suppress_family=True)
        if cand is not None:
            all_candidates.append(cand)

        # 4. Explicit current use
        cand = _find_best(CURRENT_PATTERNS, text, note, section, "Current", 0.91, suppress_family=True)
        if cand is not None:
            all_candidates.append(cand)

        # 5. Explicit never
        cand = _find_best(NEVER_PATTERNS, text, note, section, "Never", 0.90, suppress_family=True)
        if cand is not None:
            all_candidates.append(cand)

        # 6. Recent quit context without exact duration
        cand = _find_recent_quit_context_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        # 7. Generic quit defaults to Former
        cand = _find_generic_quit_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        # 8. Lower-confidence screening/template never
        cand = _find_best(SCREENING_NEVER_PATTERNS, text, note, section, "Never", 0.70, suppress_family=True)
        if cand is not None:
            all_candidates.append(cand)

    if not all_candidates:
        return []

    def sort_key(c):
        value_priority = 0 if c.value == "Former" else 1 if c.value == "Current" else 2
        return (
            value_priority,
            _section_priority(c.section),
            -float(getattr(c, "confidence", 0.0) or 0.0),
            len(getattr(c, "evidence", "") or "")
        )

    best = sorted(all_candidates, key=sort_key)[0]
    return [best]
