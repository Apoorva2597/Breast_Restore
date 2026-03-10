# extractors/smoking.py
import re
from datetime import datetime
from typing import List, Optional

from models import Candidate, SectionedNote
from .utils import window_around

# ----------------------------------------------
# Smoking extraction
#
# Dataset rule:
# Patients reported as having smoked in the past
# 3 months are considered Current smokers.
#
# Main principles from QA:
# 1. Strong present-tense smoking wins:
#    - current smoker
#    - currently smoking
#    - still smoking
#    - down to 4-5 cigs daily
#    - smokes approximately two cigarettes a day
#
# 2. Strong former wins when supported by:
#    - Smoking Status: Former Smoker
#    - quit date
#    - years since quitting
#    - quit/stopped ... X years ago
#
# 3. Quit date / years-since-quitting are interpreted
#    relative to the note date:
#    - <= 90 days -> Current
#    - > 90 days  -> Former
#
# 4. Questionnaire/template phrases like:
#    - resources to help quit smoking
#    - advised to quit smoking
#    - referral to MHealthy
#    should NOT create Former.
#
# Python 3.6.8 compatible.
# ----------------------------------------------

CURRENT_PATTERNS = [
    re.compile(r"\bcurrent smoker\b", re.IGNORECASE),
    re.compile(r"\bactive smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking currently\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\b", re.IGNORECASE),
    re.compile(r"\bcurrently smokes\b", re.IGNORECASE),
    re.compile(r"\bstill smoking\b", re.IGNORECASE),
    re.compile(r"\bcontinues to smoke\b", re.IGNORECASE),
    re.compile(r"\bdown to\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*cigs?\s+(?:daily|per day)\b", re.IGNORECASE),
    re.compile(r"\busing chantix[^\.]{0,80}\b(?:\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*cigs?\s+(?:daily|per day)|down to\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*cigs?)", re.IGNORECASE),
    re.compile(r"\bsmokes approximately\s+\d+(?:\.\d+)?\s+cigarettes?\s+a\s+day\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+(?:a\s+)?(?:couple|few)\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+\d+(?:\.\d+)?\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+\d+(?:\.\d+)?\s*packs?\s*/?\s*(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\btobacco use\s*[:\-]?\s*current\b", re.IGNORECASE),
]

FORMER_PATTERNS = [
    re.compile(r"\bformer smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking status\s*[:\-]?\s*former(?:\s+smoker)?\b", re.IGNORECASE),
    re.compile(r"\bhistory smoking status\s*[:\-]?\s*former(?:\s+smoker)?\b", re.IGNORECASE),
    re.compile(r"\bex[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\bformer user\b", re.IGNORECASE),
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
    re.compile(r"\bnever used tobacco\b", re.IGNORECASE),
]

SCREENING_NEVER_PATTERNS = [
    re.compile(r"\bactive tobacco use\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bactive tobacco use\s*[:\-]?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrent tobacco use\s*[:\-]?\s*no\b", re.IGNORECASE),
    re.compile(r"\bno tobacco use\b", re.IGNORECASE),
]

# Explicit quit-with-duration
QUIT_TIME_PATTERN = re.compile(
    r"(quit|stopped)\s+(smoking|tobacco)[^\.]{0,60}?(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)",
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

YEARS_SINCE_QUITTING_PATTERN = re.compile(
    r"\byears?\s+since\s+quitting\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE
)

QUIT_DATE_PATTERN = re.compile(
    r"\bquit date\s*[:\-]?\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{1,2}/[0-9]{4}|(?:19|20)[0-9]{2})\b",
    re.IGNORECASE
)

GENERIC_QUIT_PATTERN = re.compile(
    r"\b(quit smoking|quit tobacco|stopped smoking|stopped tobacco)\b",
    re.IGNORECASE
)

RECENT_QUIT_CONTEXT_PATTERN = re.compile(
    r"\b(?:since\s+(?:our|the)\s+last\s+visit[^\.]{0,80}?quit|recently\s+quit)\b",
    re.IGNORECASE
)

# Template/questionnaire quit phrases that should NOT count as Former
QUESTIONNAIRE_QUIT_PATTERN = re.compile(
    r"\b(resources?\s+to\s+help\s+quit\s+smoking|interested\s+in\s+resources?\s+to\s+help\s+quit\s+smoking|referral\s+to\s+mhealthy|referred\s+to\s+mhealthy|advised\s+by\s+provider\s+to\s+quit\s+smoking|plans?\s+to\s+quit)\b",
    re.IGNORECASE
)

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


def _parse_date_safe(x):
    s = str(x or "").strip()
    if not s:
        return None

    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%m/%d/%y",
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
        ts = datetime.strptime(s[:10], "%Y-%m-%d")
        return ts
    except Exception:
        return None


def _parse_quit_date(raw):
    s = str(raw or "").strip()
    if not s:
        return None

    # MM/DD/YYYY or MM/DD/YY
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    # MM/YYYY -> first day of month
    m = re.match(r"^([0-9]{1,2})/([0-9]{4})$", s)
    if m:
        try:
            return datetime(int(m.group(2)), int(m.group(1)), 1)
        except Exception:
            pass

    # YYYY -> first day of year
    m = re.match(r"^((?:19|20)[0-9]{2})$", s)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1)
        except Exception:
            pass

    return None


def _days_between(d1, d2):
    if d1 is None or d2 is None:
        return None
    return (d1.date() - d2.date()).days


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


def _is_questionnaire_quit_context(text, start, end):
    left = max(0, start - 160)
    right = min(len(text), end + 160)
    ctx = text[left:right]
    return QUESTIONNAIRE_QUIT_PATTERN.search(ctx) is not None


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


def _find_recent_quit_context_candidate(text, note, section):
    for m in RECENT_QUIT_CONTEXT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        return _candidate(note, section, "Current", text, m.start(), m.end(), 0.97)
    return None


def _find_quit_time_candidate(text, note, section):
    best = None
    best_key = None

    for m in QUIT_TIME_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        if _is_questionnaire_quit_context(text, m.start(), m.end()):
            continue

        number = float(m.group(3))
        unit = m.group(4).lower()

        if unit.startswith("day"):
            value = "Current" if number <= 90 else "Former"
        elif unit.startswith("week"):
            value = "Current" if number <= 12 else "Former"
        elif unit.startswith("month"):
            value = "Current" if number <= 3 else "Former"
        else:
            value = "Former"

        cand = _candidate(note, section, value, text, m.start(), m.end(), 0.97)
        key = (m.start(),)

        if best is None or key < best_key:
            best = cand
            best_key = key

    return best


def _find_quit_years_ago_candidate(text, note, section):
    for rx in [QUIT_YEARS_AGO_PATTERN, QUIT_MONTHS_AGO_PATTERN, QUIT_WEEKS_AGO_PATTERN, QUIT_DAYS_AGO_PATTERN]:
        for m in rx.finditer(text):
            if _is_family_history_context(text, m.start(), m.end()):
                continue
            if _is_questionnaire_quit_context(text, m.start(), m.end()):
                continue

            if rx == QUIT_YEARS_AGO_PATTERN:
                value = "Former"
            elif rx == QUIT_MONTHS_AGO_PATTERN:
                months = float(m.group(1))
                value = "Current" if months <= 3.0 else "Former"
            elif rx == QUIT_WEEKS_AGO_PATTERN:
                weeks = float(m.group(1))
                value = "Current" if weeks <= 12.0 else "Former"
            else:
                days = float(m.group(1))
                value = "Current" if days <= 90.0 else "Former"

            return _candidate(note, section, value, text, m.start(), m.end(), 0.98)

    return None


def _find_years_since_quit_candidate(text, note, section):
    for m in YEARS_SINCE_QUITTING_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        yrs = float(m.group(1))
        value = "Current" if yrs < 0.25 else "Former"
        conf = 0.98 if value == "Current" else 0.97
        return _candidate(note, section, value, text, m.start(), m.end(), conf)

    return None


def _find_quit_date_candidate(text, note, section):
    note_dt = _parse_date_safe(getattr(note, "note_date", ""))

    for m in QUIT_DATE_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        quit_dt = _parse_quit_date(m.group(1))
        if note_dt is not None and quit_dt is not None:
            dd = _days_between(note_dt, quit_dt)
            if dd is not None and dd >= 0 and dd <= 90:
                return _candidate(note, section, "Current", text, m.start(), m.end(), 0.99)
            if dd is not None and dd > 90:
                return _candidate(note, section, "Former", text, m.start(), m.end(), 0.97)

        return _candidate(note, section, "Former", text, m.start(), m.end(), 0.95)

    return None


def _find_generic_quit_candidate(text, note, section):
    for m in GENERIC_QUIT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        if _is_questionnaire_quit_context(text, m.start(), m.end()):
            continue
        return _candidate(note, section, "Former", text, m.start(), m.end(), 0.86)
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

        # 1. Explicit present-tense smoking: strongest signal
        cand = _find_best(CURRENT_PATTERNS, text, note, section, "Current", 0.99, suppress_family=True)
        if cand is not None:
            all_candidates.append(cand)

        # 2. Recent quit / explicit quit timing relative to note date
        cand = _find_recent_quit_context_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        cand = _find_quit_years_ago_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        cand = _find_quit_time_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        cand = _find_years_since_quit_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        cand = _find_quit_date_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        # 3. Strong structured former labels
        cand = _find_best(FORMER_PATTERNS, text, note, section, "Former", 0.96, suppress_family=True)
        if cand is not None:
            all_candidates.append(cand)

        # 4. Explicit never
        cand = _find_best(NEVER_PATTERNS, text, note, section, "Never", 0.93, suppress_family=True)
        if cand is not None:
            all_candidates.append(cand)

        # 5. Generic quit only if it is not just a questionnaire phrase
        cand = _find_generic_quit_candidate(text, note, section)
        if cand is not None:
            all_candidates.append(cand)

        # 6. Lower-confidence screening/template never
        cand = _find_best(SCREENING_NEVER_PATTERNS, text, note, section, "Never", 0.70, suppress_family=True)
        if cand is not None:
            all_candidates.append(cand)

    # if nothing matched, create a weak candidate so evidence is preserved
        if not all_candidates:
            text_blob = " ".join([note.sections.get(s, "") for s in note.sections])
        if text_blob.strip():
            cand = Candidate(
                field="SmokingStatus",
                value="Unknown",
                status="present",
                evidence=text_blob[:200],
                section="UNKNOWN",
                note_type=note.note_type,
                note_id=note.note_id,
                note_date=note.note_date,
                confidence=0.10,
                )
            return [cand]
        return []
        

    def sort_key(c):
        return (
            _section_priority(c.section),
            -float(getattr(c, "confidence", 0.0) or 0.0),
            len(getattr(c, "evidence", "") or "")
        )

    best = sorted(all_candidates, key=sort_key)[0]
    return [best]
