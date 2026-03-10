# extractors/smoking.py
import re
from datetime import datetime
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
# Updated principles:
# 1. Explicit current-use evidence wins within the same note.
# 2. "Quit" language is NOT Former unless timing supports it.
# 3. Recent quit (<= 90 days / <= 12 weeks / <= 3 months) => Current.
# 4. Former requires either:
#       - explicit Former label, OR
#       - quit timing > threshold, OR
#       - quit date sufficiently before note
# 5. Never requires explicit never/denies tobacco language.
# 6. Questionnaire/template prompts should not create Former.
# 7. Social history / H&P sections are preferred over generic FULL.
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
    re.compile(r"\bsmoker\b[^\.]{0,40}\bcurrent\b", re.IGNORECASE),
    re.compile(r"\bcurrent\b[^\.]{0,40}\bsmoker\b", re.IGNORECASE),
    re.compile(r"\bsmokes approximately\s+\d+(?:\.\d+)?\s+cigarettes?\s+a\s+day\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+(?:a\s+)?(?:couple|few)\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+\d+(?:\.\d+)?\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+\d+(?:\.\d+)?\s*packs?\s*/?\s*(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmoking\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*(?:cigs?|cigarettes?)\s+(?:daily|per day|a day)\b", re.IGNORECASE),
    re.compile(r"\bdown to\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*(?:cigs?|cigarettes?)\s+(?:daily|per day|a day)\b", re.IGNORECASE),
    re.compile(r"\btrying to quit\b[^\.]{0,100}\b(?:smok|cigarette|tobacco)\b", re.IGNORECASE),
    re.compile(r"\brecently has been smoking\b", re.IGNORECASE),
    re.compile(r"\blight tobacco smoker\b", re.IGNORECASE),
    re.compile(r"\btobacco use\s*[:\-]?\s*current\b", re.IGNORECASE),
]

FORMER_PATTERNS = [
    re.compile(r"\bformer smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking status\s*[:\-]?\s*former(?:\s+smoker)?\b", re.IGNORECASE),
    re.compile(r"\bhistory smoking status\s*[:\-]?\s*former(?:\s+smoker)?\b", re.IGNORECASE),
    re.compile(r"\bex[- ]smoker\b", re.IGNORECASE),
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
    re.compile(r"\bsmokeless tobacco\s*[:\-]?\s*never used\b", re.IGNORECASE),
    re.compile(r"\btobacco use\s*[:\-]?\s*never\b", re.IGNORECASE),
]

SCREENING_NEVER_PATTERNS = [
    re.compile(r"\bactive tobacco use\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bis patient currently smoking\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bactive tobacco use\s*[:\-]?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrent tobacco use\s*[:\-]?\s*no\b", re.IGNORECASE),
    re.compile(r"\bno tobacco use\b", re.IGNORECASE),
]

QUIT_TIME_PATTERN = re.compile(
    r"(quit|stopped)\s+(smoking|tobacco)[^\.]{0,60}?(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)",
    re.IGNORECASE
)

QUIT_YEARS_AGO_PATTERN = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.\d+)?)\s+years?\s+ago\b",
    re.IGNORECASE
)

QUIT_MONTHS_AGO_PATTERN = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.\d+)?)\s+months?\s+ago\b",
    re.IGNORECASE
)

QUIT_WEEKS_AGO_PATTERN = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.\d+)?)\s+weeks?\s+ago\b",
    re.IGNORECASE
)

QUIT_DAYS_AGO_PATTERN = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.\d+)?)\s+days?\s+ago\b",
    re.IGNORECASE
)

YEARS_SINCE_QUITTING_PATTERN = re.compile(
    r"\byears?\s+since\s+quitting\s*[:\-]?\s*([0-9]+(?:\.\d+)?)\b",
    re.IGNORECASE
)

LAST_ATTEMPT_TO_QUIT_PATTERN = re.compile(
    r"\blast attempt to quit\s*[:\-]?\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{1,2}/[0-9]{4}|(?:19|20)[0-9]{2})\b",
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

QUESTIONNAIRE_QUIT_PATTERN = re.compile(
    r"\b(resources?\s+to\s+help\s+quit\s+smoking|interested\s+in\s+resources?\s+to\s+help\s+quit\s+smoking|referral\s+to\s+mhealthy|referred\s+to\s+mhealthy|advised\s+by\s+provider\s+to\s+quit\s+smoking|plans?\s+to\s+quit)\b",
    re.IGNORECASE
)

QUESTIONNAIRE_FALSE_CURRENT_PATTERN = re.compile(
    r"\b(is patient currently smoking\?\s*no|currently smoking\?\s*no|active tobacco use\?\s*no|current tobacco use\?\s*no|if active smoker\b|was patient advised by provider to quit smoking\?|was patient referred to mhealthy\?)\b",
    re.IGNORECASE
)

FAMILY_HISTORY_PATTERN = re.compile(
    r"\bfamily history\b|\bmother\b|\bfather\b|\bgrandmother\b|\bgrandfather\b|\baunt\b|\buncle\b|\bsister\b|\bbrother\b",
    re.IGNORECASE
)

PREFERRED_SECTIONS = {
    "SOCIAL HISTORY",
    "HISTORY",
    "FULL",
    "SUBSTANCE AND SEXUAL ACTIVITY",
}

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "ALLERGIES",
}

NEGATED_CURRENT_CONTEXT = re.compile(
    r"\b(not currently smoking|no current tobacco use|not smoking currently)\b",
    re.IGNORECASE
)


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
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except Exception:
        return None


def _parse_quit_date(raw):
    s = str(raw or "").strip()
    if not s:
        return None

    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    m = re.match(r"^([0-9]{1,2})/([0-9]{4})$", s)
    if m:
        try:
            return datetime(int(m.group(2)), int(m.group(1)), 1)
        except Exception:
            pass

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
    ctx = window_around(text, start, end, 140)
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
    if s == "SUBSTANCE AND SEXUAL ACTIVITY":
        return 1
    if s == "HISTORY":
        return 2
    if s == "FULL":
        return 3
    return 4


def _value_priority(value):
    # Current should win within-note over Former/Never when explicit use is present.
    v = str(value or "").strip()
    if v == "Current":
        return 0
    if v == "Former":
        return 1
    if v == "Never":
        return 2
    return 9


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


def _is_questionnaire_false_current_context(text, start, end):
    left = max(0, start - 200)
    right = min(len(text), end + 200)
    ctx = text[left:right]
    return QUESTIONNAIRE_FALSE_CURRENT_PATTERN.search(ctx) is not None


def _is_negated_current_context(text, start, end):
    left = max(0, start - 120)
    right = min(len(text), end + 120)
    ctx = text[left:right]
    return NEGATED_CURRENT_CONTEXT.search(ctx) is not None


def _add_first_match(patterns, text, note, section, value, confidence, out_list, suppress_family=True):
    for rx in patterns:
        for m in rx.finditer(text):
            if suppress_family and _is_family_history_context(text, m.start(), m.end()):
                continue
            if value == "Current":
                if _is_questionnaire_false_current_context(text, m.start(), m.end()):
                    continue
                if _is_negated_current_context(text, m.start(), m.end()):
                    continue
            out_list.append(_candidate(note, section, value, text, m.start(), m.end(), confidence))


def _find_recent_quit_context_candidates(text, note, section, out_list):
    for m in RECENT_QUIT_CONTEXT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        out_list.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.98))


def _find_quit_time_candidates(text, note, section, out_list):
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

        conf = 0.99 if value == "Current" else 0.98
        out_list.append(_candidate(note, section, value, text, m.start(), m.end(), conf))


def _find_quit_relative_candidates(text, note, section, out_list):
    for rx in [QUIT_YEARS_AGO_PATTERN, QUIT_MONTHS_AGO_PATTERN, QUIT_WEEKS_AGO_PATTERN, QUIT_DAYS_AGO_PATTERN]:
        for m in rx.finditer(text):
            if _is_family_history_context(text, m.start(), m.end()):
                continue
            if _is_questionnaire_quit_context(text, m.start(), m.end()):
                continue

            if rx == QUIT_YEARS_AGO_PATTERN:
                value = "Former"
                conf = 0.98
            elif rx == QUIT_MONTHS_AGO_PATTERN:
                months = float(m.group(1))
                value = "Current" if months <= 3.0 else "Former"
                conf = 0.99 if value == "Current" else 0.98
            elif rx == QUIT_WEEKS_AGO_PATTERN:
                weeks = float(m.group(1))
                value = "Current" if weeks <= 12.0 else "Former"
                conf = 0.99 if value == "Current" else 0.98
            else:
                days = float(m.group(1))
                value = "Current" if days <= 90.0 else "Former"
                conf = 0.99 if value == "Current" else 0.98

            out_list.append(_candidate(note, section, value, text, m.start(), m.end(), conf))


def _find_years_since_quit_candidates(text, note, section, out_list):
    for m in YEARS_SINCE_QUITTING_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        yrs = float(m.group(1))
        value = "Current" if yrs < 0.25 else "Former"
        conf = 0.99 if value == "Current" else 0.98
        out_list.append(_candidate(note, section, value, text, m.start(), m.end(), conf))


def _find_quit_date_candidates(text, note, section, out_list):
    note_dt = _parse_date_safe(getattr(note, "note_date", ""))

    for rx in [QUIT_DATE_PATTERN, LAST_ATTEMPT_TO_QUIT_PATTERN]:
        for m in rx.finditer(text):
            if _is_family_history_context(text, m.start(), m.end()):
                continue

            quit_dt = _parse_quit_date(m.group(1))
            if note_dt is not None and quit_dt is not None:
                dd = _days_between(note_dt, quit_dt)
                if dd is not None and dd >= 0 and dd <= 90:
                    out_list.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.99))
                elif dd is not None and dd > 90:
                    out_list.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.98))
                else:
                    # Future or malformed relation to note date -> do not force label
                    pass
            else:
                # Date exists but note date unavailable; keep lower-confidence Former
                out_list.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.90))


def _find_generic_quit_candidates(text, note, section, out_list):
    # Do NOT force Former from generic quit language.
    # Only allow generic quit to imply Current when clearly recent.
    for m in GENERIC_QUIT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        if _is_questionnaire_quit_context(text, m.start(), m.end()):
            continue

        left = max(0, m.start() - 140)
        right = min(len(text), m.end() + 140)
        ctx = text[left:right]

        if re.search(r"\b(recently|since last visit|last visit|today|this week|few days ago|few weeks ago)\b", ctx, re.IGNORECASE):
            out_list.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.92))
        # Else: ignore generic quit if timing is not supported.


def _best_candidate_within_note(candidates):
    if not candidates:
        return None

    def sort_key(c):
        return (
            _value_priority(getattr(c, "value", "")),
            _section_priority(getattr(c, "section", "")),
            -float(getattr(c, "confidence", 0.0) or 0.0),
            len(getattr(c, "evidence", "") or ""),
        )

    return sorted(candidates, key=sort_key)[0]


def extract_smoking(note: SectionedNote) -> List[Candidate]:
    section_order = []

    for s in note.sections.keys():
        su = str(s or "").upper().strip()
        if su in PREFERRED_SECTIONS and su not in SUPPRESS_SECTIONS:
            section_order.append(s)

    for s in note.sections.keys():
        su = str(s or "").upper().strip()
        if s not in section_order and su not in SUPPRESS_SECTIONS:
            section_order.append(s)

    note_candidates = []

    for section in section_order:
        raw_text = note.sections.get(section, "") or ""
        if not raw_text:
            continue

        text = _normalize_text(raw_text)
        section_candidates = []

        # 1. Explicit current-use evidence
        _add_first_match(CURRENT_PATTERNS, text, note, section, "Current", 0.99, section_candidates, suppress_family=True)
        _find_recent_quit_context_candidates(text, note, section, section_candidates)
        _find_quit_relative_candidates(text, note, section, section_candidates)
        _find_quit_time_candidates(text, note, section, section_candidates)
        _find_years_since_quit_candidates(text, note, section, section_candidates)
        _find_quit_date_candidates(text, note, section, section_candidates)
        _find_generic_quit_candidates(text, note, section, section_candidates)

        # 2. Strong structured former labels
        _add_first_match(FORMER_PATTERNS, text, note, section, "Former", 0.96, section_candidates, suppress_family=True)

        # 3. Explicit never
        _add_first_match(NEVER_PATTERNS, text, note, section, "Never", 0.94, section_candidates, suppress_family=True)

        # 4. Lower-confidence screening/template never
        _add_first_match(SCREENING_NEVER_PATTERNS, text, note, section, "Never", 0.70, section_candidates, suppress_family=True)

        # Keep best per section, not just first regex hit.
        best_section = _best_candidate_within_note(section_candidates)
        if best_section is not None:
            note_candidates.append(best_section)

    if not note_candidates:
        return []

    best_note = _best_candidate_within_note(note_candidates)
    return [best_note]
