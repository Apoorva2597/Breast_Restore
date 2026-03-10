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
# This revision targets the highest-yield pipeline fixes:
# 1. Strong structured tobacco/social-history block parsing
# 2. Within-note conflict resolution based on evidence type
# 3. Numeric quit-date / years-since-quitting handling
# 4. Aggressive suppression of questionnaire/template prompts
# 5. Historical descriptors (pack years / packs per day) do NOT
#    count as current by themselves
#
# Priority inside a note:
#   1. Explicit current behavior
#   2. Computed recent quit evidence
#   3. Strong former evidence
#   4. Explicit never
#   5. Weak screening/template answers
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
    re.compile(r"\blight tobacco smoker\b", re.IGNORECASE),
    re.compile(r"\bheavy tobacco smoker\b", re.IGNORECASE),
    re.compile(r"\bevery day smoker\b", re.IGNORECASE),
    re.compile(r"\bsome day smoker\b", re.IGNORECASE),
    re.compile(r"\bsmokes approximately\s+\d+(?:\.\d+)?\s+cigarettes?\s+a\s+day\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+(?:a\s+)?(?:couple|few)\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+\d+(?:\.\d+)?\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmoking\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*(?:cigs?|cigarettes?)\s+(?:daily|per day|a day)\b", re.IGNORECASE),
    re.compile(r"\bdown to\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*(?:cigs?|cigarettes?)\s+(?:daily|per day|a day)\b", re.IGNORECASE),
    re.compile(r"\brecently has been smoking\b", re.IGNORECASE),
    re.compile(r"\btrying to quit\b[^\.]{0,100}\b(?:smok|cigarette|tobacco)\b", re.IGNORECASE),
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
    r"(quit|stopped)\s+(smoking|tobacco)[^\.]{0,80}?(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)",
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

QUIT_DATE_PATTERN = re.compile(
    r"\bquit date\s*[:\-]?\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{1,2}/[0-9]{4}|(?:19|20)[0-9]{2})\b",
    re.IGNORECASE
)

LAST_ATTEMPT_TO_QUIT_PATTERN = re.compile(
    r"\blast attempt to quit\s*[:\-]?\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{1,2}/[0-9]{4}|(?:19|20)[0-9]{2})\b",
    re.IGNORECASE
)

GENERIC_QUIT_PATTERN = re.compile(
    r"\b(quit smoking|quit tobacco|stopped smoking|stopped tobacco)\b",
    re.IGNORECASE
)

RECENT_QUIT_CONTEXT_PATTERN = re.compile(
    r"\b(?:since\s+(?:our|the)\s+last\s+visit[^\.]{0,100}?quit|recently\s+quit)\b",
    re.IGNORECASE
)

QUESTIONNAIRE_QUIT_PATTERN = re.compile(
    r"\b("
    r"resources?\s+to\s+help\s+quit\s+smoking|"
    r"interested\s+in\s+resources?\s+to\s+help\s+quit\s+smoking|"
    r"referral\s+to\s+mhealthy|"
    r"referred\s+to\s+mhealthy|"
    r"advised\s+by\s+provider\s+to\s+quit\s+smoking|"
    r"plans?\s+to\s+quit(?:\s+smoking)?|"
    r"provider\s+advised\s+to\s+quit\s+smoking|"
    r"would\s+like\s+to\s+quit|"
    r"interest(?:ed)?\s+in\s+quitting"
    r")\b",
    re.IGNORECASE
)

QUESTIONNAIRE_FALSE_CURRENT_PATTERN = re.compile(
    r"\b("
    r"is patient currently smoking\?\s*no|"
    r"currently smoking\?\s*no|"
    r"active tobacco use\?\s*no|"
    r"current tobacco use\?\s*no|"
    r"if active smoker\b|"
    r"was patient advised by provider to quit smoking\?|"
    r"was patient referred to mhealthy\?"
    r")\b",
    re.IGNORECASE
)

NEGATED_CURRENT_CONTEXT = re.compile(
    r"\b(not currently smoking|no current tobacco use|not smoking currently)\b",
    re.IGNORECASE
)

FAMILY_HISTORY_PATTERN = re.compile(
    r"\bfamily history\b|\bmother\b|\bfather\b|\bgrandmother\b|\bgrandfather\b|\baunt\b|\buncle\b|\bsister\b|\bbrother\b",
    re.IGNORECASE
)

PACK_HISTORY_PATTERN = re.compile(
    r"\b(pack years?|packs?/day|packs?\s*/\s*day|types?\s*:\s*cigarettes?|years?\s*:\s*\d+(?:\.\d+)?)\b",
    re.IGNORECASE
)

PREFERRED_SECTIONS = {
    "SOCIAL HISTORY",
    "HISTORY",
    "FULL",
    "SUBSTANCE AND SEXUAL ACTIVITY",
    "TOBACCO USE",
    "SOCIAL HISTORY/TOBACCO USE",
}

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "ALLERGIES",
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


def _candidate(note, section, value, text, start, end, confidence, source_type):
    ctx = window_around(text, start, end, 160)
    return Candidate(
        field="SmokingStatus",
        value=value,
        status=source_type,
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
    if s == "TOBACCO USE":
        return 1
    if s == "SOCIAL HISTORY/TOBACCO USE":
        return 1
    if s == "SUBSTANCE AND SEXUAL ACTIVITY":
        return 2
    if s == "HISTORY":
        return 3
    if s == "FULL":
        return 4
    return 5


def _evidence_priority(cand):
    status = str(getattr(cand, "status", "") or "").strip().lower()
    value = str(getattr(cand, "value", "") or "").strip()

    if value == "Current" and status == "explicit_current":
        return 0
    if value == "Current" and status == "computed_recent_quit":
        return 1
    if value == "Former" and status == "strong_former":
        return 2
    if value == "Never" and status == "explicit_never":
        return 3
    if value == "Former" and status == "generic_quit":
        return 4
    if value == "Never" and status == "screening_never":
        return 5
    return 9


def _is_family_history_context(text, start, end):
    left = max(0, start - 180)
    right = min(len(text), end + 180)
    ctx = text[left:right]
    return FAMILY_HISTORY_PATTERN.search(ctx) is not None


def _is_questionnaire_quit_context(text, start, end):
    left = max(0, start - 200)
    right = min(len(text), end + 200)
    ctx = text[left:right]
    return QUESTIONNAIRE_QUIT_PATTERN.search(ctx) is not None


def _is_questionnaire_false_current_context(text, start, end):
    left = max(0, start - 220)
    right = min(len(text), end + 220)
    ctx = text[left:right]
    return QUESTIONNAIRE_FALSE_CURRENT_PATTERN.search(ctx) is not None


def _is_negated_current_context(text, start, end):
    left = max(0, start - 160)
    right = min(len(text), end + 160)
    ctx = text[left:right]
    return NEGATED_CURRENT_CONTEXT.search(ctx) is not None


def _has_only_pack_history_context(text, start, end):
    left = max(0, start - 120)
    right = min(len(text), end + 120)
    ctx = text[left:right]
    return PACK_HISTORY_PATTERN.search(ctx) is not None


def _add_candidates_from_patterns(patterns, text, note, section, value, confidence, source_type, out_list, suppress_family=True):
    for rx in patterns:
        for m in rx.finditer(text):
            if suppress_family and _is_family_history_context(text, m.start(), m.end()):
                continue
            if value == "Current":
                if _is_questionnaire_false_current_context(text, m.start(), m.end()):
                    continue
                if _is_negated_current_context(text, m.start(), m.end()):
                    continue
            out_list.append(_candidate(note, section, value, text, m.start(), m.end(), confidence, source_type))


def _find_recent_quit_context_candidates(text, note, section, out_list):
    for m in RECENT_QUIT_CONTEXT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        out_list.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.98, "computed_recent_quit"))


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
        src = "computed_recent_quit" if value == "Current" else "strong_former"
        out_list.append(_candidate(note, section, value, text, m.start(), m.end(), conf, src))


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
                src = "strong_former"
            elif rx == QUIT_MONTHS_AGO_PATTERN:
                months = float(m.group(1))
                value = "Current" if months <= 3.0 else "Former"
                conf = 0.99 if value == "Current" else 0.98
                src = "computed_recent_quit" if value == "Current" else "strong_former"
            elif rx == QUIT_WEEKS_AGO_PATTERN:
                weeks = float(m.group(1))
                value = "Current" if weeks <= 12.0 else "Former"
                conf = 0.99 if value == "Current" else 0.98
                src = "computed_recent_quit" if value == "Current" else "strong_former"
            else:
                days = float(m.group(1))
                value = "Current" if days <= 90.0 else "Former"
                conf = 0.99 if value == "Current" else 0.98
                src = "computed_recent_quit" if value == "Current" else "strong_former"

            out_list.append(_candidate(note, section, value, text, m.start(), m.end(), conf, src))


def _find_years_since_quit_candidates(text, note, section, out_list):
    for m in YEARS_SINCE_QUITTING_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        yrs = float(m.group(1))
        value = "Current" if yrs < 0.25 else "Former"
        conf = 0.99 if value == "Current" else 0.98
        src = "computed_recent_quit" if value == "Current" else "strong_former"
        out_list.append(_candidate(note, section, value, text, m.start(), m.end(), conf, src))


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
                    out_list.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.99, "computed_recent_quit"))
                elif dd is not None and dd > 90:
                    out_list.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.98, "strong_former"))
            else:
                out_list.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.90, "strong_former"))


def _find_generic_quit_candidates(text, note, section, out_list):
    # Generic quit statements should default to Former, not Current.
    for m in GENERIC_QUIT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        if _is_questionnaire_quit_context(text, m.start(), m.end()):
            continue

        left = max(0, m.start() - 160)
        right = min(len(text), m.end() + 160)
        ctx = text[left:right]

        if re.search(r"\b(recently|since last visit|last visit|today|this week)\b", ctx, re.IGNORECASE):
            out_list.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.91, "computed_recent_quit"))
        else:
            out_list.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.90, "generic_quit"))


def _find_structured_tobacco_block_candidates(text, note, section, out_list):
    # Strong structured parsing for social history / tobacco blocks.
    # Examples:
    # Smoking status: Former Smoker
    # Quit date: 4/9/2009
    # Years since quitting: 11.6
    # Smokeless tobacco: Never

    status_m = re.search(
        r"\bsmoking status\s*[:\-]?\s*(current smoker|former smoker|never smoker|current|former|never)\b",
        text,
        re.IGNORECASE
    )
    quit_m = QUIT_DATE_PATTERN.search(text)
    years_m = YEARS_SINCE_QUITTING_PATTERN.search(text)
    recent_quit_context_m = RECENT_QUIT_CONTEXT_PATTERN.search(text)
    current_m = None
    for rx in CURRENT_PATTERNS:
        mm = rx.search(text)
        if mm is not None:
            current_m = mm
            break

    never_m = None
    for rx in NEVER_PATTERNS:
        mm = rx.search(text)
        if mm is not None:
            never_m = mm
            break

    if status_m:
        raw = status_m.group(1).strip().lower()
        s_start = status_m.start()
        s_end = status_m.end()

        if "current" in raw:
            if not _is_questionnaire_false_current_context(text, s_start, s_end) and not _is_negated_current_context(text, s_start, s_end):
                out_list.append(_candidate(note, section, "Current", text, s_start, s_end, 0.995, "explicit_current"))
            return

        if "never" in raw:
            # Keep explicit never lower than strong former if same block contains quit history.
            out_list.append(_candidate(note, section, "Never", text, s_start, s_end, 0.95, "explicit_never"))

        if "former" in raw:
            # If same block has explicit current behavior, current should still win.
            if current_m is not None and not _has_only_pack_history_context(text, current_m.start(), current_m.end()):
                if not _is_questionnaire_false_current_context(text, current_m.start(), current_m.end()) and not _is_negated_current_context(text, current_m.start(), current_m.end()):
                    out_list.append(_candidate(note, section, "Current", text, current_m.start(), current_m.end(), 0.995, "explicit_current"))

            # If quit date / years since quitting support recent quit, classify Current.
            note_dt = _parse_date_safe(getattr(note, "note_date", ""))
            if quit_m is not None:
                quit_dt = _parse_quit_date(quit_m.group(1))
                if note_dt is not None and quit_dt is not None:
                    dd = _days_between(note_dt, quit_dt)
                    if dd is not None and dd >= 0 and dd <= 90:
                        out_list.append(_candidate(note, section, "Current", text, quit_m.start(), quit_m.end(), 0.995, "computed_recent_quit"))
                    elif dd is not None and dd > 90:
                        out_list.append(_candidate(note, section, "Former", text, s_start, s_end, 0.995, "strong_former"))
                else:
                    out_list.append(_candidate(note, section, "Former", text, s_start, s_end, 0.985, "strong_former"))
            elif years_m is not None:
                yrs = float(years_m.group(1))
                if yrs < 0.25:
                    out_list.append(_candidate(note, section, "Current", text, years_m.start(), years_m.end(), 0.995, "computed_recent_quit"))
                else:
                    out_list.append(_candidate(note, section, "Former", text, s_start, s_end, 0.995, "strong_former"))
            elif recent_quit_context_m is not None:
                out_list.append(_candidate(note, section, "Current", text, recent_quit_context_m.start(), recent_quit_context_m.end(), 0.99, "computed_recent_quit"))
            else:
                out_list.append(_candidate(note, section, "Former", text, s_start, s_end, 0.985, "strong_former"))


def _best_candidate_within_note(candidates):
    if not candidates:
        return None

    def sort_key(c):
        return (
            _evidence_priority(c),
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

        # 0. Structured tobacco/social history block parsing
        _find_structured_tobacco_block_candidates(text, note, section, section_candidates)

        # 1. Explicit current behavior
        _add_candidates_from_patterns(
            CURRENT_PATTERNS, text, note, section,
            "Current", 0.99, "explicit_current", section_candidates, suppress_family=True
        )

        # 2. Computed recent-quit evidence
        _find_recent_quit_context_candidates(text, note, section, section_candidates)
        _find_quit_relative_candidates(text, note, section, section_candidates)
        _find_quit_time_candidates(text, note, section, section_candidates)
        _find_years_since_quit_candidates(text, note, section, section_candidates)
        _find_quit_date_candidates(text, note, section, section_candidates)

        # 3. Strong former evidence
        _add_candidates_from_patterns(
            FORMER_PATTERNS, text, note, section,
            "Former", 0.97, "strong_former", section_candidates, suppress_family=True
        )

        # 4. Explicit never
        _add_candidates_from_patterns(
            NEVER_PATTERNS, text, note, section,
            "Never", 0.94, "explicit_never", section_candidates, suppress_family=True
        )

        # 5. Generic quit only after stronger evidence
        _find_generic_quit_candidates(text, note, section, section_candidates)

        # 6. Low-confidence screening/template never
        _add_candidates_from_patterns(
            SCREENING_NEVER_PATTERNS, text, note, section,
            "Never", 0.60, "screening_never", section_candidates, suppress_family=True
        )

        best_section = _best_candidate_within_note(section_candidates)
        if best_section is not None:
            note_candidates.append(best_section)

    if not note_candidates:
        return []

    best_note = _best_candidate_within_note(note_candidates)
    return [best_note]
