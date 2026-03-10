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
# This revision adds stronger support for:
# - structured EHR checkbox / labeled tobacco blocks
# - "does not smoke" style negation
# - "does not drink alcohol or smoke"
# - nonsmoker / non-smoker
# - "denies tobacco or alcohol use"
# - repeated structured Never/Former/Current patterns seen in QA
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
    re.compile(r"\bcurrent every day smoker\b", re.IGNORECASE),
    re.compile(r"\bcurrent some day smoker\b", re.IGNORECASE),
    re.compile(r"\bsome day smoker\b", re.IGNORECASE),
    re.compile(r"\bsmokes approximately\s+\d+(?:\.\d+)?\s+cigarettes?\s+a\s+day\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+(?:a\s+)?(?:couple|few)\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+\d+(?:\.\d+)?\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+\d+(?:\.\d+)?\s*packs?\s*/?\s*(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmoking\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*(?:cigs?|cigarettes?)\s+(?:daily|per day|a day)\b", re.IGNORECASE),
    re.compile(r"\bdown to\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*(?:cigs?|cigarettes?)\s+(?:daily|per day|a day)\b", re.IGNORECASE),
    re.compile(r"\brecently has been smoking\b", re.IGNORECASE),
    re.compile(r"\btrying to quit\b[^\.]{0,100}\b(?:smok|cigarette|tobacco)\b", re.IGNORECASE),
    re.compile(r"\bcomment\s*[:\-]?\s*states?\s+she\s+smokes?\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+every\s+once\s+in\s+a\s+while\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+every\s+once\s+in\s+a\s+while\s+currently\b", re.IGNORECASE),
    re.compile(r"\btobacco use\s*[:\-]?\s*current\b", re.IGNORECASE),
]

FORMER_PATTERNS = [
    re.compile(r"\bformer smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking status\s*[:\-]?\s*former(?:\s+smoker)?\b", re.IGNORECASE),
    re.compile(r"\bhistory smoking status\s*[:\-]?\s*former(?:\s+smoker)?\b", re.IGNORECASE),
    re.compile(r"\bex[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\bformer user\b", re.IGNORECASE),
    re.compile(r"\bquit smoking about\s+[0-9]+(?:\.\d+)?\s+years?\s+ago\b", re.IGNORECASE),
    re.compile(r"\breports?\s+(?:she|he)\s+quit\s+smoking\b", re.IGNORECASE),
    re.compile(r"\bformer smoker who quit in [A-Za-z]+\s+(?:19|20)\d{2}\b", re.IGNORECASE),
    re.compile(r"\bremote history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bquit as a teenager\b", re.IGNORECASE),
    re.compile(r"\bquit in [A-Za-z]+\s+(?:19|20)\d{2}\b", re.IGNORECASE),
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
    re.compile(r"\bdenies use of tobacco products\b", re.IGNORECASE),
    re.compile(r"\bdenies use of tobacco\b", re.IGNORECASE),
    re.compile(r"\bdoes not smoke\b", re.IGNORECASE),
    re.compile(r"\bdoesn't smoke\b", re.IGNORECASE),
    re.compile(r"\bno smoking\b", re.IGNORECASE),
    re.compile(r"\bdoes not smoke or use nicotine\b", re.IGNORECASE),
    re.compile(r"\bdoesn't smoke or use nicotine\b", re.IGNORECASE),
    re.compile(r"\bdoes not drink alcohol or smoke\b", re.IGNORECASE),
    re.compile(r"\bdoesn't drink alcohol or smoke\b", re.IGNORECASE),
    re.compile(r"\bdoes not drink alcohol or use tobacco\b", re.IGNORECASE),
    re.compile(r"\bdenied the use of tobacco\b", re.IGNORECASE),
    re.compile(r"\bdenies tobacco or drug use\b", re.IGNORECASE),
    re.compile(r"\bdenies tobacco or alcohol use\b", re.IGNORECASE),
    re.compile(r"\bdenied the use of tobacco,\s*alcohol\s*or\s*illicit drug use\b", re.IGNORECASE),
    re.compile(r"\bdenies use of tobacco,\s*alcohol\s*or\s*recreational drug use\b", re.IGNORECASE),
    re.compile(r"\bdenies use of tobacco,\s*alcohol\s*or\s*illicit drug use\b", re.IGNORECASE),
    re.compile(r"\bno history of tobacco\b", re.IGNORECASE),
    re.compile(r"\bno history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bpassive smoke exposure\s*[-:]\s*never smoker\b", re.IGNORECASE),
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

COUNSELING_ONLY_PATTERN = re.compile(
    r"\b(avoid tobacco use|avoid smoking|encouraged to avoid tobacco use|counseled to avoid tobacco use)\b",
    re.IGNORECASE
)

PASSIVE_SMOKE_PATTERN = re.compile(
    r"\bpassive smoke exposure\b",
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

# Added stronger structured EHR block helpers
STRUCTURED_STATUS_PATTERN = re.compile(
    r"\bsmoking status\s*[:\-]?\s*(current every day smoker|current some day smoker|current smoker|former smoker|never smoker|current|former|never|not on file)\b",
    re.IGNORECASE
)

STRUCTURED_TOBACCO_USE_PATTERN = re.compile(
    r"\btobacco use\s*[:\-]?\s*(current|former|never|history|not on file)\b",
    re.IGNORECASE
)

STRUCTURED_SMOKELESS_PATTERN = re.compile(
    r"\bsmokeless tobacco\s*[:\-]?\s*(never used|current user|former user|not on file)\b",
    re.IGNORECASE
)

STRUCTURED_COMMENT_CURRENT_PATTERN = re.compile(
    r"\bcomment\s*[:\-]?\s*(?:states?\s+)?(?:she|he|pt|patient)\s+smokes?\b",
    re.IGNORECASE
)

STRUCTURED_QUIT_AS_TEENAGER_PATTERN = re.compile(
    r"\bcomment\s*[:\-]?\s*quit as a teenager\b",
    re.IGNORECASE
)

DOES_NOT_SMOKE_COMPOUND_PATTERN = re.compile(
    r"\b(?:does\s+not|doesn't)\s+(?:drink\s+alcohol\s+or\s+)?smoke\b",
    re.IGNORECASE
)

DID_NOT_USE_TOBACCO_PATTERN = re.compile(
    r"\b(?:does|did)\s+not\s+use\s+tobacco\b",
    re.IGNORECASE
)

PREFERRED_SECTIONS = {
    "SOCIAL HISTORY",
    "HISTORY",
    "FULL",
    "SUBSTANCE AND SEXUAL ACTIVITY",
    "TOBACCO USE",
    "SOCIAL HISTORY/TOBACCO USE",
    "SOCIAL HISTORY MAIN TOPICS",
    "SOCIAL & SUBSTANCE USE HISTORY",
    "SOCIAL & SUBSTANCE USE TOPICS",
    "SUBSTANCE USE TOPICS",
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
    ctx = window_around(text, start, end, 180)
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
    if s == "SOCIAL HISTORY MAIN TOPICS":
        return 1
    if s == "SOCIAL & SUBSTANCE USE HISTORY":
        return 1
    if s == "SOCIAL & SUBSTANCE USE TOPICS":
        return 2
    if s == "SUBSTANCE USE TOPICS":
        return 2
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

    if value == "Current" and status == "structured_current":
        return 0
    if value == "Current" and status == "explicit_current":
        return 1
    if value == "Current" and status == "quantified_current":
        return 2
    if value == "Current" and status == "computed_recent_quit":
        return 3
    if value == "Former" and status == "structured_former":
        return 4
    if value == "Former" and status == "strong_former":
        return 5
    if value == "Former" and status == "generic_quit":
        return 6
    if value == "Never" and status == "structured_never":
        return 7
    if value == "Never" and status == "explicit_never":
        return 8
    if value == "Never" and status == "screening_never":
        return 9
    return 99


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


def _local_context(text, start, end, pad=180):
    left = max(0, start - pad)
    right = min(len(text), end + pad)
    return text[left:right]


def _is_family_history_context(text, start, end):
    ctx = _local_context(text, start, end, 180)
    return FAMILY_HISTORY_PATTERN.search(ctx) is not None


def _is_questionnaire_quit_context(text, start, end):
    ctx = _local_context(text, start, end, 220)
    return QUESTIONNAIRE_QUIT_PATTERN.search(ctx) is not None


def _is_questionnaire_false_current_context(text, start, end):
    ctx = _local_context(text, start, end, 220)
    return QUESTIONNAIRE_FALSE_CURRENT_PATTERN.search(ctx) is not None


def _is_negated_current_context(text, start, end):
    ctx = _local_context(text, start, end, 160)
    return NEGATED_CURRENT_CONTEXT.search(ctx) is not None


def _is_counseling_only_context(text, start, end):
    ctx = _local_context(text, start, end, 180)
    return COUNSELING_ONLY_PATTERN.search(ctx) is not None


def _add_candidates_from_patterns(patterns, text, note, section, value, confidence, source_type, out_list, suppress_family=True):
    for rx in patterns:
        for m in rx.finditer(text):
            if suppress_family and _is_family_history_context(text, m.start(), m.end()):
                continue

            if _is_counseling_only_context(text, m.start(), m.end()):
                continue

            if PASSIVE_SMOKE_PATTERN.search(_local_context(text, m.start(), m.end(), 120)) is not None:
                if value == "Current":
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
    for m in GENERIC_QUIT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        if _is_questionnaire_quit_context(text, m.start(), m.end()):
            continue

        ctx = _local_context(text, m.start(), m.end(), 160)

        if re.search(r"\b(recently|since last visit|last visit|today|this week)\b", ctx, re.IGNORECASE):
            out_list.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.91, "computed_recent_quit"))
        else:
            out_list.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.90, "generic_quit"))


def _find_quantified_current_candidates(text, note, section, out_list):
    patterns = [
        re.compile(r"\bpacks?/day\s*[:\-]?\s*[0-9]+(?:\.[0-9]+)?\b", re.IGNORECASE),
        re.compile(r"\b[0-9]+(?:\.\d+)?\s*pack[- ]years?\b", re.IGNORECASE),
        re.compile(r"\btypes?\s*:\s*cigarettes\b", re.IGNORECASE),
    ]

    for rx in patterns:
        for m in rx.finditer(text):
            if _is_family_history_context(text, m.start(), m.end()):
                continue
            ctx = _local_context(text, m.start(), m.end(), 200)

            if re.search(r"\bnever smoker\b", ctx, re.IGNORECASE):
                continue
            if re.search(r"\bformer smoker\b", ctx, re.IGNORECASE) and re.search(r"\bquit date\b|\byears since quitting\b|\blast attempt to quit\b", ctx, re.IGNORECASE):
                continue

            if re.search(r"\bcurrent every day smoker\b|\bcurrent some day smoker\b|\bcurrent smoker\b", ctx, re.IGNORECASE):
                out_list.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.992, "quantified_current"))
            elif re.search(r"\bcomment\s*:\s*states?\s+she\s+smokes?\b|\bsmokes?\s+every\s+once\s+in\s+a\s+while\b", ctx, re.IGNORECASE):
                out_list.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.99, "quantified_current"))


def _find_structured_block_candidates(text, note, section, out_list):
    note_dt = _parse_date_safe(getattr(note, "note_date", ""))

    status_m = STRUCTURED_STATUS_PATTERN.search(text)
    tobacco_use_m = STRUCTURED_TOBACCO_USE_PATTERN.search(text)
    smokeless_m = STRUCTURED_SMOKELESS_PATTERN.search(text)
    quit_date_m = QUIT_DATE_PATTERN.search(text)
    last_attempt_m = LAST_ATTEMPT_TO_QUIT_PATTERN.search(text)
    years_since_m = YEARS_SINCE_QUITTING_PATTERN.search(text)
    current_comment_m = STRUCTURED_COMMENT_CURRENT_PATTERN.search(text)
    quit_teen_m = STRUCTURED_QUIT_AS_TEENAGER_PATTERN.search(text)

    if status_m:
        raw = status_m.group(1).strip().lower()

        if "current every day smoker" in raw or "current some day smoker" in raw or raw == "current smoker" or raw == "current":
            if not _is_questionnaire_false_current_context(text, status_m.start(), status_m.end()) and not _is_negated_current_context(text, status_m.start(), status_m.end()):
                out_list.append(_candidate(note, section, "Current", text, status_m.start(), status_m.end(), 0.997, "structured_current"))
            return

        if raw == "former smoker" or raw == "former":
            if quit_date_m is not None:
                quit_dt = _parse_quit_date(quit_date_m.group(1))
                if note_dt is not None and quit_dt is not None:
                    dd = _days_between(note_dt, quit_dt)
                    if dd is not None and dd >= 0 and dd <= 90:
                        out_list.append(_candidate(note, section, "Current", text, quit_date_m.start(), quit_date_m.end(), 0.996, "computed_recent_quit"))
                    elif dd is not None and dd > 90:
                        out_list.append(_candidate(note, section, "Former", text, status_m.start(), status_m.end(), 0.996, "structured_former"))
                    else:
                        out_list.append(_candidate(note, section, "Former", text, status_m.start(), status_m.end(), 0.992, "structured_former"))
                else:
                    out_list.append(_candidate(note, section, "Former", text, status_m.start(), status_m.end(), 0.992, "structured_former"))
            elif last_attempt_m is not None:
                quit_dt = _parse_quit_date(last_attempt_m.group(1))
                if note_dt is not None and quit_dt is not None:
                    dd = _days_between(note_dt, quit_dt)
                    if dd is not None and dd >= 0 and dd <= 90:
                        out_list.append(_candidate(note, section, "Current", text, last_attempt_m.start(), last_attempt_m.end(), 0.996, "computed_recent_quit"))
                    elif dd is not None and dd > 90:
                        out_list.append(_candidate(note, section, "Former", text, status_m.start(), status_m.end(), 0.996, "structured_former"))
                    else:
                        out_list.append(_candidate(note, section, "Former", text, status_m.start(), status_m.end(), 0.992, "structured_former"))
                else:
                    out_list.append(_candidate(note, section, "Former", text, status_m.start(), status_m.end(), 0.992, "structured_former"))
            elif years_since_m is not None:
                yrs = float(years_since_m.group(1))
                if yrs < 0.25:
                    out_list.append(_candidate(note, section, "Current", text, years_since_m.start(), years_since_m.end(), 0.996, "computed_recent_quit"))
                else:
                    out_list.append(_candidate(note, section, "Former", text, status_m.start(), status_m.end(), 0.996, "structured_former"))
            elif quit_teen_m is not None:
                out_list.append(_candidate(note, section, "Former", text, quit_teen_m.start(), quit_teen_m.end(), 0.994, "structured_former"))
            else:
                out_list.append(_candidate(note, section, "Former", text, status_m.start(), status_m.end(), 0.992, "structured_former"))
            return

        if raw == "never smoker" or raw == "never":
            out_list.append(_candidate(note, section, "Never", text, status_m.start(), status_m.end(), 0.994, "structured_never"))
            return

        if raw == "not on file":
            # Not enough by itself.
            pass

    if tobacco_use_m:
        raw = tobacco_use_m.group(1).strip().lower()
        if raw == "never":
            out_list.append(_candidate(note, section, "Never", text, tobacco_use_m.start(), tobacco_use_m.end(), 0.985, "structured_never"))
        elif raw == "former":
            out_list.append(_candidate(note, section, "Former", text, tobacco_use_m.start(), tobacco_use_m.end(), 0.980, "structured_former"))
        elif raw == "current":
            if not _is_questionnaire_false_current_context(text, tobacco_use_m.start(), tobacco_use_m.end()) and not _is_negated_current_context(text, tobacco_use_m.start(), tobacco_use_m.end()):
                out_list.append(_candidate(note, section, "Current", text, tobacco_use_m.start(), tobacco_use_m.end(), 0.985, "structured_current"))

    if smokeless_m:
        raw = smokeless_m.group(1).strip().lower()
        if raw == "never used":
            if status_m and re.search(r"\bnever smoker\b", status_m.group(0), re.IGNORECASE):
                out_list.append(_candidate(note, section, "Never", text, smokeless_m.start(), smokeless_m.end(), 0.985, "structured_never"))

    if current_comment_m:
        out_list.append(_candidate(note, section, "Current", text, current_comment_m.start(), current_comment_m.end(), 0.992, "structured_current"))

    if quit_teen_m:
        out_list.append(_candidate(note, section, "Former", text, quit_teen_m.start(), quit_teen_m.end(), 0.992, "structured_former"))


def _find_compound_negation_candidates(text, note, section, out_list):
    for rx in [DOES_NOT_SMOKE_COMPOUND_PATTERN, DID_NOT_USE_TOBACCO_PATTERN]:
        for m in rx.finditer(text):
            if _is_family_history_context(text, m.start(), m.end()):
                continue
            out_list.append(_candidate(note, section, "Never", text, m.start(), m.end(), 0.975, "explicit_never"))


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

        # 0. Strong structured EHR tobacco block parsing first
        _find_structured_block_candidates(text, note, section, section_candidates)

        # 0b. Common negation variants
        _find_compound_negation_candidates(text, note, section, section_candidates)

        # 1. Explicit current
        _add_candidates_from_patterns(
            CURRENT_PATTERNS, text, note, section,
            "Current", 0.99, "explicit_current", section_candidates, suppress_family=True
        )

        # 1b. Quantified/structured current
        _find_quantified_current_candidates(text, note, section, section_candidates)

        # 2. Computed recent-quit evidence
        _find_recent_quit_context_candidates(text, note, section, section_candidates)
        _find_quit_relative_candidates(text, note, section, section_candidates)
        _find_quit_time_candidates(text, note, section, section_candidates)
        _find_years_since_quit_candidates(text, note, section, section_candidates)
        _find_quit_date_candidates(text, note, section, section_candidates)

        # 3. Strong former
        _add_candidates_from_patterns(
            FORMER_PATTERNS, text, note, section,
            "Former", 0.97, "strong_former", section_candidates, suppress_family=True
        )

        # 4. Explicit never
        _add_candidates_from_patterns(
            NEVER_PATTERNS, text, note, section,
            "Never", 0.96, "explicit_never", section_candidates, suppress_family=True
        )

        # 5. Generic quit
        _find_generic_quit_candidates(text, note, section, section_candidates)

        # 6. Weak screening/template never
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
