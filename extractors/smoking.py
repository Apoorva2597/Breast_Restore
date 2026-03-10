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
# Key design:
# - A note may contain multiple smoking clues.
# - We collect all of them first.
# - Then we choose the best smoking interpretation using
#   smoking-specific priority, not generic section/confidence only.
#
# Python 3.6.8 compatible.
# ----------------------------------------------

CHECKBOX_CHARS = [
    u"\u25A1",  # □
    u"\u25A0",  # ■
    u"\u25AA",  # ▪
    u"\u25AB",  # ▫
    u"\u2610",  # ☐
    u"\u2611",  # ☑
    u"\u2612",  # ☒
    u"\u2022",  # •
    u"\u00B7",  # ·
    u"\uf0a7",
    u"\uf0b7",
    u"\uf0fc",
]

CURRENT_PATTERNS = [
    re.compile(r"\bcurrent smoker\b", re.IGNORECASE),
    re.compile(r"\bactive smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking currently\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\b", re.IGNORECASE),
    re.compile(r"\bcurrently smokes\b", re.IGNORECASE),
    re.compile(r"\bstill smoking\b", re.IGNORECASE),
    re.compile(r"\bcontinues to smoke\b", re.IGNORECASE),
    re.compile(r"\bcurrent every day smoker\b", re.IGNORECASE),
    re.compile(r"\bcurrent some day smoker\b", re.IGNORECASE),
    re.compile(r"\blight tobacco smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoker,\s*current\b", re.IGNORECASE),
    re.compile(r"\bdown to\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*cigs?\s+(?:daily|per day)\b", re.IGNORECASE),
    re.compile(r"\busing chantix[^\.]{0,100}\b(?:\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*cigs?\s+(?:daily|per day)|down to\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*cigs?)", re.IGNORECASE),
    re.compile(r"\bsmokes approximately\s+\d+(?:\.\d+)?\s+cigarettes?\s+a\s+day\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+(?:a\s+)?(?:couple|few)\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+\d+(?:\.\d+)?\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+\d+(?:\.\d+)?\s*packs?\s*/?\s*(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+every\s+once\s+in\s+a\s+while\b", re.IGNORECASE),
    re.compile(r"\bcomment\s*[:\-]?\s*(?:states?\s+)?(?:she|he|pt|patient)\s+smokes\b", re.IGNORECASE),
    re.compile(r"\btobacco use\s*[:\-]?\s*current\b", re.IGNORECASE),
]

FORMER_PATTERNS = [
    re.compile(r"\bformer smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking status\s*[:\-]?\s*former(?:\s+smoker)?\b", re.IGNORECASE),
    re.compile(r"\bhistory smoking status\s*[:\-]?\s*former(?:\s+smoker)?\b", re.IGNORECASE),
    re.compile(r"\bex[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\bformer user\b", re.IGNORECASE),
    re.compile(r"\bremote history of tobacco use\b", re.IGNORECASE),
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
    re.compile(r"\bdenies use of tobacco products\b", re.IGNORECASE),
    re.compile(r"\bdoes not smoke\b", re.IGNORECASE),
    re.compile(r"\bdoesn't smoke\b", re.IGNORECASE),
    re.compile(r"\bdoes not smoke or use nicotine\b", re.IGNORECASE),
    re.compile(r"\bdoesn't smoke or use nicotine\b", re.IGNORECASE),
    re.compile(r"\bnever used tobacco\b", re.IGNORECASE),
]

SCREENING_NEVER_PATTERNS = [
    re.compile(r"\bactive tobacco use\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bis patient currently smoking\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bactive tobacco use\s*[:\-]?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrent tobacco use\s*[:\-]?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\s*[:\-]?\s*no\b", re.IGNORECASE),
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

LAST_ATTEMPT_PATTERN = re.compile(
    r"\blast attempt to quit\s*[:\-]?\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{1,2}/[0-9]{4}|(?:19|20)[0-9]{2})\b",
    re.IGNORECASE
)

GENERIC_QUIT_PATTERN = re.compile(
    r"\b(quit smoking|quit tobacco|stopped smoking|stopped tobacco)\b",
    re.IGNORECASE
)

RECENT_QUIT_CONTEXT_PATTERN = re.compile(
    r"\b(?:since\s+(?:our|the)\s+last\s+visit[^\.]{0,120}?quit|recently\s+quit)\b",
    re.IGNORECASE
)

QUESTIONNAIRE_QUIT_PATTERN = re.compile(
    r"\b(resources?\s+to\s+help\s+quit\s+smoking|interested\s+in\s+resources?\s+to\s+help\s+quit\s+smoking|referral\s+to\s+mhealthy|referred\s+to\s+mhealthy|advised\s+by\s+provider\s+to\s+quit\s+smoking|plans?\s+to\s+quit)\b",
    re.IGNORECASE
)

QUESTIONNAIRE_FALSE_CURRENT_PATTERN = re.compile(
    r"\b(is patient currently smoking\?\s*no|currently smoking\?\s*no|active tobacco use\?\s*no|current tobacco use\?\s*no|if active smoker\b|was patient advised by provider to quit smoking\?|was patient referred to mhealthy\?|not currently smoking|no current tobacco use)\b",
    re.IGNORECASE
)

FAMILY_HISTORY_PATTERN = re.compile(
    r"\bfamily history\b|\bmother\b|\bfather\b|\bgrandmother\b|\bgrandfather\b|\baunt\b|\buncle\b|\bsister\b|\bbrother\b",
    re.IGNORECASE
)

STRUCTURED_BLOCK_START_PATTERN = re.compile(
    r"\b(?:social history|social & substance use history|substance use topics|social history main topics|history smoking status|smoking status|tobacco use)\b",
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
    for ch in CHECKBOX_CHARS:
        text = text.replace(ch, " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("|", " | ")
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


def _candidate(note, section, value, text, start, end, confidence, status="present"):
    ctx = window_around(text, start, end, 160)
    return Candidate(
        field="SmokingStatus",
        value=value,
        status=status,
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
    left = max(0, start - 180)
    right = min(len(text), end + 180)
    ctx = text[left:right]
    return FAMILY_HISTORY_PATTERN.search(ctx) is not None


def _is_questionnaire_quit_context(text, start, end):
    left = max(0, start - 180)
    right = min(len(text), end + 180)
    ctx = text[left:right]
    return QUESTIONNAIRE_QUIT_PATTERN.search(ctx) is not None


def _is_questionnaire_false_current_context(text, start, end):
    left = max(0, start - 220)
    right = min(len(text), end + 220)
    ctx = text[left:right]
    return QUESTIONNAIRE_FALSE_CURRENT_PATTERN.search(ctx) is not None


def _has_strong_former_support(txt):
    txt = txt.lower()
    return (
        ("former smoker" in txt or "smoking status former" in txt or "history smoking status former" in txt or "ex-smoker" in txt or "ex smoker" in txt) and
        ("quit date" in txt or "years since quitting" in txt or "last attempt to quit" in txt or "quit " in txt or "stopped smoking" in txt or "stopped tobacco" in txt)
    )


def _is_smokeless_only_never(cand):
    val = str(getattr(cand, "value", "") or "").strip()
    if val != "Never":
        return False

    txt = str(getattr(cand, "evidence", "") or "").lower()
    if "smokeless tobacco" not in txt:
        return False

    if "never smoker" in txt or "never smoked" in txt or "nonsmoker" in txt or "non-smoker" in txt:
        return False

    return True


def _find_best(patterns, text, note, section, value, confidence, suppress_family=True, status="present"):
    out = []
    for rx in patterns:
        for m in rx.finditer(text):
            if suppress_family and _is_family_history_context(text, m.start(), m.end()):
                continue
            if value == "Current" and _is_questionnaire_false_current_context(text, m.start(), m.end()):
                continue
            out.append(_candidate(note, section, value, text, m.start(), m.end(), confidence, status=status))
    return out


def _find_recent_quit_context_candidates(text, note, section):
    out = []
    for m in RECENT_QUIT_CONTEXT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        out.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.97, status="recent_quit_current"))
    return out


def _find_quit_time_candidates(text, note, section):
    out = []
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

        status = "recent_quit_current" if value == "Current" else "quit_supported_former"
        conf = 0.98 if value == "Current" else 0.97
        out.append(_candidate(note, section, value, text, m.start(), m.end(), conf, status=status))
    return out


def _find_quit_years_ago_candidates(text, note, section):
    out = []
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

            status = "recent_quit_current" if value == "Current" else "quit_supported_former"
            conf = 0.98 if value == "Current" else 0.98
            out.append(_candidate(note, section, value, text, m.start(), m.end(), conf, status=status))
    return out


def _find_years_since_quit_candidates(text, note, section):
    out = []
    for m in YEARS_SINCE_QUITTING_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        yrs = float(m.group(1))
        value = "Current" if yrs < 0.25 else "Former"
        status = "recent_quit_current" if value == "Current" else "quit_supported_former"
        conf = 0.98 if value == "Current" else 0.97
        out.append(_candidate(note, section, value, text, m.start(), m.end(), conf, status=status))
    return out


def _find_quit_date_candidates(text, note, section):
    out = []
    note_dt = _parse_date_safe(getattr(note, "note_date", ""))

    for m in QUIT_DATE_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        quit_dt = _parse_quit_date(m.group(1))
        if note_dt is not None and quit_dt is not None:
            dd = _days_between(note_dt, quit_dt)
            if dd is not None and dd >= 0 and dd <= 90:
                out.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.99, status="recent_quit_current"))
            elif dd is not None and dd > 90:
                out.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.98, status="quit_supported_former"))
            else:
                out.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.96, status="quit_supported_former"))
        else:
            out.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.95, status="quit_supported_former"))
    return out


def _find_last_attempt_candidates(text, note, section):
    out = []
    note_dt = _parse_date_safe(getattr(note, "note_date", ""))

    for m in LAST_ATTEMPT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue

        quit_dt = _parse_quit_date(m.group(1))
        if note_dt is not None and quit_dt is not None:
            dd = _days_between(note_dt, quit_dt)
            if dd is not None and dd >= 0 and dd <= 90:
                out.append(_candidate(note, section, "Current", text, m.start(), m.end(), 0.99, status="recent_quit_current"))
            elif dd is not None and dd > 90:
                out.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.97, status="quit_supported_former"))
            else:
                out.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.95, status="quit_supported_former"))
        else:
            out.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.95, status="quit_supported_former"))
    return out


def _find_generic_quit_candidates(text, note, section):
    out = []
    for m in GENERIC_QUIT_PATTERN.finditer(text):
        if _is_family_history_context(text, m.start(), m.end()):
            continue
        if _is_questionnaire_quit_context(text, m.start(), m.end()):
            continue
        out.append(_candidate(note, section, "Former", text, m.start(), m.end(), 0.86, status="generic_quit_former"))
    return out


def _find_structured_block_candidates(text, note, section):
    candidates = []

    current_rx = re.compile(
        r"\bsmoking status\s*[:\-]?\s*(current every day smoker|current some day smoker|current smoker|current|light tobacco smoker)\b",
        re.IGNORECASE
    )
    former_rx = re.compile(
        r"\bsmoking status\s*[:\-]?\s*(former smoker|former)\b|\bhistory smoking status\s*[:\-]?\s*former(?:\s+smoker)?\b",
        re.IGNORECASE
    )
    never_rx = re.compile(
        r"\bsmoking status\s*[:\-]?\s*(never smoker|never)\b",
        re.IGNORECASE
    )
    smokeless_never_rx = re.compile(
        r"\bsmokeless tobacco\s*[:\-]?\s*never used\b",
        re.IGNORECASE
    )
    passive_smoke_rx = re.compile(
        r"\bpassive smoke exposure\s*[:\-]?\s*never smoker\b",
        re.IGNORECASE
    )
    comment_current_rx = re.compile(
        r"\bcomment\s*[:\-]?\s*(?:states?\s+)?(?:she|he|pt|patient)\s+smokes\b|\bsmokes\s+every\s+once\s+in\s+a\s+while\s+currently\b",
        re.IGNORECASE
    )
    structured_never_rx = re.compile(
        r"\b(does not smoke|doesn't smoke|does not smoke or use nicotine|denies tobacco use|denies use of tobacco products|never smoked|never smoker|nonsmoker|non[- ]smoker)\b",
        re.IGNORECASE
    )

    for m in STRUCTURED_BLOCK_START_PATTERN.finditer(text):
        s = m.start()
        e = min(len(text), m.end() + 320)
        chunk = text[s:e]

        if _is_family_history_context(text, s, e):
            continue

        m2 = current_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            if not _is_questionnaire_false_current_context(text, start, end):
                candidates.append(_candidate(note, section, "Current", text, start, end, 0.997, status="structured_current"))

        m2 = former_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            if ("quit date" in chunk.lower()) or ("years since quitting" in chunk.lower()) or ("last attempt to quit" in chunk.lower()) or ("quit " in chunk.lower()) or ("stopped smoking" in chunk.lower()):
                candidates.append(_candidate(note, section, "Former", text, start, end, 0.998, status="structured_former_supported"))
            else:
                candidates.append(_candidate(note, section, "Former", text, start, end, 0.993, status="structured_former"))

        m2 = never_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Never", text, start, end, 0.994, status="structured_never"))

        m2 = passive_smoke_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Never", text, start, end, 0.992, status="structured_never"))

        m2 = comment_current_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Current", text, start, end, 0.996, status="structured_current"))

        m2 = structured_never_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Never", text, start, end, 0.987, status="narrative_never"))

        # keep smokeless-only signal as a low-priority helper only;
        # later ranking will never let it stand alone as Never smoker
        m2 = smokeless_never_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Never", text, start, end, 0.900, status="smokeless_only_never"))

    return candidates


def _smoking_priority(c):
    val = str(getattr(c, "value", "") or "").strip()
    status = str(getattr(c, "status", "") or "").strip().lower()
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    section_pri = _section_priority(getattr(c, "section", ""))
    txt = str(getattr(c, "evidence", "") or "").lower()

    if _is_smokeless_only_never(c):
        return (99, section_pri, -conf)

    # Current first
    if status in {"structured_current"}:
        return (0, section_pri, -conf)
    if status in {"recent_quit_current"}:
        return (1, section_pri, -conf)
    if val == "Current":
        return (2, section_pri, -conf)

    # Former next, especially when supported by quit evidence
    if status in {"structured_former_supported"}:
        return (3, section_pri, -conf)
    if status in {"quit_supported_former"}:
        return (4, section_pri, -conf)
    if status in {"structured_former"}:
        return (5, section_pri, -conf)
    if val == "Former" and _has_strong_former_support(txt):
        return (6, section_pri, -conf)
    if val == "Former":
        return (7, section_pri, -conf)

    # Never last
    if status in {"structured_never"}:
        return (8, section_pri, -conf)
    if val == "Never" and ("never smoker" in txt or "never smoked" in txt or "passive smoke exposure" in txt):
        return (9, section_pri, -conf)
    if status in {"narrative_never"}:
        return (10, section_pri, -conf)
    if val == "Never":
        return (11, section_pri, -conf)

    return (50, section_pri, -conf)


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

        all_candidates.extend(_find_structured_block_candidates(text, note, section))

        all_candidates.extend(_find_best(CURRENT_PATTERNS, text, note, section, "Current", 0.99, suppress_family=True, status="present_current"))
        all_candidates.extend(_find_recent_quit_context_candidates(text, note, section))
        all_candidates.extend(_find_quit_years_ago_candidates(text, note, section))
        all_candidates.extend(_find_quit_time_candidates(text, note, section))
        all_candidates.extend(_find_years_since_quit_candidates(text, note, section))
        all_candidates.extend(_find_quit_date_candidates(text, note, section))
        all_candidates.extend(_find_last_attempt_candidates(text, note, section))
        all_candidates.extend(_find_best(FORMER_PATTERNS, text, note, section, "Former", 0.96, suppress_family=True, status="present_former"))
        all_candidates.extend(_find_best(NEVER_PATTERNS, text, note, section, "Never", 0.93, suppress_family=True, status="present_never"))
        all_candidates.extend(_find_generic_quit_candidates(text, note, section))
        all_candidates.extend(_find_best(SCREENING_NEVER_PATTERNS, text, note, section, "Never", 0.70, suppress_family=True, status="screening_never"))

    if not all_candidates:
        return []

    best = sorted(all_candidates, key=_smoking_priority)[0]
    return [best]
