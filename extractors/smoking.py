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
# 5. Structured EHR social history blocks are common:
#    - Tobacco Use: History  Smoking status □ Never Smoker
#    - Smoking status □ Current Every Day Smoker
#    - Smokeless tobacco □ Never Used
#    These should be handled explicitly after unicode cleanup.
#
# Python 3.6.8 compatible.
# ----------------------------------------------

# unicode / template cleanup
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
    u"\uf0a7",  # private-use bullets often from exports
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
    re.compile(r"\bday smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoker,\s*current\b", re.IGNORECASE),
    re.compile(r"\bdown to\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*cigs?\s+(?:daily|per day)\b", re.IGNORECASE),
    re.compile(r"\busing chantix[^\.]{0,100}\b(?:\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*cigs?\s+(?:daily|per day)|down to\s+\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*cigs?)", re.IGNORECASE),
    re.compile(r"\bsmokes approximately\s+\d+(?:\.\d+)?\s+cigarettes?\s+a\s+day\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+(?:a\s+)?(?:couple|few)\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+\d+(?:\.\d+)?\s+cigarettes?\s+(?:a|per)\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+\d+(?:\.\d+)?\s*packs?\s*/?\s*(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bsmokes\s+every\s+once\s+in\s+a\s+while\b", re.IGNORECASE),
    re.compile(r"\bcomment\s*[:\-]?\s*states\s+she\s+smokes\b", re.IGNORECASE),
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
    re.compile(r"\bdenies use of tobacco products\b", re.IGNORECASE),
    re.compile(r"\bdoes not smoke\b", re.IGNORECASE),
    re.compile(r"\bdoesn't smoke\b", re.IGNORECASE),
    re.compile(r"\bno smoking\b", re.IGNORECASE),
    re.compile(r"\bdoes not smoke or use nicotine\b", re.IGNORECASE),
    re.compile(r"\bnever used tobacco\b", re.IGNORECASE),
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
    r"\b(is patient currently smoking\?\s*no|currently smoking\?\s*no|active tobacco use\?\s*no|current tobacco use\?\s*no|if active smoker\b|was patient advised by provider to quit smoking\?|was patient referred to mhealthy\?)\b",
    re.IGNORECASE
)

FAMILY_HISTORY_PATTERN = re.compile(
    r"\bfamily history\b|\bmother\b|\bfather\b|\bgrandmother\b|\bgrandfather\b|\baunt\b|\buncle\b|\bsister\b|\bbrother\b",
    re.IGNORECASE
)

# explicit structured social-history windows
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


def _find_best(patterns, text, note, section, value, confidence, suppress_family=True):
    best = None
    best_key = None

    for rx in patterns:
        for m in rx.finditer(text):
            if suppress_family and _is_family_history_context(text, m.start(), m.end()):
                continue

            if value == "Current" and _is_questionnaire_false_current_context(text, m.start(), m.end()):
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


def _local_window_has(text, anchor_start, anchor_end, pattern, left=80, right=180):
    s = max(0, anchor_start - left)
    e = min(len(text), anchor_end + right)
    ctx = text[s:e]
    return pattern.search(ctx) is not None, s, e


def _find_structured_block_candidates(text, note, section):
    """
    Recover structured EHR smoking blocks like:
      Tobacco Use: History Smoking status Never Smoker Smokeless tobacco Never Used
      Smoking status Current Every Day Smoker
      Tobacco comment exposed to second hand smoke ...
    """
    candidates = []

    current_rx = re.compile(
        r"\b(current every day smoker|current some day smoker|current smoker|light tobacco smoker|day smoker)\b",
        re.IGNORECASE
    )
    former_rx = re.compile(
        r"\b(former smoker|ex[- ]smoker)\b",
        re.IGNORECASE
    )
    never_rx = re.compile(
        r"\b(never smoker|never smoked|nonsmoker|non[- ]smoker)\b",
        re.IGNORECASE
    )
    smokeless_never_rx = re.compile(
        r"\bsmokeless tobacco\s*[:\-]?\s*never used\b",
        re.IGNORECASE
    )
    tobacco_not_on_file_rx = re.compile(
        r"\btobacco use\s*[:\-]?\s*(history|not on file)\b",
        re.IGNORECASE
    )
    passive_smoke_rx = re.compile(
        r"\bpassive smoke exposure\s*[:\-]?\s*never smoker\b",
        re.IGNORECASE
    )

    for m in STRUCTURED_BLOCK_START_PATTERN.finditer(text):
        s = m.start()
        e = min(len(text), m.end() + 260)
        chunk = text[s:e]

        if _is_family_history_context(text, s, e):
            continue

        m2 = current_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            if not _is_questionnaire_false_current_context(text, start, end):
                candidates.append(_candidate(note, section, "Current", text, start, end, 0.995))

        m2 = former_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Former", text, start, end, 0.992))

        m2 = never_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Never", text, start, end, 0.991))

        # structured "not on file" alone should not force a smoking class
        # but paired with smokeless never or explicit never smoker it is okay
        m2 = smokeless_never_rx.search(chunk)
        if m2 is not None:
            has_never, _, _ = _local_window_has(chunk, m2.start(), m2.end(), never_rx, left=120, right=120)
            if has_never:
                start = s + m2.start()
                end = s + m2.end()
                candidates.append(_candidate(note, section, "Never", text, start, end, 0.989))

        # passive smoke exposure = never smoker, not current/former smoker
        m2 = passive_smoke_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Never", text, start, end, 0.988))

        # explicit comment in structured block can indicate current
        comment_current_rx = re.compile(
            r"\bcomment\s*[:\-]?\s*states\s+she\s+smokes\b|\bsmokes\s+every\s+once\s+in\s+a\s+while\s+currently\b",
            re.IGNORECASE
        )
        m2 = comment_current_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Current", text, start, end, 0.994))

        # explicit "does not smoke" / "no smoking" in nearby social history window
        structured_never_rx = re.compile(
            r"\b(does not smoke|doesn't smoke|no smoking|does not smoke or use nicotine|denies use of tobacco products|denies tobacco use|denies tobacco)\b",
            re.IGNORECASE
        )
        m2 = structured_never_rx.search(chunk)
        if m2 is not None:
            start = s + m2.start()
            end = s + m2.end()
            candidates.append(_candidate(note, section, "Never", text, start, end, 0.987))

    # also allow direct global structured matches not necessarily anchored by start term
    direct_structured = [
        (re.compile(r"\bsmoking status\s*[:\-]?\s*current every day smoker\b", re.IGNORECASE), "Current", 0.996),
        (re.compile(r"\bsmoking status\s*[:\-]?\s*current some day smoker\b", re.IGNORECASE), "Current", 0.996),
        (re.compile(r"\bsmoking status\s*[:\-]?\s*former smoker\b", re.IGNORECASE), "Former", 0.995),
        (re.compile(r"\bsmoking status\s*[:\-]?\s*never smoker\b", re.IGNORECASE), "Never", 0.995),
        (re.compile(r"\bsmokeless tobacco\s*[:\-]?\s*never used\b", re.IGNORECASE), "Never", 0.985),
    ]

    for rx, value, conf in direct_structured:
        for m in rx.finditer(text):
            if _is_family_history_context(text, m.start(), m.end()):
                continue
            if value == "Current" and _is_questionnaire_false_current_context(text, m.start(), m.end()):
                continue
            candidates.append(_candidate(note, section, value, text, m.start(), m.end(), conf))

    return candidates


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

        # 0. Structured EHR blocks first: high-yield remaining misses
        block_candidates = _find_structured_block_candidates(text, note, section)
        if block_candidates:
            all_candidates.extend(block_candidates)

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

    if not all_candidates:
        return []

    def sort_key(c):
        return (
            _section_priority(c.section),
            -float(getattr(c, "confidence", 0.0) or 0.0),
            len(getattr(c, "evidence", "") or "")
        )

    best = sorted(all_candidates, key=sort_key)[0]
    return [best]
