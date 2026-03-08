# extractors/bmi.py
#
# BMI extractor designed for OP NOTE / BRIEF OP NOTE wording, with clinic fallback support.
#
# Captures patterns such as:
#   BMI 30.4
#   BMI: 30.4
#   BMI (30.4)
#   BMI=30.4
#   BMI 30
#   BMI of 30.4
#   body mass index 30.4
#   obesity, BMI 30
#   morbid obesity-BMI 48.4
#   morbid obesity (BMI 37.8)
#
# Avoids threshold / rule language such as:
#   BMI >= 35
#   BMI greater than 35
#   if BMI ...
#
# Python 3.6.8 compatible.

import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around


BMI_PATTERNS = [
    # BMI 30.4 / BMI: 30.4 / BMI=30 / BMI 30 kg/m2
    re.compile(
        r"\bBMI\b\s*[:=]?\s*\(?\s*(\d{2,3}(?:\.\d+)?)\s*\)?\s*(?:kg\s*/\s*m2|kg\s*/\s*m\^?2|kg/m2|kg/m\^?2)?\b",
        re.IGNORECASE
    ),

    # BMI of 30.4
    re.compile(
        r"\bBMI\s+of\s+\(?\s*(\d{2,3}(?:\.\d+)?)\s*\)?\b",
        re.IGNORECASE
    ),

    # body mass index 30.4 / body mass index: 30.4
    re.compile(
        r"\bbody\s+mass\s+index\b\s*[:=]?\s*\(?\s*(\d{2,3}(?:\.\d+)?)\s*\)?\b",
        re.IGNORECASE
    ),

    # obesity, BMI 30 / obesity - BMI 30 / obesity (BMI 30)
    re.compile(
        r"\bobesity\b[\s,\-:;()]{0,12}\bBMI\b[\s:=()\-]{0,8}\(?\s*(\d{2,3}(?:\.\d+)?)\s*\)?\b",
        re.IGNORECASE
    ),

    # morbid obesity-BMI 48.4 / morbid obesity (BMI 37.8)
    re.compile(
        r"\bmorbid\s+obesity\b[\s,\-:;()]{0,12}\bBMI\b[\s:=()\-]{0,8}\(?\s*(\d{2,3}(?:\.\d+)?)\s*\)?\b",
        re.IGNORECASE
    ),
]


THRESHOLD_FALSE_POS = re.compile(
    r"\bBMI\b.{0,50}\b("
    r"greater\s+than|less\s+than|greater\s+than\s+or\s+equal\s+to|"
    r"less\s+than\s+or\s+equal\s+to|at\s+least|more\s+than|under|over|"
    r"above|below|minimum|maximum|threshold|criteria|cutoff|eligib"
    r")\b|"
    r"\bBMI\s*(>=|<=|>|<)\s*\d{1,3}(?:\.\d+)?\b",
    re.IGNORECASE,
)

CONDITIONAL_FALSE_POS = re.compile(
    r"\b(if|when|because|due\s+to|given|for\s+patients?\s+with)\b.{0,50}\bBMI\b",
    re.IGNORECASE,
)

NON_MEASURED_FALSE_POS = re.compile(
    r"\b(weight\s+loss|diet|exercise|counsel|counseling|goal\s+bmi|target\s+bmi)\b",
    re.IGNORECASE,
)

PREFERRED_SECTIONS = {
    "FULL",
    "PREOPERATIVE DIAGNOSES",
    "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSES",
    "POSTOPERATIVE DIAGNOSIS",
    "PHYSICAL EXAM",
    "OBJECTIVE",
    "ASSESSMENT",
    "ASSESSMENT/PLAN",
}

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "SOCIAL HISTORY",
    "ALLERGIES",
}


def _normalize_text(text):
    text = text or ""
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace(u"\xa0", " ")
    text = text.replace(u"\u25a1", " ")
    text = text.replace(u"\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _section_order(note):
    order = []

    for s in note.sections.keys():
        if s in PREFERRED_SECTIONS and s not in SUPPRESS_SECTIONS:
            order.append(s)

    for s in note.sections.keys():
        if s not in order and s not in SUPPRESS_SECTIONS:
            order.append(s)

    return order


def _confidence_for_note_type(note_type):
    nt = str(note_type or "").lower().strip()

    if "brief op note" in nt:
        return 0.96
    if "op note" in nt:
        return 0.95
    if "operative" in nt or "operation" in nt:
        return 0.94
    if "clinic" in nt or "progress" in nt or "h&p" in nt:
        return 0.88
    return 0.84


def _valid_bmi_value(val):
    try:
        x = float(val)
    except Exception:
        return False
    return 10.0 <= x <= 80.0


def extract_bmi(note: SectionedNote) -> List[Candidate]:
    """
    Returns at most one BMI candidate per note.
    """
    section_order = _section_order(note)

    best_candidate = None
    best_score = -1.0

    for section in section_order:
        raw_text = note.sections.get(section, "") or ""
        if not raw_text:
            continue

        text = _normalize_text(raw_text)
        if not text:
            continue

        for rx in BMI_PATTERNS:
            for m in rx.finditer(text):
                raw_val = m.group(1)

                if not _valid_bmi_value(raw_val):
                    continue

                bmi_val = float(raw_val)
                ctx = window_around(text, m.start(), m.end(), 180)

                if not ctx:
                    continue

                if THRESHOLD_FALSE_POS.search(ctx):
                    continue

                if CONDITIONAL_FALSE_POS.search(ctx):
                    ctx_l = ctx.lower()
                    if ("morbid obesity" not in ctx_l) and ("obesity" not in ctx_l):
                        continue

                if NON_MEASURED_FALSE_POS.search(ctx):
                    continue

                score = _confidence_for_note_type(note.note_type)
                ctx_l = ctx.lower()

                if "morbid obesity" in ctx_l:
                    score += 0.04
                elif "obesity" in ctx_l:
                    score += 0.02

                if section in {
                    "PREOPERATIVE DIAGNOSES",
                    "PREOPERATIVE DIAGNOSIS",
                    "POSTOPERATIVE DIAGNOSES",
                    "POSTOPERATIVE DIAGNOSIS",
                }:
                    score += 0.03

                cand = Candidate(
                    field="BMI",
                    value=bmi_val,
                    status="measured",
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=score,
                )

                if best_candidate is None or score > best_score:
                    best_candidate = cand
                    best_score = score

    if best_candidate is None:
        return []

    return [best_candidate]
