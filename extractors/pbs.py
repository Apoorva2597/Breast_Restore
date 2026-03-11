# extractors/pbs.py
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "REVIEW OF SYSTEMS",
    "ALLERGIES"
}

NEGATE_PATTERNS = [
    re.compile(r"\bno\s+prior\s+breast\s+surgery\b", re.I),
    re.compile(r"\bno\s+history\s+of\s+breast\s+surgery\b", re.I),
    re.compile(r"\bdenies\s+prior\s+breast\s+surgery\b", re.I),
    re.compile(r"\bnever\s+had\s+breast\s+surgery\b", re.I),
]

# Strong history cues
STRONG_HISTORY_PATTERNS = [
    re.compile(r"\bs/p\b", re.I),
    re.compile(r"\bstatus\s+post\b", re.I),
    re.compile(r"\bhistory\s+of\b", re.I),
    re.compile(r"\bprior\b", re.I),
    re.compile(r"\bprevious\b", re.I),
    re.compile(r"\bremote\b", re.I),
    re.compile(r"\bwith\s+a\s+history\s+of\b", re.I),
]

# More permissive history cue for lumpectomy-only context
LUMPECTOMY_HISTORY_PATTERNS = [
    re.compile(r"\bs/p\b", re.I),
    re.compile(r"\bstatus\s+post\b", re.I),
    re.compile(r"\bhistory\s+of\b", re.I),
    re.compile(r"\bprior\b", re.I),
    re.compile(r"\bprevious\b", re.I),
    re.compile(r"\brecurrent\b", re.I),
    re.compile(r"\bafter\b", re.I),
]

LUMP_PATTERNS = [
    re.compile(r"\blumpectomy\b", re.I),
    re.compile(r"\bpartial\s+mastectomy\b", re.I),
    re.compile(r"\bsegmental\s+mastectomy\b", re.I),
    re.compile(r"\bbreast[- ]conserving\s+surgery\b", re.I),
    re.compile(r"\bwide\s+local\s+excision\b", re.I),
]

REDUCTION_PATTERNS = [
    re.compile(r"\bbreast\s+reduction\b", re.I),
    re.compile(r"\breduction\s+mammaplasty\b", re.I),
    re.compile(r"\breduction\b", re.I),
]

MASTOPEXY_PATTERNS = [
    re.compile(r"\bmastopexy\b", re.I),
    re.compile(r"\bbreast\s+lift\b", re.I),
]

# -----------------------------
# REVISED AUGMENTATION PATTERNS
# -----------------------------
AUGMENT_PATTERNS = [
    re.compile(r"\bbreast\s+augmentation\b", re.I),
    re.compile(r"\baugmentation\s+mammaplasty\b", re.I),
    re.compile(r"\bcosmetic\s+breast\s+augmentation\b", re.I),
    re.compile(r"\bcosmetic\s+augmentation\b", re.I),
    re.compile(r"\bs/p\s+(?:bilateral\s+|left\s+|right\s+)?augmentation\b", re.I),
    re.compile(r"\bhistory\s+of\s+(?:bilateral\s+|left\s+|right\s+)?breast\s+augmentation\b", re.I),
    re.compile(r"\bprior\s+(?:bilateral\s+|left\s+|right\s+)?breast\s+augmentation\b", re.I),
    re.compile(r"\bprevious\s+(?:bilateral\s+|left\s+|right\s+)?breast\s+augmentation\b", re.I),
    re.compile(r"\bbreast\s+implants?\s+for\s+augmentation\b", re.I),
    re.compile(r"\bcosmetic\s+breast\s+implants?\b", re.I),
    re.compile(r"\bsaline\s+breast\s+implants?\b", re.I),
    re.compile(r"\bsilicone\s+breast\s+implants?\b", re.I),
]

OTHER_PATTERNS = [
    re.compile(r"\bexcisional\s+biopsy\b", re.I),
    re.compile(r"\bopen\s+breast\s+biopsy\b", re.I),
    re.compile(r"\bduct\s+excision\b", re.I),
    re.compile(r"\bfibroadenoma\s+excision\b", re.I),
    re.compile(r"\bbenign\s+breast\s+excision\b", re.I),
]

FIELD_CONFIG = [
    ("PBS_Lumpectomy", LUMP_PATTERNS, 0.92, "lumpectomy"),
    ("PBS_Breast Reduction", REDUCTION_PATTERNS, 0.84, "strict_history"),
    ("PBS_Mastopexy", MASTOPEXY_PATTERNS, 0.84, "strict_history"),
    ("PBS_Augmentation", AUGMENT_PATTERNS, 0.86, "augmentation"),
    ("PBS_Other", OTHER_PATTERNS, 0.80, "strict_history"),
]


def _normalize_text(text):
    text = text or ""
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_negation_near(text, start, end):
    ctx = window_around(text, start, end, 160)
    for rx in NEGATE_PATTERNS:
        if rx.search(ctx):
            return True
    return False


def _has_any(patterns, text):
    for rx in patterns:
        if rx.search(text):
            return True
    return False


def _lumpectomy_history_near(text, start, end):
    ctx = window_around(text, start, end, 180)
    return _has_any(LUMPECTOMY_HISTORY_PATTERNS, ctx)


def _strict_history_near(text, start, end):
    ctx = window_around(text, start, end, 180)
    return _has_any(STRONG_HISTORY_PATTERNS, ctx)


def _augmentation_history_near(text, start, end):
    ctx = window_around(text, start, end, 220)

    # positive historical / cosmetic cues
    positive = [
        re.compile(r"\bs/p\b", re.I),
        re.compile(r"\bhistory\s+of\b", re.I),
        re.compile(r"\bprior\b", re.I),
        re.compile(r"\bprevious\b", re.I),
        re.compile(r"\bcosmetic\b", re.I),
        re.compile(r"\bfor\s+augmentation\b", re.I),
        re.compile(r"\baugmentation\b", re.I),
    ]

    # reconstruction-related cues we do NOT want augmentation to fire on
    negative = [
        re.compile(r"\breconstruction\b", re.I),
        re.compile(r"\bimplant[- ]based\s+reconstruction\b", re.I),
        re.compile(r"\btissue\s+expander\b", re.I),
        re.compile(r"\bexpander\b", re.I),
        re.compile(r"\bimplant\s+exchange\b", re.I),
        re.compile(r"\bexchange\s+of\s+(?:the\s+)?(?:tissue\s+expanders?|implants?)\b", re.I),
        re.compile(r"\bpermanent\s+(?:silicone|saline)\s+breast\s+implants?\b", re.I),
        re.compile(r"\bbreast\s+implant\s+reconstruction\b", re.I),
        re.compile(r"\bpost[- ]mastectomy\b", re.I),
        re.compile(r"\bmastectomy\b", re.I),
    ]

    has_positive = False
    for rx in positive:
        if rx.search(ctx):
            has_positive = True
            break

    has_negative = False
    for rx in negative:
        if rx.search(ctx):
            has_negative = True
            break

    # Require a positive cue. If strong reconstruction cues dominate,
    # do not label as augmentation history here.
    if has_positive and not has_negative:
        return True

    # allow explicit augmentation phrasing even if surrounding note is noisy
    explicit = [
        re.compile(r"\bbreast\s+augmentation\b", re.I),
        re.compile(r"\baugmentation\s+mammaplasty\b", re.I),
        re.compile(r"\bcosmetic\s+augmentation\b", re.I),
        re.compile(r"\bbreast\s+implants?\s+for\s+augmentation\b", re.I),
    ]
    return _has_any(explicit, ctx)


def _emit(field, text, m, section, note, conf, status):
    return Candidate(
        field=field,
        value=True,
        status=status,
        evidence=window_around(text, m.start(), m.end(), 220),
        section=section,
        note_type=note.note_type,
        note_id=note.note_id,
        note_date=note.note_date,
        confidence=conf,
    )


def extract_pbs(note: SectionedNote) -> List[Candidate]:
    cands = []

    for section, raw_text in note.sections.items():
        if section in SUPPRESS_SECTIONS:
            continue
        if not raw_text:
            continue

        text = _normalize_text(raw_text)

        for field, patterns, conf, mode in FIELD_CONFIG:
            for rx in patterns:
                for m in rx.finditer(text):
                    if _has_negation_near(text, m.start(), m.end()):
                        continue

                    if mode == "lumpectomy":
                        hist = _lumpectomy_history_near(text, m.start(), m.end())
                        status = "history_possible" if hist else "procedure_mention"
                        cands.append(_emit(field, text, m, section, note, conf, status))

                    elif mode == "strict_history":
                        hist = _strict_history_near(text, m.start(), m.end())
                        status = "history_possible" if hist else "procedure_mention"
                        cands.append(_emit(field, text, m, section, note, conf, status))

                    elif mode == "augmentation":
                        hist = _augmentation_history_near(text, m.start(), m.end())
                        status = "history_possible" if hist else "procedure_mention"
                        cands.append(_emit(field, text, m, section, note, conf, status))

                    else:
                        cands.append(_emit(field, text, m, section, note, conf, "procedure_mention"))

    return cands
