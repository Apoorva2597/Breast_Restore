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

STRONG_HISTORY_PATTERNS = [
    re.compile(r"\bs/p\b", re.I),
    re.compile(r"\bstatus\s+post\b", re.I),
    re.compile(r"\bhistory\s+of\b", re.I),
    re.compile(r"\bprior\b", re.I),
    re.compile(r"\bprevious\b", re.I),
    re.compile(r"\bremote\b", re.I),
    re.compile(r"\bwith\s+a\s+history\s+of\b", re.I),
    re.compile(r"\bpreviously\b", re.I),
    re.compile(r"\bund?erwent\b", re.I),
    re.compile(r"\btreated\s+with\b", re.I),
    re.compile(r"\bpast\s+surgical\s+history\b", re.I),
    re.compile(r"\bpast\s+medical\s+history\b", re.I),
]

LUMPECTOMY_CONTEXT_PATTERNS = [
    re.compile(r"\bs/p\b", re.I),
    re.compile(r"\bstatus\s+post\b", re.I),
    re.compile(r"\bhistory\s+of\b", re.I),
    re.compile(r"\bprior\b", re.I),
    re.compile(r"\bprevious\b", re.I),
    re.compile(r"\bpreviously\b", re.I),
    re.compile(r"\bund?erwent\b", re.I),
    re.compile(r"\btreated\s+with\b", re.I),
    re.compile(r"\bradiation\b", re.I),
    re.compile(r"\bchemotherapy\b", re.I),
    re.compile(r"\bchemo\b", re.I),
    re.compile(r"\bxrt\b", re.I),
    re.compile(r"\bsentinel\s+lymph\s+node\b", re.I),
    re.compile(r"\bslnb\b", re.I),
    re.compile(r"\balnd\b", re.I),
    re.compile(r"\bdcis\b", re.I),
    re.compile(r"\binvasive\s+ductal\s+carcinoma\b", re.I),
    re.compile(r"\bductal\s+carcinoma\b", re.I),
    re.compile(r"\blobular\s+carcinoma\b", re.I),
    re.compile(r"\bbreast\s+cancer\b", re.I),
    re.compile(r"\bpast\s+surgical\s+history\b", re.I),
    re.compile(r"\blumpectomy\s+scar\b", re.I),
    re.compile(r"\bwell[- ]healed\s+lumpectomy\s+scar\b", re.I),
    re.compile(r"\bin\s+(?:19|20)\d{2}\b", re.I),
    re.compile(r"\b(?:19|20)\d{2}\b", re.I),
]

LUMP_PATTERNS = [
    re.compile(r"\blumpectomy\b", re.I),
    re.compile(r"\bpartial\s+mastectomy\b", re.I),
    re.compile(r"\bsegmental\s+mastectomy\b", re.I),
    re.compile(r"\bbreast[- ]conserving\s+surgery\b", re.I),
    re.compile(r"\bwide\s+local\s+excision\b", re.I),
    re.compile(r"\bwire[- ]localized\s+lumpectomy\b", re.I),
    re.compile(r"\bbracketed\s+lumpectomy\b", re.I),
    re.compile(r"\bre[- ]excision\s+lumpectomy\b", re.I),
]

REDUCTION_PATTERNS = [
    re.compile(r"\bbreast\s+reduction\b", re.I),
    re.compile(r"\breduction\s+mammaplasty\b", re.I),
    re.compile(r"\bbreast\s+reduction\s+surgery\b", re.I),
    re.compile(r"\breduction\b", re.I),
]

MASTOPEXY_PATTERNS = [
    re.compile(r"\bmastopexy\b", re.I),
    re.compile(r"\bbreast\s+lift\b", re.I),
]

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
    re.compile(r"\bsubmuscular\s+saline\s+breast\s+augmentation\b", re.I),
    re.compile(r"\bsubmuscular\s+silicone\s+breast\s+augmentation\b", re.I),
    re.compile(r"\baugmentation\s+mammaplasty\s+(?:19|20)\d{2}\b", re.I),
]

OTHER_PATTERNS = [
    re.compile(r"\bexcisional\s+biopsy\b", re.I),
    re.compile(r"\bopen\s+breast\s+biopsy\b", re.I),
    re.compile(r"\bduct\s+excision\b", re.I),
    re.compile(r"\bfibroadenoma\s+excision\b", re.I),
    re.compile(r"\bbenign\s+breast\s+excision\b", re.I),
]

FIELD_CONFIG = [
    ("PBS_Lumpectomy", LUMP_PATTERNS, 0.94, "lumpectomy"),
    ("PBS_Breast Reduction", REDUCTION_PATTERNS, 0.84, "reduction_strict"),
    ("PBS_Mastopexy", MASTOPEXY_PATTERNS, 0.84, "mastopexy_strict"),
    ("PBS_Augmentation", AUGMENT_PATTERNS, 0.86, "augmentation_strict"),
    ("PBS_Other", OTHER_PATTERNS, 0.80, "strict_history"),
]


def _normalize_text(text):
    text = text or ""
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_any(patterns, text):
    for rx in patterns:
        if rx.search(text):
            return True
    return False


def _has_negation_near(text, start, end):
    ctx = window_around(text, start, end, 180)
    return _has_any(NEGATE_PATTERNS, ctx)


def _strict_history_near(text, start, end):
    ctx = window_around(text, start, end, 220)
    return _has_any(STRONG_HISTORY_PATTERNS, ctx)


def _lumpectomy_history_near(text, start, end):
    ctx = window_around(text, start, end, 280)
    return _has_any(LUMPECTOMY_CONTEXT_PATTERNS, ctx)


def _reduction_history_near(text, start, end):
    ctx = window_around(text, start, end, 240)
    if _has_any(STRONG_HISTORY_PATTERNS, ctx):
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", ctx, re.I):
        return True
    if re.search(r"\bbilateral\s+breast\s+reduction\b", ctx, re.I):
        return True
    if re.search(r"\bbreast\s+reduction\s+surgery\b", ctx, re.I):
        return True
    return False


def _mastopexy_history_near(text, start, end):
    ctx = window_around(text, start, end, 240)
    if _has_any(STRONG_HISTORY_PATTERNS, ctx):
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", ctx, re.I):
        return True
    if re.search(r"\bfor\s+symmetry\b", ctx, re.I):
        return True
    return False


def _augmentation_history_near(text, start, end):
    ctx = window_around(text, start, end, 280)

    positive = [
        re.compile(r"\bs/p\b", re.I),
        re.compile(r"\bhistory\s+of\b", re.I),
        re.compile(r"\bprior\b", re.I),
        re.compile(r"\bprevious\b", re.I),
        re.compile(r"\bpreviously\b", re.I),
        re.compile(r"\bcosmetic\b", re.I),
        re.compile(r"\bfor\s+augmentation\b", re.I),
        re.compile(r"\baugmentation\b", re.I),
        re.compile(r"\bin\s+(?:19|20)\d{2}\b", re.I),
        re.compile(r"\b(?:19|20)\d{2}\b", re.I),
        re.compile(r"\bsubmuscular\b", re.I),
        re.compile(r"\bsaline\b", re.I),
        re.compile(r"\bsilicone\b", re.I),
        re.compile(r"\baugmentation\s+mammaplasty\b", re.I),
        re.compile(r"\bpast\s+surgical\s+history\b", re.I),
    ]

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
        re.compile(r"\btissue\s+expander\s+placement\b", re.I),
        re.compile(r"\bexpander\s+placement\b", re.I),
        re.compile(r"\bdirect\s+implant\b", re.I),
    ]

    explicit = [
        re.compile(r"\bbreast\s+augmentation\b", re.I),
        re.compile(r"\baugmentation\s+mammaplasty\b", re.I),
        re.compile(r"\bcosmetic\s+augmentation\b", re.I),
        re.compile(r"\bbreast\s+implants?\s+for\s+augmentation\b", re.I),
        re.compile(r"\bprevious\s+submuscular\s+(?:saline|silicone)\s+breast\s+augmentation\b", re.I),
        re.compile(r"\baugmentation\s+mammaplasty\s+(?:19|20)\d{2}\b", re.I),
    ]

    if _has_any(explicit, ctx):
        return True

    has_positive = _has_any(positive, ctx)
    has_negative = _has_any(negative, ctx)

    return has_positive and not has_negative


def _emit(field, text, m, section, note, conf, status):
    return Candidate(
        field=field,
        value=True,
        status=status,
        evidence=window_around(text, m.start(), m.end(), 260),
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

                    elif mode == "reduction_strict":
                        hist = _reduction_history_near(text, m.start(), m.end())
                        status = "history_possible" if hist else "procedure_mention"
                        cands.append(_emit(field, text, m, section, note, conf, status))

                    elif mode == "mastopexy_strict":
                        hist = _mastopexy_history_near(text, m.start(), m.end())
                        status = "history_possible" if hist else "procedure_mention"
                        cands.append(_emit(field, text, m, section, note, conf, status))

                    elif mode == "augmentation_strict":
                        hist = _augmentation_history_near(text, m.start(), m.end())
                        status = "history_possible" if hist else "procedure_mention"
                        cands.append(_emit(field, text, m, section, note, conf, status))

                    elif mode == "strict_history":
                        hist = _strict_history_near(text, m.start(), m.end())
                        status = "history_possible" if hist else "procedure_mention"
                        cands.append(_emit(field, text, m, section, note, conf, status))

                    else:
                        cands.append(_emit(field, text, m, section, note, conf, "procedure_mention"))

    return cands
