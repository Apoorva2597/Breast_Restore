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
]


STRONG_HISTORY_PATTERNS = [
    re.compile(r"\bs/p\b", re.I),
    re.compile(r"\bstatus\s+post\b", re.I),
    re.compile(r"\bhistory\s+of\b", re.I),
    re.compile(r"\bprior\b", re.I),
    re.compile(r"\bprevious\b", re.I),
    re.compile(r"\bremote\b", re.I),
    re.compile(r"\bpreviously\b", re.I),
    re.compile(r"\btreated\s+with\b", re.I),
    re.compile(r"\bund?erwent\b", re.I),
    re.compile(r"\bpast\s+surgical\s+history\b", re.I),
]


# --------------------------------
# Lumpectomy-specific refinements
# --------------------------------

LUMPECTOMY_PLANNING_PATTERNS = [
    re.compile(r"\bcandidate\s+for\s+lumpectomy\b", re.I),
    re.compile(r"\bnot\s+(?:felt\s+to\s+be\s+)?a\s+lumpectomy\s+candidate\b", re.I),
    re.compile(r"\blumpectomy\s+vs\.?\s+mastectomy\b", re.I),
    re.compile(r"\bbreast\s+conservation\s+surgery[-–]lumpectomy\b", re.I),
    re.compile(r"\bdiscussion\s+of\s+lumpectomy\b", re.I),
    re.compile(r"\bdiscussed\s+lumpectomy\b", re.I),
    re.compile(r"\brecommend(?:ed)?\s+lumpectomy\b", re.I),
    re.compile(r"\bplanned\s+lumpectomy\b", re.I),
    re.compile(r"\bscheduled\s+for\s+lumpectomy\b", re.I),
    re.compile(r"\belect(?:ed|ing)\s+to\s+proceed\b.{0,60}\blumpectomy\b", re.I),
    re.compile(r"\bplan(?:ned)?\s+of\s+care\b.{0,100}\blumpectomy\b", re.I),
    re.compile(r"\btreatment\s+options\b.{0,120}\blumpectomy\b", re.I),
]

LUMPECTOMY_CURRENT_EPISODE_PATTERNS = [
    re.compile(r"\bpre[- ]op\b.{0,80}\blumpectomy\b", re.I),
    re.compile(r"\bplanned\s+procedure\b.{0,100}\blumpectomy\b", re.I),
    re.compile(r"\bindications?\s+for\s+procedure\b.{0,120}\blumpectomy\b", re.I),
    re.compile(r"\boperative\s+diagnosis\b.{0,120}\blumpectomy\b", re.I),
    re.compile(r"\bpostoperative\s+diagnosis\b.{0,120}\blumpectomy\b", re.I),
    re.compile(r"\bpathology\b.{0,120}\blumpectomy\b", re.I),
    re.compile(r"\bspecimen\b.{0,120}\blumpectomy\b", re.I),
    re.compile(r"\bresidual\s+invasive\s+carcinoma\b.{0,120}\blumpectomy\b", re.I),
    re.compile(r"\bwire[- ]localized\s+lumpectomy\b", re.I),
    re.compile(r"\bwire\s+locali[sz]ation\b.{0,80}\blumpectomy\b", re.I),
]

LUMPECTOMY_HISTORY_PATTERNS = [
    re.compile(r"\bs/p\b", re.I),
    re.compile(r"\bstatus\s+post\b", re.I),
    re.compile(r"\bhistory\s+of\b", re.I),
    re.compile(r"\bprior\b", re.I),
    re.compile(r"\bprevious\b", re.I),
    re.compile(r"\bpreviously\b", re.I),
    re.compile(r"\btreated\s+with\b", re.I),
    re.compile(r"\bund?erwent\b", re.I),
    re.compile(r"\bpast\s+surgical\s+history\b", re.I),
    re.compile(r"\blumpectomy\s+scar\b", re.I),
    re.compile(r"\blumpectomy\s+site\b", re.I),
    re.compile(r"\bwell[- ]healed\s+lumpectomy\s+scar\b", re.I),
    re.compile(r"\bafter\s+lumpectomy\b", re.I),
    re.compile(r"\bfollowed\s+by\s+radiation\b", re.I),
    re.compile(r"\blumpectomy\/lymph\s+node\s+dissection\b", re.I),
]

LUMPECTOMY_CANCER_CONTEXT = [
    re.compile(r"\bradiation\b", re.I),
    re.compile(r"\bchemo\b", re.I),
    re.compile(r"\bchemotherapy\b", re.I),
    re.compile(r"\bxrt\b", re.I),
    re.compile(r"\bbreast\s+cancer\b", re.I),
    re.compile(r"\bdcis\b", re.I),
    re.compile(r"\bductal\s+carcinoma\b", re.I),
    re.compile(r"\blobular\s+carcinoma\b", re.I),
    re.compile(r"\bsentinel\s+lymph\s+node\b", re.I),
    re.compile(r"\bslnb\b", re.I),
    re.compile(r"\balnd\b", re.I),
]

LUMPECTOMY_SIDE_CONTEXT = [
    re.compile(r"\bleft\b.{0,140}\blumpectomy\b", re.I),
    re.compile(r"\bright\b.{0,140}\blumpectomy\b", re.I),
    re.compile(r"\blumpectomy\b.{0,140}\bleft\b", re.I),
    re.compile(r"\blumpectomy\b.{0,140}\bright\b", re.I),
    re.compile(r"\bleft\b.{0,140}\b(?:breast\s+cancer|dcis|carcinoma|lumpectomy\s+scar|lumpectomy\s+site)\b", re.I),
    re.compile(r"\bright\b.{0,140}\b(?:breast\s+cancer|dcis|carcinoma|lumpectomy\s+scar|lumpectomy\s+site)\b", re.I),
    re.compile(r"\b(?:breast\s+cancer|dcis|carcinoma|lumpectomy\s+scar|lumpectomy\s+site)\b.{0,140}\bleft\b", re.I),
    re.compile(r"\b(?:breast\s+cancer|dcis|carcinoma|lumpectomy\s+scar|lumpectomy\s+site)\b.{0,140}\bright\b", re.I),
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
]


MASTOPEXY_PATTERNS = [
    re.compile(r"\bmastopexy\b", re.I),
    re.compile(r"\bbreast\s+lift\b", re.I),
]


AUGMENT_PATTERNS = [
    re.compile(r"\bbreast\s+augmentation\b", re.I),
    re.compile(r"\baugmentation\s+mammaplasty\b", re.I),
]


OTHER_PATTERNS = [
    re.compile(r"\bexcisional\s+biopsy\b", re.I),
    re.compile(r"\bopen\s+breast\s+biopsy\b", re.I),
]


FIELD_CONFIG = [
    ("PBS_Lumpectomy", LUMP_PATTERNS, 0.95, "lumpectomy"),
    ("PBS_Breast Reduction", REDUCTION_PATTERNS, 0.90, "strict_history"),
    ("PBS_Mastopexy", MASTOPEXY_PATTERNS, 0.90, "strict_history"),
    ("PBS_Augmentation", AUGMENT_PATTERNS, 0.92, "strict_history"),
    ("PBS_Other", OTHER_PATTERNS, 0.85, "strict_history"),
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
    ctx = window_around(text, start, end, 200)
    return _has_any(NEGATE_PATTERNS, ctx)


def _strict_history_near(text, start, end):
    ctx = window_around(text, start, end, 200)
    return _has_any(STRONG_HISTORY_PATTERNS, ctx)


def _lumpectomy_planning_context(text, start, end):
    ctx = window_around(text, start, end, 220)
    return _has_any(LUMPECTOMY_PLANNING_PATTERNS, ctx)


def _lumpectomy_current_episode_context(text, start, end):
    ctx = window_around(text, start, end, 240)
    return _has_any(LUMPECTOMY_CURRENT_EPISODE_PATTERNS, ctx)


def _lumpectomy_history_near(text, start, end):
    ctx = window_around(text, start, end, 320)

    if _has_any(LUMPECTOMY_HISTORY_PATTERNS, ctx):
        return True

    if _has_any(LUMPECTOMY_CANCER_CONTEXT, ctx):
        return True

    if re.search(r"\b(?:19|20)\d{2}\b", ctx, re.I):
        return True

    if _has_any(LUMPECTOMY_SIDE_CONTEXT, ctx):
        return True

    return False


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
                        # Reject treatment-planning and current-episode framing
                        if _lumpectomy_planning_context(text, m.start(), m.end()):
                            continue

                        if _lumpectomy_current_episode_context(text, m.start(), m.end()):
                            continue

                        hist = _lumpectomy_history_near(text, m.start(), m.end())
                        status = "history_possible" if hist else "procedure_mention"
                        cands.append(_emit(field, text, m, section, note, conf, status))

                    elif mode == "strict_history":
                        hist = _strict_history_near(text, m.start(), m.end())
                        if not hist:
                            continue

                        status = "history_possible"
                        cands.append(_emit(field, text, m, section, note, conf, status))

                    else:
                        cands.append(_emit(field, text, m, section, note, conf, "procedure_mention"))

    return cands
