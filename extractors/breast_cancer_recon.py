#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_cancer_recon.py

Rule-based extractor for:
- Indication_Left
- Indication_Right
- LymphNode

Key updates:
1. ALND trumps SLNB
2. Search across all notes, but prioritize op/procedure notes
3. Tighter side-local windows for indication
4. Explicit contralateral prophylactic handling
5. Negation filtering
6. Python 3.6.8 compatible
"""

import re
from collections import defaultdict

# ---------------------------------------------------
# Regex helpers
# ---------------------------------------------------

RE_WS = re.compile(r"\s+")
RE_SENT_SPLIT = re.compile(r"(?<=[\.\?\!\;])\s+|\n+")

LEFT_TERMS = [
    r"\bleft\b", r"\blt\b", r"\bl\b", r"\bleft[- ]sided\b"
]
RIGHT_TERMS = [
    r"\bright\b", r"\brt\b", r"\br\b", r"\bright[- ]sided\b"
]
BILAT_TERMS = [
    r"\bbilateral\b", r"\bbilat\b", r"\bbi[- ]lateral\b"
]

CANCER_TERMS = [
    r"\bcancer\b",
    r"\bcarcinoma\b",
    r"\bmalignan\w*\b",
    r"\bdcis\b",
    r"\bidc\b",
    r"\bilc\b",
    r"\brecurrent\b",
    r"\binvasive\b",
    r"\bbreast neoplasm\b",
    r"\bhistory of breast cancer\b",
    r"\bknown breast cancer\b"
]

PROPHY_TERMS = [
    r"\bprophylactic\b",
    r"\brisk[- ]reducing\b",
    r"\brisk[- ]reduction\b",
    r"\bpreventive\b",
    r"\bcontralateral prophylactic\b",
    r"\bcpm\b"
]

ALND_TERMS = [
    r"\baxillary lymph node dissection\b",
    r"\balnd\b",
    r"\baxillary dissection\b",
    r"\blevel [i1l][i1l]?(?:/| and )?[i1l]?[i1l]?\s+axillary dissection\b"
]

SLNB_TERMS = [
    r"\bsentinel lymph node biopsy\b",
    r"\bsentinel node biopsy\b",
    r"\bslnb\b",
    r"\bsentinel node excision\b",
    r"\blymphatic mapping\b",
    r"\bsentinel lymphadenectomy\b"
]

NEGATION_TERMS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bdeclined\b",
    r"\bdefer(?:red)?\b",
    r"\bwas not performed\b",
    r"\bwere not performed\b",
    r"\bnot performed\b",
    r"\bnegative for\b"
]

MASTECTOMY_TERMS = [
    r"\bmastectom\w*\b",
    r"\bnipple sparing mastectom\w*\b",
    r"\bskin sparing mastectom\w*\b",
    r"\bsimple mastectom\w*\b",
    r"\bmodified radical mastectom\w*\b"
]

SECTION_PRIORITY = {
    "operative report": 4,
    "operation note": 4,
    "brief op note": 4,
    "procedure": 4,
    "procedures": 4,
    "hospital course": 2,
    "assessment": 2,
    "plan": 2,
    "oncology": 3,
    "clinic": 2,
    "history": 0,
    "past surgical history": 0,
    "pmh": 0,
    "psh": 0
}


def norm_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace(u"\xa0", " ")
    text = RE_WS.sub(" ", text)
    return text.strip()


def lower_text(text):
    return norm_text(text).lower()


def split_sentences(text):
    text = norm_text(text)
    if not text:
        return []
    parts = RE_SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def has_any(patterns, text):
    for pat in patterns:
        if re.search(pat, text, flags=re.I):
            return True
    return False


def find_matches(patterns, text):
    spans = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            spans.append((m.start(), m.end(), m.group(0)))
    spans.sort(key=lambda x: x[0])
    return spans


def is_negated(text, start_idx, end_idx, window=50):
    left = max(0, start_idx - window)
    context = text[left:end_idx]
    for pat in NEGATION_TERMS:
        if re.search(pat, context, flags=re.I):
            return True
    return False


def get_note_priority(note):
    """
    note keys supported:
    - note_type
    - section
    """
    score = 1
    note_type = lower_text(note.get("note_type", ""))
    section = lower_text(note.get("section", ""))

    for k, v in SECTION_PRIORITY.items():
        if k in note_type:
            score = max(score, v)
        if k in section:
            score = max(score, v)
    return score


def sentence_side_labels(sent):
    sent_l = sent.lower()

    left = has_any(LEFT_TERMS, sent_l)
    right = has_any(RIGHT_TERMS, sent_l)
    bilat = has_any(BILAT_TERMS, sent_l)

    return {
        "left": left or bilat,
        "right": right or bilat,
        "bilat": bilat
    }


def get_side_windows(sent, side):
    """
    Returns sentence fragments likely tied to one side.
    Uses punctuation/conjunction splitting for tighter local parsing.
    """
    chunks = re.split(r"[,:;]|\band\b|\bwith\b", sent)
    out = []
    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue
        c_l = c.lower()
        labels = sentence_side_labels(c_l)

        if side == "left" and labels["left"] and not (labels["right"] and not labels["bilat"]):
            out.append(c)
        elif side == "right" and labels["right"] and not (labels["left"] and not labels["bilat"]):
            out.append(c)

    if not out:
        out = [sent]
    return out


def score_indication_fragment(fragment, side):
    """
    Returns:
    - "Therapeutic"
    - "Prophylactic"
    - None
    """
    frag = lower_text(fragment)

    has_mast = has_any(MASTECTOMY_TERMS, frag)
    has_cancer = has_any(CANCER_TERMS, frag)
    has_prophy = has_any(PROPHY_TERMS, frag)

    # Contralateral prophylactic logic:
    # Example:
    # "left breast cancer with contralateral prophylactic mastectomy"
    # if current side has cancer and "contralateral prophylactic" appears,
    # opposite side should be prophylactic (handled outside too).
    #
    # Here, simple local scoring:
    if has_prophy and has_mast:
        return "Prophylactic"

    if has_cancer:
        return "Therapeutic"

    return None


def collect_indication_votes(notes):
    """
    Returns:
        {
          "left": [(label, priority, evidence), ...],
          "right": [(label, priority, evidence), ...]
        }
    """
    votes = {"left": [], "right": []}

    for note in notes:
        text = norm_text(note.get("text", ""))
        if not text:
            continue

        pr = get_note_priority(note)
        sentences = split_sentences(text)

        for sent in sentences:
            sent_l = sent.lower()
            if not has_any(MASTECTOMY_TERMS + CANCER_TERMS + PROPHY_TERMS + LEFT_TERMS + RIGHT_TERMS + BILAT_TERMS, sent_l):
                continue

            # Explicit paired pattern:
            # "left breast cancer, right prophylactic mastectomy"
            left_chunks = get_side_windows(sent, "left")
            right_chunks = get_side_windows(sent, "right")

            for chunk in left_chunks:
                lbl = score_indication_fragment(chunk, "left")
                if lbl:
                    votes["left"].append((lbl, pr, chunk.strip()))

            for chunk in right_chunks:
                lbl = score_indication_fragment(chunk, "right")
                if lbl:
                    votes["right"].append((lbl, pr, chunk.strip()))

            # Contralateral prophylactic handling
            if re.search(r"\bcontralateral prophylactic\b", sent_l, flags=re.I):
                if has_any(LEFT_TERMS, sent_l) and has_any(CANCER_TERMS, sent_l):
                    votes["left"].append(("Therapeutic", pr + 1, sent.strip()))
                    votes["right"].append(("Prophylactic", pr + 1, sent.strip()))
                elif has_any(RIGHT_TERMS, sent_l) and has_any(CANCER_TERMS, sent_l):
                    votes["right"].append(("Therapeutic", pr + 1, sent.strip()))
                    votes["left"].append(("Prophylactic", pr + 1, sent.strip()))

    return votes


def resolve_side_label(votes_for_side):
    """
    Priority:
    1. Higher note priority
    2. Therapeutic beats Prophylactic if both have same priority and evidence is cancer-side specific
       BUT if there is stronger prophylactic side-local evidence, keep prophylactic.
    Practical rule used here:
    - pick highest-priority vote
    - tie-break: Therapeutic > Prophylactic
    - else None
    """
    if not votes_for_side:
        return "None", ""

    sorted_votes = sorted(
        votes_for_side,
        key=lambda x: (x[1], 1 if x[0] == "Therapeutic" else 0),
        reverse=True
    )
    top = sorted_votes[0]
    return top[0], top[2]


def extract_indications(notes):
    votes = collect_indication_votes(notes)

    left_label, left_evid = resolve_side_label(votes["left"])
    right_label, right_evid = resolve_side_label(votes["right"])

    # Safety rule:
    # if one side therapeutic and other missing in same patient, keep other as None
    # if both prophylactic, keep as-is; downstream can flag exclusion
    return {
        "Indication_Left": left_label,
        "Indication_Right": right_label,
        "Indication_Left_Evidence": left_evid,
        "Indication_Right_Evidence": right_evid
    }


def extract_lymphnode(notes):
    """
    Final rule:
    - If ALND anywhere -> ALND
    - Else if SLNB anywhere -> SLNB
    - Else none

    Uses note priority, but ALND still globally trumps.
    Ignores negated mentions.
    """
    alnd_hits = []
    slnb_hits = []

    for note in notes:
        text = norm_text(note.get("text", ""))
        if not text:
            continue

        pr = get_note_priority(note)
        text_l = text.lower()

        for pat in ALND_TERMS:
            for m in re.finditer(pat, text_l, flags=re.I):
                if not is_negated(text_l, m.start(), m.end()):
                    alnd_hits.append((pr, m.group(0), text[max(0, m.start()-60):m.end()+60]))

        for pat in SLNB_TERMS:
            for m in re.finditer(pat, text_l, flags=re.I):
                if not is_negated(text_l, m.start(), m.end()):
                    slnb_hits.append((pr, m.group(0), text[max(0, m.start()-60):m.end()+60]))

    if alnd_hits:
        alnd_hits = sorted(alnd_hits, key=lambda x: x[0], reverse=True)
        return {
            "LymphNode": "ALND",
            "LymphNode_Evidence": alnd_hits[0][2].strip()
        }

    if slnb_hits:
        slnb_hits = sorted(slnb_hits, key=lambda x: x[0], reverse=True)
        return {
            "LymphNode": "SLNB",
            "LymphNode_Evidence": slnb_hits[0][2].strip()
        }

    return {
        "LymphNode": "none",
        "LymphNode_Evidence": ""
    }


def extract_cancer_recon_from_notes(notes):
    """
    Input: list of note dicts. Each note dict should ideally contain:
        {
            "note_id": ...,
            "mrn": ...,
            "note_date": ...,
            "note_type": ...,
            "section": ...,
            "text": ...
        }

    Returns dict of extracted variables.
    """
    out = {}

    ind = extract_indications(notes)
    out.update(ind)

    ln = extract_lymphnode(notes)
    out.update(ln)

    return out


# Backward-compatible wrapper name if your build imports this symbol
def extract(notes):
    return extract_cancer_recon_from_notes(notes)
