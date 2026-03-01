# =========================
# TIGHTENED EXCHANGE_TIGHT
# =========================

import re

# -------------------------
# 1. INTRAOPERATIVE SIGNALS
# -------------------------
INTRAOP_SIGNALS = re.compile(
    r"""
    \b(
        estimated\ blood\ loss|ebl|
        urine\ output|
        drains?\ (were\ )?(placed|inserted)|
        specimens?\ removed|
        anesthesia|
        general\ anesthesia|
        operating\ room|
        intraop(erative)?|
        incision\ was\ made|
        procedure:\ |
        pre[- ]?op|post[- ]?op
    )\b
    """,
    re.I | re.X,
)

# -------------------------
# 2. ACTION + OBJECT PAIRS
# -------------------------
EXCHANGE_ACTION_OBJECT = re.compile(
    r"""
    \b(
        (remove(d)?|explant(ed)?|exchange(d)?|replace(d)?|revision\ of)
        [^\.]{0,80}?
        (implant|expander|prosthesis)
    )\b
    """,
    re.I | re.X,
)

# -------------------------
# 3. NEGATION / PLANNING FILTER
# -------------------------
PLANNING_OR_HYPOTHETICAL = re.compile(
    r"""
    \b(
        possible|
        discuss(ed|ion)?|
        candidate\ for|
        will\ plan|
        plan(ned)?\ to|
        considering|
        may\ need|
        would\ like|
        risk\ of|
        if\ needed
    )\b
    """,
    re.I | re.X,
)

# -------------------------
# 4. HISTORICAL FILTER
# -------------------------
HISTORICAL_CONTEXT = re.compile(
    r"""
    \b(
        status\ post|
        s/p|
        history\ of|
        previously|
        prior\ to|
        in\ \d{4}
    )\b
    """,
    re.I | re.X,
)

# -------------------------
# 5. NOTE TYPE HARD FILTER
# -------------------------
VALID_NOTE_TYPES = {
    "OP NOTE",
    "BRIEF OP NOTES",
}

# =========================
# UPDATED EXCHANGE_TIGHT
# =========================

def detect_exchange_tight(note_text: str, note_type: str) -> bool:
    """
    Stricter EXCHANGE_TIGHT:
    - Must be operative note type
    - Must contain intraoperative signals
    - Must contain action+object pair
    - Must NOT contain planning/hypothetical language
    - Must NOT be historical-only mention
    """

    if note_type not in VALID_NOTE_TYPES:
        return False

    if not INTRAOP_SIGNALS.search(note_text):
        return False

    if not EXCHANGE_ACTION_OBJECT.search(note_text):
        return False

    if PLANNING_OR_HYPOTHETICAL.search(note_text):
        return False

    if HISTORICAL_CONTEXT.search(note_text):
        return False

    return True
