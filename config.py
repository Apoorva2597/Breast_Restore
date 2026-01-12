# config.py
# Pattern lists and normalisation dictionaries used by the extractors.
# Python 3.6–compatible: no type annotations, no fancy syntax.

# ------------------------------------------------------------
# Status cue patterns (used by classify_status in utils.py)
# ------------------------------------------------------------

NEGATION_CUES = [
    r"\bno\b",
    r"\bdenies?\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bnone\b",
]

PLANNED_CUES = [
    r"\bplanned\b",
    r"\bscheduled\b",
    r"\bto undergo\b",
    r"\bwill undergo\b",
    r"\bplanned for\b",
    r"\bconsidering\b",
]

PERFORMED_CUES = [
    r"\bstatus\s+post\b",
    r"\bs/p\b",
    r"\bhistory of\b",
    r"\bhas\b",
    r"\bhad\b",
    r"\bwas done\b",
    r"\bwas performed\b",
]

# ------------------------------------------------------------
# Smoking normalisation
# ------------------------------------------------------------

SMOKING_NORMALIZE = {
    "never smoker": "never",
    "nonsmoker": "never",
    "non smoker": "never",
    "no tobacco use": "never",
    "denies tobacco": "never",

    "former smoker": "former",
    "quit smoking": "former",
    "stopped smoking": "former",
    "ex-smoker": "former",

    "current every day smoker": "current",
    "current some day smoker": "current",
    "current smoker": "current",
    "smokes daily": "current",
    "tobacco use: yes": "current",
}

# ------------------------------------------------------------
# Diabetes, Hypertension, Cardiac disease
# Each *_POS is a list of regex strings that indicate presence.
# Each *_EXCLUDE is a list of regex strings that should suppress a hit.
# ------------------------------------------------------------

DM_POS = [
    r"\bdiabetes\b",
    r"\bdm\b",
    r"\btype\s*1\s+diabetes\b",
    r"\btype\s*2\s+diabetes\b",
]

DM_EXCLUDE = [
    r"\bgestational\b",
    r"\bpre[- ]diabetes\b",
]

HTN_POS = [
    r"\bhypertension\b",
    r"\bhtn\b",
    r"\bhigh blood pressure\b",
]

HTN_EXCLUDE = [
    r"\bno history of hypertension\b",
]

CARDIAC_POS = [
    r"\bcoronary artery disease\b",
    r"\bcad\b",
    r"\bmyocardial infarction\b",
    r"\bmi\b",
    r"\bcongestive heart failure\b",
    r"\bchf\b",
    r"\bischemic heart disease\b",
]

# Things that look cardiac but we want to ignore (template noise, etc.)
CARDIAC_EXCLUDE = [
    r"\brisk of\b",
    r"\brisk factors?\b",
    r"\bcardiac clearance\b",
]

# ------------------------------------------------------------
# Venous thromboembolism (VTE)
# ------------------------------------------------------------

VTE_POS = [
    r"\bdeep venous thrombosis\b",
    r"\bdeep vein thrombosis\b",
    r"\bdvt\b",
    r"\bpulmonary embol(ism)?\b",
    r"\bpe\b",
    r"\bvenous thromboembol(ism)?\b",
]

# ------------------------------------------------------------
# Chronic steroid use
# ------------------------------------------------------------

STEROID_POS = [
    r"\bchronic steroid\b",
    r"\bprednisone\b",
    r"\bprednisolone\b",
    r"\bdexamethasone\b",
    r"\bmedrol\b",
]

STEROID_EXCLUDE = [
    r"\ballergy\b",
    r"\ballergies\b",
    r"\ballergic\b",
]

# ------------------------------------------------------------
# Other cancer history (non-index breast cancer)
# ------------------------------------------------------------

CANCER_OTHER_POS = [
    r"\bhistory of (.+) cancer\b",
    r"\bprior (.+) malignancy\b",
    r"\bmelanoma\b",
    r"\blymphoma\b",
    r"\bleukemia\b",
]
