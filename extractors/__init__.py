# extractors/__init__.py
from typing import List

from models import SectionedNote, Candidate

from .bmi import extract_bmi
from .smoking import extract_smoking
from .comorbidities import extract_comorbidities
from .procedures import extract_reconstruction, extract_lymph_node_mgmt
from .pbs import extract_pbs
from .mastectomy import extract_mastectomy
from .age import extract_age
from .cancer_treatment import extract_cancer_treatment
from .complications import extract_stage1_outcomes



def extract_all(sec: SectionedNote) -> List[Candidate]:
    """
    Run all extractors and return a flat list of candidates.
    """
    cands = []  # type: List[Candidate]

    # Tier 1
    cands.extend(extract_bmi(sec))
    cands.extend(extract_smoking(sec))
    cands.extend(extract_comorbidities(sec))
    cands.extend(extract_reconstruction(sec))
    cands.extend(extract_lymph_node_mgmt(sec))

    # Tier 1.5 – shared demographic variable
    cands.extend(extract_age(sec))

    # Tier 2
    cands.extend(extract_pbs(sec))
    cands.extend(extract_mastectomy(sec))
    cands.extend(extract_cancer_treatment(sec))
    cands.extend(extract_stage1_outcomes(sec))

    return cands
