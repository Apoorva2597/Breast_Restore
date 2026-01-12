from typing import List

from models import SectionedNote, Candidate

from .bmi import extract_bmi
from .smoking import extract_smoking
from .comorbidities import extract_comorbidities
from .procedures import extract_reconstruction, extract_lymph_node_mgmt
from .pbs import extract_pbs
from .mastectomy import extract_mastectomy


def extract_all(sec: SectionedNote) -> List[Candidate]:
    """
    Run all extractors and return a flat list of candidates.

    Tier 1:
      - BMI
      - SmokingStatus
      - Comorbidities (DM, HTN, Cardiac, VTE, Steroid, CancerOther)
      - Reconstruction (type, laterality, timing, performed/planned)
      - Lymph node management

    Tier 2:
      - PBS_Lumpectomy / PBS_Other
      - Mastectomy_Laterality / Mastectomy_Type / Mastectomy_Performed
    """
    cands = []  # type: List[Candidate]

    # Tier 1
    cands.extend(extract_bmi(sec))
    cands.extend(extract_smoking(sec))
    cands.extend(extract_comorbidities(sec))
    cands.extend(extract_reconstruction(sec))
    cands.extend(extract_lymph_node_mgmt(sec))

    # Tier 2
    cands.extend(extract_pbs(sec))
    cands.extend(extract_mastectomy(sec))

    return cands
