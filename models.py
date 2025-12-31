from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class NoteDocument:
    note_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SectionedNote:
    note_id: str
    note_type: str
    sections: Dict[str, str]
    note_date: Optional[str] = None

@dataclass
class Candidate:
    field: str
    value: Any
    status: str
    evidence: str
    section: str
    note_type: str
    note_id: str
    note_date: Optional[str] = None
    confidence: float = 1.0

@dataclass
class FinalField:
    field: str
    value: Any
    status: str
    evidence: str
    section: str
    note_type: str
    note_id: str
    note_date: Optional[str]
    rule: str
