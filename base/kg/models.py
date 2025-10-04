from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Entity:
    id: int | None
    name: str
    type: str  # e.g., PERSON, ORG, LOC


@dataclass
class Relation:
    id: int | None
    source_id: int
    target_id: int
    relation: str
    confidence: float = 1.0
