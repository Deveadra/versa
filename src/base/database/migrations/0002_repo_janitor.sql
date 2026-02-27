from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

Risk = Literal["low", "medium", "high"]
Scope = Literal["read", "write", "network", "exec", "git", "pr"]

@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    scopes: tuple[Scope, ...]
    risk: Risk
    fn: Callable[..., Any]
    # Optional: JSON-schema-like descriptors for tool calling
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

class CapabilityRegistry:
    def __init__(self) -> None:
        self._caps: dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        if cap.name in self._caps:
            raise ValueError(f"Capability already registered: {cap.name}")
        self._caps[cap.name] = cap

    def get(self, name: str) -> Capability | None:
        return self._caps.get(name)

    def list(self) -> list[Capability]:
        return sorted(self._caps.values(), key=lambda c: c.name)
