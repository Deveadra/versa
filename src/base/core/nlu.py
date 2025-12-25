# base/core/nlu.py
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_DIAG_TRIGGERS = {
    "diagnostic",
    "diagnostics",
    "diag",
    "scan",
    "health check",
    "self check",
    "self-check",
    "self test",
    "self-test",
    "system check",
    "check yourself",
    "run checks",
    "laggy",
    "slow",
    "sluggish",
    "stuttering",
    "too slow",
    "optimize yourself",
}

_FULL_TRIGGERS = {"full", "all", "entire", "deep", "everything", "whole repo", "all files"}
_QUICK_TRIGGERS = {"quick", "fast", "shallow", "recent", "changed", "diff only", "changed files"}
_FIX_TRIGGERS = {
    "fix",
    "auto-fix",
    "autofix",
    "format",
    "tidy",
    "clean up",
    "cleanup",
    "repair",
    "optimize",
}

_WORD_RE = re.compile(r"[a-z0-9\-]+")


def _normalize(text: str) -> str:
    # strip code blocks to avoid false positives
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`.*?`", "", text)
    return " ".join(text.lower().split())


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _fuzzy_contains(text: str, phrases: set[str], threshold: float = 0.82) -> bool:
    # rough fuzzy check; cheap and good enough for short commands
    t = _normalize(text)
    for p in phrases:
        s = SequenceMatcher(None, t, p).ratio()
        if s >= threshold:
            return True
    # token-level approximation (handles small typos like "diagnositc")
    toks = _tokenize(text)
    for p in phrases:
        for tok in toks:
            s = SequenceMatcher(None, tok, p).ratio()
            if s >= threshold:
                return True
    return False


def _contains_any(text: str, phrases: set[str]) -> bool:
    t = _normalize(text)
    # fast path exact/substring
    for p in phrases:
        if p in t:
            return True
    # fallback fuzzy
    return _fuzzy_contains(t, phrases)


def parse_diagnostic_intent(text: str) -> dict[str, Any] | None:
    """
    Returns:
      { "name": "diagnostic", "mode": "all|changed", "fix": bool, "confidence": float }
    or None if not a diagnostic request.
    """
    if not _contains_any(text, _DIAG_TRIGGERS):
        return None

    mode = "all" if _contains_any(text, _FULL_TRIGGERS) else "changed"
    fix = _contains_any(text, _FIX_TRIGGERS)

    # Confidence heuristic: base + contributions
    conf = 0.55
    if mode == "all":
        conf += 0.15
    if fix:
        conf += 0.15
    # if multiple diag triggers appear (e.g., "diagnostic" + "laggy"), bump a bit
    diag_hits = sum(1 for p in _DIAG_TRIGGERS if p in _normalize(text))
    if diag_hits >= 2:
        conf += 0.1
    conf = min(conf, 0.95)

    return {"name": "diagnostic", "mode": mode, "fix": fix, "confidence": conf}
