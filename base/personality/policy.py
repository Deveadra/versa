from __future__ import annotations

import random
from typing import Any

POLICIES = [
    {"id": "casual", "max_words": 120},
    {"id": "formal", "max_words": 140},
    {"id": "playful", "max_words": 120},
    {"id": "succinct", "max_words": 60},
]


class PolicyBandit:
    def __init__(self):
        self.values = {p["id"]: 0.0 for p in POLICIES}
        self.counts = {p["id"]: 0 for p in POLICIES}

    def select(self, prior: str | None) -> dict[str, Any]:
        if prior and prior in self.values:
            return next(p for p in POLICIES if p["id"] == prior)
        eps = 0.1
        if random.random() < eps:
            return random.choice(POLICIES)
        return max(POLICIES, key=lambda p: self.values[p["id"]])

    def update(self, policy_id: str, reward: float):
        n = self.counts[policy_id] = self.counts[policy_id] + 1
        v = self.values[policy_id]
        self.values[policy_id] = v + (reward - v) / n
