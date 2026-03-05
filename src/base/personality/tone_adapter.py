# src/base/personality/tone_adapter.py (augment)
from __future__ import annotations

from typing import Any

from .policy import PolicyBandit


class ToneAdapter:
    def __init__(self, profile: Any | None = None):
        # Accept None / non-dict profiles gracefully (helps tests and bootstrapping)
        if isinstance(profile, dict):
            self.profile = profile
        else:
            self.profile = {}
        self.bandit = PolicyBandit()

    # def __init__(self, profile: Any | None = None):
    #     # Accept None / non-dict profiles gracefully (helps tests and bootstrapping)
    #     self.profile = profile if isinstance(profile, dict) else {}
    #     self.bandit = PolicyBandit()

    # def __init__(self, profile: dict[str, Any]):
    #     self.profile = profile
    #     self.bandit = PolicyBandit()

    @staticmethod
    def adapt(polarity: str) -> str:
        if polarity == "positive":
            return "Be upbeat and encouraging."
        if polarity == "negative":
            return "Be patient, warm, and reassuring."
        return "Use your usual adaptive style."

    def choose_policy(self) -> dict[str, Any]:
        prior = self.profile.get("greeting_style")
        return self.bandit.select(prior)

    def reward(self, policy_id: str, sentiment_score: float):
        reward = 0.5 + 0.5 * sentiment_score
        self.bandit.update(policy_id, reward)
