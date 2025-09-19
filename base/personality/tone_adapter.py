from __future__ import annotations
from typing import Dict, Any
from .policy import PolicyBandit

class ToneAdapter:
    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile
        self.bandit = PolicyBandit()

    def choose_policy(self) -> Dict[str, Any]:
        prior = self.profile.get("greeting_style")
        return self.bandit.select(prior)

    def reward(self, policy_id: str, sentiment_score: float):
        reward = 0.5 + 0.5 * sentiment_score
        self.bandit.update(policy_id, reward)
