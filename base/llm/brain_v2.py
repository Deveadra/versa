
from __future__ import annotations
import openai

from openai import OpenAI
from assistant.config.config import settings

class Brain:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()