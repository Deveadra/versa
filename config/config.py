from __future__ import annotations
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class Settings(BaseModel):
    db_path: str = Field(default=os.getenv("ULTRON_DB_PATH", "./ultron.db"))
    memory_ttl_days: int = int(os.getenv("ULTRON_MEMORY_TTL_DAYS", 30))
    importance_threshold: int = int(os.getenv("ULTRON_IMPORTANCE_THRESHOLD", 25))


    # LLM
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")


    # Embeddings
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "sentence_transformers")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")


    # ElevenLabs
    eleven_api_key: str | None = os.getenv("ELEVENLABS_API_KEY")
    eleven_voice_id: str | None = os.getenv("ELEVENLABS_VOICE_ID")


    # Home Assistant
    ha_base_url: str | None = os.getenv("HA_BASE_URL")
    ha_token: str | None = os.getenv("HA_TOKEN")


    # Consolidation cron
    consolidation_hour: int = int(os.getenv("ULTRON_CONSOLIDATION_HOUR", 3))
    consolidation_minute: int = int(os.getenv("ULTRON_CONSOLIDATION_MINUTE", 0))

class Settings(BaseModel):
    mode: str = os.getenv("ULTRON_MODE", "text")

settings = Settings()