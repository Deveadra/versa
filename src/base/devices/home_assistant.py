from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx

from config.config import settings


class HomeAssistantError(Exception):
    """Custom exception for Home Assistant API errors."""


class HomeAssistant:
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: int = 10):
        if not (settings.ha_base_url and settings.ha_token) and not (base_url and token):
            raise RuntimeError("Home Assistant not configured: missing URL or token")

        self.base_url = cast(str, (base_url or settings.ha_base_url or "")).rstrip("/")
        self.token = cast(str, (token or settings.ha_token or ""))
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, headers=self.headers, timeout=self.timeout
            )
        return self._client

    async def validate_connection(self) -> bool:
        """Check if HA API is reachable."""
        client = await self._get_client()
        try:
            r = await client.get("/api/")
            r.raise_for_status()
            return "API" in r.text
        except Exception as e:
            raise HomeAssistantError(f"Failed to connect to Home Assistant: {e}")

    async def call_service(
        self, domain: str, service: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Call a Home Assistant service.
        Example: await call_service("light", "turn_on", {"entity_id": "light.living_room"})
        """
        client = await self._get_client()
        try:
            r = await client.post(f"/api/services/{domain}/{service}", json=data or {})
            r.raise_for_status()
            return r.json() if r.content else {}
        except Exception as e:
            raise HomeAssistantError(f"Service call {domain}.{service} failed: {e}")

    async def get_state(self, entity_id: str) -> dict[str, Any]:
        """Fetch the current state of an entity."""
        client = await self._get_client()
        try:
            r = await client.get(f"/api/states/{entity_id}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise HomeAssistantError(f"Failed to get state for {entity_id}: {e}")

    async def set_state(
        self, entity_id: str, state: str, attributes: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Manually override entity state (not common, but supported)."""
        client = await self._get_client()
        try:
            payload = {"state": state, "attributes": attributes or {}}
            r = await client.post(f"/api/states/{entity_id}", json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise HomeAssistantError(f"Failed to set state for {entity_id}: {e}")

    async def close(self):
        """Cleanly close the HTTP session."""
        if self._client:
            await self._client.aclose()
            self._client = None


# --- Example usage ---
async def _demo():
    ha = HomeAssistant()
    ok = await ha.validate_connection()
    print("HA Connected:", ok)

    state = await ha.get_state("light.living_room")
    print("Light state:", state["state"])

    await ha.call_service("light", "turn_on", {"entity_id": "light.living_room"})


# Run demo manually if needed
if __name__ == "__main__":
    asyncio.run(_demo())
