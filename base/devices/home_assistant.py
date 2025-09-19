
from __future__ import annotations
import httpx
import os
import requests

from assistant.config.config import settings

HA_URL = os.getenv("HA_URL")
HA_TOKEN = os.getenv("HA_TOKEN")

headers = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json"
}


class HomeAssistant:
    def __init__(self, base="http://localhost:8123", token=""):
        self.base = base.rstrip("/")
        self.h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if not settings.ha_base_url or not settings.ha_token:
            raise RuntimeError("Home Assistant not configured")
        self.base = settings.ha_base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {settings.ha_token}", "Content-Type": "application/json"}

        
    def health(self):
        r = requests.get(f"{self.base}/api/", headers=self.h, timeout=10)
        r.raise_for_status()
        return r.text  # "API running."

    # def get_state(self, entity_id):
    #     r = requests.get(f"{self.base}/api/states/{entity_id}", headers=self.h, timeout=10)
    #     r.raise_for_status()
    #     return r.json()

    def call_service(self, domain, service, data):
        r = requests.post(f"{self.base}/api/services/{domain}/{service}",
                          headers=self.h, json=data, timeout=10)
        r.raise_for_status()
        return r.json() if r.content else True

    def toggle(self, entity_id):
        return self.call_service("homeassistant", "toggle", {"entity_id": entity_id})
    
  
    # # def call_service(domain, service, data=None):
    # async def call_service(self, domain: str, service: str, data: dict) -> dict:
    #     """
    #     Generic HA service call.
    #     Example: call_service("media_player", "turn_on", {"entity_id": "media_player.living_room_tv"})
    #     """
    #     # url = f"{self.base}/api/services/{domain}/{service}"
    #     url = f"{HA_URL}/services/{domain}/{service}"
    #     async with httpx.AsyncClient(timeout=10) as client:
    #         r = await client.post(url, headers=self.headers, json=data)
    #         # r = requests.post(url, headers=headers, json=data or {})
    #         r.raise_for_status()
            
    #         # if r.status_code not in [200, 201]:
    #         #     return False, f"Error calling {domain}.{service}: {r.text}"
    #         # return True, "OK"
        
    #         return r.json()
        
            
    def get_state(entity_id):
        url = f"{HA_URL}/states/{entity_id}"
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return r.json()["state"]
        return None
