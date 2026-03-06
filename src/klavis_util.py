import os
from typing import Dict, Optional
import logging
import httpx


logger = logging.getLogger(__name__)

KLAVIS_API_BASE = "https://api.klavis.ai"

class KlavisSandbox:
    """Klavis MCP Sandbox API client."""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("KLAVIS_API_KEY")
        if not self.api_key:
            raise ValueError("KLAVIS_API_KEY is required")
        self.acquired_sandbox = None

    def acquire(self, server_name: str, extra_params: Optional[Dict] = None) -> Optional[Dict]:
        """Acquire an individual sandbox for a non-local-sandbox server."""
        url = f"{KLAVIS_API_BASE}/sandbox/{server_name}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {"benchmark": "MCP_Mark", "ttl_seconds": 7200}
        if extra_params:
            body.update(extra_params)
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            self.acquired_sandbox = data
            return data
        except Exception as e:
            logger.error(f"Failed to acquire sandbox for '{server_name}': {e}")
            return None

    def get_sandbox_info(self) -> Optional[Dict]:
        """Get info for a specific sandbox."""
        if not self.acquired_sandbox:
            logger.warning("No sandbox acquired yet.")
            return None
        server_name = self.acquired_sandbox.get("server_name")
        sandbox_id = self.acquired_sandbox.get("sandbox_id")
        url = f"{KLAVIS_API_BASE}/sandbox/{server_name}/{sandbox_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = httpx.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get sandbox info for '{server_name}/{sandbox_id}': {e}")
            return None

    def release(self) -> Optional[Dict]:
        """Release a sandbox back to idle state."""
        if not self.acquired_sandbox:
            logger.warning("No sandbox acquired yet.")
            return None
        server_name = self.acquired_sandbox.get("server_name")
        sandbox_id = self.acquired_sandbox.get("sandbox_id")
        url = f"{KLAVIS_API_BASE}/sandbox/{server_name}/{sandbox_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = httpx.delete(url, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to release sandbox '{server_name}/{sandbox_id}': {e}")
            return None
