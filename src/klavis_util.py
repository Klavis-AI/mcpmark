import io
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import logging
import httpx


logger = logging.getLogger(__name__)

KLAVIS_API_BASE = "https://api.klavis.ai"


class KlavisSandbox:
    """Klavis MCP Sandbox API client for individual (non-local) sandboxes."""
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

    def get_notion_auth(self) -> Optional[Dict]:
        """Extract Notion-specific auth credentials from sandbox details.

        Returns a dict with integration keys, hub page URLs, and the OAuth
        access token for the official Notion MCP server, or None on failure.
        """
        details = self.get_sandbox_info()
        if not details:
            return None
        metadata = details.get("metadata") or {}
        mcp_auth = metadata.get("mcp_auth_data") or {}
        return {
            "integration_key": metadata.get("mcpmark_notion_integration_key"),
            "integration_key_eval": metadata.get("mcpmark_notion_integration_key_eval"),
            "source_page_url": metadata.get("mcpmark_source_notion_page_url"),
            "eval_page_url": metadata.get("mcpmark_eval_notion_page_url"),
            "official_mcp_token": (mcp_auth.get("token") or {}).get("access_token"),
        }


class KlavisLocalSandbox:
    """Klavis Local Sandbox API client.

    A Local Sandbox is a specialized VM that hosts multiple interconnected
    MCP servers simultaneously (e.g. filesystem + git + terminal).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("KLAVIS_API_KEY")
        if not self.api_key:
            raise ValueError("KLAVIS_API_KEY is required")
        self.sandbox_id: Optional[str] = None
        self.acquired_sandbox: Optional[Dict] = None

    # -----------------------------------------------------------------
    # Acquire / Release
    # -----------------------------------------------------------------

    def acquire(self, server_names: List[str]) -> Optional[Dict]:
        """Acquire a local sandbox VM with the requested MCP servers.

        Args:
            server_names: List of MCP server names to spin up
                          (e.g. ``["filesystem", "git", "terminal"]``).

        Returns:
            The response JSON dict on success, or ``None`` on failure.
        """
        url = f"{KLAVIS_API_BASE}/local-sandbox"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {"server_names": server_names}
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            self.sandbox_id = data.get("local_sandbox_id")
            # Build a convenient server_urls mapping from the servers array
            # so callers can do acquired_sandbox["server_urls"]["filesystem"]
            data["server_urls"] = {
                s["server_name"]: s["mcp_server_url"]
                for s in data.get("servers", [])
                if "server_name" in s and "mcp_server_url" in s
            }
            self.acquired_sandbox = data
            logger.info(f"Acquired local sandbox: {self.sandbox_id}")
            return data
        except Exception as e:
            logger.error(f"Failed to acquire local sandbox: {e}")
            return None

    def release(self) -> Optional[Dict]:
        """Release the local sandbox VM and clean up."""
        if not self.sandbox_id:
            logger.warning("No local sandbox acquired yet.")
            return None
        url = f"{KLAVIS_API_BASE}/local-sandbox/{self.sandbox_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = httpx.delete(url, headers=headers, timeout=60)
            resp.raise_for_status()
            logger.info(f"Released local sandbox: {self.sandbox_id}")
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to release local sandbox '{self.sandbox_id}': {e}")
            return None

    # -----------------------------------------------------------------
    # Upload / Initialize
    # -----------------------------------------------------------------

    def upload_directory(self, directory: Path) -> bool:
        """Pack *directory* into a ``tar.gz`` archive and upload it into the sandbox.

        Three-step process:
        1. Obtain a signed upload URL.
        2. PUT the archive to that URL.
        3. POST ``/initialize`` to extract the archive inside the sandbox.

        Args:
            directory: Local directory whose contents should be uploaded.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if not self.sandbox_id:
            logger.error("No local sandbox acquired yet.")
            return False

        headers = {"Authorization": f"Bearer {self.api_key}"}

        # --- Step 1: get signed upload URL ---
        try:
            resp = httpx.post(
                f"{KLAVIS_API_BASE}/local-sandbox/{self.sandbox_id}/upload-url",
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            upload_url = resp.json()["upload_url"]
            logger.info("Obtained signed upload URL for local sandbox")
        except Exception as e:
            logger.error(f"Failed to get upload URL: {e}")
            return False

        # --- Step 2: create tar.gz and upload ---
        try:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(str(directory), arcname=".")
            buf.seek(0)
            archive_bytes = buf.getvalue()
            logger.info(
                f"Created tar.gz archive: {len(archive_bytes)} bytes from {directory}"
            )

            resp = httpx.put(
                upload_url,
                headers={"Content-Type": "application/gzip"},
                content=archive_bytes,
                timeout=300,
            )
            resp.raise_for_status()
            logger.info("Uploaded archive to local sandbox")
        except Exception as e:
            logger.error(f"Failed to upload archive: {e}")
            return False

        # --- Step 3: initialize (extract archive in sandbox) ---
        try:
            resp = httpx.post(
                f"{KLAVIS_API_BASE}/local-sandbox/{self.sandbox_id}/initialize",
                headers=headers,
                timeout=300,
            )
            resp.raise_for_status()
            logger.info("Initialized local sandbox with uploaded data")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize sandbox: {e}")
            return False

    # -----------------------------------------------------------------
    # Dump (export sandbox filesystem state)
    # -----------------------------------------------------------------

    def dump_to_directory(self, destination: Path) -> bool:
        """Download the sandbox filesystem state and extract it to *destination*.

        Two-step process:
        1. GET ``/dump`` to obtain a signed download URL.
        2. Download and extract the ``tar.gz`` archive.

        Args:
            destination: Local directory to extract the sandbox state into.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if not self.sandbox_id:
            logger.error("No local sandbox acquired yet.")
            return False

        headers = {"Authorization": f"Bearer {self.api_key}"}

        # --- Step 1: get signed download URL ---
        try:
            resp = httpx.get(
                f"{KLAVIS_API_BASE}/local-sandbox/{self.sandbox_id}/dump",
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            download_url = resp.json()["download_url"]
            logger.info("Obtained signed download URL for sandbox dump")
        except Exception as e:
            logger.error(f"Failed to get dump URL: {e}")
            return False

        # --- Step 2: download and extract ---
        try:
            resp = httpx.get(download_url, timeout=300)
            resp.raise_for_status()

            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
                tar.extractall(path=str(destination))
            logger.info(f"Dumped sandbox state to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to dump sandbox state: {e}")
            return False

    # -----------------------------------------------------------------
    # Info helpers
    # -----------------------------------------------------------------

    def get_sandbox_info(self) -> Optional[Dict]:
        """Return the acquired sandbox metadata (server URLs, etc.)."""
        return self.acquired_sandbox
