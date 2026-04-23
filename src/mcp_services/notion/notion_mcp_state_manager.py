"""
Notion MCP State Manager for MCPMark
=====================================

Creates the per-task page under eval_hub via a single Notion REST call
(POST /v1/pages with template.template_id), authenticated with the sandbox's
integration key. Replaces the previous duplicate-and-move flow that went
through the official Notion MCP and needed an OAuth access_token plus a
separate move step with up to ~12 retries.
"""

import os
import re
import time
from typing import Any, Dict, Optional, Set

import httpx
from notion_client import Client

from src.base.state_manager import BaseStateManager, InitialStateInfo
from src.base.task_manager import BaseTask
from src.logger import get_logger
from src.mcp_services.notion.notion_task_manager import NotionTask

logger = get_logger(__name__)

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"
ORPHAN_PAGE_PATTERN = re.compile(r".+\s+\(\d+\)$")


class NotionMCPStateManager(BaseStateManager):
    """Manages Notion task state. Creates the per-task page under eval_hub via
    POST /v1/pages with template.template_id (single Notion REST call) and uses
    the Notion API for verification/cleanup.
    """

    def __init__(self):
        super().__init__(service_name="notion")
        self._notion_auth: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------
    # Public helpers called by the evaluator
    # ------------------------------------------------------------------

    def set_sandbox_auth(self, notion_auth: Dict[str, str]) -> None:
        """Store sandbox-provided Notion credentials for the current task."""
        self._notion_auth = notion_auth

    def get_service_config_for_agent(self) -> dict:
        if self._notion_auth and self._notion_auth.get("integration_key_eval"):
            return {"notion_key": self._notion_auth["integration_key_eval"]}
        return {}

    def set_verification_environment(self, messages_path: str = None) -> None:
        super().set_verification_environment(messages_path)
        if self._notion_auth and self._notion_auth.get("integration_key_eval"):
            os.environ["EVAL_NOTION_API_KEY"] = self._notion_auth["integration_key_eval"]

    # ------------------------------------------------------------------
    # BaseStateManager template methods
    # ------------------------------------------------------------------

    def _create_initial_state(self, task: BaseTask) -> Optional[InitialStateInfo]:
        if not isinstance(task, NotionTask):
            logger.error("Task must be NotionTask for Notion state manager")
            return None

        if not self._notion_auth:
            logger.error("No sandbox auth set – call set_sandbox_auth() first")
            return None

        integration_key = self._notion_auth["integration_key"]
        eval_key = self._notion_auth["integration_key_eval"]
        source_page_url = self._notion_auth["source_page_url"]
        eval_page_url = self._notion_auth["eval_page_url"]

        notion = Client(auth=integration_key)
        eval_notion = Client(auth=eval_key)

        source_hub_id = self._extract_page_id(source_page_url)
        eval_hub_id = self._extract_page_id(eval_page_url)

        # Clean up leftover pages in the eval hub (ideally not needed because Klavis Sandbox release will clean up but just in case)
        self._cleanup_eval_hub(eval_notion, eval_hub_id)
        # Clean up orphan "(n)" pages in the source hub
        self._cleanup_source_hub_orphans(notion, source_hub_id)

        try:
            title = self._category_to_title(task.category_id)
            child_page_id = self._find_child_by_title(notion, source_hub_id, title)
            if not child_page_id:
                logger.error(
                    "| ✗ Source template '%s' not found under source hub", title
                )
                return None

            # Create a new page under eval_hub from the template — single Notion
            # REST call. Retry up to 5 times on transient 5xx / rate-limit errors.
            max_attempts = 5
            new_page_id: Optional[str] = None
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    new_page_id = self._create_page_from_template(
                        integration_key, child_page_id, eval_hub_id
                    )
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < max_attempts:
                        wait = min(2 ** attempt, 60)
                        logger.warning(
                            "| ✗ Create-from-template attempt %d/%d failed "
                            "(%s: %s). Retrying in %ds …",
                            attempt, max_attempts,
                            type(e).__name__, str(e)[:200], wait,
                        )
                        time.sleep(wait)
            if new_page_id is None:
                raise RuntimeError(
                    f"Create-from-template failed after {max_attempts} attempts: "
                    f"{last_exc}"
                )

            # Notion applies the template asynchronously after the response
            # returns — poll until the page is API-accessible.
            if not self._wait_for_page_ready(eval_notion, new_page_id):
                logger.error(
                    "| ✗ New page %s not accessible after create", new_page_id
                )
                try:
                    eval_notion.pages.update(page_id=new_page_id, archived=True)
                except Exception:
                    pass
                return None

            new_page_url = f"https://www.notion.so/{new_page_id.replace('-', '')}"
            return InitialStateInfo(
                state_id=new_page_id,
                state_url=new_page_url,
                metadata={
                    "category": task.category_id,
                    "task_name": task.name,
                },
            )
        except Exception as e:
            logger.error("| ✗ Failed to create initial state for %s: %s", task.name, e)
            return None

    def _store_initial_state_info(
        self, task: BaseTask, state_info: InitialStateInfo
    ) -> None:
        if isinstance(task, NotionTask):
            task.duplicated_initial_state_id = state_info.state_id
            task.duplicated_initial_state_url = state_info.state_url
            self.track_resource("page", state_info.state_id, state_info.metadata)

    def _cleanup_task_initial_state(self, task: BaseTask) -> bool:
        if not isinstance(task, NotionTask):
            return True
        page_id = task.duplicated_initial_state_id
        if not page_id:
            logger.warning("| ✗ No duplicated page ID for task %s", task.name)
            return False
        try:
            eval_key = (self._notion_auth or {}).get("integration_key_eval")
            if not eval_key:
                logger.warning("| ✗ No eval key available for cleanup")
                return False
            Client(auth=eval_key).pages.update(page_id=page_id, archived=True)
            logger.info("| ✓ Archived duplicated page: %s", page_id)
            self.tracked_resources = [
                r for r in self.tracked_resources
                if not (r["type"] == "page" and r["id"] == page_id)
            ]
            return True
        except Exception as e:
            logger.error("| ✗ Failed to archive page %s: %s", page_id, e)
            return False

    def _cleanup_single_resource(self, resource: Dict[str, Any]) -> bool:
        if resource["type"] == "page":
            try:
                eval_key = (self._notion_auth or {}).get("integration_key_eval")
                if not eval_key:
                    return False
                Client(auth=eval_key).pages.update(
                    page_id=resource["id"], archived=True
                )
                logger.info("| ✓ Archived Notion page: %s", resource["id"])
                return True
            except Exception as e:
                logger.error(
                    "| ✗ Failed to archive Notion page %s: %s", resource["id"], e
                )
                return False
        logger.warning("| ? Unknown resource type: %s", resource["type"])
        return False

    # ------------------------------------------------------------------
    # Create page from template (Notion REST)
    # ------------------------------------------------------------------

    @staticmethod
    def _create_page_from_template(
        integration_key: str, template_id: str, eval_hub_id: str
    ) -> str:
        """Create a new page under eval_hub_id using template_id as the template.

        Single POST /v1/pages authenticated with the sandbox's integration key.
        Notion applies the template asynchronously after the response returns,
        so the caller must still poll `_wait_for_page_ready` on the returned id.
        """
        headers = {
            "Authorization": f"Bearer {integration_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }
        body = {
            "parent": {"type": "page_id", "page_id": eval_hub_id},
            "template": {"type": "template_id", "template_id": template_id},
        }
        t0 = time.time()
        logger.info(
            "| ○ Creating page from template %s → parent=%s …",
            template_id, eval_hub_id,
        )
        resp = httpx.post(
            f"{NOTION_API_BASE}/pages",
            headers=headers,
            json=body,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"POST /pages HTTP {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        new_page_id = data.get("id")
        if not new_page_id:
            raise RuntimeError(f"POST /pages returned no id: body={resp.text[:300]}")
        logger.info(
            "| ✓ Created page %s (time=%.2fs)", new_page_id, time.time() - t0
        )
        return new_page_id

    # ------------------------------------------------------------------
    # Notion API helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_page_id(url: str) -> str:
        slug = url.split("?")[0].split("#")[0].rstrip("/").split("/")[-1]
        compact = "".join(c for c in slug if c.isalnum())
        if len(compact) < 32:
            raise ValueError(f"Could not parse page ID from URL: {url}")
        compact = compact[-32:]
        return f"{compact[:8]}-{compact[8:12]}-{compact[12:16]}-{compact[16:20]}-{compact[20:]}"

    @staticmethod
    def _category_to_title(category: str) -> str:
        return " ".join(word.capitalize() for word in category.split("_"))

    @staticmethod
    def _find_child_by_title(
        notion: Client, parent_id: str, title: str
    ) -> Optional[str]:
        next_cursor = None
        while True:
            kwargs: Dict[str, Any] = {"block_id": parent_id}
            if next_cursor:
                kwargs["start_cursor"] = next_cursor
            children = notion.blocks.children.list(**kwargs)
            for child in children.get("results", []):
                if child.get("type") != "child_page":
                    continue
                child_title = (
                    (child.get("child_page", {}) or {}).get("title", "").strip()
                )
                if child_title == title:
                    return child.get("id")
            if not children.get("has_more"):
                break
            next_cursor = children.get("next_cursor")
        return None

    @staticmethod
    def _wait_for_page_ready(
        notion: Client, page_id: str, timeout: int = 120, poll_interval: int = 2
    ) -> bool:
        logger.info("| ○ Waiting for page %s to be accessible …", page_id)
        elapsed = 0
        attempt = 0
        while elapsed < timeout:
            attempt += 1
            try:
                result = notion.pages.retrieve(page_id=page_id)
                if result and isinstance(result, dict) and "properties" in result:
                    logger.info(
                        "| ✓ Page ready (attempt %d, %ds elapsed)", attempt, elapsed
                    )
                    return True
            except Exception as e:
                logger.debug(
                    "| ✗ Not ready (attempt %d, %ds elapsed): %s", attempt, elapsed, e
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        logger.error("| ✗ Page not ready after %ds", timeout)
        return False

    @staticmethod
    def _cleanup_eval_hub(notion: Client, eval_hub_id: str) -> None:
        try:
            children = notion.blocks.children.list(block_id=eval_hub_id)
            count = 0
            for child in children.get("results", []):
                if child.get("type") == "child_page":
                    try:
                        notion.pages.update(page_id=child["id"], archived=True)
                        count += 1
                    except Exception:
                        pass
            if count:
                logger.info("| ✓ Cleaned up %d page(s) from eval hub", count)
        except Exception as e:
            logger.warning("Eval hub cleanup failed (non-critical): %s", e)

    @staticmethod
    def _cleanup_source_hub_orphans(
        notion: Client,
        source_hub_id: str,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        exclude = exclude or set()
        count = 0
        next_cursor = None
        try:
            while True:
                kwargs: Dict[str, Any] = {"block_id": source_hub_id}
                if next_cursor:
                    kwargs["start_cursor"] = next_cursor
                children = notion.blocks.children.list(**kwargs)
                for child in children.get("results", []):
                    if child.get("type") != "child_page":
                        continue
                    cid = child.get("id")
                    if cid in exclude:
                        continue
                    title = (
                        (child.get("child_page", {}) or {}).get("title", "").strip()
                    )
                    if ORPHAN_PAGE_PATTERN.match(title):
                        try:
                            notion.pages.update(page_id=cid, archived=True)
                            count += 1
                        except Exception:
                            pass
                if not children.get("has_more"):
                    break
                next_cursor = children.get("next_cursor")
            if count:
                logger.info("| ✓ Cleaned up %d orphan(s) from source hub", count)
        except Exception as e:
            logger.warning("Source hub orphan cleanup failed (non-critical): %s", e)
        return count
