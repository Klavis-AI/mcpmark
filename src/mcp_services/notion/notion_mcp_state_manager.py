"""
Notion MCP State Manager for MCPMark
=====================================

Replaces Playwright-based page duplication with the official Notion MCP server
(https://mcp.notion.com/mcp).  All credentials are sourced from the Klavis
sandbox metadata instead of local environment variables.
"""

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, Optional, Set

from notion_client import Client

from src.base.state_manager import BaseStateManager, InitialStateInfo
from src.base.task_manager import BaseTask
from src.agents.mcp.http_server import MCPHttpServer
from src.logger import get_logger
from src.mcp_services.notion.notion_task_manager import NotionTask

logger = get_logger(__name__)

NOTION_OFFICIAL_MCP_URL = "https://mcp.notion.com/mcp"
ORPHAN_PAGE_PATTERN = re.compile(r".+\s+\(\d+\)$")


class NotionMCPStateManager(BaseStateManager):
    """Manages Notion task state using the official Notion MCP server for
    page duplication and the Notion API for verification/cleanup.
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
        official_token = self._notion_auth["official_mcp_token"]

        notion = Client(auth=integration_key)
        eval_notion = Client(auth=eval_key)

        source_hub_id = self._extract_page_id(source_page_url)
        eval_hub_id = self._extract_page_id(eval_page_url)

        # Clean up leftover pages in the eval hub
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

            duplicated_id = asyncio.run(
                self._duplicate_and_move(
                    official_token, child_page_id, eval_hub_id, title, notion
                )
            )

            # Wait for the page to be accessible via the eval integration
            if not self._wait_for_page_ready(eval_notion, duplicated_id):
                logger.error(
                    "| ✗ Duplicated page %s not accessible after move", duplicated_id
                )
                try:
                    eval_notion.pages.update(page_id=duplicated_id, archived=True)
                except Exception:
                    pass
                return None

            time.sleep(5)

            duplicated_url = f"https://www.notion.so/{duplicated_id.replace('-', '')}"
            return InitialStateInfo(
                state_id=duplicated_id,
                state_url=duplicated_url,
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
    # MCP duplication logic
    # ------------------------------------------------------------------

    async def _duplicate_and_move(
        self,
        official_token: str,
        source_page_id: str,
        eval_hub_id: str,
        page_title: str,
        notion: Client,
    ) -> str:
        """Duplicate a page via the official Notion MCP server, then move it."""
        server = MCPHttpServer(
            url=NOTION_OFFICIAL_MCP_URL,
            headers={"Authorization": f"Bearer {official_token}"},
            timeout=120,
        )
        async with server:
            # 1. Duplicate
            logger.info("| ○ Duplicating page via official MCP server …")
            result = await server.call_tool(
                "notion-duplicate-page", {"page_id": source_page_id}
            )
            dup_data = json.loads(result["content"][0]["text"])
            if dup_data.get("name") == "APIResponseError":
                body = json.loads(dup_data.get("body", "{}"))
                raise RuntimeError(f"Duplication failed: {body.get('message')}")
            duplicated_id = dup_data["page_id"]
            logger.info("| ✓ Duplicated page: %s", duplicated_id)

            # 2. Wait for page to be API-accessible before moving
            if not self._wait_for_page_ready(notion, duplicated_id):
                raise RuntimeError(
                    f"Duplicated page {duplicated_id} not ready after timeout"
                )

            # 3. Move with retry + exponential backoff
            logger.info("| ○ Moving page to eval hub …")
            max_attempts = 8
            for attempt in range(1, max_attempts + 1):
                move_result = await server.call_tool(
                    "notion-move-pages",
                    {
                        "page_or_database_ids": [duplicated_id],
                        "new_parent": {"page_id": eval_hub_id},
                    },
                )
                move_data = json.loads(move_result["content"][0]["text"])

                if move_data.get("name") == "APIResponseError":
                    body = json.loads(move_data.get("body", "{}"))
                    msg = body.get("message", "")
                    if "not a page or database" in msg.lower() or body.get("code") == "validation_error":
                        if attempt < max_attempts:
                            wait = 2 ** attempt
                            logger.info(
                                "| ○ Page not ready for move (attempt %d/%d). Waiting %ds …",
                                attempt, max_attempts, wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                    raise RuntimeError(f"Move failed: {msg}")

                if "result" in move_data and move_data["result"].startswith("Success"):
                    logger.info("| ✓ Page moved to eval hub")
                    break
            else:
                raise RuntimeError(f"Move failed after {max_attempts} attempts")

        # 4. Rename (strip the "(1)" suffix)
        logger.info("| ○ Renaming duplicated page to '%s' …", page_title)
        notion.pages.update(
            page_id=duplicated_id,
            properties={"title": {"title": [{"text": {"content": page_title}}]}},
        )
        return duplicated_id

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
        notion: Client, page_id: str, max_retries: int = 10, delay: int = 2
    ) -> bool:
        logger.info("| ○ Waiting for page %s to be accessible …", page_id)
        for attempt in range(max_retries):
            try:
                result = notion.pages.retrieve(page_id=page_id)
                if result and isinstance(result, dict) and "properties" in result:
                    logger.info(
                        "| ✓ Page ready (attempt %d/%d)", attempt + 1, max_retries
                    )
                    return True
            except Exception as e:
                logger.debug(
                    "| ✗ Not ready (attempt %d/%d): %s", attempt + 1, max_retries, e
                )
            if attempt < max_retries - 1:
                time.sleep(delay)
        logger.error("| ✗ Page not ready after %d attempts", max_retries)
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
