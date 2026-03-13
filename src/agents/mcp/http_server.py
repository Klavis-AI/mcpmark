"""
Minimal MCP HTTP Server Implementation
=======================================

Provides HTTP-based MCP server communication for services like GitHub.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

_RETRYABLE_BASE = (ConnectionError, TimeoutError, OSError)


def _is_retryable(exc: BaseException) -> bool:
    """Return True if exc looks like a transient connection/TLS error.

    Handles plain exceptions, httpcore.ConnectError (OSError subclass),
    and anyio/asyncio BaseExceptionGroup wrapping connection errors.
    """
    if isinstance(exc, _RETRYABLE_BASE):
        return True
    # BaseExceptionGroup is Python 3.11+; fall back gracefully on older versions
    if isinstance(exc, BaseException) and hasattr(exc, "exceptions"):
        return any(_is_retryable(e) for e in exc.exceptions)
    return False


class MCPHttpServer:
    """
    HTTP-based MCP client using the official MCP Python SDK
    (Streamable HTTP transport).
    """

    DEFAULT_RETRIES = 3
    RETRY_BACKOFF = 2.0  # seconds; doubled on each subsequent attempt

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        max_retries: int = DEFAULT_RETRIES,
    ):
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.max_retries = max_retries

        self._stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    async def start(self):
        """Open Streamable HTTP transport and initialize MCP session.

        Retries up to *max_retries* times with exponential backoff on
        transient connection / TLS errors.
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                delay = self.RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "MCP connect attempt %d/%d in %.1fs (last error: %s)",
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                    last_exc,
                )
                await asyncio.sleep(delay)

            stack = AsyncExitStack()
            try:
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamablehttp_client(self.url, headers=self.headers)
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await asyncio.wait_for(session.initialize(), timeout=self.timeout)
                self._stack = stack
                self.session = session
                return
            except BaseException as e:
                await stack.aclose()
                last_exc = e
                if not _is_retryable(e) or attempt >= self.max_retries:
                    raise

        raise last_exc  # unreachable, satisfies type checkers

    async def stop(self):
        """Close the session/transport cleanly.

        Teardown errors (e.g. the background SSE listener failing on a
        already-broken connection) are logged and suppressed so they do not
        mask the real exception that caused the shutdown.
        """
        if self._stack:
            try:
                await self._stack.aclose()
            except BaseException as e:
                logger.warning("Error during MCP session teardown (ignored): %s", e)
        self._stack = None
        self.session = None
        self._tools_cache = None

    async def _reconnect(self):
        """Tear down and re-establish the MCP session."""
        logger.warning("MCP connection lost; reconnecting...")
        await self.stop()
        await self.start()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions (cached)."""
        if self._tools_cache is not None:
            return self._tools_cache
        if not self.session:
            raise RuntimeError("MCP HTTP client not started")

        resp = await asyncio.wait_for(self.session.list_tools(), timeout=self.timeout)
        self._tools_cache = [t.model_dump() for t in resp.tools]
        return self._tools_cache

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke a remote tool and return the structured result.

        On transient connection errors, reconnects and retries up to
        *max_retries* times with exponential backoff.
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            if not self.session:
                raise RuntimeError("MCP HTTP client not started")
            try:
                result = await asyncio.wait_for(
                    self.session.call_tool(name, arguments), timeout=self.timeout
                )
                return result.model_dump()
            except asyncio.CancelledError as e:
                task = asyncio.current_task()
                if task and task.cancelling() > 0:
                    raise  # outer timeout is cancelling us, don't retry
                # internal anyio cancellation — treat like a connection error
                last_exc = e
                if attempt < self.max_retries:
                    delay = self.RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Tool call '%s' cancelled internally (attempt %d/%d), reconnecting in %.1fs",
                        name,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    try:
                        await self._reconnect()
                    except BaseException as re:
                        logger.error("Reconnect failed: %s", re)
                        raise re
                else:
                    raise
            except BaseException as e:
                last_exc = e
                if _is_retryable(e) and attempt < self.max_retries:
                    delay = self.RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Tool call '%s' failed (attempt %d/%d), reconnecting in %.1fs: %s",
                        name,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    try:
                        await self._reconnect()
                    except BaseException as re:
                        logger.error("Reconnect failed: %s", re)
                        raise re
                else:
                    raise

        raise last_exc  # unreachable
