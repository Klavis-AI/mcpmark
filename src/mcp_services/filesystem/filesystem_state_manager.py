"""
Filesystem State Manager for MCPMark
=====================================

This module handles filesystem state management for consistent task evaluation.
It uses the Klavis Local Sandbox to host the filesystem MCP server and uploads
the test environment data into the sandbox VM for each task.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.base.state_manager import BaseStateManager
from src.base.task_manager import BaseTask
from src.logger import get_logger

logger = get_logger(__name__)


class FilesystemStateManager(BaseStateManager):
    """
    Manages filesystem state for task evaluation via Klavis Local Sandbox.

    For each task the manager:
    1. Downloads the category-specific test environment locally (if not cached).
    2. Uploads it into the Klavis Local Sandbox VM.
    3. Before verification, dumps the sandbox state back to a local directory
       so that existing verification scripts (which read ``FILESYSTEM_TEST_DIR``)
       work unchanged.
    """

    # URL mapping for downloadable test environment archives
    TEST_ENV_URLS = {
        "desktop": "https://storage.mcpmark.ai/filesystem/desktop.zip",
        "file_context": "https://storage.mcpmark.ai/filesystem/file_context.zip",
        "file_property": "https://storage.mcpmark.ai/filesystem/file_property.zip",
        "folder_structure": "https://storage.mcpmark.ai/filesystem/folder_structure.zip",
        "papers": "https://storage.mcpmark.ai/filesystem/papers.zip",
        "student_database": "https://storage.mcpmark.ai/filesystem/student_database.zip",
        "threestudio": "https://storage.mcpmark.ai/filesystem/threestudio.zip",
        "votenet": "https://storage.mcpmark.ai/filesystem/votenet.zip",
        "legal_document": "https://storage.mcpmark.ai/filesystem/legal_document.zip",
        "desktop_template": "https://storage.mcpmark.ai/filesystem/desktop_template.zip",
    }

    def _get_project_root(self) -> Path:
        """Find project root by looking for marker files."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").exists() or (parent / "pipeline.py").exists():
                return parent
        return Path(__file__).parent / "../../../"

    def __init__(self, templates_root: str = "./test_environments"):
        """
        Initialize filesystem state manager.

        Args:
            templates_root: Local directory that stores downloaded test
                            environment archives (one sub-dir per category).
        """
        super().__init__(service_name="filesystem")

        self.templates_root = Path(templates_root).expanduser().resolve()
        self.current_task_dir: Optional[Path] = None
        self.created_resources: List[Path] = []
        self._dump_dir: Optional[Path] = None  # temp dir for sandbox dump

        logger.info(
            f"Initialized FilesystemStateManager (templates_root={self.templates_root})"
        )

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def initialize(self, **kwargs) -> bool:
        """Ensure local templates root exists."""
        self.templates_root.mkdir(parents=True, exist_ok=True)
        return True

    def set_up(self, task: BaseTask) -> bool:
        """Upload the test environment into the Klavis Local Sandbox.

        The sandbox object is expected to be attached to ``task.sandbox``
        (a ``KlavisLocalSandbox`` instance) by the evaluator before calling
        this method.
        """
        try:
            sandbox = task.sandbox
            if sandbox is None:
                logger.error("No sandbox attached to task")
                return False

            # Determine category and resolve local test environment path
            category = task.category_id or "desktop"
            self._current_task_category = category
            test_root = self.templates_root / category

            # Download test environment locally if not already cached
            if not test_root.exists():
                logger.info(f"| Test environment not cached locally, downloading for category '{category}'")
                if not self._download_and_extract_test_environment(category, test_root):
                    return False

            # Upload into the sandbox
            logger.info(f"| Uploading test environment ({test_root}) into sandbox …")
            if not sandbox.upload_directory(test_root):
                logger.error("| Failed to upload test environment into sandbox")
                return False
            logger.info("| ✓ Test environment uploaded into sandbox")

            # Store task directory reference (will be set after dump for verification)
            if hasattr(task, "__dict__"):
                task.test_directory = None  # set after dump

            return True

        except Exception as e:
            logger.error(f"Failed to set up filesystem state for {task.name}: {e}")
            return False

    def set_verification_environment(self, messages_path: str = None) -> None:
        """Dump the sandbox state locally so verification scripts can read it.

        Verification scripts rely on the ``FILESYSTEM_TEST_DIR`` environment
        variable pointing to a local directory that contains the files the
        agent worked on.
        """
        import os
        if messages_path:
            os.environ["MCP_MESSAGES"] = str(messages_path)

        # The dump is performed by the evaluator *before* calling verification.
        # Here we just make sure the env-var is set if a dump dir exists.
        if self._dump_dir and self._dump_dir.exists():
            os.environ["FILESYSTEM_TEST_DIR"] = str(self._dump_dir)

    def dump_sandbox_state(self, sandbox, task: BaseTask) -> bool:
        """Dump sandbox filesystem state to a local temp directory.

        After this call ``FILESYSTEM_TEST_DIR`` will point at the dumped
        directory so that verification scripts can inspect the result.

        Args:
            sandbox: ``KlavisLocalSandbox`` instance.
            task: Current task (used for naming).

        Returns:
            ``True`` on success.
        """
        try:
            project_root = self._get_project_root()
            dump_root = (project_root / ".mcpmark_dumps").resolve()
            dump_root.mkdir(exist_ok=True)

            task_id = f"{task.service}_{task.category_id}_{task.task_id}"
            self._dump_dir = dump_root / f"dump_{task_id}_{os.getpid()}"

            # Clean previous dump if present
            if self._dump_dir.exists():
                shutil.rmtree(self._dump_dir)

            if not sandbox.dump_to_directory(self._dump_dir):
                logger.error("| Failed to dump sandbox state")
                return False

            self.current_task_dir = self._dump_dir
            if hasattr(task, "__dict__"):
                task.test_directory = str(self._dump_dir)
            os.environ["FILESYSTEM_TEST_DIR"] = str(self._dump_dir)
            logger.info(f"| ✓ Sandbox state dumped to {self._dump_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to dump sandbox state: {e}")
            return False

    def clean_up(self, task: Optional[BaseTask] = None, **kwargs) -> bool:
        """Clean up local dump directory after verification."""
        try:
            if self._dump_dir and self._dump_dir.exists():
                shutil.rmtree(self._dump_dir)
                logger.info(
                    f"| ✓ Cleaned up dump directory for task "
                    f"{task.name if task else 'unknown'}"
                )
                self._dump_dir = None

            self.current_task_dir = None
            self.created_resources.clear()
            return True
        except Exception as e:
            logger.error(f"Filesystem cleanup failed: {e}")
            return False

    # -----------------------------------------------------------------
    # Service config for agent
    # -----------------------------------------------------------------

    def get_service_config_for_agent(self) -> dict:
        """Return service configuration for the agent."""
        return {}

    # -----------------------------------------------------------------
    # Abstract method implementations
    # -----------------------------------------------------------------

    def _create_initial_state(self, task: BaseTask) -> Optional[Dict[str, Any]]:
        if self.current_task_dir and self.current_task_dir.exists():
            return {"task_directory": str(self.current_task_dir)}
        return None

    def _store_initial_state_info(
        self, task: BaseTask, state_info: Dict[str, Any]
    ) -> None:
        if state_info and "task_directory" in state_info:
            if hasattr(task, "__dict__"):
                task.test_directory = state_info["task_directory"]

    def _cleanup_task_initial_state(self, task: BaseTask) -> bool:
        return True  # sandbox handles isolation

    def _cleanup_single_resource(self, resource: Dict[str, Any]) -> bool:
        if "path" in resource:
            resource_path = Path(resource["path"])
            if resource_path.exists():
                try:
                    if resource_path.is_dir():
                        shutil.rmtree(resource_path)
                    else:
                        resource_path.unlink()
                    return True
                except Exception as e:
                    logger.error(f"Failed to clean up {resource_path}: {e}")
                    return False
        return True

    # -----------------------------------------------------------------
    # Download helpers
    # -----------------------------------------------------------------

    def _download_and_extract_test_environment(
        self, category: str, dest: Path
    ) -> bool:
        """Download and extract a test environment zip for *category* into *dest*."""
        import subprocess

        url = os.getenv("TEST_ENVIRONMENT_URL", self.TEST_ENV_URLS.get(category, ""))
        if not url:
            logger.error(f"| No URL mapping found for category: {category}")
            return False

        logger.info(f"| ○ Downloading test environment from: {url}")

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "test_environment.zip"

            # Download
            try:
                try:
                    subprocess.run(
                        ["wget", "-O", str(zip_path), url],
                        capture_output=True, text=True, check=True,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError):
                    subprocess.run(
                        ["curl", "-L", "-o", str(zip_path), url],
                        capture_output=True, text=True, check=True,
                    )
                logger.info("| ✓ Download completed")
            except Exception as e:
                logger.error(f"| Download failed: {e}")
                return False

            # Extract
            try:
                subprocess.run(
                    ["unzip", "-o", str(zip_path), "-d", str(dest.parent)],
                    capture_output=True, text=True, check=True,
                )
                logger.info("| ✓ Extraction completed")
            except Exception as e:
                logger.error(f"| Extraction failed: {e}")
                return False

            # Cleanup macOS artefacts
            macosx_path = dest.parent / "__MACOSX"
            if macosx_path.exists():
                shutil.rmtree(macosx_path, ignore_errors=True)

            if not dest.exists():
                logger.error(f"| Extracted directory not found: {dest}")
                return False

            logger.info(f"| ✓ Test environment ready: {dest}")
            return True
