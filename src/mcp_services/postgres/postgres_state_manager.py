"""
PostgreSQL State Manager for MCPMark
=====================================

Manages database state for PostgreSQL tasks including schema setup,
test data creation, and cleanup.
"""

import os
import subprocess
import sys
import psycopg2
from psycopg2 import sql
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.base.state_manager import BaseStateManager, InitialStateInfo
from src.base.task_manager import BaseTask
from src.logger import get_logger

logger = get_logger(__name__)


class PostgresStateManager(BaseStateManager):
    """Manages PostgreSQL database state for task evaluation."""

    def __init__(
        self,
    ):
        """Initialize PostgreSQL state manager.
        """
        super().__init__(service_name="postgres")

        self.database = "postgres"
        self.host = ""
        self.port = 0
        self.username = ""
        self.password = ""
        self.conn_params = {}

        # Track created databases for cleanup
        self.created_databases: List[str] = []

        # Track current task database for agent configuration
        self._current_task_database: Optional[str] = None

    def _setup_database(self, db_name: str):
        """Setup all required databases by downloading and restoring from backup."""
        databases = ['employees', 'chinook', 'dvdrental', 'sports', 'lego']
        if db_name not in databases:
            return

        if not self._database_exists(db_name):
            logger.info(f"Setting up {db_name} database...")

            # Path to backup file
            backup_dir = Path(__file__).parent.parent.parent.parent / "postgres_state"
            backup_file = backup_dir / f"{db_name}.backup"

            # Download backup if not exists
            if not backup_file.exists():
                backup_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading {db_name} backup...")
                try:
                    import urllib.request
                    urllib.request.urlretrieve(
                        f'https://storage.mcpmark.ai/postgres/{db_name}.backup',
                        str(backup_file)
                    )
                    logger.info(f"{db_name} backup downloaded")
                except Exception as e:
                    logger.warning(f"Failed to download {db_name} backup: {e}")
                    return

            # Create database
            try:
                self._create_empty_database(db_name)
                logger.info(f"Created {db_name} database")
            except Exception as e:
                logger.warning(f"Failed to create {db_name} database: {e}")
                return

            # Restore from backup
            env = os.environ.copy()
            env['PGPASSWORD'] = self.password

            try:
                result = subprocess.run([
                    'pg_restore',
                    '-h', str(self.host),
                    '-p', str(self.port),
                    '-U', self.username,
                    '-d', db_name,
                    '-v',
                    str(backup_file)
                ], env=env, capture_output=True, text=True)

                if result.returncode != 0 and "ERROR" in result.stderr:
                    logger.warning(f"pg_restore had errors for {db_name}: {result.stderr}")
                else:
                    logger.info(f"{db_name} database restored successfully")
            except Exception as e:
                logger.warning(f"Failed to restore {db_name} database: {e}")
        else:
            logger.debug(f"{db_name} database already exists")

    def _create_initial_state(self, task: BaseTask) -> Optional[InitialStateInfo]:
        """Create initial database state for a task."""
        try:
            # Generate unique database name
            db_name = "sandbox"
            sandbox = task.sandbox
            sandbox_info = sandbox.get_sandbox_info()

            postgres_uri = sandbox_info.get("auth_data", {}).get("api_key")

            if postgres_uri:
                self._configure_from_uri(postgres_uri, db_name)

            self._setup_database(task.category_id)
            # Create database from template if exists, otherwise empty
            if self._database_exists(task.category_id):
                self._create_database_from_template(db_name, task.category_id)
                logger.info(
                    f"| Created database '{db_name}' from template '{task.category_id}'"
                )
            else:
                self._create_empty_database(db_name)
                logger.info(f"| Created empty database '{db_name}'")
                # Run prepare_environment.py if it exists
                self._run_prepare_environment(db_name, task)
                logger.info(f"| Prepared environment for database '{db_name}'")

            # Track for cleanup
            self.created_databases.append(db_name)
            self.track_resource("database", db_name, {"task": task.name})


            return InitialStateInfo(
                state_id=db_name,
                state_url=f"postgresql://{self.username}@{self.host}:{self.port}/{db_name}",
                metadata={
                    "database": db_name,
                    "category": task.category_id,
                    "task_id": task.task_id,
                },
            )

        except Exception as e:
            logger.error(f"Failed to create initial state for {task.name}: {e}")
            return None

    def _store_initial_state_info(
        self, task: BaseTask, state_info: InitialStateInfo
    ) -> None:
        """Store database info in task object."""
        if hasattr(task, "__dict__"):
            task.database_name = state_info.state_id
            task.database_url = state_info.state_url
            # Store current task database for agent configuration
            self._current_task_database = state_info.state_id

    def _cleanup_task_initial_state(self, task: BaseTask) -> bool:
        """Clean up task database."""
        if hasattr(task, "database_name") and task.database_name:
            try:
                self._drop_database(task.database_name)
                logger.info(f"| Dropped database: {task.database_name}")

                # Remove from tracking
                self.created_databases = [
                    db for db in self.created_databases if db != task.database_name
                ]
                # Clear current task database
                if self._current_task_database == task.database_name:
                    self._current_task_database = None
                return True
            except Exception as e:
                logger.error(f"Failed to drop database {task.database_name}: {e}")
                return False
        return True

    def _cleanup_single_resource(self, resource: Dict[str, Any]) -> bool:
        """Clean up a single PostgreSQL resource."""
        if resource["type"] == "database":
            try:
                self._drop_database(resource["id"])
                logger.info(f"| Dropped database: {resource['id']}")
                return True
            except Exception as e:
                logger.error(f"| Failed to drop database {resource['id']}: {e}")
                return False
        return False

    def clean_up(self, task=None, **kwargs) -> bool:
        """Delete repositories that were imported for tasks."""
        success = True
        sandbox = task.sandbox if task else None
        if sandbox:
            try:
                sandbox.release()
            except Exception as e:
                logger.error(f"| Failed to release sandbox: {e}")
                success = False
        return success

    def _database_exists(self, db_name: str) -> bool:
        """Check if database exists."""
        conn = psycopg2.connect(**self.conn_params, database="postgres")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
                return cur.fetchone() is not None
        finally:
            conn.close()

    def _create_database_from_template(self, new_db: str, template_db: str):
        """Create database from template."""
        import time
        conn = psycopg2.connect(**self.conn_params, database="postgres")
        conn.autocommit = True
        logger.info(f"Creating database '{new_db}' from template '{template_db}'")
        try:
            with conn.cursor() as cur:
                # Block new connections to the template DB
                cur.execute(
                    sql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(
                        sql.Identifier(template_db)
                    )
                )
                last_exc = None
                for attempt in range(40):
                    cur.execute(
                        sql.SQL("""
                        SELECT pg_terminate_backend(pid)
                        FROM pg_stat_activity
                        WHERE datname = %s AND pid <> pg_backend_pid()
                    """),
                        (template_db,),
                    )
                    try:
                        cur.execute(
                            sql.SQL("CREATE DATABASE {} WITH TEMPLATE {}").format(
                                sql.Identifier(new_db), sql.Identifier(template_db)
                            )
                        )
                        last_exc = None
                        break
                    except psycopg2.errors.ObjectInUse as e:
                        last_exc = e
                        logger.warning(f"Template DB still in use (attempt {attempt + 1}/40), retrying in 5s...")
                        time.sleep(5)
                if last_exc:
                    raise last_exc
        except Exception as e:
            logger.error(f"_create_database_from_template error: {type(e).__name__}: {e}", exc_info=True)
            raise
        finally:
            # Re-enable connections to the template DB regardless of outcome
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO PUBLIC").format(
                        sql.Identifier(template_db)
                    )
                )
            conn.close()

    def _create_empty_database(self, db_name: str):
        """Create empty database."""
        conn = psycopg2.connect(**self.conn_params, database="postgres")
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                )
        finally:
            conn.close()

    def _drop_database(self, db_name: str):
        """Drop database."""
        conn = psycopg2.connect(**self.conn_params, database="postgres")
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                # Terminate connections
                cur.execute(
                    sql.SQL("""
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                """),
                    (db_name,),
                )

                # Drop database
                cur.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(db_name)
                    )
                )
        finally:
            conn.close()

    def _run_prepare_environment(self, db_name: str, task: BaseTask):
        """Run prepare_environment.py script if it exists in the task directory."""
        # Find the task directory containing prepare_environment.py
        task_dir = task.task_instruction_path.parent
        prepare_script = task_dir / "prepare_environment.py"

        if not prepare_script.exists():
            logger.debug(f"No prepare_environment.py found for task {task.name}")
            return

        logger.info(f"| Running prepare_environment.py for task {task.name}")

        # Set up environment variables for the script
        env = os.environ.copy()
        env.update({
            "POSTGRES_HOST": str(self.host),
            "POSTGRES_PORT": str(self.port),
            "POSTGRES_DATABASE": db_name,
            "POSTGRES_USERNAME": self.username,
            "POSTGRES_PASSWORD": self.password or "",
        })

        try:
            # Run the prepare_environment.py script
            result = subprocess.run(
                [sys.executable, str(prepare_script)],
                cwd=str(task_dir),  # Run from task directory to access data/ folder
                env=env,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"| ✓ Environment preparation completed for {task.name}")
                if result.stdout.strip():
                    logger.debug(f"| prepare_environment.py output: {result.stdout}")
            else:
                logger.error(f"| ❌ Environment preparation failed for {task.name}")
                logger.error(f"| Error output: {result.stderr}")
                raise RuntimeError(f"prepare_environment.py failed with exit code {result.returncode}")

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Environment preparation timed out for {task.name}")
            raise RuntimeError("prepare_environment.py execution timed out")
        except Exception as e:
            logger.error(f"❌ Failed to run prepare_environment.py for {task.name}: {e}")
            raise

    def _setup_task_specific_data(self, db_name: str, task: BaseTask):
        """Set up task-specific schema and data."""
        conn = psycopg2.connect(**self.conn_params, database=db_name)
        try:
            with conn.cursor() as cur:
                if task.category_id == "basic_queries":
                    self._setup_basic_queries_data(cur)
                elif task.category_id == "data_manipulation":
                    self._setup_data_manipulation_data(cur)
                elif task.category_id == "table_operations":
                    self._setup_table_operations_data(cur)
                # Add more categories as needed

            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to setup task data: {e}")
            raise
        finally:
            conn.close()

    def _setup_basic_queries_data(self, cursor):
        """Set up data for basic query tasks."""
        cursor.execute("""
            CREATE TABLE employees (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                department VARCHAR(50),
                salary DECIMAL(10, 2),
                hire_date DATE
            );

            INSERT INTO employees (name, department, salary, hire_date) VALUES
            ('John Doe', 'Engineering', 75000.00, '2020-01-15'),
            ('Jane Smith', 'Marketing', 65000.00, '2019-03-22'),
            ('Bob Johnson', 'Engineering', 80000.00, '2018-07-01'),
            ('Alice Brown', 'HR', 55000.00, '2021-02-10');
        """)

    def _setup_data_manipulation_data(self, cursor):
        """Set up data for data manipulation tasks."""
        cursor.execute("""
            CREATE TABLE products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                category VARCHAR(50),
                price DECIMAL(10, 2),
                stock INTEGER DEFAULT 0
            );

            CREATE TABLE orders (
                id SERIAL PRIMARY KEY,
                product_id INTEGER REFERENCES products(id),
                quantity INTEGER NOT NULL,
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    def _setup_table_operations_data(self, cursor):
        """Set up for table operation tasks."""
        # Start with minimal schema that tasks will modify
        cursor.execute("""
            CREATE TABLE test_table (
                id SERIAL PRIMARY KEY,
                data VARCHAR(255)
            );
        """)

    def _get_timestamp(self) -> str:
        """Get timestamp for unique naming."""
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d%H%M%S")

    def _configure_from_uri(self, postgres_uri: str, db_name: str):
        """Parse postgres_uri and configure connection params and environment variables."""
        parsed = psycopg2.extensions.parse_dsn(postgres_uri)
        self.host = parsed.get("host", "localhost")
        self.port = int(parsed.get("port", 5432))
        self.username = parsed.get("user", "postgres")
        self.password = parsed.get("password")
        self.conn_params = {k: v for k, v in parsed.items() if k != "dbname"}
        os.environ["POSTGRES_HOST"] = self.host
        os.environ["POSTGRES_PORT"] = str(self.port)
        os.environ["POSTGRES_DATABASE"] = parsed.get("dbname", db_name)
        os.environ["POSTGRES_USERNAME"] = self.username
        os.environ["POSTGRES_PASSWORD"] = self.password or ""

    def get_service_config_for_agent(self) -> dict:
        """Get configuration for agent execution."""
        config = {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
        }

        # If there's a current task database, include it
        if hasattr(self, "_current_task_database") and self._current_task_database:
            config["current_database"] = self._current_task_database
            config["database_url"] = (
                f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self._current_task_database}"
            )
        else:
            # Fallback to default database
            config["database"] = self.database
            config["database_url"] = (
                f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            )

        return config
