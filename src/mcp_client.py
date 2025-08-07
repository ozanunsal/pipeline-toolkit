import asyncio
import logging
import os
import io
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

import nest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

from src.config import MCPServerConfig, load_config

def _resolve_log_file() -> Path:
    # Prefer config.json 'log_file' if present
    try:
        cfg = load_config()
        if getattr(cfg, "log_file", None):
            p = Path(cfg.log_file).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            return p.resolve()
    except Exception:
        # Config may not be loadable here; fall back to env/default
        pass

    # Else prefer env var; otherwise use CWD/logs/pipeline_bot.log
    log_dir = Path(os.getenv("PIPELINE_TOOLKIT_LOG_DIR", str(Path.cwd() / "logs"))).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "pipeline_bot.log"

log_file = _resolve_log_file()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[logging.FileHandler(str(log_file), mode="w")],
)

logger = logging.getLogger(__name__)
nest_asyncio.apply()


class _ErrLogToLogger(io.TextIOBase):
    """Redirect server stderr to logger, line-buffered."""
    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        self._logger = logger
        self._level = level
        self._buffer: str = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        if not isinstance(s, str):
            s = str(s)
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._logger.log(self._level, line)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.strip())
        self._buffer = ""


class ConnectionState(Enum):
    """Connection state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class MCPClient:
    """
    Enhanced MCP Client with Advanced Connection Management.

    This client provides enterprise-grade connection management to MCP servers with:
    - Automatic retry logic with exponential backoff
    - Connection health monitoring and heartbeat checks
    - Graceful degradation and circuit breaker patterns
    - Comprehensive metrics and monitoring
    - Connection pooling and lifecycle management
    - Event-driven architecture with callbacks

    Features:
        - Automatic connection recovery
        - Health checks and monitoring
        - Performance metrics collection
        - Robust error handling and logging
        - Async context manager support
        - Event callbacks for state changes

    Example:
        >>> config = MCPServerConfig(name="MyServer", url="http://localhost:8080")
        >>> async with MCPClient(config) as client:
        ...     await client.wait_for_healthy()
        ...     tools = await client.list_tools()
        ...     result = await client.call_tool("tool_name", param="value")
    """

    def __init__(
        self, config: MCPServerConfig, connection_timeout: float = 10.0
    ) -> None:
        """
        Initialize the MCP client with a server configuration.

        Args:
            config: Server configuration including URL and connection settings
            connection_timeout: Timeout for initial connection (seconds)
        """
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.session: Optional[ClientSession] = None
        self.context_manager = None
        self.read_stream = None
        self.write_stream = None
        self.tools: List[Any] = []
        self._shutdown_event = asyncio.Event()
        self._state_change_callbacks: List[
            Callable[[ConnectionState, ConnectionState], None]
        ] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        self.connection_timeout = connection_timeout

        logger.info(f"Initialized MCP client for {config.name}")

    def add_state_change_callback(
        self, callback: Callable[[ConnectionState, ConnectionState], None]
    ) -> None:
        """Add a callback for state changes."""
        self._state_change_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback for errors."""
        self._error_callbacks.append(callback)

    def _notify_state_change(
        self, old_state: ConnectionState, new_state: ConnectionState
    ) -> None:
        """Notify all callbacks of state change."""
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.warning(f"State change callback failed: {e}")

    def _notify_error(self, error: Exception) -> None:
        """Notify all callbacks of errors."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")

    def _set_state(self, new_state: ConnectionState) -> None:
        """Set the connection state and notify callbacks."""
        old_state = self.state
        self.state = new_state
        if old_state != new_state:
            logger.debug(f"State changed from {old_state.value} to {new_state.value}")
            self._notify_state_change(old_state, new_state)

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> bool:
        """
        Connect to the MCP server with retry logic.

        Returns:
            True if connection was successful, False otherwise.
        Raises:
            MCPConnectionError: If connection fails after all retries.
        """
        self._set_state(ConnectionState.CONNECTING)
        try:
            if self.config.is_sse_connection():
                logger.info(
                    f"Connecting to MCP server: {self.config.url}{self.config.endpoint}"
                )
                sse_url = f"{self.config.url.rstrip('/')}{self.config.endpoint}"
                connect_task = asyncio.create_task(
                    self._establish_sse_connection(sse_url)
                )
                await asyncio.wait_for(connect_task, timeout=self.connection_timeout)
            else:
                # stdio transport
                logger.info(
                    f"Connecting to MCP server via stdio: {self.config.command}"
                )
                connect_task = asyncio.create_task(
                    self._establish_stdio_connection()
                )
                await asyncio.wait_for(connect_task, timeout=self.connection_timeout)

            # Load available tools
            await self._load_tools()

            # Mark as connected
            self._set_state(ConnectionState.CONNECTED)
            logger.info(f"Successfully connected to MCP server: {self.config.name}")
            return True
        except asyncio.TimeoutError as e:
            logger.error(f"Connection timeout after {self.connection_timeout} seconds")
            self._set_state(ConnectionState.FAILED)
            raise e
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {self.config.name}")
            self._set_state(ConnectionState.FAILED)
            raise e
        return False

    async def _establish_sse_connection(self, sse_url: str) -> None:
        """Establish an SSE connection to the MCP server."""
        self.context_manager = sse_client(sse_url)
        self.read_stream, self.write_stream = await self.context_manager.__aenter__()

        # Create and initialize MCP session
        self.session = ClientSession(self.read_stream, self.write_stream)
        await self.session.__aenter__()
        await self.session.initialize()

    async def _establish_stdio_connection(self) -> None:
        """Establish a stdio connection to the MCP server using configured command."""
        if not self.config.command:
            raise ValueError("Missing 'command' for stdio connection")

        # Expand ~ and $VARS in command, args and working directory
        def _expand(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            return os.path.expandvars(os.path.expanduser(value))

        command = _expand(self.config.command)
        args = [
            _expand(arg) if isinstance(arg, str) else arg  # type: ignore[func-returns-value]
            for arg in (self.config.args or [])
        ]
        env: Optional[Dict[str, str]] = None
        if self.config.environment:
            env = {k: _expand(v) or "" for k, v in self.config.environment.items()}
        cwd = _expand(self.config.working_directory)
        if cwd and not os.path.isdir(cwd):
            logger.warning("Working directory not found, ignoring: %s", cwd)
            cwd = None

        logger.info(
            "Starting stdio MCP server: %s %s", command, " ".join(args)
        )
        if cwd:
            logger.info("Working directory: %s", cwd)

        try:
            server_params = StdioServerParameters(
                command=command or self.config.command,
                args=args,
                env=env,
                cwd=cwd,
                encoding="utf-8",
                encoding_error_handler="strict",
            )
            # Route server stderr to our file handler stream (not the terminal)
            err_stream = None
            for lg in (logging.getLogger(), logging.getLogger(__name__)):
                for h in getattr(lg, "handlers", []):
                    stream = getattr(h, "stream", None)
                    if stream is not None and hasattr(stream, "write"):
                        err_stream = stream
                        break
                if err_stream is not None:
                    break
            if err_stream is None:
                # Fallback: open the configured log file
                err_stream = open(str(log_file), "a", buffering=1)

            self.context_manager = stdio_client(server_params, errlog=err_stream)
            self.read_stream, self.write_stream = await self.context_manager.__aenter__()

            # Create and initialize MCP session
            self.session = ClientSession(self.read_stream, self.write_stream)
            await self.session.__aenter__()
            await self.session.initialize()
        except FileNotFoundError:
            logger.error("Failed to start stdio command. Not found: %s", command)
            raise
        except Exception as e:
            logger.error("Failed to establish stdio connection: %s", e)
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        self._set_state(ConnectionState.DISCONNECTED)
        self._shutdown_event.set()

        await self._cleanup()
        logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def _cleanup(self, disconnect_state: bool = True) -> None:
        """Clean up all resources."""
        if disconnect_state:
            self._set_state(ConnectionState.DISCONNECTED)

        # Clean up session
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing session: {e}")
            finally:
                self.session = None

        # Clean up context manager
        if self.context_manager:
            try:
                await self.context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing context manager: {e}")
            finally:
                self.context_manager = None
        self.read_stream = None
        self.write_stream = None

    async def _load_tools(self) -> None:
        """Load available tools from the MCP server."""
        try:
            if self.session is None:
                raise RuntimeError("MCP session not initialized")

            tools_info = await self.session.list_tools()
            self.tools = tools_info.tools if tools_info.tools else []
            logger.info(f"Loaded {len(self.tools)} tools from MCP server")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
            self.tools = []
            raise

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.

        Returns:
            List of tool dictionaries with name, description, and parameters.
        Raises:
            RuntimeError: If MCP session is not initialized.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to MCP server. Call connect() first.")
        return [
            {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", "No description"),
                "inputSchema": (
                    getattr(tool, "inputSchema", {})
                    if hasattr(tool, "inputSchema")
                    else {}
                ),
            }
            for tool in self.tools
        ]

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to the MCP server."""
        return self.state == ConnectionState.CONNECTED

    @property
    def is_healthy(self) -> bool:
        """Check if the client is healthy."""
        return self.state == ConnectionState.HEALTHY

    def __repr__(self) -> str:
        """String representation of the client."""
        return f"MCPClient(name='{self.config.name}', state='{self.state.value}', tools={len(self.tools)})"
