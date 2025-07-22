"""
Enhanced MCP Client - Professional SSE Transport with Advanced Connection Management.

This module provides a robust, enterprise-grade client for connecting to MCP servers using SSE
(Server-Sent Events) transport with comprehensive error handling, retry logic, health checks,
and connection lifecycle management.
"""

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta

import nest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

from src.exceptions import MCPConnectionError, ToolExecutionError
from src.config import MCPServerConfig

# Configure logging
log_file = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'pipeline_bot.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='a')]
)

logger = logging.getLogger(__name__)
nest_asyncio.apply()


class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class ConnectionMetrics:
    """Connection metrics and statistics."""
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_connection_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    average_response_time: float = 0.0
    uptime_percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "connection_attempts": self.connection_attempts,
            "successful_connections": self.successful_connections,
            "failed_connections": self.failed_connections,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "last_connection_time": self.last_connection_time.isoformat() if self.last_connection_time else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "average_response_time": self.average_response_time,
            "uptime_percentage": self.uptime_percentage,
        }


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
        self,
        config: MCPServerConfig,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        heartbeat_interval: float = 30.0,
        health_check_timeout: float = 5.0,
        connection_timeout: float = 10.0
    ) -> None:
        """
        Initialize the enhanced MCP client.

        Args:
            config: Server configuration including URL and connection settings
            max_retries: Maximum number of connection retry attempts
            retry_delay: Initial delay between retries (seconds)
            max_retry_delay: Maximum delay between retries (seconds)
            heartbeat_interval: Interval for heartbeat checks (seconds)
            health_check_timeout: Timeout for health checks (seconds)
            connection_timeout: Timeout for initial connection (seconds)
        """
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self.heartbeat_interval = heartbeat_interval
        self.health_check_timeout = health_check_timeout
        self.connection_timeout = connection_timeout

        # Connection state management
        self.state = ConnectionState.DISCONNECTED
        self.session: Optional[ClientSession] = None
        self.tools: List[Any] = []
        self.read_stream = None
        self.write_stream = None
        self._context_manager = None

        # Metrics and monitoring
        self.metrics = ConnectionMetrics()
        self._start_time = datetime.now()

        # Health monitoring
        self._health_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Event callbacks
        self._state_change_callbacks: List[Callable[[ConnectionState, ConnectionState], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        logger.info(f"Initialized MCP client for {config.name} with enhanced features")

    def add_state_change_callback(self, callback: Callable[[ConnectionState, ConnectionState], None]) -> None:
        """Add a callback for state changes."""
        self._state_change_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback for errors."""
        self._error_callbacks.append(callback)

    def _notify_state_change(self, old_state: ConnectionState, new_state: ConnectionState) -> None:
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
            True if connection successful, False otherwise

        Raises:
            MCPConnectionError: If connection fails after all retries
        """
        self._set_state(ConnectionState.CONNECTING)
        self.metrics.connection_attempts += 1

        for attempt in range(self.max_retries + 1):
            try:
                if self.config.is_stdio_connection():
                    logger.info(f"Starting stdio MCP server: {self.config.name} (attempt {attempt + 1}/{self.max_retries + 1})")
                    # Use asyncio.wait_for for connection timeout
                    connect_task = asyncio.create_task(self._establish_stdio_connection())
                    await asyncio.wait_for(connect_task, timeout=self.connection_timeout)
                else:
                    logger.info(f"Connecting to MCP server: {self.config.url}{self.config.endpoint} (attempt {attempt + 1}/{self.max_retries + 1})")
                    # Create SSE connection with timeout
                    sse_url = f"{self.config.url.rstrip('/')}{self.config.endpoint}"
                    # Use asyncio.wait_for for connection timeout
                    connect_task = asyncio.create_task(self._establish_http_connection(sse_url))
                    await asyncio.wait_for(connect_task, timeout=self.connection_timeout)

                # Load available tools
                await self._load_tools()

                # Mark as connected and healthy
                self._set_state(ConnectionState.CONNECTED)
                self.metrics.successful_connections += 1
                self.metrics.last_connection_time = datetime.now()

                # Start health monitoring
                await self._start_health_monitoring()

                logger.info(f"Successfully connected to MCP server: {self.config.name}")
                return True

            except asyncio.TimeoutError:
                error_msg = f"Connection timeout after {self.connection_timeout}s"
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")

                if attempt < self.max_retries:
                    delay = min(self.retry_delay * (2 ** attempt), self.max_retry_delay)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self._set_state(ConnectionState.FAILED)
                    self.metrics.failed_connections += 1
                    error = MCPConnectionError(f"Failed to connect to {self.config.name} after {self.max_retries + 1} attempts: {error_msg}")
                    self._notify_error(error)
                    raise error

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries:
                    delay = min(self.retry_delay * (2 ** attempt), self.max_retry_delay)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self._set_state(ConnectionState.FAILED)
                    self.metrics.failed_connections += 1
                    error = MCPConnectionError(f"Failed to connect to {self.config.name} after {self.max_retries + 1} attempts: {e}")
                    self._notify_error(error)
                    await self._cleanup()
                    raise error

        return False

    async def _establish_http_connection(self, sse_url: str) -> None:
        """Establish the SSE connection and MCP session."""
        self._context_manager = sse_client(sse_url)
        self.read_stream, self.write_stream = await self._context_manager.__aenter__()

        # Create and initialize MCP session
        self.session = ClientSession(self.read_stream, self.write_stream)
        await self.session.__aenter__()
        await self.session.initialize()

    async def _establish_stdio_connection(self) -> None:
        """Establish a stdio connection to a local MCP server."""
        # Build command with arguments
        cmd = [self.config.command] + (self.config.args or [])

        # Set up environment
        env = os.environ.copy()
        if self.config.environment:
            env.update(self.config.environment)

        # Set working directory
        cwd = self.config.working_directory or os.getcwd()

        logger.info(f"Starting stdio MCP server: {self.config.name}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Working directory: {cwd}")
        if self.config.environment:
            logger.info(f"Environment variables: {list(self.config.environment.keys())}")

        try:
            # Verify command exists
            import shutil
            command_path = shutil.which(self.config.command)
            if not command_path:
                raise MCPConnectionError(f"Command not found: {self.config.command}")

            # Verify working directory exists
            if not os.path.exists(cwd):
                raise MCPConnectionError(f"Working directory does not exist: {cwd}")

            # Create stdio client - this will launch the subprocess
            logger.debug(f"Creating stdio client with parameters")
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=env,
                cwd=cwd
            )
            self._context_manager = stdio_client(server_params)

            logger.debug("Entering stdio client context...")
            self.read_stream, self.write_stream = await self._context_manager.__aenter__()

            logger.debug("Creating MCP session...")
            # Create and initialize MCP session
            self.session = ClientSession(self.read_stream, self.write_stream)
            await self.session.__aenter__()

            logger.debug("Initializing MCP session...")
            await self.session.initialize()

            logger.info(f"Successfully connected to stdio MCP server: {self.config.name}")

        except Exception as e:
            logger.error(f"Failed to establish stdio connection to {self.config.name}: {e}")
            logger.error(f"Command attempted: {' '.join(cmd)}")
            logger.error(f"Working directory: {cwd}")
            if hasattr(e, '__cause__') and e.__cause__:
                logger.error(f"Underlying error: {e.__cause__}")
            raise MCPConnectionError(f"Failed to start stdio MCP server {self.config.name}: {e}")

    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()

        self._health_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self) -> None:
        """Background task for health monitoring."""
        while not self._shutdown_event.is_set() and self.state != ConnectionState.DISCONNECTED:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                if self.state in [ConnectionState.CONNECTED, ConnectionState.HEALTHY]:
                    is_healthy = await self._perform_health_check()

                    if is_healthy:
                        if self.state != ConnectionState.HEALTHY:
                            self._set_state(ConnectionState.HEALTHY)
                        self.metrics.last_heartbeat = datetime.now()
                    else:
                        self._set_state(ConnectionState.UNHEALTHY)
                        # Start reconnection process
                        if not self._reconnect_task or self._reconnect_task.done():
                            self._reconnect_task = asyncio.create_task(self._reconnect())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health monitor error: {e}")
                self._notify_error(e)

    async def _perform_health_check(self) -> bool:
        """Perform a health check on the connection."""
        try:
            if not self.session:
                return False

            # Simple health check - try to list tools with timeout
            health_task = asyncio.create_task(self.session.list_tools())
            await asyncio.wait_for(health_task, timeout=self.health_check_timeout)

            return True

        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the server."""
        if self.state == ConnectionState.RECONNECTING:
            return  # Already reconnecting

        self._set_state(ConnectionState.RECONNECTING)
        logger.info(f"Attempting to reconnect to {self.config.name}")

        try:
            await self._cleanup(disconnect_state=False)
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self._set_state(ConnectionState.FAILED)
            self._notify_error(e)

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and cleanup resources."""
        self._set_state(ConnectionState.DISCONNECTED)
        self._shutdown_event.set()

        # Cancel background tasks
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        await self._cleanup()
        logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def _cleanup(self, disconnect_state: bool = True) -> None:
        """Clean up all resources."""
        if disconnect_state:
            self._set_state(ConnectionState.DISCONNECTED)

        # Cleanup session first
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing session: {e}")
            finally:
                self.session = None

        # Cleanup context manager (this will stop stdio processes/containers)
        if self._context_manager:
            try:
                # For stdio connections, this will terminate the subprocess/container
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing context manager: {e}")
            finally:
                self._context_manager = None

        self.read_stream = None
        self.write_stream = None

    async def _load_tools(self) -> None:
        """Load available tools from the MCP server."""
        try:
            if self.session is None:
                raise MCPConnectionError("Session not initialized")

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
            List of tool definitions with name, description, and full inputSchema

        Raises:
            MCPConnectionError: If not connected to server
        """
        if not self.is_connected:
            raise MCPConnectionError("Not connected to MCP server. Call connect() first.")

        return [
            {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", "No description"),
                "inputSchema": getattr(tool, "inputSchema", {}) if hasattr(tool, "inputSchema") else {},
            }
            for tool in self.tools
        ]

    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool with the given name and arguments.

        Args:
            tool_name: The name of the tool to call
            **kwargs: Tool-specific arguments

        Returns:
            Tool execution result with success status and content

        Raises:
            MCPConnectionError: If not connected to server
            ToolExecutionError: If tool execution fails
        """
        if not self.is_connected:
            raise MCPConnectionError("Not connected to MCP server. Call connect() first.")

        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            if self.session is None:
                raise MCPConnectionError("Session not initialized")

            logger.debug(f"Calling tool: {tool_name} with arguments: {kwargs}")

            # MCP session expects arguments as a dict
            result = await self.session.call_tool(tool_name, arguments=kwargs)

            # Calculate response time
            response_time = time.time() - start_time

            # Update successful requests count first
            self.metrics.successful_requests += 1
            self._update_response_time(response_time)

            # Extract the actual result content for the AI agent
            if result.content and len(result.content) > 0:
                # Combine all text content from the response
                text_content = "\n".join([content.text for content in result.content if hasattr(content, 'text')])
            else:
                text_content = ""

            success_result = {
                "success": True,
                "result": text_content,  # AI agent expects "result" field
                "content": [{"text": content.text} for content in result.content] if result.content else [],  # Keep original for compatibility
                "tool_name": tool_name,
                "arguments": kwargs,
                "response_time": response_time,
            }

            logger.debug(f"Tool {tool_name} executed successfully in {response_time:.3f}s")
            return success_result

        except Exception as e:
            error_msg = f"Failed to call tool {tool_name}: {e}"
            logger.error(error_msg)

            self.metrics.failed_requests += 1
            self._notify_error(e)

            error_result = {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "arguments": kwargs,
                "response_time": time.time() - start_time,
            }

            return error_result

    def _update_response_time(self, response_time: float) -> None:
        """Update average response time metric."""
        if self.metrics.successful_requests == 1:
            self.metrics.average_response_time = response_time
        else:
            # Moving average
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.successful_requests - 1) + response_time)
                / self.metrics.successful_requests
            )

    async def wait_for_healthy(self, timeout: float = 30.0) -> bool:
        """
        Wait for the connection to become healthy.

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            True if connection becomes healthy, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.state == ConnectionState.HEALTHY:
                return True
            elif self.state in [ConnectionState.FAILED, ConnectionState.DISCONNECTED]:
                return False

            await asyncio.sleep(0.1)

        return False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to MCP server."""
        return self.state in [ConnectionState.CONNECTED, ConnectionState.HEALTHY]

    @property
    def is_healthy(self) -> bool:
        """Check if the connection is healthy."""
        return self.state == ConnectionState.HEALTHY

    def get_server_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the connected server."""
        uptime = datetime.now() - self._start_time

        return {
            "name": self.config.name,
            "url": self.config.url,
            "endpoint": self.config.endpoint,
            "state": self.state.value,
            "connected": self.is_connected,
            "healthy": self.is_healthy,
            "tools_count": len(self.tools),
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics.to_dict(),
        }

    def get_metrics(self) -> ConnectionMetrics:
        """Get current connection metrics."""
        # Update uptime percentage
        uptime = datetime.now() - self._start_time
        if self.metrics.last_connection_time:
            connected_time = datetime.now() - self.metrics.last_connection_time
            self.metrics.uptime_percentage = min(100.0, (connected_time.total_seconds() / uptime.total_seconds()) * 100)

        return self.metrics

    async def reset_connection(self) -> bool:
        """Reset the connection (disconnect and reconnect)."""
        logger.info(f"Resetting connection to {self.config.name}")
        await self.disconnect()
        return await self.connect()

    def __repr__(self) -> str:
        """String representation of the client."""
        return f"MCPClient(name='{self.config.name}', state='{self.state.value}', tools={len(self.tools)})"