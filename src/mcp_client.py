"""
MCP Client - Professional SSE Transport for Model Context Protocol servers.

This module provides a robust client for connecting to MCP servers using SSE (Server-Sent Events)
transport with comprehensive error handling and logging.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import nest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

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


class MCPClient:
    """
    Professional MCP Client with SSE Transport.

    This client provides a robust connection to MCP servers using Server-Sent Events
    transport with comprehensive error handling, logging, and resource management.

    Features:
        - Automatic connection management
        - Tool discovery and schema validation
        - Robust error handling and logging
        - Async context manager support
        - Resource cleanup

    Example:
        >>> config = MCPServerConfig(name="MyServer", url="http://localhost:8080")
        >>> async with MCPClient(config) as client:
        ...     tools = await client.list_tools()
        ...     result = await client.call_tool("tool_name", param="value")
    """

    def __init__(self, config: MCPServerConfig) -> None:
        """
        Initialize the MCP client.

        Args:
            config: Server configuration including URL and connection settings
        """
        self.config = config
        self.session: Optional[ClientSession] = None
        self.connected = False
        self.tools: List[Any] = []
        self.read_stream = None
        self.write_stream = None
        self._context_manager = None

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> bool:
        """
        Connect to the MCP server using SSE transport.

        Returns:
            True if connection successful, False otherwise

        Raises:
            MCPConnectionError: If connection fails after retries
        """
        try:
            logger.info(f"Connecting to MCP server: {self.config.url}{self.config.endpoint}")

            # Create SSE connection
            sse_url = f"{self.config.url.rstrip('/')}{self.config.endpoint}"
            self._context_manager = sse_client(sse_url)
            self.read_stream, self.write_stream = await self._context_manager.__aenter__()

            # Create and initialize MCP session
            self.session = ClientSession(self.read_stream, self.write_stream)
            await self.session.__aenter__()
            await self.session.initialize()

            # Load available tools
            await self._load_tools()

            self.connected = True
            logger.info(f"Successfully connected to MCP server: {self.config.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self._cleanup()
            raise MCPConnectionError(f"Failed to connect to {self.config.name}: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and cleanup resources."""
        await self._cleanup()
        logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def _cleanup(self) -> None:
        """Clean up all resources."""
        self.connected = False

        # Cleanup session
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            finally:
                self.session = None

        # Cleanup context manager
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing context manager: {e}")
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

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.

        Returns:
            List of tool definitions with name, description, and full inputSchema

        Raises:
            MCPConnectionError: If not connected to server
        """
        if not self.connected:
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
        if not self.connected:
            raise MCPConnectionError("Not connected to MCP server. Call connect() first.")

        try:
            if self.session is None:
                raise MCPConnectionError("Session not initialized")

            logger.info(f"Calling tool: {tool_name} with arguments: {kwargs}")

            # MCP session expects arguments as a dict
            result = await self.session.call_tool(tool_name, arguments=kwargs)

            success_result = {
                "success": True,
                "content": [{"text": content.text} for content in result.content] if result.content else [],
                "tool_name": tool_name,
                "arguments": kwargs,
            }

            logger.info(f"Tool {tool_name} executed successfully")
            return success_result

        except Exception as e:
            error_msg = f"Failed to call tool {tool_name}: {e}"
            logger.error(error_msg)

            error_result = {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "arguments": kwargs
            }

            return error_result

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to MCP server."""
        return self.connected

    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the connected server."""
        return {
            "name": self.config.name,
            "url": self.config.url,
            "endpoint": self.config.endpoint,
            "connected": self.connected,
            "tools_count": len(self.tools)
        }