from src.config import MCPServerConfig

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
    def __init__(self, config: MCPServerConfig):
        """
        Initialize the MCP client with a server configuration.

        Args:
            config: Server configuration including URL and connection settings
        """
        self.config = config
