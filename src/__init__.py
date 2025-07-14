"""
Pipeline Toolkit - AI-powered MCP client for intelligent tool selection and query processing.

This package provides a generic MCP (Model Context Protocol) client that uses Gemini AI
to intelligently select and call tools from any MCP server based on natural language queries.
"""

__version__ = "0.1.0"
__author__ = "Ozan Unsal"
__email__ = "ounsal@redhat.com"

from src.ai_agent import AIAgent
from src.mcp_client import MCPClient
from src.config import MCPServerConfig, Config, load_config
from src.exceptions import PipelineToolkitError, MCPConnectionError, GeminiError, ToolExecutionError, ConfigurationError

__all__ = [
    "AIAgent",
    "MCPClient",
    "MCPServerConfig",
    "Config",
    "load_config",
    "PipelineToolkitError",
    "MCPConnectionError",
    "GeminiError",
    "ToolExecutionError",
    "ConfigurationError",
]