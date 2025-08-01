"""
Enhanced Configuration Management - Advanced Configuration Loading and Validation.

This module provides robust configuration management with comprehensive validation,
environment variable support, schema validation, and intelligent defaults.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MCPServerConfig:
    """Enhanced MCP Server Configuration with support for sse and stdio connections."""
    name: str
    connection_type: str = "sse"
    enabled: bool = True
    description: str = ""

    # SSE connection settings
    url: Optional[str] = None
    endpoint: str = "/sse"

    # stdio connection settings
    command: Optional[str] = None
    args: Optional[List[str]] = None
    working_directory: Optional[str] = None
    environment: Optional[Dict[str, str]] = None