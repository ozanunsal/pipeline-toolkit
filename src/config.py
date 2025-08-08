"""
Enhanced Configuration Management - Advanced Configuration Loading and Validation.

This module provides robust configuration management with comprehensive validation,
environment variable support, schema validation, and intelligent defaults.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Enhanced MCP Server Configuration with support for sse and stdio connections."""

    name: str
    connection_type: str = "sse"
    enabled: bool = False
    description: str = ""

    # SSE connection settings
    url: Optional[str] = None
    endpoint: str = "/sse"

    # stdio connection settings
    command: Optional[str] = None
    args: Optional[List[str]] = None
    working_directory: Optional[str] = None
    environment: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Validate MCP server configuration based on connection type."""
        if not self.name or not self.name.strip():
            raise ValueError("MCP server name is required.")

        if self.connection_type not in ["sse", "stdio"]:
            raise ValueError(
                f"Connection type must be either 'sse' or 'stdio', got: {self.connection_type}"
            )

        # Validate based on connection type
        if self.connection_type == "sse":
            self._validate_sse_connection()
        elif self.connection_type == "stdio":
            self._validate_stdio_connection()

    def _validate_sse_connection(self):
        """Validate SSE connection settings."""
        if not self.url or not self.url.strip():
            raise ValueError("Server URL is required for SSE connections.")
        if not self._is_valid_url(self.url):
            raise ValueError(f"Invalid URL format: {self.url}")

    def _validate_stdio_connection(self):
        """Validate stdio connection settings."""
        if not self.command or not self.command.strip():
            raise ValueError("Command is required for stdio connections.")
        if self.working_directory and not Path(self.working_directory).exists():
            logger.warning(
                f"Working directory does not exist: {self.working_directory}"
            )
        if self.args is not None and not isinstance(self.args, list):
            raise ValueError("Args must be a list of strings.")
        if self.environment is not None and not isinstance(self.environment, dict):
            raise ValueError(
                "Environment must be a dictionary of string ket-value pairs"
            )
        # Normalize args to list of strings
        if self.args is None:
            self.args = []
        else:
            self.args = [str(a) for a in self.args]

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        return bool(url_pattern.match(url))

    def is_sse_connection(self) -> bool:
        """Check if this is an SSE connection."""
        return self.connection_type == "sse"

    def is_stdio_connection(self) -> bool:
        """Check if this is a stdio connection."""
        return self.connection_type == "stdio"


@dataclass
class Config:
    """Enhanced main configuration class for Pipeline Toolkit."""

    mcp_servers: List[MCPServerConfig]
    log_file: Optional[str] = None
    gemini: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate main configuration."""
        if not self.mcp_servers:
            raise ValueError("At least one MCP server must be configured.")
        enabled_servers = [server for server in self.mcp_servers if server.enabled]
        if not enabled_servers:
            raise ValueError("No enabled MCP servers found in configuration.")

    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get all enabled MCP servers."""
        return [server for server in self.mcp_servers if server.enabled]

    def get_server_by_name(self, name: str) -> Optional[MCPServerConfig]:
        """Get server configuration by name."""
        for server in self.mcp_servers:
            if server.name == name:
                return server
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "mcp_servers": [
                {
                    "name": server.name,
                    "enabled": server.enabled,
                    "description": server.description,
                    "connection_type": server.connection_type,
                    "url": server.url,
                    "endpoint": server.endpoint,
                    "command": server.command,
                }
                for server in self.mcp_servers
            ],
            "log_file": self.log_file,
            "gemini": self.gemini,
        }
        return config_dict

    def validate(self) -> List[str]:
        """Comprehensive validation of the configuration."""
        issues = []

        # Validate MCP servers
        server_names = [s.name for s in self.mcp_servers]
        if len(server_names) != len(set(server_names)):
            issues.append("Duplicate server names found")

        for server in self.mcp_servers:
            if server.enabled and server.is_sse_connection():
                if not server.url or not server.url.startswith(("http", "https")):
                    issues.append(
                        f"Server {server.name} URL should start with http:// or https://"
                    )
        return issues


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""

    pass


class EnhancedConfigLoader:
    """Enhanced configuration loader with validation and environment support."""

    def __init__(self):
        self.config_file = self._get_config_file_path()
        self.env_prefix = "PIPELINE_TOOLKIT_"

    def _get_config_file_path(self) -> Path:
        """Get the path to the configuration file."""
        config_path = os.getenv("PIPELINE_TOOLKIT_CONFIG_PATH", "config/config.json")
        return Path(config_path).resolve()

    def load_config(self) -> Config:
        """Load and validate the configuration from file and environment."""
        try:
            config_dict = self._load_config_file()
            # config_dict = self._apply_environment_overrides(config_dict)
            config = self._create_config(config_dict)
            issues = config.validate()
            if issues:
                logger.warning(f"Configuration validation issues: {', '.join(issues)}")
            logger.info(f"Configuration loaded successfully from {self.config_file}")
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.config_file}"
            )
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            logger.debug(f"Loaded configuration from {self.config_file}")
            return config_dict

        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file: {e}")

    # def _apply_environment_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    #     """Apply environment variable overrides to the configuration."""
    #     for key, value in config_dict.items():
    #         env_key = f"{self.env_prefix}{key.upper()}"
    #         if env_key in os.environ:
    #             config_dict[key] = os.environ[env_key]
    #     return config_dict

    def _create_config(self, config_dict: Dict[str, Any]) -> Config:
        """Create a Config object from the configuration dictionary."""
        try:

            # Create MCP server configuration
            mcp_servers = []
            for server_dict in config_dict.get("mcp_servers", []):
                # Only include the enabled servers
                if server_dict.get("enabled", True):
                    try:
                        mcp_server = MCPServerConfig(**server_dict)
                        mcp_servers.append(mcp_server)
                        logger.debug(f"Added MCP server: {mcp_server.name}")
                    except Exception as e:
                        logger.warning(
                            f"Invalid MCP server configuration {server_dict.get('name', 'unknown')}: {e}"
                        )

            return Config(
                mcp_servers=mcp_servers,
                log_file=config_dict.get("log_file"),
                gemini=config_dict.get("gemini"),
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration object: {e}")


# Global config loader instance
_config_loader = EnhancedConfigLoader()


def load_config() -> Config:
    """Load configuration from using the global loader."""
    return _config_loader.load_config()
