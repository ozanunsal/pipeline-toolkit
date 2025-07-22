"""
Enhanced Configuration Management - Advanced Configuration Loading and Validation.

This module provides robust configuration management with comprehensive validation,
environment variable support, schema validation, and intelligent defaults.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import re
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    """Configuration for Gemini AI integration with enhanced validation."""
    api_key: str
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    timeout: int = 30
    max_tokens: int = 4096
    max_retries: int = 3

    def __post_init__(self):
        """Validate Gemini configuration."""
        if not self.api_key:
            raise ValueError("Gemini API key is required")

        if not isinstance(self.temperature, (int, float)) or not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")

        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise ValueError("Timeout must be a positive integer")

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("Max tokens must be a positive integer")

    def is_valid_api_key(self) -> bool:
        """Validate API key format."""
        # Basic API key format validation
        return bool(self.api_key and len(self.api_key) > 10 and self.api_key.strip())


@dataclass
class MCPServerConfig:
    """Enhanced MCP server configuration with support for HTTP and stdio connections."""
    name: str
    connection_type: str = "http"  # "http" or "stdio"
    enabled: bool = True
    description: str = ""
    timeout: int = 30
    max_retries: int = 3
    health_check_interval: int = 30
    connection_timeout: int = 10

    # HTTP connection fields
    url: Optional[str] = None
    endpoint: str = "/sse"

    # Stdio connection fields
    command: Optional[str] = None
    args: Optional[List[str]] = None
    working_directory: Optional[str] = None
    environment: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Validate MCP server configuration based on connection type."""
        if not self.name or not self.name.strip():
            raise ValueError("Server name is required")

        if self.connection_type not in ["http", "stdio"]:
            raise ValueError(f"Connection type must be 'http' or 'stdio', got: {self.connection_type}")

        # Validate based on connection type
        if self.connection_type == "http":
            self._validate_http_config()
        elif self.connection_type == "stdio":
            self._validate_stdio_config()

        # Common validations
        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise ValueError("Timeout must be a positive integer")

        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("Max retries must be a non-negative integer")

        if not isinstance(self.health_check_interval, int) or self.health_check_interval <= 0:
            raise ValueError("Health check interval must be a positive integer")

    def _validate_http_config(self):
        """Validate HTTP-specific configuration."""
        if not self.url or not self.url.strip():
            raise ValueError("Server URL is required for HTTP connections")

        if not self._is_valid_url(self.url):
            raise ValueError(f"Invalid URL format: {self.url}")

    def _validate_stdio_config(self):
        """Validate stdio-specific configuration."""
        if not self.command or not self.command.strip():
            raise ValueError("Command is required for stdio connections")

        if self.working_directory and not Path(self.working_directory).exists():
            logger.warning(f"Working directory does not exist: {self.working_directory}")

        if self.args is not None and not isinstance(self.args, list):
            raise ValueError("Args must be a list of strings")

        if self.environment is not None and not isinstance(self.environment, dict):
            raise ValueError("Environment must be a dictionary of string key-value pairs")

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        return bool(url_pattern.match(url))

    def get_full_url(self) -> str:
        """Get the full URL including endpoint (HTTP connections only)."""
        if self.connection_type != "http":
            raise ValueError("get_full_url() is only valid for HTTP connections")
        if not self.url:
            raise ValueError("URL is required for HTTP connections")
        return f"{self.url.rstrip('/')}{self.endpoint}"

    def get_command_with_args(self) -> List[str]:
        """Get the full command with arguments (stdio connections only)."""
        if self.connection_type != "stdio":
            raise ValueError("get_command_with_args() is only valid for stdio connections")
        if not self.command:
            raise ValueError("Command is required for stdio connections")

        cmd_parts = [self.command]
        if self.args:
            cmd_parts.extend(self.args)
        return cmd_parts

    def is_http_connection(self) -> bool:
        """Check if this is an HTTP connection."""
        return self.connection_type == "http"

    def is_stdio_connection(self) -> bool:
        """Check if this is a stdio connection."""
        return self.connection_type == "stdio"


@dataclass
class LoggingConfig:
    """Enhanced logging configuration."""
    level: str = "INFO"
    file: str = "logs/pipeline_toolkit.log"
    max_log_lines: int = 1000
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_console: bool = False

    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")

        if not isinstance(self.max_log_lines, int) or self.max_log_lines <= 0:
            raise ValueError("Max log lines must be a positive integer")

        if not isinstance(self.max_file_size, int) or self.max_file_size <= 0:
            raise ValueError("Max file size must be a positive integer")

    def get_log_file_path(self) -> Path:
        """Get the absolute path to the log file."""
        return Path(self.file).resolve()


@dataclass
class UIConfig:
    """Enhanced UI configuration."""
    show_banner: bool = True
    show_tool_preview: bool = True
    max_tools_preview: int = 5
    theme: str = "default"
    show_progress_bars: bool = True
    show_performance_metrics: bool = True
    enable_colors: bool = True
    prompt_style: str = "fancy"

    def __post_init__(self):
        """Validate UI configuration."""
        if not isinstance(self.max_tools_preview, int) or self.max_tools_preview <= 0:
            raise ValueError("Max tools preview must be a positive integer")

        valid_themes = ["default", "dark", "light", "minimal"]
        if self.theme not in valid_themes:
            raise ValueError(f"Invalid theme: {self.theme}. Must be one of {valid_themes}")

        valid_prompt_styles = ["simple", "fancy", "minimal"]
        if self.prompt_style not in valid_prompt_styles:
            raise ValueError(f"Invalid prompt style: {self.prompt_style}. Must be one of {valid_prompt_styles}")


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    enable_api_key_validation: bool = True
    mask_sensitive_logs: bool = True
    max_request_size: int = 1024 * 1024  # 1MB
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    allow_insecure_connections: bool = False

    def __post_init__(self):
        """Validate security configuration."""
        if not isinstance(self.max_request_size, int) or self.max_request_size <= 0:
            raise ValueError("Max request size must be a positive integer")

        if not isinstance(self.rate_limit_requests, int) or self.rate_limit_requests <= 0:
            raise ValueError("Rate limit requests must be a positive integer")


@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    max_concurrent_requests: int = 10
    connection_pool_size: int = 5
    request_timeout: int = 30
    retry_backoff_factor: float = 1.5
    max_retry_delay: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes

    def __post_init__(self):
        """Validate performance configuration."""
        if not isinstance(self.max_concurrent_requests, int) or self.max_concurrent_requests <= 0:
            raise ValueError("Max concurrent requests must be a positive integer")

        if not isinstance(self.connection_pool_size, int) or self.connection_pool_size <= 0:
            raise ValueError("Connection pool size must be a positive integer")

        if not isinstance(self.retry_backoff_factor, (int, float)) or self.retry_backoff_factor < 1.0:
            raise ValueError("Retry backoff factor must be >= 1.0")


@dataclass
class Config:
    """Enhanced main configuration class for Pipeline Toolkit."""
    gemini: GeminiConfig
    mcp_servers: List[MCPServerConfig]
    logging: LoggingConfig
    ui: UIConfig
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    def __post_init__(self):
        """Validate main configuration."""
        if not self.mcp_servers:
            raise ValueError("At least one MCP server must be configured")

        enabled_servers = [s for s in self.mcp_servers if s.enabled]
        if not enabled_servers:
            raise ValueError("At least one MCP server must be enabled")

    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get list of enabled MCP servers."""
        return [server for server in self.mcp_servers if server.enabled]

    def get_server_by_name(self, name: str) -> Optional[MCPServerConfig]:
        """Get server configuration by name."""
        for server in self.mcp_servers:
            if server.name == name:
                return server
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for debugging/export)."""
        config_dict = {
            "gemini": {
                "model": self.gemini.model,
                "temperature": self.gemini.temperature,
                "timeout": self.gemini.timeout,
                "max_tokens": self.gemini.max_tokens,
                "api_key": "***masked***" if self.security.mask_sensitive_logs else self.gemini.api_key
            },
            "mcp_servers": [
                {
                    "name": server.name,
                    "url": server.url,
                    "endpoint": server.endpoint,
                    "enabled": server.enabled,
                    "timeout": server.timeout,
                    "description": server.description
                }
                for server in self.mcp_servers
            ],
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file,
                "max_log_lines": self.logging.max_log_lines
            },
            "ui": {
                "show_banner": self.ui.show_banner,
                "show_tool_preview": self.ui.show_tool_preview,
                "theme": self.ui.theme
            }
        }
        return config_dict

    def validate(self) -> List[str]:
        """Comprehensive configuration validation."""
        issues = []

        # Validate Gemini config
        if not self.gemini.is_valid_api_key():
            issues.append("Invalid Gemini API key format")

        # Validate MCP servers
        server_names = [s.name for s in self.mcp_servers]
        if len(server_names) != len(set(server_names)):
            issues.append("Duplicate server names found")

        # Check for common issues
        for server in self.mcp_servers:
            if server.enabled and server.is_http_connection():
                if not server.url or not server.url.startswith(('http://', 'https://')):
                    issues.append(f"Server {server.name} URL should start with http:// or https://")

        return issues


class ConfigurationError(Exception):
    """Configuration-related error."""
    pass


class EnhancedConfigLoader:
    """Enhanced configuration loader with validation and environment support."""

    def __init__(self):
        self.config_file = self._get_config_file_path()
        self.env_prefix = "PIPELINE_TOOLKIT_"

    def _get_config_file_path(self) -> Path:
        """Get configuration file path from environment or default."""
        config_path = os.getenv('OLS_CONFIG_FILE', 'config/config.json')
        return Path(config_path).resolve()

    def load_config(self) -> Config:
        """Load and validate configuration from file and environment."""
        try:
            # Load base config from file
            config_dict = self._load_config_file()

            # Apply environment variable overrides
            config_dict = self._apply_environment_overrides(config_dict)

            # Create configuration objects
            config = self._create_config(config_dict)

            # Validate configuration
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
            raise ConfigurationError(f"Configuration file not found: {self.config_file}")

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            logger.debug(f"Loaded configuration from {self.config_file}")
            return config_dict

        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")

    def _apply_environment_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Gemini API key override
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if api_key:
            config_dict.setdefault('gemini', {})['api_key'] = api_key
            logger.debug("Applied API key from environment variable")

        # Model override
        model = os.getenv(f'{self.env_prefix}GEMINI_MODEL')
        if model:
            config_dict.setdefault('gemini', {})['model'] = model

        # Log level override
        log_level = os.getenv(f'{self.env_prefix}LOG_LEVEL')
        if log_level:
            config_dict.setdefault('logging', {})['level'] = log_level

        # Server URL override (for single server setups)
        server_url = os.getenv(f'{self.env_prefix}MCP_SERVER_URL')
        server_name = os.getenv(f'{self.env_prefix}MCP_SERVER_NAME', 'Default Server')
        if server_url:
            config_dict['mcp_servers'] = [{
                'name': server_name,
                'url': server_url,
                'enabled': True
            }]
            logger.debug("Applied MCP server configuration from environment")

        # Security settings
        if os.getenv(f'{self.env_prefix}ALLOW_INSECURE') == 'true':
            config_dict.setdefault('security', {})['allow_insecure_connections'] = True

        return config_dict

    def _create_config(self, config_dict: Dict[str, Any]) -> Config:
        """Create Config object from dictionary with enhanced validation."""
        try:
            # Create Gemini configuration
            gemini_dict = config_dict.get('gemini', {})
            if not gemini_dict.get('api_key'):
                # Try to get from environment as fallback
                api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ConfigurationError(
                        "Gemini API key is required. Set it in config.json or via "
                        "GEMINI_API_KEY/GOOGLE_API_KEY environment variable."
                    )
                gemini_dict['api_key'] = api_key

            gemini_config = GeminiConfig(**gemini_dict)

            # Create MCP server configurations
            mcp_servers = []
            for server_dict in config_dict.get('mcp_servers', []):
                # Only include enabled servers or if enabled field is not specified
                if server_dict.get('enabled', True):
                    try:
                        mcp_server = MCPServerConfig(**server_dict)
                        mcp_servers.append(mcp_server)
                        logger.debug(f"Added MCP server: {mcp_server.name}")
                    except Exception as e:
                        logger.warning(f"Invalid MCP server configuration {server_dict.get('name', 'unknown')}: {e}")

            if not mcp_servers:
                raise ConfigurationError("At least one valid enabled MCP server must be configured.")

            # Create logging configuration
            logging_dict = config_dict.get('logging', {})
            logging_config = LoggingConfig(**logging_dict)

            # Create UI configuration
            ui_dict = config_dict.get('ui', {})
            ui_config = UIConfig(**ui_dict)

            # Create security configuration
            security_dict = config_dict.get('security', {})
            security_config = SecurityConfig(**security_dict)

            # Create performance configuration
            performance_dict = config_dict.get('performance', {})
            performance_config = PerformanceConfig(**performance_dict)

            return Config(
                gemini=gemini_config,
                mcp_servers=mcp_servers,
                logging=logging_config,
                ui=ui_config,
                security=security_config,
                performance=performance_config
            )

        except Exception as e:
            raise ConfigurationError(f"Error creating configuration: {e}")

    def create_example_config(self, output_path: Optional[Path] = None) -> Path:
        """Create an example configuration file."""
        if output_path is None:
            output_path = Path("config/config.json.example")

        example_config = {
            "gemini": {
                "api_key": "your_gemini_api_key_here",
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.1,
                "timeout": 30,
                "max_tokens": 4096
            },
            "mcp_servers": [
                {
                    "name": "Example Server 1",
                    "url": "http://localhost:8080",
                    "endpoint": "/sse",
                    "timeout": 30,
                    "enabled": True,
                    "description": "Example MCP server"
                },
                {
                    "name": "Example Server 2",
                    "url": "http://localhost:8090",
                    "endpoint": "/sse",
                    "timeout": 30,
                    "enabled": False,
                    "description": "Another example MCP server"
                }
            ],
            "logging": {
                "level": "INFO",
                "file": "logs/pipeline_toolkit.log",
                "max_log_lines": 1000,
                "enable_console": False
            },
            "ui": {
                "show_banner": True,
                "show_tool_preview": True,
                "max_tools_preview": 5,
                "theme": "default",
                "show_progress_bars": True
            },
            "security": {
                "enable_api_key_validation": True,
                "mask_sensitive_logs": True,
                "allow_insecure_connections": False
            },
            "performance": {
                "max_concurrent_requests": 10,
                "connection_pool_size": 5,
                "enable_caching": True
            }
        }

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(example_config, f, indent=2)

        logger.info(f"Example configuration created at {output_path}")
        return output_path

    def validate_config_file(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Validate a configuration file and return validation results."""
        if config_path is None:
            config_path = self.config_file

        validation_result = {
            "valid": False,
            "issues": [],
            "warnings": [],
            "file_exists": config_path.exists(),
            "file_readable": False,
            "json_valid": False,
            "config_valid": False
        }

        if not validation_result["file_exists"]:
            validation_result["issues"].append(f"Configuration file not found: {config_path}")
            return validation_result

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            validation_result["file_readable"] = True
            validation_result["json_valid"] = True
        except json.JSONDecodeError as e:
            validation_result["issues"].append(f"Invalid JSON: {e}")
            return validation_result
        except Exception as e:
            validation_result["issues"].append(f"Cannot read file: {e}")
            return validation_result

        try:
            config = self._create_config(config_dict)
            config_issues = config.validate()
            validation_result["config_valid"] = True
            validation_result["warnings"].extend(config_issues)

            if not config_issues:
                validation_result["valid"] = True

        except Exception as e:
            validation_result["issues"].append(f"Configuration error: {e}")

        return validation_result


# Global configuration loader instance
_config_loader = EnhancedConfigLoader()


def load_config() -> Config:
    """Load configuration using the global loader."""
    return _config_loader.load_config()


def create_example_config(output_path: Optional[Path] = None) -> Path:
    """Create an example configuration file."""
    return _config_loader.create_example_config(output_path)


def validate_config_file(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Validate a configuration file."""
    return _config_loader.validate_config_file(config_path)


def get_config_file_path() -> Path:
    """Get the current configuration file path."""
    return _config_loader.config_file