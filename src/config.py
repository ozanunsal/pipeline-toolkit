"""
Configuration management for Pipeline Toolkit.

This module provides centralized configuration management with support for
JSON configuration files, environment variables, default values, and validation.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GeminiConfig:
    """Gemini AI configuration."""
    api_key: str
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    timeout: int = 30


@dataclass
class MCPServerConfig:
    """MCP Server configuration."""
    name: str
    url: str
    endpoint: str = "/sse"
    timeout: int = 30
    max_retries: int = 3
    enabled: bool = True
    description: str = ""


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/pipeline_bot.log"
    max_log_lines: int = 1000


@dataclass
class UIConfig:
    """UI configuration."""
    show_banner: bool = True
    show_tool_preview: bool = True
    max_tools_preview: int = 5


@dataclass
class Config:
    """Main configuration class for Pipeline Toolkit."""
    gemini: GeminiConfig
    mcp_servers: List[MCPServerConfig]
    logging: LoggingConfig
    ui: UIConfig


class ConfigLoader:
    """Configuration loader with JSON support and environment variable overrides."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, "config.json")

    def load_config(self) -> Config:
        """Load configuration from config.json with environment variable overrides."""
        # Load configuration from config.json
        config_data = self._load_json_file(self.config_file)

        # Apply environment variable overrides
        final_config = self._apply_env_overrides(config_data)

        # Validate and create Config object
        return self._create_config(final_config)

    def _load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load JSON configuration file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration file {file_path}: {e}")

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Gemini API key override
        if gemini_api_key := (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')):
            config.setdefault('gemini', {})['api_key'] = gemini_api_key

        # Gemini model override
        if gemini_model := os.getenv('GEMINI_MODEL'):
            config.setdefault('gemini', {})['model'] = gemini_model

        # Logging level override
        if log_level := os.getenv('LOG_LEVEL'):
            config.setdefault('logging', {})['level'] = log_level

        # Single MCP server override (for backward compatibility)
        if mcp_server_url := os.getenv('MCP_SERVER_URL'):
            override_server = {
                'name': os.getenv('MCP_SERVER_NAME', 'Environment MCP Server'),
                'url': mcp_server_url,
                'endpoint': os.getenv('MCP_SERVER_ENDPOINT', '/sse'),
                'enabled': True
            }
            config['mcp_servers'] = [override_server]

        return config

    def _create_config(self, config_dict: Dict[str, Any]) -> Config:
        """Create Config object from dictionary."""
        # Create Gemini configuration
        gemini_dict = config_dict.get('gemini', {})
        if not gemini_dict.get('api_key'):
            raise ValueError("Gemini API key is required. Set it in config.json or GEMINI_API_KEY environment variable.")

        gemini_config = GeminiConfig(
            api_key=gemini_dict['api_key'],
            model=gemini_dict.get('model', 'gemini-2.0-flash-exp'),
            temperature=gemini_dict.get('temperature', 0.1),
            timeout=gemini_dict.get('timeout', 30)
        )

        # Create MCP server configurations
        mcp_servers = []
        for server_dict in config_dict.get('mcp_servers', []):
            if server_dict.get('enabled', True):
                mcp_server = MCPServerConfig(
                    name=server_dict['name'],
                    url=server_dict['url'],
                    endpoint=server_dict.get('endpoint', '/sse'),
                    timeout=server_dict.get('timeout', 30),
                    max_retries=server_dict.get('max_retries', 3),
                    enabled=server_dict.get('enabled', True),
                    description=server_dict.get('description', '')
                )
                mcp_servers.append(mcp_server)

        if not mcp_servers:
            raise ValueError("At least one enabled MCP server must be configured.")

        # Create logging configuration
        logging_dict = config_dict.get('logging', {})
        logging_config = LoggingConfig(
            level=logging_dict.get('level', 'INFO'),
            file=logging_dict.get('file', 'logs/pipeline_bot.log'),
            max_log_lines=logging_dict.get('max_log_lines', 1000)
        )

        # Create UI configuration
        ui_dict = config_dict.get('ui', {})
        ui_config = UIConfig(
            show_banner=ui_dict.get('show_banner', True),
            show_tool_preview=ui_dict.get('show_tool_preview', True),
            max_tools_preview=ui_dict.get('max_tools_preview', 5)
        )

        return Config(
            gemini=gemini_config,
            mcp_servers=mcp_servers,
            logging=logging_config,
            ui=ui_config
        )


def load_config(config_dir: str = "config") -> Config:
    """
    Load configuration from JSON files with environment variable overrides.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        Config instance
    """
    loader = ConfigLoader(config_dir)
    return loader.load_config()


def create_sample_config(config_dir: str = "config") -> None:
    """Create sample configuration files."""
    os.makedirs(config_dir, exist_ok=True)

    # Create example user config
    example_file = os.path.join(config_dir, "config.json.example")
    config_file = os.path.join(config_dir, "config.json")

    if os.path.exists(example_file):
        print(f"Sample configuration already exists at {example_file}")
        print(f"📝 Copy {example_file} to {config_file} and edit as needed")
        print(f"🔑 Don't forget to set your Gemini API key in the config file!")
        return

    print(f"✅ Configuration files created in {config_dir}/")
    print(f"📝 Copy {example_file} to {config_file} and edit as needed")
    print(f"🔑 Don't forget to set your Gemini API key in the config file!")


if __name__ == "__main__":
    # Create sample configuration when run as script
    create_sample_config()