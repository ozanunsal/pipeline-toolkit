"""
Custom exceptions for Pipeline Toolkit.
"""


class PipelineToolkitError(Exception):
    """Base exception for all Pipeline Toolkit errors."""
    pass


class MCPConnectionError(PipelineToolkitError):
    """Raised when MCP server connection fails."""
    pass


class GeminiError(PipelineToolkitError):
    """Raised when Gemini AI operations fail."""
    pass


class ToolExecutionError(PipelineToolkitError):
    """Raised when tool execution fails."""
    pass


class ConfigurationError(PipelineToolkitError):
    """Raised when configuration is invalid."""
    pass