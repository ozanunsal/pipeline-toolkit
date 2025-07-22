"""
Enhanced Exception Classes - Comprehensive Error Handling for Pipeline Toolkit.

This module provides a hierarchy of specialized exception classes for different
components of the Pipeline Toolkit with enhanced error context and debugging capabilities.
"""

import traceback
from typing import Any, Dict, Optional, List
from datetime import datetime


class PipelineToolkitError(Exception):
    """
    Base exception class for all Pipeline Toolkit errors.

    Provides enhanced error context and debugging capabilities.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        inner_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.inner_exception = inner_exception
        self.timestamp = datetime.now()
        self.traceback_info = traceback.format_exc() if inner_exception else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "inner_exception": str(self.inner_exception) if self.inner_exception else None,
            "traceback": self.traceback_info
        }

    def __str__(self) -> str:
        """Enhanced string representation with context."""
        base_msg = f"{self.error_code}: {self.message}"
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            base_msg += f" (Context: {context_str})"
        return base_msg


class ConfigurationError(PipelineToolkitError):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_section: Optional[str] = None,
        invalid_keys: Optional[List[str]] = None,
        **kwargs
    ):
        context = {
            "config_file": config_file,
            "config_section": config_section,
            "invalid_keys": invalid_keys
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class MCPConnectionError(PipelineToolkitError):
    """Exception raised for MCP connection-related errors."""

    def __init__(
        self,
        message: str,
        server_name: Optional[str] = None,
        server_url: Optional[str] = None,
        connection_attempt: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ):
        context = {
            "server_name": server_name,
            "server_url": server_url,
            "connection_attempt": connection_attempt,
            "max_retries": max_retries
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="MCP_CONNECTION_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class ToolExecutionError(PipelineToolkitError):
    """Exception raised for tool execution errors."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        server_name: Optional[str] = None,
        execution_time: Optional[float] = None,
        **kwargs
    ):
        context = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "server_name": server_name,
            "execution_time": execution_time
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="TOOL_EXECUTION_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class GeminiError(PipelineToolkitError):
    """Exception raised for Gemini AI-related errors."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
        rate_limited: bool = False,
        quota_exceeded: bool = False,
        **kwargs
    ):
        context = {
            "model": model,
            "request_id": request_id,
            "rate_limited": rate_limited,
            "quota_exceeded": quota_exceeded
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="GEMINI_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class ValidationError(PipelineToolkitError):
    """Exception raised for validation errors."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        validation_rules: Optional[List[str]] = None,
        **kwargs
    ):
        context = {
            "field_name": field_name,
            "field_value": str(field_value) if field_value is not None else None,
            "expected_type": expected_type,
            "validation_rules": validation_rules
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class AuthenticationError(PipelineToolkitError):
    """Exception raised for authentication-related errors."""

    def __init__(
        self,
        message: str,
        auth_type: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key_valid: Optional[bool] = None,
        **kwargs
    ):
        context = {
            "auth_type": auth_type,
            "user_id": user_id,
            "api_key_valid": api_key_valid
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="AUTH_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class RateLimitError(PipelineToolkitError):
    """Exception raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        rate_limit: Optional[int] = None,
        current_usage: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        **kwargs
    ):
        context = {
            "rate_limit": rate_limit,
            "current_usage": current_usage,
            "reset_time": reset_time.isoformat() if reset_time else None
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="RATE_LIMIT_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class PerformanceError(PipelineToolkitError):
    """Exception raised for performance-related issues."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        duration: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs
    ):
        context = {
            "operation": operation,
            "duration": duration,
            "threshold": threshold
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="PERFORMANCE_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class SecurityError(PipelineToolkitError):
    """Exception raised for security-related issues."""

    def __init__(
        self,
        message: str,
        security_rule: Optional[str] = None,
        attempted_action: Optional[str] = None,
        risk_level: Optional[str] = None,
        **kwargs
    ):
        context = {
            "security_rule": security_rule,
            "attempted_action": attempted_action,
            "risk_level": risk_level
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class ResourceError(PipelineToolkitError):
    """Exception raised for resource-related issues."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        available: Optional[bool] = None,
        **kwargs
    ):
        context = {
            "resource_type": resource_type,
            "resource_id": resource_id,
            "available": available
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="RESOURCE_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


class NetworkError(PipelineToolkitError):
    """Exception raised for network-related issues."""

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        **kwargs
    ):
        context = {
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time": response_time
        }
        context.update(kwargs.get("context", {}))

        super().__init__(
            message,
            error_code="NETWORK_ERROR",
            context=context,
            inner_exception=kwargs.get("inner_exception")
        )


def handle_exception(exception: Exception, context: Optional[Dict[str, Any]] = None) -> PipelineToolkitError:
    """
    Convert a generic exception to a Pipeline Toolkit exception with enhanced context.

    Args:
        exception: The original exception
        context: Additional context to include

    Returns:
        PipelineToolkitError: Enhanced exception with context
    """
    if isinstance(exception, PipelineToolkitError):
        return exception

    # Map common exception types to Pipeline Toolkit exceptions
    exception_mapping = {
        ConnectionError: MCPConnectionError,
        TimeoutError: NetworkError,
        ValueError: ValidationError,
        KeyError: ConfigurationError,
        PermissionError: SecurityError,
        OSError: ResourceError,
    }

    exception_class = exception_mapping.get(type(exception), PipelineToolkitError)

    return exception_class(
        message=str(exception),
        context=context,
        inner_exception=exception
    )


def format_exception_for_user(exception: PipelineToolkitError) -> str:
    """
    Format exception for user-friendly display.

    Args:
        exception: Pipeline Toolkit exception

    Returns:
        str: User-friendly error message
    """
    user_messages = {
        "CONFIG_ERROR": "Configuration Error: Please check your configuration file and settings.",
        "MCP_CONNECTION_ERROR": "Connection Error: Unable to connect to MCP server. Please check your server configuration and network connectivity.",
        "TOOL_EXECUTION_ERROR": "Tool Error: The requested tool could not be executed successfully.",
        "GEMINI_ERROR": "AI Service Error: There was an issue with the AI service. Please check your API key and try again.",
        "VALIDATION_ERROR": "Input Error: The provided input is invalid or incomplete.",
        "AUTH_ERROR": "Authentication Error: Please check your credentials and permissions.",
        "RATE_LIMIT_ERROR": "Rate Limit Exceeded: Too many requests. Please wait before trying again.",
        "PERFORMANCE_ERROR": "Performance Issue: The operation took longer than expected.",
        "SECURITY_ERROR": "Security Error: The requested action violates security policies.",
        "RESOURCE_ERROR": "Resource Error: Required resources are not available.",
        "NETWORK_ERROR": "Network Error: Unable to connect to the service. Please check your internet connection."
    }

    base_message = user_messages.get(exception.error_code, "An unexpected error occurred.")

    if exception.context:
        relevant_context = []
        if "server_name" in exception.context and exception.context["server_name"]:
            relevant_context.append(f"Server: {exception.context['server_name']}")
        if "tool_name" in exception.context and exception.context["tool_name"]:
            relevant_context.append(f"Tool: {exception.context['tool_name']}")

        if relevant_context:
            base_message += f" ({', '.join(relevant_context)})"

    return base_message


def log_exception(exception: PipelineToolkitError, logger) -> None:
    """
    Log exception with structured information.

    Args:
        exception: Pipeline Toolkit exception to log
        logger: Logger instance
    """
    exception_dict = exception.to_dict()

    logger.error(
        f"{exception.error_code}: {exception.message}",
        extra={"exception_context": exception_dict}
    )

    if exception.inner_exception and exception.traceback_info:
        logger.debug(f"Traceback: {exception.traceback_info}")


# Backward compatibility aliases
MCPError = MCPConnectionError  # Legacy alias
AIError = GeminiError  # Legacy alias