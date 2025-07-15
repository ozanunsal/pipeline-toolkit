# Configuration Guide

This directory contains the configuration files for Pipeline Toolkit.

## Configuration Files

### `config.json` (Main configuration file)
Your main configuration file that contains all Pipeline Toolkit settings. Copy from `config.json.example` and customize as needed.

### `config.json.example`
Example configuration file showing how to customize your settings.

## Setup Instructions

1. **Copy the example configuration:**
   ```bash
   cp config/config.json.example config/config.json
   ```

2. **Edit your configuration:**
   ```bash
   nano config/config.json  # or use your favorite editor
   ```

3. **Set your Gemini API key:**
   Replace `"your_gemini_api_key_here"` with your actual Gemini API key.

4. **Configure your MCP servers:**
   - Update server URLs and ports as needed
   - Enable/disable servers by setting `"enabled": true/false`
   - Add new servers by adding entries to the `mcp_servers` array

## Configuration Structure

```json
{
  "gemini": {
    "api_key": "your_gemini_api_key_here",
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.1,
    "timeout": 30
  },
  "mcp_servers": [
    {
      "name": "Server Name",
      "url": "http://localhost:8080",
      "endpoint": "/sse",
      "timeout": 30,
      "max_retries": 3,
      "enabled": true,
      "description": "Description of what this server does"
    }
  ],
  "logging": {
    "level": "INFO",
    "file": "logs/pipeline_bot.log",
    "max_log_lines": 1000
  },
  "ui": {
    "show_banner": true,
    "show_tool_preview": true,
    "max_tools_preview": 5
  }
}
```

## Environment Variable Overrides

You can override certain configuration values using environment variables:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` - Override Gemini API key
- `GEMINI_MODEL` - Override Gemini model
- `LOG_LEVEL` - Override logging level
- `MCP_SERVER_URL` - Override MCP servers (single server mode)
- `MCP_SERVER_NAME` - Override MCP server name
- `MCP_SERVER_ENDPOINT` - Override MCP server endpoint

## Configuration Options

### Gemini Settings
- `api_key`: Your Gemini AI API key (required)
- `model`: Gemini model to use (default: "gemini-2.0-flash-exp")
- `temperature`: Controls randomness (0.0-1.0, default: 0.1)
- `timeout`: Request timeout in seconds (default: 30)

### MCP Server Settings
- `name`: Display name for the server
- `url`: Server URL including protocol and port
- `endpoint`: SSE endpoint (default: "/sse")
- `timeout`: Connection timeout (default: 30)
- `max_retries`: Maximum retry attempts (default: 3)
- `enabled`: Whether to connect to this server (default: true)
- `description`: Optional description of the server

### Logging Settings
- `level`: Log level (DEBUG, INFO, WARNING, ERROR)
- `file`: Log file path (default: "logs/pipeline_bot.log")
- `max_log_lines`: Maximum lines to show when displaying logs (default: 1000)

### UI Settings
- `show_banner`: Show startup banner (default: true)
- `show_tool_preview`: Show available tools on startup (default: true)
- `max_tools_preview`: Maximum tools to preview (default: 5)

## Security Notes

- Never commit your actual API keys to version control
- Use environment variables for sensitive configuration in production
- Keep your `config.json` file secure and don't share it publicly
- The example file shows the structure but uses placeholder values