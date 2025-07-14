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

## Configuration Loading

The configuration system works as follows:

1. **Main Configuration**: Loaded from `config.json`
2. **Environment Overrides**: Applied from environment variables
3. **Final Configuration**: Result used by the application

Environment variables override the JSON configuration settings.

## Troubleshooting

### "Gemini API key is required" Error
- Make sure you've set your API key in `config.json`
- Or set the `GEMINI_API_KEY` environment variable

### "No MCP servers connected" Error
- Check that your MCP servers are running
- Verify the URLs and ports in your configuration
- Make sure at least one server has `"enabled": true`

### "Configuration file not found" Error
- Make sure `config/config.json` exists
- Copy `config/config.json.example` to `config/config.json`
- Check that you're running the command from the correct directory

## Adding New MCP Servers

To add a new MCP server:

1. Add a new entry to the `mcp_servers` array in your `config.json`:
   ```json
   {
     "name": "My Custom Server",
     "url": "http://localhost:9000",
     "endpoint": "/sse",
     "timeout": 30,
     "max_retries": 3,
     "enabled": true,
     "description": "Custom MCP server for specific tasks"
   }
   ```

2. Restart the Pipeline Bot

The system will automatically discover and use tools from all enabled servers.