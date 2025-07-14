# Pipeline Toolkit

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Alpha-orange)

**Pipeline Toolkit** is an AI-powered MCP (Model Context Protocol) client that uses Google's Gemini AI to intelligently select and execute tools from any MCP server based on natural language queries.

## 🚀 Features

- **🧠 AI-Powered Tool Selection**: Uses Gemini AI to intelligently understand queries and select appropriate tools
- **🔗 Universal MCP Client**: Works with any MCP server that implements the standard protocol
- **📝 Natural Language Interface**: Ask questions in plain English and get intelligent responses
- **🛠️ Dynamic Tool Discovery**: Automatically discovers and adapts to available tools from connected MCP servers
- **📊 Comprehensive Logging**: File-based logging with on-demand log viewing
- **⚡ Async Architecture**: Built with modern async/await patterns for optimal performance
- **🎯 Professional CLI**: Beautiful, user-friendly command-line interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   AI Agent      │───▶│   MCP Client    │
│ Natural Language│    │ (Gemini AI)     │    │ (MCP Protocol)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲                       ▲
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Tool Selection  │    │   MCP Server    │
                       │ & Execution     │    │ (e.g., BrewMCP) │
                       └─────────────────┘    └─────────────────┘
```

The Pipeline Toolkit acts as an intelligent bridge between users and MCP servers:

1. **User Input**: Natural language queries from users
2. **AI Processing**: Gemini AI analyzes the query and selects appropriate tools
3. **Tool Execution**: Selected tools are executed on the target MCP server
4. **Response Generation**: Results are formatted and returned to the user

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Google Gemini API key
- Running MCP server (e.g., BrewMCP, JiraMCP, etc.)

### Method 1: Using pip (Recommended)

```bash
pip install pipeline-toolkit
```

### Method 2: From Source

```bash
git clone https://github.com/your-org/pipeline-toolkit.git
cd pipeline-toolkit
pip install -e .
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (with defaults)
GEMINI_MODEL=gemini-2.0-flash-exp
MCP_SERVER_URL=http://localhost:8080
MCP_SERVER_NAME=My MCP Server
MCP_SERVER_ENDPOINT=/sse
LOG_LEVEL=INFO
```

### Configuration File

Generate a sample configuration:

```bash
python -m src.config
```

This creates a `.env.example` file that you can customize.

## 🚀 Usage

### Interactive Mode

Start the interactive CLI:

```bash
pipeline-bot
```

Or if installed from source:

```bash
python src/cli.py
```

### Command Line Mode

Execute a single query:

```bash
pipeline-bot --query "Get tag info for release-1.0"
```

### Example Queries

Once connected to an MCP server, you can ask natural language questions:

```bash
🔍 Enter your query: Get tag info for release-1.0
🔍 Enter your query: List all builds in the main-branch tag
🔍 Enter your query: Show me packages containing 'kernel'
🔍 Enter your query: What is the latest version of package-name?
```

### Special Commands

- `help` - Show available commands
- `tools` - List all available tools from connected MCP servers
- `stats` - Show agent statistics
- `show me the logs` - View recent log entries
- `exit` or `quit` - Exit the application

## 🛠️ Development

### Project Structure

```
pipeline-toolkit/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── ai_agent.py          # Gemini AI integration
│   ├── mcp_client.py        # MCP protocol client
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration management
│   └── exceptions.py        # Custom exceptions
├── config/                  # Configuration files
│   ├── config.json          # Main configuration
│   ├── config.json.example  # Example configuration
│   └── README.md           # Configuration guide
├── logs/                    # Log files
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
└── README.md               # This file
```

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/pipeline-toolkit.git
cd pipeline-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/
```

### Adding New Features

1. **New MCP Servers**: The toolkit automatically adapts to any MCP server
2. **Custom Tools**: Add new tools by implementing them in your MCP server
3. **AI Models**: Modify `src/ai_agent.py` to use different AI models
4. **Extensions**: Use the modular architecture to add new functionality

## 📚 API Reference

### AIAgent Class

```python
from src import AIAgent

agent = AIAgent(model="gemini-2.0-flash-exp")
result = await agent.process_query("Your query here")
```

### MCPClient Class

```python
from src import MCPClient, MCPServerConfig

config = MCPServerConfig(
    name="My Server",
    url="http://localhost:8080"
)

async with MCPClient(config) as client:
    tools = await client.list_tools()
    result = await client.call_tool("tool_name", param="value")
```

## 🔍 How It Works

### Tool Discovery

1. **Connection**: Pipeline Toolkit connects to your MCP server
2. **Discovery**: Automatically discovers available tools and their schemas
3. **Conversion**: Converts MCP tool schemas to Gemini AI function declarations
4. **Registration**: Registers tools with the AI agent for intelligent selection

### Query Processing

1. **Input**: User provides natural language query
2. **Analysis**: Gemini AI analyzes the query and intent
3. **Selection**: AI selects appropriate tools based on query content
4. **Execution**: Selected tools are executed with extracted parameters
5. **Response**: Results are formatted and returned to user

### Parameter Extraction

The AI agent intelligently extracts parameters from natural language:

- **"Get tag info for release-1.0"** → `get_brew_tag_info(tag_name="release-1.0")`
- **"List builds in main-branch"** → `list_brew_builds(tag_name="main-branch")`
- **"Show package with ID 123"** → `get_package_info(package_id=123)`

## 🧪 Testing

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
pytest tests/integration/
```

### Manual Testing

1. Start an MCP server (e.g., BrewMCP)
2. Set your environment variables
3. Run the Pipeline Toolkit
4. Test various queries

## 📊 Logging

All operations are logged to `logs/pipeline_bot.log` by default. View logs:

```bash
# In interactive mode
🔍 Enter your query: show me the logs

# Or directly
tail -f logs/pipeline_bot.log
```

## 🐛 Troubleshooting

### Common Issues

1. **"Gemini API key not found"**
   - Set `GEMINI_API_KEY` environment variable
   - Ensure the key is valid and has appropriate permissions

2. **"Failed to connect to MCP server"**
   - Verify MCP server is running
   - Check `MCP_SERVER_URL` configuration
   - Ensure server supports SSE transport

3. **"No tools available"**
   - Verify MCP server is properly configured
   - Check server logs for errors
   - Ensure tools are properly registered in the MCP server

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=DEBUG pipeline-bot
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI**: For providing the intelligent query processing capabilities
- **MCP Protocol**: For the standardized server communication protocol
- **Python Community**: For the excellent asyncio and typing support

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/your-org/pipeline-toolkit/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-org/pipeline-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/pipeline-toolkit/discussions)

---

**Pipeline Toolkit** - Making MCP servers accessible through natural language! 🚀
