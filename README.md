# Pipeline Toolkit - Enhanced Interactive MCP Client

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Beta-blue)
![Version](https://img.shields.io/badge/Version-0.2.0-brightgreen)

**Pipeline Toolkit** is an advanced, AI-powered MCP (Model Context Protocol) client featuring an interactive CLI, intelligent tool orchestration, and comprehensive conversation management using Google's Gemini AI.

## 🚀 Enhanced Features

### 🧠 Advanced AI Capabilities
- **Intelligent Tool Selection**: Uses Gemini AI with enhanced dynamic analysis for smart tool selection
- **Conversation Management**: Persistent conversation history with context-aware responses
- **Adaptive Learning**: Learns from usage patterns to improve tool recommendations
- **Multi-step Planning**: Advanced execution planning for complex queries

### 💻 Interactive CLI Experience
- **Rich UI Components**: Beautiful, colorful interface with progress bars and tables
- **Command History**: Persistent command history with auto-suggestions
- **Tab Completion**: Smart auto-completion for commands and parameters
- **Real-time Feedback**: Live status updates and performance metrics

### 🔗 Robust MCP Integration
- **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **Dual Connection Types**: Support for both HTTP and stdio (local process) connections
- **Health Monitoring**: Automatic health checks and connection recovery
- **Performance Tracking**: Detailed metrics and performance analytics
- **Circuit Breaker**: Intelligent failover and retry mechanisms

### 🛡️ Enterprise-Grade Features
- **Enhanced Security**: Comprehensive validation and secure credential handling
- **Configurable Logging**: Structured logging with multiple output formats
- **Error Handling**: Detailed exception handling with user-friendly messages
- **Performance Monitoring**: Real-time performance metrics and optimization

## 🏗️ Enhanced Architecture

```mermaid
graph TB
    User[User] --> CLI[Interactive CLI]
    CLI --> Agent[Enhanced AI Agent]
    Agent --> Analyzer[Dynamic Tool Analyzer]
    Agent --> Context[Conversation Context]
    Agent --> Gemini[Gemini AI]

    Agent --> Client1[MCP Client 1]
    Agent --> Client2[MCP Client 2]
    Agent --> ClientN[MCP Client N]

    Client1 --> Server1[MCP Server 1]
    Client2 --> Server2[MCP Server 2]
    ClientN --> ServerN[MCP Server N]

    CLI --> Config[Enhanced Config]
    CLI --> Logging[Advanced Logging]
    CLI --> Metrics[Performance Metrics]
```

### Key Components

1. **Interactive CLI**: Rich terminal interface with advanced features
2. **Enhanced AI Agent**: Intelligent conversation and tool management
3. **Advanced MCP Clients**: Robust connection management with health monitoring
4. **Dynamic Tool Analyzer**: Smart tool categorization and selection
5. **Configuration System**: Comprehensive configuration with validation
6. **Exception Handling**: Detailed error management and user feedback

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Google Gemini API key
- Running MCP server(s)

### Method 1: Using pip (Recommended)

```bash
pip install pipeline-toolkit
```

### Method 2: From Source

```bash
git clone https://github.com/pipeline-toolkit/pipeline-toolkit.git
cd pipeline-toolkit
pip install -e .
```

### Method 3: Development Installation

```bash
git clone https://github.com/pipeline-toolkit/pipeline-toolkit.git
cd pipeline-toolkit
pip install -e ".[dev,full]"
```

## 🔧 Configuration

### Quick Start Configuration

1. **Generate example configuration:**
   ```bash
   pipeline-toolkit --help
   ```

2. **Create your configuration:**
   ```bash
   cp config/config.json.example config/config.json
   ```

3. **Edit configuration:**
   ```json
   {
     "gemini": {
       "api_key": "your_gemini_api_key_here",
       "model": "gemini-2.0-flash-exp",
       "temperature": 0.1
     },
     "mcp_servers": [
       {
         "name": "HTTP MCP Server",
         "connection_type": "http",
         "url": "http://localhost:8080",
         "enabled": true
       },
       {
         "name": "Local Filesystem MCP",
         "connection_type": "stdio",
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
         "enabled": true
       }
     ]
   }
   ```

### Environment Variables

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
PIPELINE_TOOLKIT_LOG_LEVEL=INFO
PIPELINE_TOOLKIT_MCP_SERVER_URL=http://localhost:8080
PIPELINE_TOOLKIT_MCP_SERVER_NAME=Default Server
```

### Connection Types

Pipeline Toolkit supports two types of MCP server connections:

#### HTTP Connections
For MCP servers running as HTTP services:
```json
{
  "name": "HTTP MCP Server",
  "connection_type": "http",
  "url": "http://localhost:8080",
  "endpoint": "/sse",
  "enabled": true
}
```

#### Stdio Connections
For locally running MCP servers that communicate via stdin/stdout:
```json
{
  "name": "Local Filesystem MCP",
  "connection_type": "stdio",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
  "working_directory": "/home/user",
  "environment": {
    "ENV_VAR": "value"
  },
  "enabled": true
}
```

**Common stdio MCP servers:**
- **Filesystem**: `npx -y @modelcontextprotocol/server-filesystem /path/to/files`
- **Git**: `python -m mcp_git` (in git repository)
- **SQLite**: `npx -y @modelcontextprotocol/server-sqlite /path/to/db.sqlite`
- **Brave Search**: `npx -y @modelcontextprotocol/server-brave-search`
- **Postgres**: `npx -y @modelcontextprotocol/server-postgres postgresql://user:pass@host/db`

### Advanced Configuration Options

```json
{
  "gemini": {
    "api_key": "your_key",
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.1,
    "max_tokens": 4096,
    "timeout": 30
  },
  "mcp_servers": [
    {
      "name": "HTTP MCP Server",
      "connection_type": "http",
      "url": "http://localhost:8080",
      "endpoint": "/sse",
      "timeout": 30,
      "max_retries": 3,
      "enabled": true,
      "health_check_interval": 30,
      "description": "HTTP-based MCP server"
    },
    {
      "name": "Local Filesystem MCP",
      "connection_type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"],
      "working_directory": "/home/user",
      "timeout": 30,
      "max_retries": 3,
      "enabled": true,
      "description": "Local filesystem access via stdio"
    },
    {
      "name": "Git MCP Server",
      "connection_type": "stdio",
      "command": "python",
      "args": ["-m", "mcp_git"],
      "working_directory": "/path/to/git/repo",
      "environment": {
        "GIT_CONFIG_GLOBAL": "/home/user/.gitconfig"
      },
      "enabled": false,
      "description": "Git operations MCP server"
    }
  ],
  "logging": {
    "level": "INFO",
    "file": "logs/pipeline_toolkit.log",
    "max_file_size": 10485760,
    "backup_count": 5,
    "enable_console": false
  },
  "ui": {
    "show_banner": true,
    "show_tool_preview": true,
    "theme": "default",
    "show_progress_bars": true,
    "prompt_style": "fancy"
  },
  "security": {
    "enable_api_key_validation": true,
    "mask_sensitive_logs": true,
    "allow_insecure_connections": false
  },
  "performance": {
    "max_concurrent_requests": 10,
    "connection_pool_size": 5,
    "enable_caching": true,
    "cache_ttl": 300
  }
}
```

## 🚀 Usage

### Interactive Mode

Start the enhanced interactive CLI:

```bash
pipeline-toolkit
# or
ptk
```

### Single Query Mode

Execute a single query and exit:

```bash
pipeline-toolkit --query "Get tag info for release-1.0"
ptk -q "List all available tools"
```

### Advanced CLI Options

```bash
pipeline-toolkit --help
ptk --config custom-config.json --debug --query "your query"
```

## 🎯 Interactive Commands

Once in interactive mode, you have access to enhanced commands:

### Basic Commands
- `help` - Show comprehensive help
- `list servers` - Show connected MCP servers with status
- `list tools` - Display all available tools grouped by server
- `stats` - Show detailed AI agent and performance statistics
- `config` - Display current configuration
- `logs [lines]` - Show recent log entries (default: 50 lines)

### Advanced Commands
- `show <query>` - Process query with AI agent
- `describe <tool>` - Get detailed information about a specific tool
- `connect` - Reconnect to all MCP servers
- `reload` - Reload configuration without restarting
- `clear` - Clear screen and refresh status

### Natural Language Queries
Simply type your questions naturally:

```
🤖 pipeline-toolkit > Get the latest build information
🤖 pipeline-toolkit > Show me packages containing 'kernel'
🤖 pipeline-toolkit > What tools are available for package management?
🤖 pipeline-toolkit > List all tags in the main branch
```

## ✨ Advanced Features

### Conversation Management

The toolkit maintains conversation context across queries:

```python
# Conversations are automatically managed per user/session
# Previous context influences future responses
# Tool usage patterns are learned and optimized
```

### Performance Monitoring

Real-time performance metrics are available:

- Connection health status
- Tool execution times
- Success rates
- Usage patterns
- Resource utilization

### Multi-Server Management

Intelligent management of multiple MCP servers:

- Automatic failover and recovery
- Load balancing across servers
- Health monitoring and alerting
- Performance optimization

## 🛠️ Development

### Project Structure

```
pipeline-toolkit/
├── src/
│   ├── cli.py               # Enhanced interactive CLI
│   ├── ai_agent.py          # Advanced AI agent with conversation management
│   ├── mcp_client.py        # Robust MCP client with health monitoring
│   ├── config.py            # Comprehensive configuration management
│   ├── exceptions.py        # Enhanced exception handling
│   └── dynamic_tool_analyzer.py  # Smart tool analysis
├── config/
│   ├── config.json.example  # Example configuration
│   └── README.md           # Configuration guide
├── examples/               # Usage examples
├── tests/                  # Comprehensive test suite
└── docs/                   # Documentation
```

### Setting Up Development Environment

```bash
# Clone and setup
git clone https://github.com/pipeline-toolkit/pipeline-toolkit.git
cd pipeline-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,full]"

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov=src

# Format code
black src/
isort src/

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Exclude slow tests

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📚 API Reference

### Enhanced AI Agent

```python
from src.ai_agent import EnhancedAIAgent

agent = EnhancedAIAgent(
    model="gemini-2.0-flash-exp",
    enable_conversation_history=True,
    enable_adaptive_learning=True
)

# Process query with user context
result = await agent.process_query(
    "Your query here",
    user_id="user123",
    conversation_id="session456"
)

# Get conversation summary
summary = agent.get_conversation_summary("user123", "session456")

# Clear conversation history
agent.clear_conversation_history("user123", "session456")
```

### Advanced MCP Client

```python
from src.mcp_client import MCPClient
from src.config import MCPServerConfig

config = MCPServerConfig(
    name="My Server",
    url="http://localhost:8080",
    health_check_interval=30,
    connection_timeout=10
)

async with MCPClient(config) as client:
    # Wait for healthy connection
    is_healthy = await client.wait_for_healthy(timeout=30)

    # Get comprehensive server info
    info = client.get_server_info()

    # Get performance metrics
    metrics = client.get_metrics()

    # Execute tools
    result = await client.call_tool("tool_name", param="value")
```

## 🔍 Advanced Usage Examples

### Multi-Server Setup

```json
{
  "mcp_servers": [
    {
      "name": "Brew Server",
      "url": "http://localhost:8080",
      "description": "Brew package management",
      "enabled": true
    },
    {
      "name": "JIRA Server",
      "url": "http://localhost:8090",
      "description": "JIRA project management",
      "enabled": true
    },
    {
      "name": "Custom Tools",
      "url": "http://localhost:9000",
      "description": "Custom business tools",
      "enabled": false
    }
  ]
}
```

### Performance Optimization

```json
{
  "performance": {
    "max_concurrent_requests": 20,
    "connection_pool_size": 10,
    "enable_caching": true,
    "cache_ttl": 600,
    "retry_backoff_factor": 2.0
  }
}
```

### Security Configuration

```json
{
  "security": {
    "enable_api_key_validation": true,
    "mask_sensitive_logs": true,
    "max_request_size": 2097152,
    "rate_limit_requests": 200,
    "rate_limit_window": 60
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Connection Issues**
   ```bash
   ptk --debug  # Enable debug logging
   # Check server status and configuration
   ```

2. **API Key Issues**
   ```bash
   export GEMINI_API_KEY=your_key
   # Verify key is valid and has permissions
   ```

3. **Performance Issues**
   ```bash
   # Check logs for performance metrics
   ptk -c logs 100
   # Adjust configuration for better performance
   ```

### Debug Mode

```bash
# Enable comprehensive debugging
pipeline-toolkit --debug --config debug-config.json

# Check configuration validation
ptk config

# View detailed connection status
ptk list servers
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) for the foundational protocol
- [Google Gemini AI](https://ai.google.dev/) for the AI capabilities
- [Rich](https://github.com/Textualize/rich) for the beautiful terminal UI
- The open-source community for inspiration and tools

## 📈 Roadmap

- [ ] Plugin system for custom tools
- [ ] Web-based UI interface
- [ ] Advanced analytics and reporting
- [ ] Integration with more AI models
- [ ] Collaborative features
- [ ] Cloud deployment options

---

**Pipeline Toolkit** - Transforming how you interact with MCP servers through intelligent AI orchestration.
