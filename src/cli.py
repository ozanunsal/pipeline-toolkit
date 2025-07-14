"""
Pipeline Bot - Multi-MCP Server CLI Interface.

This module provides a simple command-line interface for interacting with multiple
MCP servers using Gemini AI for intelligent tool selection.
"""

import asyncio
import logging
import os
import sys
from typing import Any, List, Tuple

from src.ai_agent import AIAgent
from src.mcp_client import MCPClient
from src.config import load_config, Config


def setup_logging(config: Config) -> None:
    """Setup file-based logging."""
    log_file = os.path.join(os.path.dirname(__file__), '..', '..', config.logging.file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a')]
    )


def read_logs(config: Config, lines: int = 100) -> str:
    """Read the last N lines from the log file."""
    log_file = os.path.join(os.path.dirname(__file__), '..', '..', config.logging.file)

    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            if lines > len(all_lines):
                return ''.join(all_lines)
            else:
                return ''.join(all_lines[-lines:])
    except FileNotFoundError:
        return "Log file not found."
    except Exception as e:
        return f"Error reading log file: {e}"


async def main() -> None:
    """Main function to run pipeline bot with multiple MCP clients."""
    try:
        # Load configuration from JSON files
        config = load_config()
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)

        # Display banner if configured
        if config.ui.show_banner:
            print("🤖 Pipeline Bot - Multi-MCP Client")
            print("=" * 40)

        # Get MCP server configurations
        mcp_configs = config.mcp_servers
        print(f"🔗 Starting {len(mcp_configs)} MCP clients\n")

        tools = []
        connected_clients: List[Tuple[MCPClient, str, List[Any]]] = []

        for server_config in mcp_configs:
            client = MCPClient(server_config)
            try:
                await client.connect()
                if client.is_connected:
                    server_tools = await client.list_tools()
                    tools.extend(server_tools)
                    connected_clients.append((client, server_config.name, server_tools))
                    print(f"✅ Connected to {server_config.name} with {len(server_tools)} tools")
                else:
                    print(f"❌ Failed to connect to {server_config.name}")
            except Exception as e:
                print(f"❌ Failed to connect to {server_config.name}: {e}")

        if not connected_clients:
            print("❌ No MCP servers connected. Please check your server configurations.")
            print(f"📝 Edit config/config.json or check config/config.json.example")
            sys.exit(1)

        # Initialize Gemini AI agent with configuration
        gemini = AIAgent(model=config.gemini.model, api_key=config.gemini.api_key)

        # Register MCP clients with the AI agent
        for client, server_name, server_tools in connected_clients:
            gemini.register_mcp_client(client, server_name, server_tools)

        # Update the Gemini AI agent with the tools
        print(f"\n🔧 Converting {len(tools)} tools to Gemini functions...")
        gemini.convert_tools_to_gemini_functions(tools)

        # Show available tools if configured
        if tools and config.ui.show_tool_preview:
            print(f"\n📋 Available tools:")
            max_preview = config.ui.max_tools_preview
            for tool in tools[:max_preview]:
                tool_name = tool.get('name', 'Unknown') if isinstance(tool, dict) else str(tool)
                tool_desc = tool.get('description', 'No description') if isinstance(tool, dict) else 'No description'
                print(f"   • {tool_name}: {tool_desc}")
            if len(tools) > max_preview:
                print(f"   ... and {len(tools) - max_preview} more tools")
        else:
            print("\n⚠️  No tools available")

        # Interactive loop
        print(f"\n🚀 Pipeline Bot ready! {len(tools)} tools available from {len(connected_clients)} servers")
        print("Type 'quit' to exit")

        while True:
            try:
                user_input = input(f"\n🤖 Pipeline Bot: ").strip()

                if user_input.lower() in {"quit", "exit", "q"}:
                    break
                if not user_input:
                    continue

                # Check for log request first
                if user_input.lower() in {"show me the logs", "show logs", "logs", "show me logs"}:
                    print("📋 Recent logs:")
                    print("-" * 60)
                    print(read_logs(config, config.logging.max_log_lines))
                    print("-" * 60)
                    continue

                print("🧠 Processing query...")
                result = await gemini.process_query(user_input)

                logger.info(f"Query result: {result}")

                if result.get("success"):
                    if result.get("gemini_response"):
                        print(f"\n💬 {result['gemini_response']}")
                    else:
                        print(f"\n💬 Query processed successfully but no response generated")

                    # Show function calls if any
                    if result.get("function_calls"):
                        print(f"\n🔧 Function calls made: {len(result['function_calls'])}")
                        for call in result['function_calls']:
                            print(f"   • {call['name']} with args: {call.get('args', {})}")
                else:
                    print(f"\n❌ Error: {result.get('error', 'Unknown error')}")

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ Error: {e}")

        # Cleanup: disconnect all clients
        print(f"\n👋 Disconnecting from {len(connected_clients)} servers...")
        for client, server_name, _ in connected_clients:
            try:
                await client.disconnect()
                print(f"✅ Disconnected from {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {server_name}: {e}")
                print(f"⚠️  Error disconnecting from {server_name}: {e}")

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        print(f"📝 Make sure config/config.json exists and is valid")
        print(f"💡 Copy config/config.json.example to config/config.json and edit it")
        print(f"🔑 Don't forget to set your Gemini API key in config/config.json")
        sys.exit(1)


def cli_main() -> None:
    """Synchronous entry point for CLI script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Terminated")
        logging.getLogger(__name__).info("Application terminated by user")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()