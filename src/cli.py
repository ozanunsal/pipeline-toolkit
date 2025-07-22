"""
Pipeline Toolkit - Enhanced Interactive MCP Client CLI.

This module provides an advanced, interactive command-line interface for connecting to multiple
MCP servers and processing queries through AI agents with rich UI components and enhanced UX.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, TaskID
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.layout import Layout
from rich.align import Align
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import yes_no_dialog
from prompt_toolkit.formatted_text import HTML

from src.ai_agent import EnhancedAIAgent as AIAgent
from src.mcp_client import MCPClient
from src.config import load_config, Config
from src.exceptions import PipelineToolkitError

# Rich console for enhanced output
console = Console()

# Command history and completion
HISTORY_FILE = Path.home() / ".pipeline_toolkit_history"
COMMANDS = [
    "help", "list servers", "list tools", "stats", "config", "logs", "clear",
    "exit", "quit", "show", "describe", "connect", "disconnect", "reload"
]

class InteractiveCLI:
    """Enhanced interactive CLI for Pipeline Toolkit."""

    def __init__(self):
        self.config: Optional[Config] = None
        self.ai_agent: Optional[AIAgent] = None
        self.connected_clients: List[Tuple[MCPClient, str, List[Any]]] = []
        self.session: Optional[PromptSession] = None
        self.running = False

        # Setup prompt session with history and completion
        self.setup_prompt_session()

        # Command registry
        self.commands = {
            "help": self.cmd_help,
            "list": self.cmd_list,
            "stats": self.cmd_stats,
            "config": self.cmd_config,
            "logs": self.cmd_logs,
            "clear": self.cmd_clear,
            "exit": self.cmd_exit,
            "quit": self.cmd_exit,
            "show": self.cmd_show,
            "describe": self.cmd_describe,
            "connect": self.cmd_connect,
            "disconnect": self.cmd_disconnect,
            "reload": self.cmd_reload,
        }

    def setup_prompt_session(self):
        """Setup enhanced prompt session with autocompletion and history."""
        try:
            completer = WordCompleter(COMMANDS + ["servers", "tools", "all"])

            # Custom key bindings
            bindings = KeyBindings()

            @bindings.add('c-c')
            def _(event):
                """Handle Ctrl+C gracefully."""
                event.app.exit(exception=KeyboardInterrupt)

            self.session = PromptSession(
                history=FileHistory(str(HISTORY_FILE)),
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
                complete_while_typing=True,
                key_bindings=bindings
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not setup enhanced prompt: {e}[/yellow]")
            self.session = None

    def display_banner(self):
        """Display an attractive banner."""
        banner_text = """
╔═══════════════════════════════════════════════════════════════╗
║                    🤖 Pipeline Toolkit                        ║
║              Enhanced Interactive MCP Client                  ║
║                                                               ║
║   AI-Powered Multi-Server Tool Orchestration Platform         ║
╚═══════════════════════════════════════════════════════════════╝
        """

        panel = Panel(
            Align.center(Text(banner_text, style="bold cyan")),
            style="bright_blue",
            title="[bold white]Welcome[/bold white]",
            subtitle="[italic]Type 'help' for available commands[/italic]"
        )
        console.print(panel)
        console.print()

    def display_status(self):
        """Display current connection status."""
        if not self.connected_clients:
            console.print("[red]⚠️  No MCP servers connected[/red]")
            return

        table = Table(title="🔗 Connected MCP Servers", show_header=True, header_style="bold magenta")
        table.add_column("Server", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Tools", style="yellow")
        table.add_column("URL", style="blue")

        for client, server_name, server_tools in self.connected_clients:
            status = "✅ Connected" if client.is_connected else "❌ Disconnected"

            # Show connection details based on type
            if client.config.is_stdio_connection():
                connection_info = f"stdio: {client.config.command}"
            else:
                connection_info = client.config.url

            table.add_row(
                server_name,
                status,
                str(len(server_tools)),
                connection_info
            )

        console.print(table)
        console.print()

    async def initialize(self, debug_mode: bool = False):
        """Initialize the CLI application."""
        try:
            with console.status("[bold green]Initializing Pipeline Toolkit...", spinner="dots"):
                self.config = load_config()
                self.setup_logging(debug_mode)

                self.ai_agent = AIAgent(
                    model=self.config.gemini.model,
                    api_key=self.config.gemini.api_key
                )

                # Connect to MCP servers
                await self.connect_to_servers()

            console.print("✅ [green]Pipeline Toolkit ready![/green]")

        except Exception as e:
            console.print(f"[red]❌ Initialization failed: {e}[/red]")
            raise

    def setup_logging(self, debug_mode: bool = False):
        """Setup enhanced logging with separate console and file levels."""
        log_file = Path(self.config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler with config level
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(getattr(logging, self.config.logging.level.upper()))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        handlers = [file_handler]

        # Add console handler only for debug mode or errors
        if debug_mode:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            handlers.append(console_handler)
        else:
            # In normal mode, only show WARNING and above in console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            handlers.append(console_handler)

        # Set root logger to lowest level to let handlers filter
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=handlers,
            force=True  # Override any existing logging configuration
        )

    async def connect_to_servers(self):
        """Connect to all configured MCP servers with progress tracking."""
        servers = self.config.mcp_servers

        for server_config in servers:
            if not server_config.enabled:
                logger.info(f"Skipping disabled server: {server_config.name}")
                continue

            logger.info(f"Attempting to connect to {server_config.name} ({server_config.connection_type})")
            client = MCPClient(server_config)
            try:
                await client.connect()
                if client.is_connected:
                    server_tools = await client.list_tools()
                    self.connected_clients.append((client, server_config.name, server_tools))

                    # Register with AI agent
                    self.ai_agent.register_mcp_client(client, server_config.name, server_tools)
                    logger.info(f"✅ Connected to {server_config.name} with {len(server_tools)} tools")
                else:
                    logger.warning(f"❌ Failed to connect to {server_config.name}")
            except Exception as e:
                logger.error(f"❌ Error connecting to {server_config.name}: {e}")
                # For debug, also print the exception details
                if self.config.logging.level.upper() == "DEBUG":
                    import traceback
                    logger.debug(f"Full traceback for {server_config.name}:\n{traceback.format_exc()}")

        if not self.connected_clients:
            console.print("[red]❌ No MCP servers connected. Check your configuration.[/red]")
            return False

        # Update AI agent with tools
        all_tools = []
        for _, _, tools in self.connected_clients:
            all_tools.extend(tools)

        self.ai_agent.convert_tools_to_gemini_functions(all_tools)

        logger.info(f"Connected to {len(self.connected_clients)} MCP servers with {len(all_tools)} tools")
        return True

    async def cmd_help(self, args: List[str]):
        """Display help information."""
        help_table = Table(title="📚 Available Commands", show_header=True, header_style="bold magenta")
        help_table.add_column("Command", style="cyan", min_width=15)
        help_table.add_column("Description", style="white")
        help_table.add_column("Examples", style="yellow")

        commands_info = [
            ("help", "Show this help message", "help"),
            ("list servers", "List connected MCP servers", "list servers"),
            ("list tools", "List available tools", "list tools"),
            ("stats", "Show AI agent statistics", "stats"),
            ("config", "Show current configuration", "config"),
            ("logs", "Show recent log entries", "logs"),
            ("show <query>", "Process query with AI agent", "show me package info"),
            ("describe <tool>", "Describe a specific tool", "describe get_package_info"),
            ("connect", "Reconnect to servers", "connect"),
            ("reload", "Reload configuration", "reload"),
            ("clear", "Clear the screen", "clear"),
            ("exit/quit", "Exit the application", "exit"),
        ]

        for cmd, desc, example in commands_info:
            help_table.add_row(cmd, desc, example)

        console.print(help_table)
        console.print("\n💡 [blue]Tip: Use Tab for auto-completion and ↑↓ for command history[/blue]")

    async def cmd_list(self, args: List[str]):
        """List servers or tools."""
        if not args:
            console.print("[yellow]Usage: list [servers|tools][/yellow]")
            return

        target = args[0].lower()

        if target == "servers":
            self.display_status()
        elif target == "tools":
            await self.list_tools()
        else:
            console.print("[yellow]Invalid option. Use: list [servers|tools][/yellow]")

    async def list_tools(self):
        """List all available tools grouped by server."""
        if not self.connected_clients:
            console.print("[red]No servers connected[/red]")
            return

        for client, server_name, tools in self.connected_clients:
            table = Table(title=f"🛠️  Tools from {server_name}", show_header=True, header_style="bold green")
            table.add_column("Tool Name", style="cyan")
            table.add_column("Description", style="white")

            for tool in tools:
                name = tool.get('name', 'Unknown') if isinstance(tool, dict) else str(tool)
                desc = tool.get('description', 'No description') if isinstance(tool, dict) else 'No description'
                table.add_row(name, desc[:80] + "..." if len(desc) > 80 else desc)

            console.print(table)
            console.print()

    async def cmd_stats(self, args: List[str]):
        """Show AI agent statistics."""
        if not self.ai_agent:
            console.print("[red]AI agent not initialized[/red]")
            return

        stats = self.ai_agent.get_stats()

        stats_table = Table(title="📊 AI Agent Statistics", show_header=True, header_style="bold blue")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Initialized", "✅ Yes" if stats["initialized"] else "❌ No")
        stats_table.add_row("Model", stats["model"])
        stats_table.add_row("Servers Connected", str(stats["servers_connected"]))
        stats_table.add_row("Tools Available", str(stats["tools_available"]))
        stats_table.add_row("Connected Servers", ", ".join(stats["servers"]))

        console.print(stats_table)

    async def cmd_config(self, args: List[str]):
        """Show current configuration."""
        if not self.config:
            console.print("[red]Configuration not loaded[/red]")
            return

        config_info = {
            "Gemini Model": self.config.gemini.model,
            "Log Level": self.config.logging.level,
            "Log File": self.config.logging.file,
            "Servers Configured": len(self.config.mcp_servers),
            "Banner Enabled": self.config.ui.show_banner,
            "Tool Preview": self.config.ui.show_tool_preview,
        }

        config_table = Table(title="⚙️  Current Configuration", show_header=True, header_style="bold purple")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")

        for key, value in config_info.items():
            config_table.add_row(key, str(value))

        console.print(config_table)

    async def cmd_logs(self, args: List[str]):
        """Show recent log entries."""
        lines = int(args[0]) if args and args[0].isdigit() else 50

        try:
            log_file = Path(self.config.logging.file)
            if not log_file.exists():
                console.print("[yellow]Log file not found[/yellow]")
                return

            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            console.print(f"📝 [bold]Recent {len(recent_lines)} log entries:[/bold]")
            console.print("-" * 80)

            for line in recent_lines:
                # Color code log levels
                if "ERROR" in line:
                    console.print(f"[red]{line.strip()}[/red]")
                elif "WARNING" in line:
                    console.print(f"[yellow]{line.strip()}[/yellow]")
                elif "INFO" in line:
                    console.print(f"[blue]{line.strip()}[/blue]")
                else:
                    console.print(line.strip())

            console.print("-" * 80)

        except Exception as e:
            console.print(f"[red]Error reading logs: {e}[/red]")

    async def cmd_clear(self, args: List[str]):
        """Clear the screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        self.display_banner()
        self.display_status()

    async def cmd_exit(self, args: List[str]):
        """Exit the application."""
        if self.session and not yes_no_dialog(
            title="Confirm Exit",
            text="Are you sure you want to exit Pipeline Toolkit?"
        ).run():
            return

        console.print("\n👋 [yellow]Disconnecting from servers...[/yellow]")

        # Cleanup connections
        for client, server_name, _ in self.connected_clients:
            try:
                await client.disconnect()
                console.print(f"✅ [green]Disconnected from {server_name}[/green]")
            except Exception as e:
                console.print(f"⚠️  [yellow]Error disconnecting from {server_name}: {e}[/yellow]")

        console.print("\n[bold green]Thank you for using Pipeline Toolkit! 🚀[/bold green]")
        self.running = False

    async def cmd_show(self, args: List[str]):
        """Process a query with the AI agent."""
        if not args:
            console.print("[yellow]Usage: show <your query>[/yellow]")
            return

        query = " ".join(args)
        await self.process_query(query)

    async def cmd_describe(self, args: List[str]):
        """Describe a specific tool."""
        if not args:
            console.print("[yellow]Usage: describe <tool_name>[/yellow]")
            return

        tool_name = args[0]
        # Implementation for describing a tool
        console.print(f"[blue]Describing tool: {tool_name}[/blue]")
        # Add tool description logic here

    async def cmd_connect(self, args: List[str]):
        """Reconnect to servers."""
        console.print("[blue]Reconnecting to MCP servers...[/blue]")
        await self.connect_to_servers()
        self.display_status()

    async def cmd_disconnect(self, args: List[str]):
        """Disconnect from a specific server or all servers."""
        # Implementation for selective disconnection
        console.print("[yellow]Disconnection functionality coming soon...[/yellow]")

    async def cmd_reload(self, args: List[str]):
        """Reload configuration."""
        try:
            console.print("[blue]Reloading configuration...[/blue]")
            self.config = load_config()
            console.print("✅ [green]Configuration reloaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to reload configuration: {e}[/red]")

    async def process_query(self, query: str):
        """Process a user query through the AI agent."""
        if not self.ai_agent:
            console.print("[red]AI agent not available[/red]")
            return

        console.print(f"\n🤔 [bold]Processing query:[/bold] {query}")

        with Live(Spinner("dots", text="[blue]AI is thinking...", style="blue"),
                  refresh_per_second=10, transient=True):
            try:
                result = await self.ai_agent.process_query(query)

                if result.get("success"):
                    response = result.get("gemini_response", "No response generated")

                    # Display response in a nice panel
                    response_panel = Panel(
                        Markdown(response) if response else "No response generated",
                        title="🤖 AI Response",
                        border_style="green"
                    )
                    console.print(response_panel)

                    # Show function call summary only if response is not comprehensive enough
                    function_calls = result.get("function_calls", [])
                    if function_calls and len(response) < 100:  # Only show if response is very short
                        successful_calls = len([call for call in function_calls if call.get("result", {}).get("success", False)])
                        total_calls = len(function_calls)

                        if total_calls > 0:
                            status_text = f"📊 Used {successful_calls}/{total_calls} tools successfully"
                            console.print(f"[dim]{status_text}[/dim]")
                else:
                    error_msg = result.get("error", "Unknown error occurred")
                    console.print(f"[red]❌ Error: {error_msg}[/red]")

            except Exception as e:
                console.print(f"[red]❌ Query processing failed: {e}[/red]")

    async def run(self):
        """Main interactive loop."""
        self.running = True

        try:
            await self.initialize(debug_mode=False)

            if self.config.ui.show_banner:
                self.display_banner()

            self.display_status()

            console.print("🚀 [bold green]Pipeline Toolkit is ready![/bold green]")
            console.print("💡 [blue]Type 'help' for available commands, or just start asking questions![/blue]\n")

            while self.running:
                try:
                    # Get user input with enhanced prompt
                    if self.session:
                        # Use prompt_toolkit's HTML formatting for colored prompt
                        prompt_text = HTML("🤖 <ansibrightcyan><b>pipeline-toolkit</b></ansibrightcyan> > ")
                        user_input = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.session.prompt(prompt_text)
                        )
                    else:
                        # Use Rich markup for Rich prompt fallback
                        user_input = Prompt.ask("🤖 [bold cyan]pipeline-toolkit[/bold cyan]")

                    if not user_input.strip():
                        continue

                    # Parse command
                    parts = user_input.strip().split()
                    cmd = parts[0].lower()
                    args = parts[1:]

                    # Handle commands
                    if cmd in self.commands:
                        await self.commands[cmd](args)
                    elif cmd.startswith(("what", "how", "show", "get", "list", "find", "tell")):
                        # Treat as natural language query
                        await self.process_query(user_input)
                    else:
                        # Try as natural language query first, then suggest commands
                        await self.process_query(user_input)

                except KeyboardInterrupt:
                    if Confirm.ask("\n[yellow]Do you want to exit?[/yellow]"):
                        await self.cmd_exit([])
                        break
                    else:
                        console.print("[blue]Use 'exit' command to quit properly[/blue]")
                        continue
                except EOFError:
                    await self.cmd_exit([])
                    break
                except Exception as e:
                    console.print(f"[red]❌ Unexpected error: {e}[/red]")
                    logging.exception("CLI error")

        except Exception as e:
            console.print(f"[red]❌ Fatal error: {e}[/red]")
            sys.exit(1)


@click.command()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode')
@click.option('--query', '-q', help='Execute single query and exit')
def main(config: Optional[str], debug: bool, query: Optional[str]):
    """Enhanced Pipeline Toolkit - Interactive MCP Client."""

    # Set config file if provided
    if config:
        os.environ['OLS_CONFIG_FILE'] = config

    # Set debug logging if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Single query mode
    if query:
        async def run_single_query():
            cli = InteractiveCLI()
            try:
                await cli.initialize(debug_mode=debug)
                await cli.process_query(query)
            finally:
                # Ensure proper cleanup of stdio connections (silent)
                for client, server_name, _ in cli.connected_clients:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass  # Silent cleanup

        asyncio.run(run_single_query())
        return

    # Interactive mode
    cli = InteractiveCLI()
    asyncio.run(cli.run())


def cli_main() -> None:
    """Synchronous entry point for CLI script."""
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n👋 [yellow]Goodbye![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()