#!/usr/bin/env python3
"""
Interactive CLI
A minimal command-line interface with help and quit functionality.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import yes_no_dialog
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from src.config import Config, load_config
from src.mcp_client import MCPClient

# Configure logging

logger = logging.getLogger(__name__)

# Rich console for enchanced output
console = Console()


class CLI:
    """An interactive CLI with Rich formatting."""

    def __init__(self):
        self.running = False
        self.config: Optional[Config] = None
        self.connected_clients: List[Tuple[MCPClient, str, List[Any]]] = []
        self.session: Optional[PromptSession] = None
        self.history_file = Path.home() / ".pipeline-toolkit-history"
        self.commands = ["help", "exit", "quit"]
        self.setup_prompt_session()

        # Command registry
        self.commands = {
            "help": self.cmd_help,
            "list": self.cmd_list,
            "exit": self.cmd_exit,
            "quit": self.cmd_exit,
            "clear": self.cmd_clear,
        }

    def setup_prompt_session(self):
        """Setup enhanced prompt session with auto-completion and history."""
        try:
            completer = WordCompleter(self.commands)
            bindings = KeyBindings()

            @bindings.add("c-c")
            def _(event):
                """Handle Ctrl+C."""
                event.app.exit(exception=KeyboardInterrupt)

            self.session = PromptSession(
                history=FileHistory(str(self.history_file)),
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
                complete_while_typing=True,
                key_bindings=bindings,
            )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not setup enhanced prompt: {e}[/yellow]"
            )
            self.session = None

    async def initialize(self):
        """Initialize the CLI."""
        try:
            with console.status(
                "[bold green]Initializing Pipeline Toolkit...", spinner="dots"
            ):
                self.config = load_config()
                # Connect to MCP servers
                await self.connect_to_servers()

            console.print("✅ [green]Pipeline Toolkit ready![/green]")
        except Exception as e:
            console.print(f"[red]❌ Initialization failed: {e}[/red]")
            raise

    async def connect_to_servers(self):
        """Connect to MCP servers."""
        servers = self.config.mcp_servers
        for server_config in servers:
            if not server_config.enabled:
                logger.info(f"Skipping disabled server: {server_config.name}")
                continue

            logger.info(
                f"Attempting to connect to {server_config.name} ({server_config.connection_type})"
            )
            client = MCPClient(server_config)
            try:
                await client.connect()
                if client.is_connected:
                    server_tools = await client.list_tools()
                    self.connected_clients.append(
                        (client, server_config.name, server_tools)
                    )

                    # TODO: Register tools to the AI agent
            except Exception as e:
                logger.error(f"Failed to connect to {server_config.name}: {e}")
                import traceback

                logger.debug(
                    f"Full traceback for {server_config.name}: \n{traceback.format_exc()}"
                )
        if not self.connected_clients:
            console.print(
                "[red]❌ No MCP servers connected. Check your configuration.[/red]"
            )
            return False

        # TODO: Update AI agent with tools
        return True

    async def process_query(self, query: str):
        """Process a user query through the AI agent."""
        # TODO: Check AI agent

        console.print(f"\n🤔 [bold]Processing query:[/bold] {query}")
        with Live(
            Spinner("dots", text="[blue]AI is thinking...", style="blue"),
            refresh_per_second=10,
            transient=True,
        ):
            try:
                # TODO: Process the query with the AI agent
                response = "This is a test response from the AI agent."
                # Display the response in a nice panel
                response_panel = Panel(
                    Markdown(response) if response else "No response generated",
                    title="🤖 AI Response",
                    border_style="green",
                )
                console.print(response_panel)

            except Exception as e:
                console.print(f"[red]❌ Error processing query: {e}[/red]")

    def display_status(self):
        """Display current connection status."""
        if not self.connected_clients:
            console.print("[red]⚠️  No MCP servers connected[/red]")
            return

        table = Table(
            title="🔗 Connected MCP Servers",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Server", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Tools", style="yellow")
        table.add_column("URL", style="blue")

        for client, server_name, server_tools in self.connected_clients:
            status = "✅ Connected" if client.is_connected else "❌ Disconnected"

            # Show connection details based on type
            if client.config.is_sse_connection():
                connection_info = client.config.url
            else:
                connection_info = f"stdio: {client.config.command}"

            table.add_row(server_name, status, str(len(server_tools)), connection_info)

        console.print(table)
        console.print()

    async def cmd_exit(self, args: List[str]):
        """Exit the application."""
        if (
            self.session
            and not yes_no_dialog(
                title="Confirm Exit",
                text="Are you sure you want to exit Pipeline-Toolkit?",
            ).run()
        ):
            return
        console.print("\n👋 [yellow]Disconnecting from servers...[/yellow]")
        # Cleanup connections
        for client, server_name, _ in self.connected_clients:
            try:
                await client.disconnect()
                console.print(f"✅ [green]Disconnected from {server_name}[/green]")
            except Exception as e:
                console.print(
                    f"⚠️  [yellow]Error disconnecting from {server_name}: {e}[/yellow]"
                )

        console.print(
            "\n[bold green]Thank you for using Pipeline Toolkit! 🚀[/bold green]"
        )
        self.running = False

    async def cmd_help(self, args: List[str]):
        """Display the help information."""
        help_table = Table(
            title="📚 Available Commands", show_header=True, header_style="bold magenta"
        )
        help_table.add_column("Command", style="cyan", min_width=15)
        help_table.add_column("Description", style="white")
        help_table.add_column("Examples", style="yellow")

        commands_info = [
            ("help", "Show this help message", "help"),
            ("list servers", "List connected MCP servers", "list servers"),
            ("list tools", "List available tools", "list tools"),
            ("clear", "Clear the screen", "clear"),
            ("exit/quit", "Exit the application", "exit"),
        ]

        for command, description, example in commands_info:
            help_table.add_row(command, description, example)
        console.print(help_table)
        console.print(
            "\n💡 [blue]Tip: Use Tab for auto-completion and ↑↓ for command history[/blue]"
        )

    async def cmd_list(self, args: List[str]):
        """List servers or tools."""
        if not args:
            console.print("[yellow]Usage: list <servers|tools> [/yellow]")
            return

        target = args[0].lower()

        if target == "servers":
            self.display_status()
        elif target == "tools":
            await self.list_tools()
        else:
            console.print("[yellow]Invalid option. Use: list [servers|tools][/yellow]")

    async def list_tools(self):
        """List all available tools grouped by MCP server."""
        if not self.connected_clients:
            console.print("[red]No MCP servers connected. Connect first.[/red]")
            return

        for client, server_name, tools in self.connected_clients:
            table = Table(
                title=f"🛠️  Tools from {server_name}",
                show_header=True,
                header_style="bold green",
            )
            table.add_column("Tool Name", style="cyan")
            table.add_column("Description", style="white")

            for tool in tools:
                name = (
                    tool.get("name", "Unknown") if isinstance(tool, dict) else str(tool)
                )
                desc = (
                    tool.get("description", "No description")
                    if isinstance(tool, dict)
                    else "No description"
                )
                table.add_row(name, desc[:80] + "..." if len(desc) > 80 else desc)
            console.print(table)
            console.print()

    async def cmd_clear(self, args: List[str]):
        """Clear the screen."""
        os.system("cls" if os.name == "nt" else "clear")
        self.display_banner()

    async def run(self):
        """Main CLI loop."""
        self.running = True

        try:
            await self.initialize()
            self.display_banner()
            self.display_status()

            console.print("🚀 [bold green]Pipeline Toolkit is ready![/bold green]")
            console.print(
                "💡 [blue]Type 'help' for available commands, or just start asking questions![/blue]\n"
            )

            while self.running:
                try:
                    if self.session:
                        prompt_text = HTML(
                            "🤖 <ansibrightcyan><b>pipeline-toolkit</b></ansibrightcyan> > "
                        )
                        user_input = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.session.prompt(prompt_text)
                        )
                    else:
                        user_input = Prompt.ask(
                            "🤖 [bold cyan]pipeline-toolkit[/bold cyan]"
                        )

                    if not user_input.strip():
                        continue
                    parts = user_input.strip().split()
                    cmd = parts[0].lower()
                    args = parts[1:]

                    if cmd in self.commands:
                        await self.commands[cmd](args)
                    else:
                        await self.process_query(user_input)
                except KeyboardInterrupt:
                    if Confirm.ask("\n[yellow]Do you want to exit?[/yellow]"):
                        await self.cmd_exit([])
                        break
                    else:
                        console.print(
                            "[blue]Use 'exit' command to quit properly[/blue]"
                        )
                        continue
                except EOFError:
                    await self.cmd_exit([])
                    break
                except Exception as e:
                    console.print(f"[red]❌ Unexpected error: {e}[/red]")
                    raise

        except Exception as e:
            console.print(f"[red]❌ Fatal error: {e}[/red]")
            sys.exit(1)

    def display_banner(self):
        """Display the banner."""
        banner = """
 ███████████   ███                     ████   ███                                 ███████████                   ████  █████       ███   █████
░░███░░░░░███ ░░░                     ░░███  ░░░                                 ░█░░░███░░░█                  ░░███ ░░███       ░░░   ░░███
 ░███    ░███ ████  ████████   ██████  ░███  ████  ████████    ██████            ░   ░███  ░   ██████   ██████  ░███  ░███ █████ ████  ███████
 ░██████████ ░░███ ░░███░░███ ███░░███ ░███ ░░███ ░░███░░███  ███░░███ ██████████    ░███     ███░░███ ███░░███ ░███  ░███░░███ ░░███ ░░░███░
 ░███░░░░░░   ░███  ░███ ░███░███████  ░███  ░███  ░███ ░███ ░███████ ░░░░░░░░░░     ░███    ░███ ░███░███ ░███ ░███  ░██████░   ░███   ░███
 ░███         ░███  ░███ ░███░███░░░   ░███  ░███  ░███ ░███ ░███░░░                 ░███    ░███ ░███░███ ░███ ░███  ░███░░███  ░███   ░███ ███
 █████        █████ ░███████ ░░██████  █████ █████ ████ █████░░██████                █████   ░░██████ ░░██████  █████ ████ █████ █████  ░░█████
░░░░░        ░░░░░  ░███░░░   ░░░░░░  ░░░░░ ░░░░░ ░░░░ ░░░░░  ░░░░░░                ░░░░░     ░░░░░░   ░░░░░░  ░░░░░ ░░░░ ░░░░░ ░░░░░    ░░░░░
                    ░███
                    █████
                   ░░░░░                                                                                                                        """
        welcome_text = Text()
        welcome_text.append(banner, style="bold bright_green")
        welcome_text.append("✨", style="bright_yellow")
        subtitle = Text(
            "AI-Powered Multi-MCP-Server Tool Orchestration Platform", style="dim cyan"
        )

        welcome_panel = Panel(
            Align.center(welcome_text),
            subtitle=subtitle,
            border_style="bright_blue",
            padding=(1, 2),
        )
        console.print(welcome_panel)
        console.print()


@click.command()
@click.option("--query", "-q", help="Execute a single query and exit")
def main(query: Optional[str]):
    """Main CLI loop."""
    if query:

        async def run_single_query():
            cli = CLI()
            try:
                await cli.initialize()
                await cli.process_query(query)
            finally:
                for client, server_name, _ in cli.connected_clients:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

        asyncio.run(run_single_query())
        return

    cli = CLI()
    asyncio.run(cli.run())


def cli_main() -> None:
    """Entry point for the CLI application."""
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
