#!/usr/bin/env python3
"""
Interactive CLI
A minimal command-line interface with help and quit functionality.
"""

import asyncio
import json
import logging
import os
import re
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

from ai_agent import GeminiAgent, GeminiConfig
from config import Config, load_config
from mcp_client import MCPClient

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
            "logs": self.cmd_logs,
            "show": self.cmd_show,
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
        console.print(f"\n🤔 [bold]Processing query:[/bold] {query}")
        with Live(
            Spinner("dots", text="[blue]AI is thinking...", style="blue"),
            refresh_per_second=10,
            transient=True,
        ):
            try:
                # Initialize Gemini agent if config allows
                gem_cfg = None
                if (
                    self.config
                    and self.config.gemini
                    and self.config.gemini.get("api_key")
                ):
                    gem_cfg = GeminiConfig(
                        api_key=self.config.gemini["api_key"],
                        model=self.config.gemini.get("model", "gemini-2.0-flash-exp"),
                    )
                agent = GeminiAgent(gem_cfg) if gem_cfg else None

                def _extract_text_from_result(result: object) -> str:
                    # Common MCP call_tool result patterns
                    # Try attributes then dict-like fallbacks
                    if result is None:
                        return ""
                    content = getattr(result, "content", None)
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            t = getattr(item, "text", None)
                            if isinstance(t, str):
                                parts.append(t)
                            elif isinstance(item, dict) and isinstance(
                                item.get("text"), str
                            ):
                                parts.append(item["text"])
                        if parts:
                            return "\n".join(parts)
                    if isinstance(content, str):
                        return content
                    if isinstance(result, dict):
                        if isinstance(result.get("content"), str):
                            return result["content"]
                        if isinstance(result.get("message"), str):
                            return result["message"]
                    return str(result)

                # Use planning if agent present, else fallback heuristic
                tool_output_text = ""
                if self.connected_clients:
                    tools_by_server = [
                        {"server": server_name, "tools": tools}
                        for _, server_name, tools in self.connected_clients
                    ]
                    plan = {"server": None}
                    if agent:
                        plan = await agent.plan_tool_call(query, tools_by_server)
                        # Update spinner to show chosen tool/server if available
                        try:
                            chosen_server = plan.get("server") or ""
                            chosen_tool = str(plan.get("tool", ""))
                            if chosen_server and chosen_tool:
                                console.print(
                                    f"[cyan]Calling tool:[/cyan] {chosen_server} :: {chosen_tool}"
                                )
                        except Exception:
                            pass

                    if plan.get("server"):
                        # Find the chosen client and tool
                        chosen_server = plan["server"].lower()
                        chosen_tool = str(plan.get("tool", ""))
                        call_args: Dict[str, Any] = (
                            plan.get("args", {})
                            if isinstance(plan.get("args"), dict)
                            else {}
                        )
                        for client, server_name, tools in self.connected_clients:
                            if (server_name or "").lower() == chosen_server:
                                try:
                                    result = await client.call_tool(
                                        chosen_tool, call_args
                                    )
                                    tool_output_text = _extract_text_from_result(result)

                                    # Generic follow-up logic: check if the query needs additional information
                                    # that can be inferred from the current result
                                    if tool_output_text and agent:
                                        followup_needed = await self._check_followup_needed(
                                            query, chosen_tool, tool_output_text, call_args, agent, tools_by_server
                                        )
                                        if followup_needed:
                                            try:
                                                followup_plan = followup_needed
                                                console.print(f"[cyan]Following up with:[/cyan] {followup_plan['tool']}")
                                                followup_result = await client.call_tool(
                                                    followup_plan['tool'], followup_plan['args']
                                                )
                                                followup_text = _extract_text_from_result(followup_result)
                                                if followup_text:
                                                    tool_output_text = f"{tool_output_text}\n\n{followup_text}"
                                            except Exception:
                                                # If follow-up fails, just use the original result
                                                pass

                                except Exception:
                                    tool_output_text = ""
                                break
                    # If planning failed or empty, try first available tool with empty args
                    if not tool_output_text:
                        for client, server_name, tools in self.connected_clients:
                            for tool in tools:
                                try:
                                    result = await client.call_tool(
                                        tool.get("name", ""), {}
                                    )
                                    tool_output_text = _extract_text_from_result(result)
                                    if tool_output_text:
                                        break
                                except Exception:
                                    continue
                            if tool_output_text:
                                break

                # If listing intent and list-like data found, render deterministically
                def _is_listing_intent(text: str) -> bool:
                    t = text.lower()
                    return any(
                        kw in t
                        for kw in [
                            "list",
                            "get ",
                            "show ",
                            "fetch",
                            "find ",
                            "latest",
                            "open ",
                            "unresolved",
                            "status",
                            "count",
                            "products",
                            "tickets",
                            "builds",
                            "pipelines",
                        ]
                    )

                def _extract_list_from_text(tool_text: str):
                    try:
                        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", tool_text)
                        if not m:
                            return None
                        obj = json.loads(m.group(1))
                        if isinstance(obj, list):
                            return {"key": "items", "items": obj}
                        if isinstance(obj, dict):
                            for key in [
                                "products",
                                "tickets",
                                "items",
                                "results",
                                "data",
                            ]:
                                val = obj.get(key)
                                if isinstance(val, list):
                                    return {"key": key, "items": val}
                        return None
                    except Exception:
                        return None

                def _render_list(items):
                    def _get(d, keys):
                        for k in keys:
                            if (
                                isinstance(d, dict)
                                and k in d
                                and isinstance(d[k], (str, int))
                            ):
                                return str(d[k])
                        return ""

                    lines: List[str] = []
                    for idx, it in enumerate(items, start=1):
                        if isinstance(it, dict):
                            id_or_key = (
                                _get(it, ["id", "key", "product", "version"]) or ""
                            )
                            name = _get(it, ["name", "title", "summary"]) or ""
                            status = _get(it, ["status", "state"]) or ""
                            parts = [
                                p
                                for p in [
                                    id_or_key,
                                    name,
                                    f"({status})" if status else "",
                                ]
                                if p
                            ]
                            line = (
                                f"{idx}. " + " - ".join(parts) if parts else f"{idx}."
                            )
                        else:
                            line = f"{idx}. {it}"
                        lines.append(line)
                    if lines:
                        console.print("\n".join(lines))

                if _is_listing_intent(query) and tool_output_text:
                    parsed = _extract_list_from_text(tool_output_text)
                    if (
                        parsed
                        and isinstance(parsed.get("items"), list)
                        and parsed["items"]
                    ):
                        _render_list(parsed["items"])
                        return

                # Otherwise, use AI summarization
                analysis_input = tool_output_text or query
                if agent:
                    response = await agent.generate_answer(query, analysis_input)
                else:
                    response = analysis_input[:500]

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
        table.add_column("Transport", style="magenta")
        table.add_column("URL", style="blue")

        for client, server_name, server_tools in self.connected_clients:
            status = "✅ Connected" if client.is_connected else "❌ Disconnected"

            # Transport type
            transport = client.config.connection_type

            # Show connection details based on type
            if client.config.is_sse_connection():
                connection_info = client.config.url
            else:
                connection_info = f"stdio: {client.config.command}"

            table.add_row(
                server_name,
                status,
                str(len(server_tools)),
                transport,
                connection_info,
            )

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
            ("show logs", "Show the saved logs", "show logs"),
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
                table.add_row(name, desc[:200] + "..." if len(desc) > 200 else desc)
            console.print(table)
            console.print()

    async def cmd_clear(self, args: List[str]):
        """Clear the screen."""
        os.system("cls" if os.name == "nt" else "clear")
        self.display_banner()

    def _resolve_log_file(self) -> Path:
        """Resolve the log file path based on config or defaults.

        - If config.log_file is set, use it (expanding ~).
        - Else, use PIPELINE_TOOLKIT_LOG_DIR/logs path or CWD/logs.
        - File name defaults to pipeline_bot.log.
        """
        try:
            if self.config and getattr(self.config, "log_file", None):
                return Path(self.config.log_file).expanduser().resolve()
        except Exception:
            pass

        base_dir = Path(
            os.getenv("PIPELINE_TOOLKIT_LOG_DIR", str(Path.cwd() / "logs"))
        ).resolve()
        return base_dir / "pipeline_bot.log"

    async def cmd_logs(self, args: List[str]):
        """Display the saved logs in the console."""
        log_path = self._resolve_log_file()
        if not log_path.exists():
            console.print(f"[yellow]No log file found at:[/yellow] {log_path}")
            return

        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            console.print(f"[red]Failed to read log file:[/red] {e}")
            return

        panel = Panel(
            content if content else "(empty)",
            title=f"📜 Logs — {log_path}",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)

    async def cmd_show(self, args: List[str]):
        """Alias for 'show logs' -> logs display."""
        if args and args[0].lower() == "logs":
            await self.cmd_logs(args[1:])
        else:
            console.print("[yellow]Usage: show logs[/yellow]")

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

    async def _check_followup_needed(self, query: str, first_tool: str, first_result: str,
                                   first_args: Dict[str, Any], agent, tools_by_server) -> Optional[Dict[str, Any]]:
        """Check if a follow-up tool call is needed based on the query and initial result."""
        import json

        # Create a follow-up planning prompt
        tools_summary = [{'server': s['server'], 'tools': [t['name'] for t in s['tools']]} for s in tools_by_server]
        tools_json = json.dumps(tools_summary, indent=2)

        followup_prompt = (
            f"Original Query: {query}\n"
            f"First Tool Called: {first_tool}\n"
            f"First Tool Result: {first_result[:500]}...\n\n"
            f"Does the original query need additional information that can be obtained from another tool?\n"
            f"If YES, respond with JSON: {{\"tool\": \"tool_name\", \"args\": {{...}}}}\n"
            f"If NO, respond with: {{\"followup_needed\": false}}\n\n"
            f"Available Tools: {tools_json}\n\n"
            "For example:\n"
            '- If query asks for "errors" but first result only shows pipeline info with ID, use get_failed_jobs\n'
            '- If query asks for "logs" but first result only shows job info with ID, use get_job_log\n'
            "Extract any needed IDs or parameters from the first result."
        )

        try:
            import re
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: agent.model.generate_content([followup_prompt])
            )
            text = getattr(resp, "text", "") or ""
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None

            obj = json.loads(m.group(0))
            if obj.get("followup_needed") is False:
                return None

            if obj.get("tool") and obj.get("args"):
                # Inherit instance_type from first call if not specified
                if "instance_type" not in obj["args"] and "instance_type" in first_args:
                    obj["args"]["instance_type"] = first_args["instance_type"]
                return obj

        except Exception as e:
            logger.debug(f"Follow-up planning failed: {e}")

        return None

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
