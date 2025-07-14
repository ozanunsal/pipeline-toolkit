"""
AI Agent - Professional Gemini AI integration for MCP tool selection and execution.

This module provides an AI agent that uses Google's Gemini AI to intelligently select
and execute tools from MCP servers based on natural language queries.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import nest_asyncio
from google import genai
from google.genai import types

from src.exceptions import GeminiError, ToolExecutionError
from src.mcp_client import MCPClient

# Configure logging
log_file = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'pipeline_bot.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='a')]
)

logger = logging.getLogger(__name__)
nest_asyncio.apply()


class AIAgent:
    """
    AI Agent for intelligent tool selection and execution using Gemini AI.

    This agent provides a natural language interface to MCP tools by:
    1. Converting MCP tool schemas to Gemini function declarations
    2. Processing natural language queries with Gemini AI
    3. Intelligently selecting and calling appropriate tools
    4. Returning formatted results

    Features:
        - Automatic tool schema conversion
        - Natural language query processing
        - Intelligent parameter extraction
        - Multi-server tool management
        - Comprehensive error handling

    Example:
        >>> agent = AIAgent()
        >>> agent.register_mcp_client(client, "MyServer", tools)
        >>> result = await agent.process_query("Get tag info for release-1.0")
    """

    def __init__(self, model: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None) -> None:
        """
        Initialize the AI Agent with Gemini AI.

        Args:
            model: Gemini model to use for processing
            api_key: Optional Gemini API key (will use env vars if not provided)
        """
        self.model = model
        self.api_key = api_key
        self.ai_agent: Optional[genai.Client] = None
        self.functions: List[types.FunctionDeclaration] = []
        self.mcp_clients: Dict[str, MCPClient] = {}
        self.tool_to_client: Dict[str, MCPClient] = {}
        self.initialized = False

        self._setup_gemini()

    def _setup_gemini(self) -> None:
        """Initialize and configure the Gemini AI client."""
        try:
            # Use provided API key or fall back to environment variables
            api_key = self.api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise GeminiError("Gemini API key is not set. Provide it via constructor or environment variables.")

            self.ai_agent = genai.Client(api_key=api_key)
            self.initialized = True
            logger.info(f"Gemini AI client initialized with model: {self.model}")

        except Exception as e:
            raise GeminiError(f"Failed to initialize Gemini AI client: {e}")

    def register_mcp_client(self, client: MCPClient, server_name: str, tools: List[Dict[str, Any]]) -> None:
        """
        Register an MCP client with its tools for tool execution.

        Args:
            client: MCP client instance
            server_name: Name of the MCP server
            tools: List of tools from this server
        """
        self.mcp_clients[server_name] = client

        # Map each tool to its client for routing
        for tool in tools:
            tool_name = None
            if isinstance(tool, dict) and 'name' in tool:
                tool_name = tool['name']
            elif hasattr(tool, 'name'):
                tool_name = getattr(tool, 'name')

            if tool_name:
                self.tool_to_client[tool_name] = client

        logger.info(f"Registered MCP client {server_name} with {len(tools)} tools")

    def convert_tools_to_gemini_functions(self, tools: List[Dict[str, Any]]) -> None:
        """
        Convert MCP tools to Gemini function declarations.

        Args:
            tools: List of tool definitions from MCP server
        """
        for tool in tools:
            try:
                # Extract tool information
                tool_name = self._extract_tool_name(tool)
                tool_description = self._extract_tool_description(tool)
                input_schema = self._extract_input_schema(tool)

                # Create Gemini function declaration
                function_declaration = self._create_function_declaration(
                    tool_name, tool_description, input_schema
                )

                self.functions.append(function_declaration)
                logger.info(f"Converted tool '{tool_name}' to Gemini function")

            except Exception as e:
                logger.warning(f"Failed to convert tool {tool} to Gemini function: {e}")
                continue

    def _extract_tool_name(self, tool: Dict[str, Any]) -> str:
        """Extract tool name from tool definition."""
        if isinstance(tool, dict):
            return tool.get("name", str(tool))
        elif hasattr(tool, "name"):
            return getattr(tool, "name")
        return str(tool)

    def _extract_tool_description(self, tool: Dict[str, Any]) -> str:
        """Extract tool description from tool definition."""
        if isinstance(tool, dict):
            return tool.get("description", "No description")
        elif hasattr(tool, "description"):
            return getattr(tool, "description", "No description")
        return "No description"

    def _extract_input_schema(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Extract input schema from tool definition."""
        if isinstance(tool, dict):
            return tool.get("inputSchema", {})
        elif hasattr(tool, "inputSchema"):
            return getattr(tool, "inputSchema", {})
        return {}

    def _create_function_declaration(
        self, tool_name: str, description: str, input_schema: Dict[str, Any]
    ) -> types.FunctionDeclaration:
        """Create a Gemini function declaration from tool information."""
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # Convert properties to Gemini format
        gemini_properties = {}
        for key, prop in properties.items():
            prop_type = self._map_type_to_gemini(prop.get("type", "string"))
            gemini_properties[key] = types.Schema(
                type=prop_type,
                description=prop.get("description", f"Parameter: {key}")
            )

        # Enhance description with parameter information
        if properties:
            param_descriptions = [
                f"{name}: {prop.get('description', f'Parameter {name}')}"
                for name, prop in properties.items()
            ]
            description = f"{description}\n\nParameters:\n" + "\n".join(f"- {desc}" for desc in param_descriptions)

        return types.FunctionDeclaration(
            name=tool_name,
            description=description,
            parameters=types.Schema(
                type="OBJECT",
                properties=gemini_properties,
                required=required
            )
        )

    def _map_type_to_gemini(self, mcp_type: str) -> str:
        """Map MCP parameter types to Gemini types."""
        type_mapping = {
            "string": "STRING",
            "str": "STRING",
            "integer": "INTEGER",
            "int": "INTEGER",
            "boolean": "BOOLEAN",
            "bool": "BOOLEAN",
            "number": "NUMBER",
            "float": "NUMBER",
            "array": "ARRAY",
            "object": "OBJECT",
        }
        return type_mapping.get(mcp_type.lower(), "STRING")

    async def process_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a natural language query using Gemini AI and available tools.

        Args:
            query: Natural language query from user
            context: Optional context to help with processing

        Returns:
            Dict containing response and any tool results
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Gemini AI not initialized. Please check your API key.",
                "query": query
            }

        try:
            # Build system instruction
            system_instruction = self._build_system_instruction(context)

            # Prepare tools for Gemini
            tools = [types.Tool(function_declarations=self.functions)] if self.functions else None

            # Generate content with Gemini
            response = await self._call_gemini(query, system_instruction, tools)

            # Parse response and execute tools
            return await self._process_gemini_response(response, query, system_instruction, tools)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    def _build_system_instruction(self, context: Optional[str] = None) -> str:
        """Build system instruction for Gemini AI."""
        instruction = """You are an AI assistant that helps users interact with MCP servers through their tools.
        You can call the available functions to get information and answer user questions.

        When a user asks a question, analyze what they want and either:
        1. Call the appropriate tool(s) with the correct parameters extracted from their query
        2. Provide a direct answer if no tools are needed

        CRITICAL: When calling tools, you MUST extract the required parameters from the user's query:
        - Carefully read the tool descriptions and parameter requirements
        - Look for values in the user's query that match the parameter names and types
        - Extract identifiers, names, IDs, or other values mentioned in the query
        - Map these values to the correct parameter names as defined in the tool schema
        - If a required parameter is not clearly stated in the query, ask the user for clarification
        - Always provide the parameters as key-value pairs in the function call

        IMPORTANT: For build/package queries, use the available tools creatively:
        - If someone asks about a specific package build in a tag, use list_brew_builds with the tag name
        - If someone asks about packages in a tag, use list_brew_packages with the tag name
        - If someone asks about tag information, use get_brew_tag_info with the tag name
        - The tools return lists that you can analyze to find specific items the user is looking for

        NOTE: Log requests are handled separately by the system - you don't need to handle "show me the logs" requests.

        Examples of parameter extraction:
        - "get info for tag ABC" → get_brew_tag_info with tag_name="ABC"
        - "show me package XYZ" → list_brew_packages with tag_name="relevant_tag"
        - "what is the kernel-automotive build in tag ABC?" → list_brew_builds with tag_name="ABC" (then find kernel-automotive in results)
        - "ID is 123" → extract id="123"

        Be helpful and provide clear responses based on the tool results or your knowledge."""

        if context:
            instruction += f"\n\nContext: {context}"

        return instruction

    async def _call_gemini(self, query: str, system_instruction: str, tools: Optional[List[types.Tool]]) -> Any:
        """Call Gemini AI with query and tools."""
        try:
            return self.ai_agent.models.generate_content(
                model=self.model,
                contents=[query],
                config=types.GenerateContentConfig(
                    tools=tools,
                    system_instruction=system_instruction,
                    temperature=0.1
                )
            )
        except Exception as e:
            raise GeminiError(f"Gemini API call failed: {e}")

    async def _process_gemini_response(
        self, response: Any, query: str, system_instruction: str, tools: Optional[List[types.Tool]]
    ) -> Dict[str, Any]:
        """Process Gemini response and execute any function calls."""
        function_calls = []
        final_response = ""

        # Parse response
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_response += part.text
                        if hasattr(part, 'function_call') and part.function_call:
                            func_call = part.function_call
                            args = dict(func_call.args) if hasattr(func_call, 'args') and func_call.args else {}
                            function_calls.append({"name": func_call.name, "args": args})

        # Execute function calls
        function_results = []
        if function_calls:
            for func_call in function_calls:
                logger.info(f"Calling tool: {func_call['name']} with args: {func_call['args']}")
                tool_result = await self.call_tool(func_call["name"], **func_call["args"])
                function_results.append({
                    "function_call": func_call,
                    "result": tool_result
                })

                # Get final response from Gemini if needed
                if tool_result.get("success") and tool_result.get("content") and not final_response:
                    final_response = await self._get_final_response(
                        query, func_call, tool_result, system_instruction, tools
                    )

        # Set default response if needed
        if not function_calls and not final_response:
            final_response = "I couldn't determine how to answer your question. Please try being more specific."

        return {
            "success": True,
            "query": query,
            "gemini_response": final_response,
            "function_calls": function_calls,
            "function_results": function_results,
            "tools_available": len(self.functions)
        }

    async def _get_final_response(
        self, query: str, func_call: Dict[str, Any], tool_result: Dict[str, Any],
        system_instruction: str, tools: Optional[List[types.Tool]]
    ) -> str:
        """Get final response from Gemini after tool execution."""
        try:
            content_text = str(tool_result.get("content", [{}])[0].get("text", ""))
            function_response_content = [
                query,
                types.Part.from_function_response(
                    name=func_call["name"],
                    response={"result": content_text}
                )
            ]

            final_model_response = self.ai_agent.models.generate_content(
                model=self.model,
                contents=function_response_content,
                config=types.GenerateContentConfig(
                    tools=tools,
                    system_instruction=system_instruction,
                    temperature=0.1
                )
            )

            response_text = ""
            if hasattr(final_model_response, 'candidates') and final_model_response.candidates:
                candidate = final_model_response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text

            return response_text

        except Exception as e:
            logger.error(f"Error getting final response from Gemini: {e}")
            return f"Tool result: {content_text}"

    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool on the appropriate MCP server.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Parameters to pass to the tool

        Returns:
            Dict containing the tool result
        """
        try:
            # Find the MCP client for this tool
            client = self.tool_to_client.get(tool_name)
            if not client:
                return {
                    "success": False,
                    "error": f"No MCP client found for tool: {tool_name}",
                    "content": []
                }

            # Call the tool on the MCP client
            result = await client.call_tool(tool_name, **kwargs)

            return {
                "success": True,
                "content": [{"text": str(result)}],
                "tool_name": tool_name
            }

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Error calling tool {tool_name}: {str(e)}",
                "content": []
            }

    @property
    def is_initialized(self) -> bool:
        """Check if Gemini AI is initialized."""
        return self.initialized

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return [
            {
                "name": func.name,
                "description": func.description,
                "server": next(
                    (server for server, client in self.mcp_clients.items()
                     if client == self.tool_to_client.get(func.name)),
                    "unknown"
                )
            }
            for func in self.functions
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "initialized": self.initialized,
            "model": self.model,
            "servers_connected": len(self.mcp_clients),
            "tools_available": len(self.functions),
            "servers": list(self.mcp_clients.keys())
        }