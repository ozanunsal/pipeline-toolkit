"""
AI Agent - Professional Gemini AI integration for MCP tool selection and execution.

This module provides an AI agent that uses Google's Gemini AI to intelligently select
and execute tools from MCP servers based on natural language queries.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import nest_asyncio
from google import genai
from google.genai import types

from src.exceptions import GeminiError, ToolExecutionError
from src.mcp_client import MCPClient
from src.dynamic_tool_analyzer import DynamicToolAnalyzer, DynamicInstructionBuilder

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

        # Dynamic tool analysis components
        self.tool_analyzer = DynamicToolAnalyzer()
        self.instruction_builder = DynamicInstructionBuilder(self.tool_analyzer)
        self.tool_analysis: Optional[Dict[str, Any]] = None

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
        Convert MCP tools to Gemini function declarations with dynamic analysis.

        Args:
            tools: List of tool definitions from MCP server
        """
        # Perform comprehensive dynamic analysis of all tools
        logger.info("Performing dynamic analysis of tool collection...")
        self.tool_analysis = self.tool_analyzer.analyze_tool_collection(tools)

        logger.info(f"Analysis complete: {self.tool_analysis['server_type']} server, "
                   f"{self.tool_analysis['primary_domain']} domain")

        for tool in tools:
            try:
                # Extract tool information
                tool_name = self._extract_tool_name(tool)
                tool_description = self._extract_tool_description(tool)
                input_schema = self._extract_input_schema(tool)

                # Create Gemini function declaration with enhanced description
                enhanced_description = self._enhance_tool_description_dynamically(
                    tool_name, tool_description, input_schema
                )

                function_declaration = self._create_function_declaration(
                    tool_name, enhanced_description, input_schema
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

    def _enhance_tool_description_dynamically(self, tool_name: str, description: str,
                                            input_schema: Dict[str, Any]) -> str:
        """Enhance tool description with dynamic semantic analysis."""
        if not self.tool_analysis:
            return description

        # Find the semantic analysis for this tool
        tool_semantic = None
        for semantic in self.tool_analysis["tool_semantics"]:
            if semantic.name == tool_name:
                tool_semantic = semantic
                break

        if not tool_semantic:
            return description

        # Build enhanced description
        enhanced_parts = [description]

        # Add capability information
        if tool_semantic.capabilities:
            primary_cap = tool_semantic.capabilities[0]
            enhanced_parts.append(f"Primary capability: {primary_cap.capability_type.value}")

            if tool_semantic.primary_entities:
                entities_str = ", ".join(list(tool_semantic.primary_entities)[:3])
                enhanced_parts.append(f"Works with entities: {entities_str}")

        # Add interaction pattern
        enhanced_parts.append(f"Interaction pattern: {tool_semantic.interaction_pattern}")

        # Add parameter semantic information
        if tool_semantic.parameter_patterns:
            param_info = []
            for param in tool_semantic.parameter_patterns[:3]:  # Limit to first 3
                param_info.append(f"{param.name} ({param.semantic_role})")
            if param_info:
                enhanced_parts.append(f"Key parameters: {', '.join(param_info)}")

        return "\n".join(enhanced_parts)

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
        """Build dynamic system instruction based on semantic tool analysis."""
        if self.tool_analysis:
            # Use dynamic instruction building based on semantic analysis
            return self.instruction_builder.build_dynamic_instruction(self.tool_analysis, context)
        else:
            # Fallback to basic instruction if analysis not available
            return self._build_fallback_instruction(context)

    def _build_fallback_instruction(self, context: Optional[str] = None) -> str:
        """Build a generic fallback instruction when dynamic analysis is not available."""
        instruction = """You are an intelligent AI assistant that analyzes user queries and automatically selects the most appropriate tools from connected MCP servers.

ADAPTIVE APPROACH:
- Analyze tool names and descriptions to understand their purposes
- Match user intent to tool capabilities through semantic understanding
- Extract parameters intelligently based on tool schemas
- Execute tools automatically to provide direct answers

INTELLIGENT TOOL SELECTION:
1. Read tool descriptions carefully to understand what each tool does
2. Match user queries to tool capabilities semantically
3. Extract required parameters from user input using context clues
4. Execute relevant tools immediately without asking permission
5. Process results to provide specific, helpful answers

PARAMETER INTELLIGENCE:
- Analyze parameter names and types to understand what values they expect
- Extract identifiers, names, filters, and options from user queries
- Use semantic understanding to map natural language to parameter values
- Consider parameter requirements and provide appropriate defaults when possible

EXECUTION STRATEGY:
- Always execute tools when they can answer the user's question
- Process tool results intelligently to find the specific information requested
- Provide direct, actionable answers based on tool outputs
- Chain multiple tools when necessary to get complete information"""

        if context:
            instruction += f"\n\nADDITIONAL CONTEXT: {context}"

        return instruction

    # Old hardcoded methods removed - now using dynamic tool analyzer

    async def _call_gemini(self, query: str, system_instruction: str, tools: Optional[List[types.Tool]]) -> Any:
        """Call Gemini AI with query and tools."""
        try:
            if not self.ai_agent:
                raise GeminiError("AI agent not initialized")
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

        # Execute function calls with intelligent processing
        function_results = []
        if function_calls:
            for func_call in function_calls:
                logger.info(f"Calling tool: {func_call['name']} with args: {func_call['args']}")
                tool_result = await self.call_tool(func_call["name"], **func_call["args"])

                # Enhance tool result with intelligent processing
                enhanced_result = self._enhance_tool_result(tool_result, func_call)

                function_results.append({
                    "function_call": func_call,
                    "result": enhanced_result
                })

                # Get intelligent final response from Gemini
                if enhanced_result.get("success") and enhanced_result.get("content") and not final_response:
                    final_response = await self._get_intelligent_final_response(
                        query, func_call, enhanced_result, system_instruction, tools
                    )

        # Set default response if needed
        if not function_calls and not final_response:
            # Check if we can answer with available tools but didn't call any
            final_response = self._generate_proactive_response(query)
        elif not final_response:
            final_response = "I executed the requested tools but couldn't generate a meaningful response."

        return {
            "success": True,
            "query": query,
            "gemini_response": final_response,
            "function_calls": function_calls,
            "function_results": function_results,
            "tools_available": len(self.functions)
        }

    def _generate_proactive_response(self, query: str) -> str:
        """Generate a proactive response when no tools were called but some might be relevant."""
        query_lower = query.lower()

        # Check if query is about capabilities we don't have
        if any(word in query_lower for word in ['weather', 'temperature', 'forecast', 'climate']):
            return "I don't have access to weather information. My capabilities are focused on the available tools for the connected MCP servers."

        # For other queries, suggest using available tools
        if self.functions:
            available_capabilities = []
            for func in self.functions:
                if hasattr(func, 'name'):
                    available_capabilities.append(func.name)

            if available_capabilities:
                return f"I have access to these tools: {', '.join(available_capabilities[:5])}. Please ask me something related to these capabilities."

        return "I couldn't determine how to answer your question with the available tools. Please try being more specific."

    def _enhance_tool_result(self, tool_result: Dict[str, Any], func_call: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance tool result with intelligent processing and formatting."""
        if not tool_result.get("success"):
            return tool_result

        # Extract and process the content intelligently
        original_content = tool_result.get("content", [])
        if not original_content:
            return tool_result

        enhanced_content = []

        for item in original_content:
            if isinstance(item, dict) and "text" in item:
                raw_text = item["text"]

                # Apply intelligent formatting based on content type
                formatted_text = self._apply_intelligent_formatting(raw_text, func_call["name"])

                # Extract structured data if possible
                structured_data = self._extract_structured_data(raw_text)

                enhanced_item = {
                    "text": formatted_text,
                    "raw_text": raw_text,
                    "type": self._classify_content_type(raw_text),
                    "tool_name": func_call["name"]
                }

                if structured_data:
                    enhanced_item["structured_data"] = structured_data

                enhanced_content.append(enhanced_item)
            else:
                enhanced_content.append(item)

        # Create enhanced result
        enhanced_result = tool_result.copy()
        enhanced_result["content"] = enhanced_content
        enhanced_result["processing"] = {
            "enhanced": True,
            "tool_name": func_call["name"],
            "parameters_used": func_call.get("args", {}),
            "content_items": len(enhanced_content)
        }

        return enhanced_result

    def _apply_intelligent_formatting(self, text: str, tool_name: str) -> str:
        """Apply intelligent formatting based on content analysis."""
        if not text or not text.strip():
            return "No data returned"

        # Detect structured content patterns
        lines = text.split('\n')

        # Format lists intelligently
        if len(lines) > 1 and any(line.strip().startswith(('-', '*', '•')) for line in lines):
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('-', '*', '•')):
                    formatted_lines.append(f"• {line}")
                elif line:
                    formatted_lines.append(line)
            return '\n'.join(formatted_lines)

        # Format key-value pairs
        if ':' in text and len(lines) > 1:
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line and ':' in line:
                    key, value = line.split(':', 1)
                    formatted_lines.append(f"**{key.strip()}**: {value.strip()}")
                elif line:
                    formatted_lines.append(line)
            return '\n'.join(formatted_lines)

        return text

    def _extract_structured_data(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from text content."""
        try:
            import json
            import re

            # Try to find JSON-like structures
            json_match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass

            # Extract key-value pairs
            if ':' in text:
                data = {}
                for line in text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        data[key.strip()] = value.strip()
                if data:
                    return data

            return None
        except:
            return None

    def _classify_content_type(self, text: str) -> str:
        """Classify the type of content for better processing."""
        if not text:
            return "empty"

        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ['error', 'failed', 'exception']):
            return "error"
        elif any(keyword in text_lower for keyword in ['list', 'items', 'packages', 'builds']):
            return "list"
        elif any(keyword in text_lower for keyword in ['info', 'details', 'description']):
            return "details"
        elif ':' in text and '\n' in text:
            return "structured"
        elif text.count('\n') > 3:
            return "multiline"
        else:
            return "text"

    async def _get_intelligent_final_response(
        self, query: str, func_call: Dict[str, Any], tool_result: Dict[str, Any],
        system_instruction: str, tools: Optional[List[types.Tool]]
    ) -> str:
        """Get intelligent final response from Gemini with enhanced context."""
        try:
            # Build intelligent context from enhanced results
            context_parts = [f"User Query: {query}"]

            # Add tool execution context
            tool_name = func_call["name"]
            parameters = func_call.get("args", {})
            context_parts.append(f"Executed Tool: {tool_name} with parameters {parameters}")

            # Process enhanced content intelligently
            content = tool_result.get("content", [])
            if content:
                context_parts.append("Tool Results:")
                for item in content:
                    if isinstance(item, dict):
                        content_type = item.get("type", "unknown")
                        text = item.get("text", "")

                        if content_type == "error":
                            context_parts.append(f"ERROR: {text}")
                        elif content_type == "list":
                            context_parts.append(f"LIST DATA: {text}")
                        elif content_type == "details":
                            context_parts.append(f"DETAILS: {text}")
                        else:
                            context_parts.append(f"DATA: {text}")

                        # Add structured data context if available
                        if "structured_data" in item:
                            context_parts.append(f"STRUCTURED: {item['structured_data']}")

            # Create enhanced response prompt
            enhanced_context = "\n".join(context_parts)

            response_prompt = f"""{enhanced_context}

INSTRUCTIONS: Based on the tool execution results above, provide an INTELLIGENT, PROCESSED response that:
1. FILTER and SEARCH through the results to find exactly what the user asked for
2. If user asked for "latest" or "newest", find the most recent item by analyzing timestamps, versions, or build numbers
3. If user asked for a specific item (like "kernel-automotive"), search through all results to find matches
4. If user asked about existence or availability, check if the item exists in the results
5. EXTRACT the specific information requested, don't just dump all data
6. Present a DIRECT ANSWER based on the processed and filtered data
7. If the specific item wasn't found after searching, explain what was checked

CRITICAL: Process the data intelligently to answer the specific question. Don't just present raw results."""

            if not self.ai_agent:
                raise GeminiError("AI agent not initialized")

            final_model_response = self.ai_agent.models.generate_content(
                model=self.model,
                contents=[response_prompt],
                config=types.GenerateContentConfig(
                    tools=tools,
                    system_instruction="You are an expert at interpreting technical data and providing clear, helpful explanations.",
                    temperature=0.2
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

            return response_text or "I processed the tool results but couldn't generate a meaningful response."

        except Exception as e:
            logger.error(f"Error getting intelligent final response: {e}")
            # Fallback to basic response
            content = tool_result.get("content", [])
            if content and isinstance(content[0], dict):
                return f"Based on the {func_call['name']} tool: {content[0].get('text', 'No data available')}"
            return "I executed the requested tool but encountered an issue processing the response."

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

            if not self.ai_agent:
                raise GeminiError("AI agent not initialized")
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
        stats = {
            "initialized": self.initialized,
            "model": self.model,
            "servers_connected": len(self.mcp_clients),
            "tools_available": len(self.functions),
            "servers": list(self.mcp_clients.keys())
        }

        # Add dynamic analysis information
        if self.tool_analysis:
            stats.update({
                "server_type": self.tool_analysis["server_type"],
                "primary_domain": self.tool_analysis["primary_domain"],
                "domain_entities": self.tool_analysis["domain_entities"][:10],  # First 10
                "capability_distribution": {
                    cap: len(tools) for cap, tools in self.tool_analysis["capability_distribution"].items()
                }
            })

        return stats

    def get_dynamic_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the complete dynamic tool analysis for debugging/transparency."""
        return self.tool_analysis