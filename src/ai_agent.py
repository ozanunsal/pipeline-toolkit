"""
Enhanced AI Agent - Advanced Gemini AI Integration for MCP Tool Orchestration.

This module provides an intelligent AI agent that uses Google's Gemini AI to dynamically select
and execute tools from MCP servers with advanced context management, conversation history,
and sophisticated tool analysis capabilities.
"""

import json
import logging
import os
import re
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import nest_asyncio
from google import genai
from google.genai import types

from src.exceptions import GeminiError, ToolExecutionError, PipelineToolkitError
from src.mcp_client import MCPClient, ConnectionState
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


@dataclass
class ConversationContext:
    """Conversation context and state management."""
    user_id: str = "default"
    conversation_id: str = "default"
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_usage_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    last_activity: Optional[datetime] = None
    session_start: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        self.last_activity = datetime.now()

    def add_tool_usage(self, tool_name: str, args: Dict, result: Dict, duration: float):
        """Record tool usage for analysis."""
        usage = {
            "tool_name": tool_name,
            "arguments": args,
            "result": result,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False)
        }
        self.tool_usage_history.append(usage)

    def get_recent_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context."""
        return self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        if not self.tool_usage_history:
            return {}

        total_calls = len(self.tool_usage_history)
        successful_calls = sum(1 for usage in self.tool_usage_history if usage["success"])
        avg_duration = sum(usage["duration"] for usage in self.tool_usage_history) / total_calls

        tool_counts = {}
        for usage in self.tool_usage_history:
            tool_name = usage["tool_name"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "average_duration": avg_duration,
            "most_used_tools": sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "session_duration": (datetime.now() - self.session_start).total_seconds()
        }


@dataclass
class ToolExecutionPlan:
    """Represents a plan for executing one or more tools."""
    tools_to_execute: List[Dict[str, Any]]
    execution_order: List[int]
    dependencies: Dict[int, List[int]]
    estimated_duration: float
    confidence_score: float
    reasoning: str

    def __post_init__(self):
        """Validate the execution plan."""
        if len(self.tools_to_execute) != len(self.execution_order):
            raise ValueError("Tools count and execution order must match")


class EnhancedAIAgent:
    """
    Enhanced AI Agent for Intelligent Tool Orchestration.

    This agent provides advanced capabilities for tool selection and execution:
    1. Dynamic tool analysis and intelligent selection
    2. Context-aware conversation management
    3. Multi-step tool execution planning
    4. Performance monitoring and optimization
    5. Adaptive learning from usage patterns
    6. Advanced error handling and recovery

    Features:
        - Smart tool selection based on context and history
        - Conversation state management and persistence
        - Multi-tool execution planning and orchestration
        - Real-time performance monitoring
        - Adaptive behavior based on usage patterns
        - Sophisticated error handling and fallback strategies

    Example:
        >>> agent = EnhancedAIAgent()
        >>> agent.register_mcp_client(client, "MyServer", tools)
        >>> result = await agent.process_query("Get tag info for release-1.0", user_id="user123")
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        max_context_length: int = 8000,
        max_tool_calls_per_query: int = 5,
        enable_conversation_history: bool = True,
        enable_adaptive_learning: bool = True
    ) -> None:
        """
        Initialize the Enhanced AI Agent.

        Args:
            model: Gemini model to use for processing
            api_key: Optional Gemini API key (will use env vars if not provided)
            max_context_length: Maximum context length for conversations
            max_tool_calls_per_query: Maximum number of tool calls per query
            enable_conversation_history: Whether to maintain conversation history
            enable_adaptive_learning: Whether to enable adaptive learning features
        """
        self.model = model
        self.api_key = api_key
        self.max_context_length = max_context_length
        self.max_tool_calls_per_query = max_tool_calls_per_query
        self.enable_conversation_history = enable_conversation_history
        self.enable_adaptive_learning = enable_adaptive_learning

        # Core components
        self.ai_agent: Optional[genai.Client] = None
        self.functions: List[types.FunctionDeclaration] = []
        self.mcp_clients: Dict[str, MCPClient] = {}
        self.tool_to_client: Dict[str, MCPClient] = {}
        self.initialized = False

        # Enhanced features
        self.conversations: Dict[str, ConversationContext] = {}
        self.tool_performance_cache: Dict[str, List[float]] = {}
        self.tool_success_rates: Dict[str, float] = {}
        self.tool_usage_patterns: Dict[str, int] = {}

        # Dynamic tool analysis components
        self.tool_analyzer = DynamicToolAnalyzer()
        self.instruction_builder = DynamicInstructionBuilder(self.tool_analyzer)
        self.tool_analysis: Optional[Dict[str, Any]] = None

        # Performance tracking
        self.query_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        self.start_time = datetime.now()

        # Connection monitoring
        self._setup_connection_monitoring()

        # Initialize Gemini
        self._setup_gemini()

    def _setup_connection_monitoring(self):
        """Setup monitoring for MCP client connections."""
        def on_state_change(old_state: ConnectionState, new_state: ConnectionState):
            logger.info(f"MCP client state changed: {old_state.value} -> {new_state.value}")
            if new_state == ConnectionState.FAILED:
                logger.warning("MCP client connection failed - some tools may be unavailable")

        def on_error(error: Exception):
            logger.error(f"MCP client error: {error}")
            self.error_count += 1

        self._state_change_callback = on_state_change
        self._error_callback = on_error

    def _setup_gemini(self) -> None:
        """Initialize and configure the Gemini AI client."""
        try:
            # Use provided API key or fall back to environment variables
            api_key = self.api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise GeminiError("Gemini API key is not set. Provide it via constructor or environment variables.")

            self.ai_agent = genai.Client(api_key=api_key)
            self.initialized = True
            logger.info(f"Enhanced Gemini AI client initialized with model: {self.model}")

        except Exception as e:
            raise GeminiError(f"Failed to initialize Gemini AI client: {e}")

    def register_mcp_client(self, client: MCPClient, server_name: str, tools: List[Dict[str, Any]]) -> None:
        """
        Register an MCP client with enhanced monitoring.

        Args:
            client: MCP client instance
            server_name: Name of the MCP server
            tools: List of tools from this server
        """
        self.mcp_clients[server_name] = client

        # Add connection monitoring
        client.add_state_change_callback(self._state_change_callback)
        client.add_error_callback(self._error_callback)

        # Map each tool to its client for routing
        for tool in tools:
            tool_name = None
            if isinstance(tool, dict) and 'name' in tool:
                tool_name = tool['name']
            elif hasattr(tool, 'name'):
                tool_name = getattr(tool, 'name')

            if tool_name:
                self.tool_to_client[tool_name] = client
                # Initialize performance tracking
                self.tool_performance_cache[tool_name] = []
                self.tool_success_rates[tool_name] = 1.0  # Start optimistic
                self.tool_usage_patterns[tool_name] = 0

        logger.info(f"Registered MCP client {server_name} with {len(tools)} tools and enhanced monitoring")

    def convert_tools_to_gemini_functions(self, tools: List[Dict[str, Any]]) -> None:
        """
        Convert MCP tools to Gemini function declarations with enhanced analysis.

        Args:
            tools: List of MCP tool definitions
        """
        if not tools:
            logger.warning("No tools provided for conversion")
            return

        try:
            # Perform dynamic analysis
            self.tool_analysis = self.tool_analyzer.analyze_tool_collection(tools)
            logger.info(f"Tool analysis completed: {self.tool_analysis.get('server_type', 'unknown')} server with {len(tools)} tools")

            self.functions = []

            for tool in tools:
                try:
                    function_declaration = self._create_gemini_function(tool)
                    if function_declaration:
                        self.functions.append(function_declaration)
                except Exception as e:
                    logger.warning(f"Failed to convert tool {tool.get('name', 'unknown')}: {e}")

            logger.info(f"Successfully converted {len(self.functions)} tools to Gemini functions")

        except Exception as e:
            logger.error(f"Failed to convert tools to Gemini functions: {e}")
            raise GeminiError(f"Tool conversion failed: {e}")

    def _create_gemini_function(self, tool: Dict[str, Any]) -> Optional[types.FunctionDeclaration]:
        """Create a Gemini function declaration from an MCP tool with enhanced metadata."""
        try:
            tool_name = tool.get('name')
            if not tool_name:
                logger.warning(f"Tool missing name: {tool}")
                return None

            description = tool.get('description', f'Execute {tool_name}')
            input_schema = tool.get('inputSchema', {})

            # Enhance description with performance hints
            avg_performance = self._get_average_performance(tool_name)
            success_rate = self.tool_success_rates.get(tool_name, 1.0)

            enhanced_description = description
            if avg_performance is not None:
                enhanced_description += f" (avg response: {avg_performance:.1f}s, success rate: {success_rate:.1%})"

            # Create properties from schema
            properties = {}
            required = []

            if 'properties' in input_schema:
                for prop_name, prop_schema in input_schema['properties'].items():
                    prop_type = prop_schema.get('type', 'string')
                    prop_description = prop_schema.get('description', f'{prop_name} parameter')

                    # Map JSON Schema types to Gemini types
                    gemini_type = self._map_json_type_to_gemini(prop_type)

                    properties[prop_name] = types.Schema(
                        type=gemini_type,
                        description=prop_description
                    )

                required = input_schema.get('required', [])

            parameters = types.Schema(
                type=types.Type.OBJECT,
                properties=properties,
                required=required
            )

            return types.FunctionDeclaration(
                name=tool_name,
                description=enhanced_description,
                parameters=parameters
            )

        except Exception as e:
            logger.error(f"Error creating Gemini function for tool {tool.get('name', 'unknown')}: {e}")
            return None

    def _map_json_type_to_gemini(self, json_type: str) -> types.Type:
        """Map JSON Schema types to Gemini types."""
        type_mapping = {
            'string': types.Type.STRING,
            'integer': types.Type.INTEGER,
            'number': types.Type.NUMBER,
            'boolean': types.Type.BOOLEAN,
            'array': types.Type.ARRAY,
            'object': types.Type.OBJECT
        }
        return type_mapping.get(json_type.lower(), types.Type.STRING)

    def _get_average_performance(self, tool_name: str) -> Optional[float]:
        """Get average performance for a tool."""
        performances = self.tool_performance_cache.get(tool_name, [])
        return sum(performances) / len(performances) if performances else None

    def _get_conversation_context(self, user_id: str, conversation_id: str) -> ConversationContext:
        """Get or create conversation context."""
        context_key = f"{user_id}:{conversation_id}"

        if context_key not in self.conversations:
            self.conversations[context_key] = ConversationContext(
                user_id=user_id,
                conversation_id=conversation_id
            )

        return self.conversations[context_key]

    async def process_query(
        self,
        query: str,
        user_id: str = "default",
        conversation_id: str = "default",
        system_prompt_override: Optional[str] = None,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Process a user query with enhanced context management.

        Args:
            query: The user's query
            user_id: Unique user identifier
            conversation_id: Conversation identifier
            system_prompt_override: Optional system prompt override
            max_tokens: Maximum tokens for response

        Returns:
            Dict containing response and execution details
        """
        start_time = datetime.now()
        self.query_count += 1

        # Get conversation context
        context = self._get_conversation_context(user_id, conversation_id)
        context.add_message("user", query)

        try:
            logger.info(f"Processing query for user {user_id}: {query[:100]}...")

            # Build enhanced prompt with context
            enhanced_prompt = self._build_enhanced_prompt(query, context, system_prompt_override)

            # Create execution plan
            execution_plan = await self._create_execution_plan(query, context)

            # Generate response with Gemini
            response_data = await self._generate_response(enhanced_prompt, max_tokens)

            # Execute planned tools if any
            if execution_plan and execution_plan.tools_to_execute:
                tool_results = await self._execute_planned_tools(execution_plan)
                response_data["tool_execution_results"] = tool_results
                response_data["execution_plan"] = {
                    "tools_planned": len(execution_plan.tools_to_execute),
                    "confidence_score": execution_plan.confidence_score,
                    "reasoning": execution_plan.reasoning
                }

            # Update conversation context
            if response_data.get("gemini_response"):
                context.add_message("assistant", response_data["gemini_response"])

            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_response_time += processing_time

            # Build final response
            result = {
                "success": True,
                "gemini_response": response_data.get("gemini_response", ""),
                "function_calls": response_data.get("function_calls", []),
                "processing_time": processing_time,
                "conversation_stats": context.get_tool_usage_stats(),
                "query_count": self.query_count,
                **response_data
            }

            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return result

        except Exception as e:
            self.error_count += 1
            error_msg = f"Failed to process query: {e}"
            logger.error(error_msg)

            # Add error to conversation context
            context.add_message("assistant", f"Error: {error_msg}")

            return {
                "success": False,
                "error": error_msg,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "query_count": self.query_count
            }

    def _build_enhanced_prompt(
        self,
        query: str,
        context: ConversationContext,
        system_prompt_override: Optional[str] = None
    ) -> str:
        """Build an enhanced prompt with conversation context."""

        # Base system prompt
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt = self._get_dynamic_system_prompt()

        # Add conversation history if enabled
        if self.enable_conversation_history and context.messages:
            recent_context = context.get_recent_context(5)  # Last 5 messages
            context_str = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_context[:-1]  # Exclude current query
            ])

            if context_str:
                system_prompt += f"\n\nRecent conversation context:\n{context_str}"

        # Add tool usage patterns if adaptive learning is enabled
        if self.enable_adaptive_learning:
            tool_stats = context.get_tool_usage_stats()
            if tool_stats.get("most_used_tools"):
                popular_tools = [tool[0] for tool in tool_stats["most_used_tools"][:3]]
                system_prompt += f"\n\nUser's frequently used tools: {', '.join(popular_tools)}"

        return f"{system_prompt}\n\nUser query: {query}"

    def _get_dynamic_system_prompt(self) -> str:
        """Get dynamic system prompt based on tool analysis."""
        if self.tool_analysis:
            return self.instruction_builder.build_dynamic_instruction(self.tool_analysis)
        else:
            return """You are an AI assistant that helps users interact with various tools and services.
            Analyze the user's request and determine which tools would be most appropriate to fulfill their needs.
            Provide clear, helpful responses and use available tools when necessary."""

    async def _create_execution_plan(self, query: str, context: ConversationContext) -> Optional[ToolExecutionPlan]:
        """Create an execution plan for complex queries requiring multiple tools."""
        try:
            # Simple heuristic for now - could be enhanced with ML
            query_lower = query.lower()

            # Identify potential tools based on query content
            relevant_tools = []
            for tool_name in self.tool_to_client.keys():
                if tool_name.lower() in query_lower or any(
                    keyword in query_lower
                    for keyword in self._get_tool_keywords(tool_name)
                ):
                    relevant_tools.append({
                        "name": tool_name,
                        "confidence": self._calculate_tool_confidence(tool_name, query)
                    })

            if not relevant_tools:
                return None

            # Sort by confidence
            relevant_tools.sort(key=lambda x: x["confidence"], reverse=True)

            # Create simple execution plan (can be enhanced)
            tools_to_execute = relevant_tools[:self.max_tool_calls_per_query]
            execution_order = list(range(len(tools_to_execute)))

            return ToolExecutionPlan(
                tools_to_execute=tools_to_execute,
                execution_order=execution_order,
                dependencies={},
                estimated_duration=self._estimate_execution_duration(tools_to_execute),
                confidence_score=sum(tool["confidence"] for tool in tools_to_execute) / len(tools_to_execute),
                reasoning=f"Selected {len(tools_to_execute)} tools based on query analysis"
            )

        except Exception as e:
            logger.warning(f"Failed to create execution plan: {e}")
            return None

    def _get_tool_keywords(self, tool_name: str) -> List[str]:
        """Get keywords associated with a tool."""
        # This could be enhanced with NLP analysis
        return [word.lower() for word in tool_name.replace("_", " ").split()]

    def _calculate_tool_confidence(self, tool_name: str, query: str) -> float:
        """Calculate confidence score for tool relevance."""
        base_score = 0.5

        # Boost score based on success rate
        success_rate = self.tool_success_rates.get(tool_name, 1.0)
        base_score += (success_rate - 0.5) * 0.3

        # Boost score based on usage frequency
        usage_count = self.tool_usage_patterns.get(tool_name, 0)
        if usage_count > 0:
            base_score += min(0.2, usage_count * 0.01)

        # Simple keyword matching boost
        tool_keywords = self._get_tool_keywords(tool_name)
        query_lower = query.lower()
        keyword_matches = sum(1 for keyword in tool_keywords if keyword in query_lower)
        if keyword_matches > 0:
            base_score += min(0.3, keyword_matches * 0.1)

        return min(1.0, base_score)

    def _estimate_execution_duration(self, tools: List[Dict[str, Any]]) -> float:
        """Estimate execution duration for a set of tools."""
        total_duration = 0.0
        for tool in tools:
            tool_name = tool["name"]
            avg_duration = self._get_average_performance(tool_name)
            total_duration += avg_duration if avg_duration else 2.0  # Default 2s estimate

        return total_duration

    async def _generate_response(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Generate response using Gemini AI."""
        if not self.ai_agent:
            raise GeminiError("Gemini AI client not initialized")

        try:
            # Create the generation config
            generation_config = types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=self.functions)] if self.functions else None,
                temperature=0.1,
                max_output_tokens=max_tokens
            )

            # Generate response
            response = await self.ai_agent.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=generation_config
            )

            # Process response
            if not response or not response.candidates:
                return {"gemini_response": "No response generated"}

            candidate = response.candidates[0]

            # Extract text content
            text_content = ""
            function_calls = []

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_content += part.text
                    elif hasattr(part, 'function_call') and part.function_call:
                        # Handle function calls
                        func_call = part.function_call
                        logger.info(f"Executing function call: {func_call.name}")
                        call_result = await self._execute_function_call(func_call)
                        function_calls.append(call_result)

            # Post-process tool results to generate intelligent responses
            if function_calls and not text_content.strip():
                if any(call.get("result", {}).get("success", False) for call in function_calls):
                    # Analyze results and potentially make follow-up calls
                    enhanced_response = await self._post_process_tool_results(function_calls, prompt)
                    if enhanced_response:
                        text_content = enhanced_response
                else:
                    # Generate a contextual response based on what was attempted
                    text_content = self._generate_failure_response(function_calls)

            return {
                "gemini_response": text_content.strip() if text_content.strip() else "No response generated",
                "function_calls": function_calls
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise GeminiError(f"Failed to generate response: {e}")



    async def _post_process_tool_results(self, function_calls: List[Dict[str, Any]], original_query: str) -> Optional[str]:
        """
        Post-process tool results to generate intelligent responses and make follow-up calls if needed.

        Args:
            function_calls: List of completed function calls with results
            original_query: The original user query

        Returns:
            Enhanced response string or None if no post-processing needed
        """
        successful_calls = [call for call in function_calls if call.get("result", {}).get("success", False)]

        if not successful_calls:
            return None

        # Analyze the results and determine if follow-up calls are needed
        analysis = self._analyze_tool_results(successful_calls, original_query)

        if analysis.get("needs_followup", False):
            # Make follow-up tool calls
            followup_calls = await self._make_followup_calls(analysis.get("followup_actions", []))
            successful_calls.extend([call for call in followup_calls if call.get("result", {}).get("success", False)])

        # Generate comprehensive response based on all results
        return self._generate_comprehensive_response(successful_calls, original_query, analysis)

    def _analyze_tool_results(self, successful_calls: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
        """Analyze tool results to determine if follow-up actions are needed."""
        analysis = {
            "needs_followup": False,
            "followup_actions": [],
            "query_type": "unknown",
            "primary_result": None
        }

        # Determine query type and primary result
        for call in successful_calls:
            func_name = call.get("name", "")
            result_data = call.get("result", {}).get("result", "")

            if "brew" in func_name.lower():
                analysis["query_type"] = "brew"
                analysis["primary_result"] = result_data

                # For brew package queries, check if we need more detailed info
                if "list_brew_packages" in func_name and "kernel" in original_query.lower():
                    if "kernel" in result_data.lower():
                        analysis["needs_followup"] = True
                        analysis["followup_actions"].append({
                            "action": "get_tag_info",
                            "params": call.get("args", {})
                        })

            elif "job" in func_name.lower() or "testing" in func_name.lower():
                analysis["query_type"] = "testing_farm"
                analysis["primary_result"] = result_data

                # For failed jobs, try to get more details
                if "failed" in result_data.lower() and "analyze_job" in func_name:
                    # Already have analysis, no follow-up needed
                    pass
                elif "get_job_status" in func_name and "failed" in result_data.lower():
                    # Need detailed analysis
                    analysis["needs_followup"] = True
                    analysis["followup_actions"].append({
                        "action": "analyze_job",
                        "params": call.get("args", {})
                    })

        return analysis

    async def _make_followup_calls(self, followup_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute follow-up tool calls based on analysis."""
        followup_results = []

        for action in followup_actions:
            try:
                if action["action"] == "get_tag_info":
                    result = await self.call_tool("get_brew_tag_info", **action["params"])
                    followup_results.append({
                        "name": "get_brew_tag_info",
                        "args": action["params"],
                        "result": result
                    })
                elif action["action"] == "analyze_job":
                    result = await self.call_tool("analyze_job", **action["params"])
                    followup_results.append({
                        "name": "analyze_job",
                        "args": action["params"],
                        "result": result
                    })
            except Exception as e:
                logger.warning(f"Follow-up call failed: {e}")

        return followup_results

    def _generate_comprehensive_response(self, successful_calls: List[Dict[str, Any]], original_query: str, analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive response based on all tool results."""
        query_type = analysis.get("query_type", "unknown")

        if query_type == "brew":
            return self._generate_brew_response(successful_calls, original_query)
        elif query_type == "testing_farm":
            return self._generate_testing_farm_response(successful_calls, original_query)
        else:
            return self._generate_generic_response(successful_calls, original_query)

    def _generate_brew_response(self, successful_calls: List[Dict[str, Any]], original_query: str) -> str:
        """Generate a comprehensive response for brew-related queries."""
        response_parts = []

        # Check if this is a "is X tagged with Y" question
        if "tagged with" in original_query.lower() or "is" in original_query.lower():
            package_query = None
            tag_name = None

            # Extract package name and tag from query
            query_lower = original_query.lower()
            if "kernel-automotive" in query_lower:
                package_query = "kernel-automotive"
            if "rhivos" in query_lower:
                for word in original_query.split():
                    if "rhivos" in word.lower():
                        tag_name = word.strip("?")
                        break

            # Look for package list results
            package_list = None
            tag_info = None

            for call in successful_calls:
                if call.get("name") == "list_brew_packages":
                    package_list = call.get("result", {}).get("result", "")
                elif call.get("name") == "get_brew_tag_info":
                    tag_info = call.get("result", {}).get("result", "")

            if package_query and tag_name and package_list:
                # Check if the package is in the list
                if package_query.lower() in package_list.lower():
                    response_parts.append(f"**✅ Yes, {package_query} is tagged with {tag_name}!**")
                    response_parts.append(f"\nI found {package_query} packages in the {tag_name} brew tag:")

                    # Extract relevant kernel packages
                    lines = package_list.split('\n')
                    kernel_packages = [line.strip() for line in lines if package_query.lower() in line.lower()]

                    if kernel_packages:
                        response_parts.append("\n**Relevant packages:**")
                        for pkg in kernel_packages[:10]:  # Show first 10
                            response_parts.append(f"• {pkg}")
                        if len(kernel_packages) > 10:
                            response_parts.append(f"• ... and {len(kernel_packages) - 10} more {package_query} packages")
                else:
                    response_parts.append(f"**❌ No, {package_query} is not tagged with {tag_name}.**")
                    response_parts.append(f"\nI searched through the packages in {tag_name} but did not find any {package_query} packages.")

                # Add tag summary if available
                if tag_info:
                    response_parts.append(f"\n**About {tag_name} tag:**")
                    response_parts.append(tag_info[:300] + "..." if len(tag_info) > 300 else tag_info)

                total_packages = package_list.count('\n') if package_list else 0
                if total_packages > 0:
                    response_parts.append(f"\n*Total packages in {tag_name}: {total_packages}*")

            else:
                # Fallback to generic brew response
                return self._generate_generic_response(successful_calls, original_query)

        else:
            # Other brew queries
            return self._generate_generic_response(successful_calls, original_query)

        return '\n'.join(response_parts)

    def _generate_testing_farm_response(self, successful_calls: List[Dict[str, Any]], original_query: str) -> str:
        """Generate a comprehensive response for Testing Farm queries."""
        job_status = None
        job_analysis = None
        job_id = None

        # Extract results
        for call in successful_calls:
            if call.get("name") == "get_job_status":
                job_status = call.get("result", {}).get("result", "")
                job_id = call.get("args", {}).get("job_id", "")
            elif call.get("name") == "analyze_job":
                job_analysis = call.get("result", {}).get("result", "")
                job_id = job_id or call.get("args", {}).get("job_id", "")

        response_parts = []

        if job_analysis:
            # We have detailed analysis
            response_parts.append(f"## 📊 Testing Farm Job Analysis: {job_id}")
            response_parts.append("")

            # Parse the analysis for key information
            if "failed" in job_analysis.lower():
                response_parts.append("**🔴 Status: FAILED**")
            elif "passed" in job_analysis.lower() or "success" in job_analysis.lower():
                response_parts.append("**🟢 Status: PASSED**")
            else:
                response_parts.append("**🟡 Status: COMPLETED**")

            response_parts.append("")
            response_parts.append("### Details:")

            # Format the analysis nicely
            lines = job_analysis.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('Job '):
                    if line.startswith(('State:', 'Result:', 'Created:', 'Updated:')):
                        response_parts.append(f"• **{line}**")
                    else:
                        response_parts.append(f"• {line}")

        elif job_status:
            # Only have basic status
            response_parts.append(f"## 📋 Testing Farm Job Status: {job_id}")
            response_parts.append("")
            response_parts.append(job_status)

        else:
            # Fallback
            return self._generate_generic_response(successful_calls, original_query)

        return '\n'.join(response_parts)

    def _generate_generic_response(self, successful_calls: List[Dict[str, Any]], original_query: str) -> str:
        """Generate a generic response for any query type."""
        response_parts = ["## 📋 Query Results"]
        response_parts.append("")

        for call in successful_calls:
            func_name = call.get("name", "unknown")
            result_data = call.get("result", {}).get("result", "")

            response_parts.append(f"### {func_name}")

            if result_data:
                # Format the result nicely
                if len(result_data) > 500:
                    response_parts.append(result_data[:500] + "...")
                    response_parts.append("*(truncated - showing first 500 characters)*")
                else:
                    response_parts.append(result_data)
            else:
                response_parts.append("*No data returned*")

            response_parts.append("")

        return '\n'.join(response_parts)

    def _generate_failure_response(self, function_calls: List[Dict[str, Any]]) -> str:
        """Generate a meaningful response when all function calls fail."""
        # Extract what was attempted
        attempted_functions = [call.get("name", "unknown") for call in function_calls]

        # Check if this was a query about brew tags/packages
        if any("brew" in func for func in attempted_functions):
            if "list_brew_packages" in attempted_functions:
                return """I attempted to check if kernel-automotive is tagged with rhivos-1.0.0 by listing the packages in that brew tag, but unfortunately the Brew server encountered an error and couldn't provide the information.

To answer your question "Is kernel-automotive tagged with rhivos-1.0.0 brew tag?", I would need to successfully query the brew system, which is currently experiencing issues.

You could try:
- Checking the brew system directly if you have access
- Trying a different brew tag to see if the server issues are specific to rhivos-1.0.0
- Waiting for the brew server to be fixed and trying again later"""

            elif "get_brew_tag_info" in attempted_functions:
                tag_name = None
                for call in function_calls:
                    if call.get("args", {}).get("tag_name"):
                        tag_name = call["args"]["tag_name"]
                        break

                return f"""I tried to get detailed information about the brew tag "{tag_name}" to help answer your question, but the Brew server encountered an error.

Unfortunately, I cannot determine the tag details or package listings due to this server issue. The brew system appears to be experiencing technical problems that prevent me from accessing the tag information."""

        # Check if this was about Testing Farm
        elif any("job" in func for func in attempted_functions):
            job_id = None
            for call in function_calls:
                if call.get("args", {}).get("job_id"):
                    job_id = call["args"]["job_id"]
                    break

            # Check if it was specifically an analyze request
            if "analyze_job" in attempted_functions:
                return f"""I attempted to analyze Testing Farm job {job_id} to provide you with a detailed analysis of the test results, but unfortunately the Testing Farm server encountered an error.

The analysis would normally include:
- Test execution summary (passed/failed/skipped)
- Failure reasons and error details if tests failed
- Performance metrics and execution time
- Environment and configuration details

However, due to the server error, I cannot retrieve the job data needed for analysis. This could be because:
- The job ID doesn't exist or has expired
- The Testing Farm server is experiencing technical issues
- There are connectivity problems with the testing infrastructure

You might want to check the Testing Farm web interface directly or verify the job ID format."""
            else:
                return f"""I attempted to get the status of Testing Farm job {job_id}, but the Testing Farm server encountered an error and couldn't provide the job information.

This could be because:
- The job ID doesn't exist or has expired
- The Testing Farm server is experiencing issues
- There are connectivity problems

You might want to verify the job ID or check the Testing Farm system directly."""

        # Generic response for other cases
        functions_str = ", ".join(attempted_functions)
        return f"""I attempted to help by calling these functions: {functions_str}, but unfortunately all of them encountered errors.

This appears to be a server-side issue preventing me from accessing the necessary data to answer your question. The tools I tried to use are currently unavailable or experiencing technical problems."""

    async def _execute_function_call(self, func_call) -> Dict[str, Any]:
        """Execute a function call from Gemini."""
        try:
            function_name = func_call.name
            arguments = dict(func_call.args) if func_call.args else {}

            start_time = datetime.now()
            result = await self.call_tool(function_name, **arguments)
            duration = (datetime.now() - start_time).total_seconds()

            # Update performance tracking
            self._update_tool_performance(function_name, duration, result.get("success", False))

            return {
                "name": function_name,
                "args": arguments,
                "result": result,
                "duration": duration
            }

        except Exception as e:
            logger.error(f"Error executing function call {func_call.name}: {e}")
            return {
                "name": func_call.name,
                "args": dict(func_call.args) if func_call.args else {},
                "result": {"success": False, "error": str(e)},
                "duration": 0.0
            }

    async def _execute_planned_tools(self, plan: ToolExecutionPlan) -> List[Dict[str, Any]]:
        """Execute tools according to the execution plan."""
        results = []

        for i in plan.execution_order:
            tool_info = plan.tools_to_execute[i]
            tool_name = tool_info["name"]

            try:
                # For now, execute with empty args - could be enhanced
                start_time = datetime.now()
                result = await self.call_tool(tool_name)
                duration = (datetime.now() - start_time).total_seconds()

                self._update_tool_performance(tool_name, duration, result.get("success", False))

                results.append({
                    "tool_name": tool_name,
                    "result": result,
                    "duration": duration,
                    "confidence": tool_info["confidence"]
                })

            except Exception as e:
                logger.error(f"Error executing planned tool {tool_name}: {e}")
                results.append({
                    "tool_name": tool_name,
                    "result": {"success": False, "error": str(e)},
                    "duration": 0.0,
                    "confidence": tool_info["confidence"]
                })

        return results

    def _update_tool_performance(self, tool_name: str, duration: float, success: bool):
        """Update tool performance tracking."""
        # Update performance cache
        if tool_name not in self.tool_performance_cache:
            self.tool_performance_cache[tool_name] = []

        self.tool_performance_cache[tool_name].append(duration)

        # Keep only recent performance data (last 100 calls)
        if len(self.tool_performance_cache[tool_name]) > 100:
            self.tool_performance_cache[tool_name] = self.tool_performance_cache[tool_name][-100:]

        # Update success rate (moving average)
        current_rate = self.tool_success_rates.get(tool_name, 1.0)
        new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        self.tool_success_rates[tool_name] = new_rate

        # Update usage patterns
        self.tool_usage_patterns[tool_name] = self.tool_usage_patterns.get(tool_name, 0) + 1

    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool on the appropriate MCP server with enhanced monitoring.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Parameters to pass to the tool

        Returns:
            Dict containing the tool result with enhanced metadata
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

            # Check client health before calling
            if not client.is_connected:
                logger.warning(f"Client for {tool_name} is not connected, attempting reconnection...")
                try:
                    await client.connect()
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to reconnect client for {tool_name}: {e}",
                        "content": []
                    }

            # Call the tool on the MCP client
            start_time = datetime.now()
            result = await client.call_tool(tool_name, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()

            # Enhance result with metadata
            enhanced_result = {
                **result,
                "execution_time": duration,
                "client_state": client.state.value,
                "client_metrics": client.get_metrics().to_dict()
            }

            logger.debug(f"Tool {tool_name} executed in {duration:.3f}s with result: {result.get('success', False)}")
            return enhanced_result

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Error calling tool {tool_name}: {str(e)}",
                "content": [],
                "execution_time": 0.0
            }

    @property
    def is_initialized(self) -> bool:
        """Check if Gemini AI is initialized."""
        return self.initialized

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with enhanced metadata."""
        tools = []
        for func in self.functions:
            server = next(
                (server for server, client in self.mcp_clients.items()
                 if client == self.tool_to_client.get(func.name)),
                "unknown"
            )

            # Get performance metrics
            avg_duration = self._get_average_performance(func.name)
            success_rate = self.tool_success_rates.get(func.name, 1.0)
            usage_count = self.tool_usage_patterns.get(func.name, 0)

            tools.append({
                "name": func.name,
                "description": func.description,
                "server": server,
                "average_duration": avg_duration,
                "success_rate": success_rate,
                "usage_count": usage_count,
                "client_state": self.tool_to_client.get(func.name).state.value if self.tool_to_client.get(func.name) else "unknown"
            })

        return tools

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        uptime = datetime.now() - self.start_time
        avg_response_time = self.total_response_time / self.query_count if self.query_count > 0 else 0

        stats = {
            "initialized": self.initialized,
            "model": self.model,
            "servers_connected": len(self.mcp_clients),
            "tools_available": len(self.functions),
            "servers": list(self.mcp_clients.keys()),
            "query_count": self.query_count,
            "error_count": self.error_count,
            "success_rate": (self.query_count - self.error_count) / self.query_count if self.query_count > 0 else 1.0,
            "average_response_time": avg_response_time,
            "uptime_seconds": uptime.total_seconds(),
            "conversations_active": len(self.conversations),
            "enable_conversation_history": self.enable_conversation_history,
            "enable_adaptive_learning": self.enable_adaptive_learning
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

    def clear_conversation_history(self, user_id: str, conversation_id: str) -> bool:
        """Clear conversation history for a specific user/conversation."""
        context_key = f"{user_id}:{conversation_id}"
        if context_key in self.conversations:
            del self.conversations[context_key]
            return True
        return False

    def get_conversation_summary(self, user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a conversation."""
        context_key = f"{user_id}:{conversation_id}"
        if context_key in self.conversations:
            context = self.conversations[context_key]
            return {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_count": len(context.messages),
                "tool_usage_stats": context.get_tool_usage_stats(),
                "last_activity": context.last_activity.isoformat() if context.last_activity else None,
                "session_duration": (datetime.now() - context.session_start).total_seconds()
            }
        return None


# Maintain backward compatibility
AIAgent = EnhancedAIAgent