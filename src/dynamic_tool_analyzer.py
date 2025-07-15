"""
Dynamic Tool Analyzer - Semantic analysis for generic MCP tool selection.

This module provides dynamic analysis of MCP tools without hardcoded assumptions,
enabling intelligent tool selection for any MCP server regardless of naming
conventions, domains, or specific tool implementations.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class ToolCapabilityType(Enum):
    """Dynamically inferred tool capability types."""
    RETRIEVAL = "retrieval"  # Gets/fetches/retrieves data
    CREATION = "creation"    # Creates/adds new entities
    MODIFICATION = "modification"  # Updates/changes existing entities
    DELETION = "deletion"    # Removes/deletes entities
    SEARCH = "search"        # Searches/queries for data
    VALIDATION = "validation"  # Validates/checks data
    TRANSFORMATION = "transformation"  # Converts/transforms data
    MONITORING = "monitoring"  # Monitors/observes state
    CONTROL = "control"      # Controls/manages operations
    UTILITY = "utility"      # General utility functions


@dataclass
class ToolCapability:
    """Represents a dynamically inferred tool capability."""
    capability_type: ToolCapabilityType
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # Evidence from analysis
    entities: Set[str]  # Entities this tool works with
    scope: str  # Single, multiple, or collection


@dataclass
class ParameterPattern:
    """Represents a parameter usage pattern."""
    name: str
    param_type: str
    purpose: str  # identifier, filter, option, data, etc.
    examples: List[str]
    is_required: bool
    semantic_role: str  # what this parameter represents


@dataclass
class ToolSemantics:
    """Complete semantic analysis of a tool."""
    name: str
    description: str
    capabilities: List[ToolCapability]
    primary_entities: Set[str]
    parameter_patterns: List[ParameterPattern]
    input_cardinality: str  # single, multiple, batch
    output_cardinality: str  # single, multiple, collection
    interaction_pattern: str  # query, command, transaction
    domain_concepts: Set[str]


class DynamicToolAnalyzer:
    """
    Analyzes MCP tools semantically to understand their capabilities
    without hardcoded assumptions about naming or domains.
    """

    def __init__(self):
        self.capability_indicators = self._build_capability_indicators()
        self.entity_extractors = self._build_entity_extractors()
        self.parameter_analyzers = self._build_parameter_analyzers()

    def _build_capability_indicators(self) -> Dict[ToolCapabilityType, Dict[str, float]]:
        """Build semantic indicators for different capability types."""
        return {
            ToolCapabilityType.RETRIEVAL: {
                # Verbs indicating retrieval
                r'\b(get|fetch|retrieve|obtain|acquire|extract|pull|read|load|show|display|view|find)\b': 0.9,
                r'\b(list|enumerate|catalog|index|browse)\b': 0.8,
                r'\b(query|lookup|search|select)\b': 0.7,
                # Nouns/contexts
                r'\b(information|data|details|content|results|records)\b': 0.6,
                r'\b(by|for|from|of|with)\s+\w+': 0.5,
            },
            ToolCapabilityType.CREATION: {
                r'\b(create|add|insert|new|build|make|generate|produce|establish|init)\b': 0.9,
                r'\b(register|enroll|setup|configure|install)\b': 0.8,
                r'\b(post|put|submit|upload|deploy)\b': 0.7,
            },
            ToolCapabilityType.MODIFICATION: {
                r'\b(update|modify|change|edit|alter|adjust|set|configure)\b': 0.9,
                r'\b(patch|fix|correct|revise|amend)\b': 0.8,
                r'\b(enable|disable|activate|deactivate|toggle)\b': 0.7,
            },
            ToolCapabilityType.DELETION: {
                r'\b(delete|remove|drop|purge|clear|clean|erase|destroy)\b': 0.9,
                r'\b(uninstall|unregister|detach|disconnect)\b': 0.8,
            },
            ToolCapabilityType.SEARCH: {
                r'\b(search|find|locate|discover|explore|scan)\b': 0.9,
                r'\b(filter|match|compare|sort|rank)\b': 0.8,
                r'\b(query|lookup|seek|hunt)\b': 0.7,
            },
            ToolCapabilityType.VALIDATION: {
                r'\b(validate|verify|check|test|confirm|ensure)\b': 0.9,
                r'\b(audit|inspect|examine|review)\b': 0.8,
                r'\b(status|health|state|condition)\b': 0.7,
            },
            ToolCapabilityType.TRANSFORMATION: {
                r'\b(convert|transform|translate|format|parse|encode|decode)\b': 0.9,
                r'\b(export|import|migrate|sync|transfer)\b': 0.8,
                r'\b(render|compile|process|analyze)\b': 0.7,
            },
            ToolCapabilityType.MONITORING: {
                r'\b(monitor|watch|observe|track|follow|listen)\b': 0.9,
                r'\b(status|metrics|stats|report|log)\b': 0.8,
                r'\b(health|performance|activity|events)\b': 0.7,
            },
            ToolCapabilityType.CONTROL: {
                r'\b(start|stop|restart|pause|resume|run|execute)\b': 0.9,
                r'\b(control|manage|orchestrate|coordinate)\b': 0.8,
                r'\b(schedule|trigger|invoke|launch)\b': 0.7,
            },
            ToolCapabilityType.UTILITY: {
                r'\b(help|info|version|ping|echo|test)\b': 0.8,
                r'\b(utility|util|tool|helper|support)\b': 0.7,
            }
        }

    def _build_entity_extractors(self) -> List[str]:
        """Build patterns to extract entities from tool descriptions."""
        return [
            # Direct object patterns
            r'\b(get|fetch|list|create|update|delete)\s+(\w+)',
            r'\b(\w+)\s+(information|data|details|list|records)',
            # Parameter-based entities
            r'\b(\w+)_?(id|name|key|identifier)',
            # Domain concepts
            r'\bfor\s+(\w+)',
            r'\bof\s+(\w+)',
            r'\bin\s+(\w+)',
        ]

    def _build_parameter_analyzers(self) -> Dict[str, str]:
        """Build patterns to understand parameter purposes."""
        return {
            r'.*_?(id|identifier|key)$': 'identifier',
            r'.*(name|title|label)$': 'name',
            r'.*(type|kind|category|class)$': 'classifier',
            r'.*(filter|query|search|match).*': 'filter',
            r'.*(limit|count|size|max|min).*': 'constraint',
            r'.*(sort|order|direction).*': 'ordering',
            r'.*(format|output|response).*': 'output_control',
            r'.*(include|exclude|with|without).*': 'inclusion',
            r'.*(date|time|timestamp|when).*': 'temporal',
            r'.*(status|state|condition).*': 'state',
            r'.*(config|setting|option|param).*': 'configuration',
        }

    def analyze_tool(self, tool: Dict[str, Any]) -> ToolSemantics:
        """
        Perform complete semantic analysis of a single tool.

        Args:
            tool: Tool definition from MCP server

        Returns:
            Complete semantic analysis of the tool
        """
        name = self._extract_tool_name(tool)
        description = self._extract_tool_description(tool)
        input_schema = self._extract_input_schema(tool)

        # Analyze capabilities
        capabilities = self._analyze_capabilities(name, description)

        # Extract entities and domain concepts
        entities, domain_concepts = self._extract_entities_and_concepts(name, description)

        # Analyze parameters
        parameter_patterns = self._analyze_parameters(input_schema)

        # Determine interaction patterns
        input_cardinality = self._determine_input_cardinality(parameter_patterns)
        output_cardinality = self._infer_output_cardinality(name, description, capabilities)
        interaction_pattern = self._determine_interaction_pattern(capabilities)

        return ToolSemantics(
            name=name,
            description=description,
            capabilities=capabilities,
            primary_entities=entities,
            parameter_patterns=parameter_patterns,
            input_cardinality=input_cardinality,
            output_cardinality=output_cardinality,
            interaction_pattern=interaction_pattern,
            domain_concepts=domain_concepts
        )

    def _extract_tool_name(self, tool: Dict[str, Any]) -> str:
        """Extract tool name generically."""
        if isinstance(tool, dict):
            return tool.get("name", str(tool))
        elif hasattr(tool, "name"):
            return getattr(tool, "name")
        return str(tool)

    def _extract_tool_description(self, tool: Dict[str, Any]) -> str:
        """Extract tool description generically."""
        if isinstance(tool, dict):
            return tool.get("description", "")
        elif hasattr(tool, "description"):
            return getattr(tool, "description", "")
        return ""

    def _extract_input_schema(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Extract input schema generically."""
        if isinstance(tool, dict):
            return tool.get("inputSchema", {})
        elif hasattr(tool, "inputSchema"):
            return getattr(tool, "inputSchema", {})
        return {}

    def _analyze_capabilities(self, name: str, description: str) -> List[ToolCapability]:
        """Analyze tool capabilities using semantic patterns."""
        capabilities = []
        text = f"{name} {description}".lower()

        for cap_type, patterns in self.capability_indicators.items():
            total_score = 0.0
            evidence = []

            for pattern, weight in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    score = len(matches) * weight
                    total_score += score
                    evidence.extend([f"Found '{match}' (weight: {weight})" for match in matches])

            if total_score > 0.5:  # Threshold for capability detection
                confidence = min(total_score, 1.0)
                capabilities.append(ToolCapability(
                    capability_type=cap_type,
                    confidence=confidence,
                    evidence=evidence,
                    entities=set(),  # Will be filled separately
                    scope="unknown"  # Will be determined later
                ))

        # Sort by confidence
        capabilities.sort(key=lambda x: x.confidence, reverse=True)
        return capabilities

    def _extract_entities_and_concepts(self, name: str, description: str) -> Tuple[Set[str], Set[str]]:
        """Extract entities and domain concepts from tool information."""
        text = f"{name} {description}"
        entities = set()
        concepts = set()

        # Extract using patterns
        for pattern in self.entity_extractors:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    entities.update([m.lower() for m in match if m and len(m) > 2])
                else:
                    if match and len(match) > 2:
                        entities.add(match.lower())

        # Extract domain concepts (nouns that appear frequently)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        for word in words:
            if word not in {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'tool', 'function'}:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Add frequent words as concepts
        for word, freq in word_freq.items():
            if freq > 1 or len(word) > 5:
                concepts.add(word)

        return entities, concepts

    def _analyze_parameters(self, input_schema: Dict[str, Any]) -> List[ParameterPattern]:
        """Analyze parameter patterns to understand their purposes."""
        patterns = []
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for param_name, param_def in properties.items():
            param_type = param_def.get("type", "string")
            param_desc = param_def.get("description", "")

            # Determine semantic role
            semantic_role = "data"
            for pattern, role in self.parameter_analyzers.items():
                if re.match(pattern, param_name, re.IGNORECASE) or \
                   re.search(pattern, param_desc, re.IGNORECASE):
                    semantic_role = role
                    break

            # Determine purpose
            purpose = self._infer_parameter_purpose(param_name, param_desc, param_type)

            # Generate examples based on type and purpose
            examples = self._generate_parameter_examples(param_name, param_type, purpose)

            patterns.append(ParameterPattern(
                name=param_name,
                param_type=param_type,
                purpose=purpose,
                examples=examples,
                is_required=param_name in required,
                semantic_role=semantic_role
            ))

        return patterns

    def _infer_parameter_purpose(self, name: str, description: str, param_type: str) -> str:
        """Infer the purpose of a parameter."""
        text = f"{name} {description}".lower()

        if param_type == "boolean":
            return "toggle"
        elif "id" in name.lower() or "identifier" in text:
            return "identifier"
        elif "name" in name.lower() or "title" in text:
            return "name"
        elif "filter" in text or "match" in text:
            return "filter"
        elif "limit" in text or "count" in text or "size" in text:
            return "constraint"
        elif "type" in name.lower() or "kind" in text or "category" in text:
            return "classifier"
        elif param_type in ["integer", "number"]:
            return "value"
        elif param_type == "array":
            return "collection"
        else:
            return "data"

    def _generate_parameter_examples(self, name: str, param_type: str, purpose: str) -> List[str]:
        """Generate example values for parameters."""
        examples = []

        if purpose == "identifier":
            examples = ["123", "abc-def", "item_001"]
        elif purpose == "name":
            examples = ["example-name", "test-item", "my-resource"]
        elif purpose == "filter":
            examples = ["active", "pending", "completed"]
        elif purpose == "constraint":
            examples = ["10", "50", "100"]
        elif purpose == "toggle":
            examples = ["true", "false"]
        elif param_type == "string":
            examples = ["example-value", "test-string"]
        elif param_type in ["integer", "number"]:
            examples = ["1", "10", "100"]
        elif param_type == "array":
            examples = ["[item1, item2]", "multiple values"]
        else:
            examples = [f"example-{name.lower()}"]

        return examples[:3]  # Limit to 3 examples

    def _determine_input_cardinality(self, parameters: List[ParameterPattern]) -> str:
        """Determine if tool operates on single items, multiple items, or batches."""
        if not parameters:
            return "none"

        # Check for batch/multiple indicators
        for param in parameters:
            if param.param_type == "array" or "list" in param.name.lower():
                return "multiple"
            if "batch" in param.name.lower() or "bulk" in param.name.lower():
                return "batch"

        return "single"

    def _infer_output_cardinality(self, name: str, description: str,
                                capabilities: List[ToolCapability]) -> str:
        """Infer the cardinality of the tool's output."""
        text = f"{name} {description}".lower()

        # Check for list/collection indicators
        if re.search(r'\b(list|all|multiple|collection|array)\b', text):
            return "collection"

        # Check capabilities
        for cap in capabilities:
            if cap.capability_type == ToolCapabilityType.RETRIEVAL:
                if "list" in name.lower() or "all" in text:
                    return "collection"
                else:
                    return "single"
            elif cap.capability_type == ToolCapabilityType.SEARCH:
                return "collection"

        return "single"

    def _determine_interaction_pattern(self, capabilities: List[ToolCapability]) -> str:
        """Determine the interaction pattern of the tool."""
        if not capabilities:
            return "query"

        primary_cap = capabilities[0].capability_type

        if primary_cap in [ToolCapabilityType.RETRIEVAL, ToolCapabilityType.SEARCH]:
            return "query"
        elif primary_cap in [ToolCapabilityType.CREATION, ToolCapabilityType.MODIFICATION,
                           ToolCapabilityType.DELETION]:
            return "command"
        elif primary_cap == ToolCapabilityType.CONTROL:
            return "transaction"
        else:
            return "query"

    def analyze_tool_collection(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a collection of tools to understand the overall MCP server capabilities.

        Args:
            tools: List of tool definitions from MCP server

        Returns:
            Comprehensive analysis of the tool collection
        """
        tool_semantics = [self.analyze_tool(tool) for tool in tools]

        # Aggregate analysis
        all_entities = set()
        all_concepts = set()
        capability_distribution = {}
        interaction_patterns = {}

        for semantic in tool_semantics:
            all_entities.update(semantic.primary_entities)
            all_concepts.update(semantic.domain_concepts)

            for cap in semantic.capabilities:
                cap_type = cap.capability_type.value
                if cap_type not in capability_distribution:
                    capability_distribution[cap_type] = []
                capability_distribution[cap_type].append(semantic.name)

            pattern = semantic.interaction_pattern
            if pattern not in interaction_patterns:
                interaction_patterns[pattern] = []
            interaction_patterns[pattern].append(semantic.name)

        return {
            "tool_count": len(tools),
            "tool_semantics": tool_semantics,
            "domain_entities": sorted(all_entities),
            "domain_concepts": sorted(all_concepts),
            "capability_distribution": capability_distribution,
            "interaction_patterns": interaction_patterns,
            "server_type": self._infer_server_type(all_concepts, capability_distribution),
            "primary_domain": self._infer_primary_domain(all_concepts, all_entities)
        }

    def _infer_server_type(self, concepts: Set[str], capabilities: Dict[str, List[str]]) -> str:
        """Infer the type of MCP server based on analysis."""
        # Check for specific domain indicators
        if any(concept in concepts for concept in ['build', 'package', 'tag', 'repository']):
            return "build_system"
        elif any(concept in concepts for concept in ['issue', 'project', 'ticket', 'sprint']):
            return "project_management"
        elif any(concept in concepts for concept in ['file', 'directory', 'path', 'content']):
            return "file_system"
        elif any(concept in concepts for concept in ['database', 'table', 'record', 'query']):
            return "database"
        elif any(concept in concepts for concept in ['service', 'api', 'endpoint', 'request']):
            return "api_service"
        else:
            return "generic"

    def _infer_primary_domain(self, concepts: Set[str], entities: Set[str]) -> str:
        """Infer the primary domain of the MCP server."""
        all_terms = concepts.union(entities)

        # Domain classification based on terminology
        domain_indicators = {
            "development": ["code", "build", "deploy", "test", "git", "repo", "branch"],
            "project_management": ["project", "issue", "task", "sprint", "board", "ticket"],
            "infrastructure": ["server", "service", "container", "cluster", "resource"],
            "data": ["database", "table", "record", "query", "data", "schema"],
            "content": ["file", "document", "content", "media", "asset"],
            "business": ["user", "customer", "order", "payment", "invoice"],
            "monitoring": ["metric", "log", "alert", "status", "health", "monitor"]
        }

        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in all_terms)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"


class DynamicInstructionBuilder:
    """
    Builds dynamic system instructions based on semantic tool analysis.
    """

    def __init__(self, analyzer: DynamicToolAnalyzer):
        self.analyzer = analyzer

    def build_dynamic_instruction(self, tool_analysis: Dict[str, Any],
                                context: Optional[str] = None) -> str:
        """
        Build a completely dynamic system instruction based on tool analysis.

        Args:
            tool_analysis: Result from analyze_tool_collection
            context: Optional additional context

        Returns:
            Dynamic system instruction optimized for the specific MCP server
        """
        semantics = tool_analysis["tool_semantics"]
        server_type = tool_analysis["server_type"]
        primary_domain = tool_analysis["primary_domain"]
        entities = tool_analysis["domain_entities"]
        capabilities = tool_analysis["capability_distribution"]

        instruction_parts = [
            self._build_core_instruction(),
            self._build_capability_description(capabilities, semantics),
            self._build_entity_awareness(entities, primary_domain),
            self._build_interaction_patterns(tool_analysis["interaction_patterns"]),
            self._build_parameter_intelligence(semantics),
            self._build_reasoning_rules(server_type, primary_domain),
            self._build_response_guidelines(server_type)
        ]

        if context:
            instruction_parts.append(f"\n\nADDITIONAL CONTEXT: {context}")

        return "\n\n".join(instruction_parts)

    def _build_core_instruction(self) -> str:
        """Build the core system instruction."""
        return """You are an intelligent AI assistant that analyzes user queries and automatically selects the most appropriate tools from connected MCP servers. Your approach is completely dynamic and adapts to any MCP server without assumptions about specific domains, naming conventions, or tool structures."""

    def _build_capability_description(self, capabilities: Dict[str, List[str]],
                                    semantics: List[ToolSemantics]) -> str:
        """Build dynamic capability description."""
        parts = ["AVAILABLE CAPABILITIES AND TOOLS:"]

        for cap_type, tool_names in capabilities.items():
            # Find tools with this capability
            relevant_tools = []
            for semantic in semantics:
                if semantic.name in tool_names:
                    relevant_tools.append(semantic)

            if relevant_tools:
                parts.append(f"\n{cap_type.upper().replace('_', ' ')} CAPABILITIES:")
                for tool in relevant_tools:
                    entities_str = ", ".join(list(tool.primary_entities)[:3])
                    if entities_str:
                        parts.append(f"  • {tool.name}: {tool.description} (works with: {entities_str})")
                    else:
                        parts.append(f"  • {tool.name}: {tool.description}")

        return "\n".join(parts)

    def _build_entity_awareness(self, entities: List[str], domain: str) -> str:
        """Build entity awareness section."""
        if not entities:
            return "ENTITY AWARENESS:\n- This server works with general-purpose entities"

        return f"""ENTITY AWARENESS:
This {domain} server works with these entities: {', '.join(entities[:10])}
- When users mention these entities, prioritize tools that work with them
- Look for entity identifiers, names, or references in user queries
- Understand relationships between these entities"""

    def _build_interaction_patterns(self, patterns: Dict[str, List[str]]) -> str:
        """Build interaction patterns section."""
        parts = ["INTERACTION PATTERNS:"]

        for pattern, tools in patterns.items():
            if pattern == "query":
                parts.append(f"- QUERY tools ({len(tools)} available): Use for retrieving information and searching")
            elif pattern == "command":
                parts.append(f"- COMMAND tools ({len(tools)} available): Use for creating, updating, or deleting entities")
            elif pattern == "transaction":
                parts.append(f"- TRANSACTION tools ({len(tools)} available): Use for complex operations requiring coordination")

        return "\n".join(parts)

    def _build_parameter_intelligence(self, semantics: List[ToolSemantics]) -> str:
        """Build parameter intelligence guidelines."""
        # Analyze common parameter patterns
        param_roles = {}
        for semantic in semantics:
            for param in semantic.parameter_patterns:
                role = param.semantic_role
                if role not in param_roles:
                    param_roles[role] = []
                param_roles[role].append(param)

        parts = ["PARAMETER INTELLIGENCE:"]

        for role, params in param_roles.items():
            examples = []
            for param in params[:3]:  # Limit examples
                examples.extend(param.examples[:2])

            if examples:
                parts.append(f"- {role.upper()} parameters: Extract from query (examples: {', '.join(examples[:5])})")

        parts.append("- Analyze user query for values that match parameter types and purposes")
        parts.append("- Use semantic understanding to map natural language to parameter values")

        return "\n".join(parts)

    def _build_reasoning_rules(self, server_type: str, domain: str) -> str:
        """Build reasoning rules based on server type and domain."""
        rules = [
            "INTELLIGENT REASONING RULES:",
            "1. SEMANTIC MATCHING: Match user intent to tool capabilities using semantic understanding",
            "2. ENTITY RECOGNITION: Identify entities mentioned in queries and route to appropriate tools",
            "3. PARAMETER EXTRACTION: Intelligently extract parameters based on their semantic roles",
            "4. CAPABILITY CHAINING: Use multiple tools when needed to answer complex queries",
            "5. CONTEXT AWARENESS: Consider relationships between entities and operations"
        ]

        # Add domain-specific rules
        if domain == "development":
            rules.append("6. VERSION AWARENESS: Understand version numbers, tags, and build identifiers")
        elif domain == "project_management":
            rules.append("6. WORKFLOW AWARENESS: Understand project states, priorities, and assignments")
        elif domain == "data":
            rules.append("6. DATA RELATIONSHIPS: Understand relationships between data entities")

        return "\n".join(rules)

    def _build_response_guidelines(self, server_type: str) -> str:
        """Build response guidelines."""
        return """RESPONSE GUIDELINES:
- Always execute relevant tools automatically without asking permission
- Process results intelligently to answer the specific question asked
- Filter and analyze data to find exactly what the user requested
- If searching for specific items, check all results and provide definitive answers
- For comparative queries, gather all necessary data and make comparisons
- Provide direct, actionable answers based on processed tool results"""