#!/usr/bin/env python3
"""
Dynamic Tool Selection Strategy Demo

This script demonstrates how the Pipeline Toolkit dynamically analyzes
MCP tools and adapts its strategy without hardcoded assumptions.
"""

import asyncio
import json
from typing import Dict, Any, List

# Mock MCP tools from different types of servers to demonstrate versatility
MOCK_BREW_TOOLS = [
    {
        "name": "list_brew_builds",
        "description": "List all builds in a specific tag from the brewery system",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag_name": {"type": "string", "description": "The tag name to list builds for"},
                "limit": {"type": "integer", "description": "Maximum number of builds to return"}
            },
            "required": ["tag_name"]
        }
    },
    {
        "name": "get_brew_tag_info",
        "description": "Get detailed information about a specific tag",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag_name": {"type": "string", "description": "Name of the tag to get info for"}
            },
            "required": ["tag_name"]
        }
    }
]

MOCK_JIRA_TOOLS = [
    {
        "name": "search_issues",
        "description": "Search for JIRA issues using JQL query language",
        "inputSchema": {
            "type": "object",
            "properties": {
                "jql": {"type": "string", "description": "JQL query to search issues"},
                "max_results": {"type": "integer", "description": "Maximum number of issues to return"}
            },
            "required": ["jql"]
        }
    },
    {
        "name": "get_issue",
        "description": "Get detailed information about a specific JIRA issue",
        "inputSchema": {
            "type": "object",
            "properties": {
                "issue_key": {"type": "string", "description": "The issue key (e.g., PROJ-123)"}
            },
            "required": ["issue_key"]
        }
    },
    {
        "name": "create_issue",
        "description": "Create a new JIRA issue in a project",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_key": {"type": "string", "description": "Project key where to create the issue"},
                "summary": {"type": "string", "description": "Issue summary/title"},
                "description": {"type": "string", "description": "Detailed issue description"},
                "issue_type": {"type": "string", "description": "Type of issue (Bug, Task, Story, etc.)"}
            },
            "required": ["project_key", "summary", "issue_type"]
        }
    }
]

MOCK_FILE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file from the filesystem",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to read"},
                "encoding": {"type": "string", "description": "File encoding (default: utf-8)"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "list_directory",
        "description": "List files and directories in a given path",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory_path": {"type": "string", "description": "Path to the directory to list"},
                "include_hidden": {"type": "boolean", "description": "Whether to include hidden files"}
            },
            "required": ["directory_path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file on the filesystem",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path where to write the file"},
                "content": {"type": "string", "description": "Content to write to the file"},
                "overwrite": {"type": "boolean", "description": "Whether to overwrite existing file"}
            },
            "required": ["file_path", "content"]
        }
    }
]


def demonstrate_dynamic_analysis():
    """Demonstrate how dynamic analysis works with different MCP servers."""
    # Import here to avoid dependency issues in demo
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from dynamic_tool_analyzer import DynamicToolAnalyzer, DynamicInstructionBuilder

    analyzer = DynamicToolAnalyzer()
    instruction_builder = DynamicInstructionBuilder(analyzer)

    print("🔍 Dynamic Tool Selection Strategy Demo")
    print("=" * 50)

    # Test with different types of servers
    test_cases = [
        ("Brew/Build System", MOCK_BREW_TOOLS),
        ("JIRA/Project Management", MOCK_JIRA_TOOLS),
        ("File System", MOCK_FILE_TOOLS)
    ]

    for server_name, tools in test_cases:
        print(f"\n📊 Analyzing {server_name} Server")
        print("-" * 40)
        
        # Perform dynamic analysis
        analysis = analyzer.analyze_tool_collection(tools)
        
        print(f"Server Type: {analysis['server_type']}")
        print(f"Primary Domain: {analysis['primary_domain']}")
        print(f"Domain Entities: {', '.join(analysis['domain_entities'][:5])}")
        
        print("\nCapability Distribution:")
        for capability, tool_list in analysis['capability_distribution'].items():
            print(f"  • {capability.title()}: {len(tool_list)} tools")
        
        print(f"\nInteraction Patterns:")
        for pattern, tool_list in analysis['interaction_patterns'].items():
            print(f"  • {pattern.title()}: {len(tool_list)} tools")

        # Show individual tool analysis
        print(f"\nIndividual Tool Analysis:")
        for semantic in analysis['tool_semantics'][:2]:  # Show first 2 tools
            print(f"  🔧 {semantic.name}")
            if semantic.capabilities:
                primary_cap = semantic.capabilities[0]
                print(f"     Primary Capability: {primary_cap.capability_type.value} (confidence: {primary_cap.confidence:.2f})")
            print(f"     Interaction Pattern: {semantic.interaction_pattern}")
            if semantic.primary_entities:
                entities = ", ".join(list(semantic.primary_entities)[:3])
                print(f"     Works with: {entities}")
            if semantic.parameter_patterns:
                params = [f"{p.name}({p.semantic_role})" for p in semantic.parameter_patterns[:3]]
                print(f"     Key Parameters: {', '.join(params)}")

        # Generate dynamic instruction
        print(f"\n📝 Generated Dynamic System Instruction (Preview):")
        instruction = instruction_builder.build_dynamic_instruction(analysis)
        lines = instruction.split('\n')
        for line in lines[:15]:  # Show first 15 lines
            print(f"     {line}")
        if len(lines) > 15:
            print(f"     ... ({len(lines) - 15} more lines)")

        print("\n" + "="*60)


def show_strategy_differences():
    """Show how the strategy adapts to different server types."""
    print("\n🎯 Strategy Adaptation Examples")
    print("=" * 50)
    
    examples = [
        {
            "server_type": "build_system",
            "query": "Show me the latest build",
            "strategy": "Automatically identify 'latest' requirement → call list tools → filter by date/version → return newest"
        },
        {
            "server_type": "project_management", 
            "query": "Find open bugs assigned to me",
            "strategy": "Parse search criteria → call search tools with appropriate filters → process results for specific subset"
        },
        {
            "server_type": "file_system",
            "query": "What's in the config directory",
            "strategy": "Identify directory request → call list/directory tools → format file listing for user"
        },
        {
            "server_type": "generic",
            "query": "Help me understand what tools are available",
            "strategy": "Analyze available capabilities → group by functionality → explain what each tool does"
        }
    ]
    
    for example in examples:
        print(f"\n📋 {example['server_type'].replace('_', ' ').title()} Server")
        print(f"Query: '{example['query']}'")
        print(f"Dynamic Strategy: {example['strategy']}")
    
    print(f"\n✨ Key Benefits:")
    print("• No hardcoded domain assumptions")
    print("• Adapts to any MCP server automatically") 
    print("• Uses semantic analysis instead of keyword matching")
    print("• Builds strategy based on actual tool capabilities")
    print("• Extracts domain entities and concepts dynamically")


def main():
    """Run the complete demonstration."""
    demonstrate_dynamic_analysis()
    show_strategy_differences()
    
    print(f"\n🎉 Demo Complete!")
    print(f"\nThe Pipeline Toolkit now uses semantic analysis to:")
    print("1. 🧠 Understand tool capabilities without hardcoded patterns")
    print("2. 🔍 Extract domain entities and concepts automatically")
    print("3. 📝 Generate tailored instructions for each MCP server")
    print("4. 🎯 Adapt reasoning strategies based on server type")
    print("5. ⚡ Work with any MCP server regardless of naming conventions")


if __name__ == "__main__":
    main() 