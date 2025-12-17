"""
Dynamic Tool Management - LangChain Tools Integration

Add these tools to agent.py to enable dynamic tool creation via the LangChain agent.

Usage in agent.py:
    from graphweaver_agent.dynamic_tools.agent_tools import (
        check_tool_exists, list_available_tools, create_dynamic_tool,
        run_dynamic_tool, get_tool_source, update_dynamic_tool, delete_dynamic_tool
    )
    
    # Add to your tools list:
    tools = [
        # ... existing tools ...
        check_tool_exists,
        list_available_tools,
        create_dynamic_tool,
        run_dynamic_tool,
        get_tool_source,
        update_dynamic_tool,
        delete_dynamic_tool,
    ]
"""

import json
from typing import Optional
from langchain_core.tools import tool

from .tool_registry import get_registry, ToolType


@tool
def check_tool_exists(tool_name: str) -> str:
    """Check if a tool exists in the registry.
    
    Use this to see if a capability already exists before creating a new tool.
    
    Args:
        tool_name: Name of the tool to check
        
    Returns:
        Whether the tool exists and its type if found
    """
    registry = get_registry()
    
    if registry.tool_exists(tool_name):
        metadata = registry.get_tool(tool_name)
        return f"✓ Tool '{tool_name}' exists\n  Type: {metadata.tool_type.value}\n  Description: {metadata.description}"
    else:
        return f"✗ Tool '{tool_name}' does not exist. You can create it with create_dynamic_tool."


@tool
def list_available_tools(tool_type: Optional[str] = None) -> str:
    """List all available tools in the registry.
    
    Use this to see what tools are available for use.
    
    Args:
        tool_type: Filter by type: 'builtin', 'dynamic', or None for all
        
    Returns:
        List of tools with their descriptions
    """
    registry = get_registry()
    
    type_filter = None
    if tool_type:
        try:
            type_filter = ToolType(tool_type.lower())
        except ValueError:
            return f"Invalid tool type: {tool_type}. Use 'builtin', 'dynamic', or leave empty for all."
    
    tools = registry.list_tools(type_filter)
    
    if not tools:
        return "No tools found."
    
    output = "## Available Tools\n\n"
    
    # Group by type
    builtin = [t for t in tools if t.tool_type == ToolType.BUILTIN]
    dynamic = [t for t in tools if t.tool_type == ToolType.DYNAMIC]
    
    if builtin and (type_filter is None or type_filter == ToolType.BUILTIN):
        output += "### Builtin Tools\n"
        for t in builtin:
            output += f"- **{t.name}**: {t.description[:100]}...\n" if len(t.description) > 100 else f"- **{t.name}**: {t.description}\n"
        output += "\n"
    
    if dynamic and (type_filter is None or type_filter == ToolType.DYNAMIC):
        output += "### Dynamic Tools\n"
        for t in dynamic:
            tags = f" [{', '.join(t.tags)}]" if t.tags else ""
            output += f"- **{t.name}**{tags}: {t.description[:100]}...\n" if len(t.description) > 100 else f"- **{t.name}**{tags}: {t.description}\n"
        output += "\n"
    
    return output


@tool
def create_dynamic_tool(
    name: str,
    code: str,
    description: str,
    dependencies: Optional[str] = None,
    tags: Optional[str] = None
) -> str:
    """Create a new dynamic tool from Python code.
    
    The code MUST define a function named 'run' that will be called when the tool is executed.
    The function should accept keyword arguments and return a string result.
    
    Example code:
```python
    def run(data: str, format: str = "json") -> str:
        # Process data
        result = do_something(data)
        return f"Processed: {result}"
```
    
    Args:
        name: Tool name (alphanumeric and underscores only)
        code: Python code with a 'run' function
        description: What the tool does
        dependencies: Comma-separated list of pip packages (e.g., "pandas,numpy")
        tags: Comma-separated tags for categorization (e.g., "data,analysis")
        
    Returns:
        Success/failure message
    """
    print(f"[CREATE_TOOL] === FUNCTION CALLED ===")
    print(f"[CREATE_TOOL] name: {name}")
    print(f"[CREATE_TOOL] description: {description}")
    print(f"[CREATE_TOOL] code length: {len(code) if code else 0}")
    print(f"[CREATE_TOOL] code:\n{code}")
    print(f"[CREATE_TOOL] dependencies: {dependencies}")
    print(f"[CREATE_TOOL] tags: {tags}")
    
    registry = get_registry()
    
    # Parse dependencies
    deps = [d.strip() for d in dependencies.split(",")] if dependencies else []
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    
    result = registry.create_tool(
        name=name,
        code=code,
        description=description,
        dependencies=deps,
        tags=tag_list,
        test_first=True,
    )
    
    print(f"[CREATE_TOOL] result: {result}")
    
    if result["success"]:
        return f"✓ Tool '{name}' created successfully!\n  File: {result['file']}\n  You can now use run_dynamic_tool to execute it."
    else:
        return f"✗ Failed to create tool: {result['error']}"

@tool
def run_dynamic_tool(tool_name: str, arguments: Optional[str] = None) -> str:
    """Execute a dynamic tool with the given arguments.
    
    Args:
        tool_name: Name of the tool to run
        arguments: JSON string of arguments (e.g., '{"data": "hello", "count": 5}')
        
    Returns:
        Tool execution result or error message
    """
    registry = get_registry()
    
    # Parse arguments
    args = {}
    if arguments:
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as e:
            return f"✗ Invalid arguments JSON: {e}"
    
    result = registry.run_tool(tool_name, args)
    
    if result["success"]:
        return f"✓ Tool Result:\n{result['result']}"
    else:
        error = result.get("error", "Unknown error")
        tb = result.get("traceback", "")
        return f"✗ Tool execution failed: {error}\n{tb}"


@tool
def get_tool_source(tool_name: str) -> str:
    """Get the source code of a dynamic tool.
    
    Use this to inspect or modify existing tools.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Source code or error message
    """
    registry = get_registry()
    
    code = registry.get_tool_code(tool_name)
    
    if code is None:
        metadata = registry.get_tool(tool_name)
        if metadata is None:
            return f"✗ Tool '{tool_name}' not found"
        elif metadata.tool_type == ToolType.BUILTIN:
            return f"✗ '{tool_name}' is a builtin tool - source not available"
        else:
            return f"✗ Source code not available for '{tool_name}'"
    
    return f"## Source code for '{tool_name}':\n\n```python\n{code}\n```"


@tool
def update_dynamic_tool(tool_name: str, code: str, description: Optional[str] = None) -> str:
    """Update an existing dynamic tool with new code.
    
    Args:
        tool_name: Name of the tool to update
        code: New Python code
        description: Optional new description
        
    Returns:
        Success/failure message
    """
    registry = get_registry()
    
    result = registry.update_tool(tool_name, code, description)
    
    if result["success"]:
        return f"✓ {result['message']}"
    else:
        return f"✗ Failed to update tool: {result['error']}"


@tool
def delete_dynamic_tool(tool_name: str) -> str:
    """Delete a dynamic tool from the registry.
    
    Args:
        tool_name: Name of the tool to delete
        
    Returns:
        Success/failure message
    """
    registry = get_registry()
    
    result = registry.delete_tool(tool_name)
    
    if result["success"]:
        return f"✓ {result['message']}"
    else:
        return f"✗ Failed to delete tool: {result['error']}"


# Export all tools
DYNAMIC_TOOL_MANAGEMENT_TOOLS = [
    check_tool_exists,
    list_available_tools,
    create_dynamic_tool,
    run_dynamic_tool,
    get_tool_source,
    update_dynamic_tool,
    delete_dynamic_tool,
]