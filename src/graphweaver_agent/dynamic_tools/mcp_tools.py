"""
Dynamic Tool Management - MCP Server Tools Integration

Add these tools to graphweaver_mcp.py to enable dynamic tool creation via MCP.

Usage in graphweaver_mcp.py:
    # At the top with other imports:
    from graphweaver_agent.dynamic_tools.mcp_tools import register_dynamic_tools
    
    # After creating the mcp instance:
    mcp = FastMCP("graphweaver")
    register_dynamic_tools(mcp)  # Add this line
"""

import json
from typing import Any, Dict, List, Optional

from .tool_registry import get_registry, ToolType, ToolMetadata


def register_dynamic_tools(mcp) -> None:
    """
    Register all dynamic tool management tools with an MCP server instance.
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    async def check_tool_exists(tool_name: str) -> str:
        """Check if a tool exists in the registry.
        
        Use this before creating a new tool to see if the capability already exists.
        
        Args:
            tool_name: Name of the tool to check
        """
        registry = get_registry()
        
        if registry.tool_exists(tool_name):
            metadata = registry.get_tool(tool_name)
            return json.dumps({
                "exists": True,
                "name": tool_name,
                "type": metadata.tool_type.value,
                "description": metadata.description,
                "version": metadata.version,
                "tags": metadata.tags,
            })
        else:
            return json.dumps({
                "exists": False,
                "name": tool_name,
                "message": "Tool does not exist. You can create it with create_dynamic_tool."
            })
    
    @mcp.tool()
    async def list_available_tools(tool_type: Optional[str] = None) -> str:
        """List all available tools in the registry.
        
        Args:
            tool_type: Filter by type: 'builtin', 'dynamic', or empty for all
        """
        registry = get_registry()
        
        type_filter = None
        if tool_type:
            try:
                type_filter = ToolType(tool_type.lower())
            except ValueError:
                return json.dumps({
                    "error": f"Invalid tool type: {tool_type}. Use 'builtin', 'dynamic', or leave empty."
                })
        
        tools = registry.list_tools(type_filter)
        
        return json.dumps({
            "count": len(tools),
            "tools": [
                {
                    "name": t.name,
                    "type": t.tool_type.value,
                    "description": t.description,
                    "version": t.version,
                    "tags": t.tags,
                }
                for t in tools
            ]
        })
    
    @mcp.tool()
    async def create_dynamic_tool(
        name: str,
        code: str,
        description: str,
        dependencies: Optional[str] = None,
        tags: Optional[str] = None,
        test_first: bool = True
    ) -> str:
        """Create a new dynamic tool from Python code.
        
        The code MUST define a function named 'run' that accepts keyword arguments.
        
        Example code:
```
        def run(data: str, format: str = "json") -> str:
            result = process(data)
            return f"Result: {result}"
```
        
        Args:
            name: Tool name (alphanumeric and underscores only)
            code: Python code with a 'run' function
            description: What the tool does
            dependencies: Comma-separated pip packages (e.g., "pandas,numpy")
            tags: Comma-separated tags (e.g., "data,analysis")
            test_first: Whether to test code before saving (default True)
        """
        registry = get_registry()
        
        deps = [d.strip() for d in dependencies.split(",")] if dependencies else []
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        
        result = registry.create_tool(
            name=name,
            code=code,
            description=description,
            dependencies=deps,
            tags=tag_list,
            test_first=test_first,
        )
        
        return json.dumps(result)
    
    @mcp.tool()
    async def run_dynamic_tool(tool_name: str, arguments: Optional[str] = None) -> str:
        """Execute a dynamic tool with arguments.
        
        Args:
            tool_name: Name of the tool to run
            arguments: JSON string of arguments (e.g., '{"data": "hello"}')
        """
        registry = get_registry()
        
        args = {}
        if arguments:
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError as e:
                return json.dumps({"success": False, "error": f"Invalid JSON: {e}"})
        
        result = registry.run_tool(tool_name, args)
        return json.dumps(result)
    
    @mcp.tool()
    async def get_tool_code(tool_name: str) -> str:
        """Get the source code of a dynamic tool.
        
        Args:
            tool_name: Name of the tool
        """
        registry = get_registry()
        
        code = registry.get_tool_code(tool_name)
        metadata = registry.get_tool(tool_name)
        
        if metadata is None:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})
        
        if code is None:
            if metadata.tool_type == ToolType.BUILTIN:
                return json.dumps({"error": "Source not available for builtin tools"})
            return json.dumps({"error": "Source code not available"})
        
        return json.dumps({
            "name": tool_name,
            "code": code,
            "metadata": metadata.to_dict(),
        })
    
    @mcp.tool()
    async def update_dynamic_tool(
        tool_name: str,
        code: str,
        description: Optional[str] = None
    ) -> str:
        """Update an existing dynamic tool with new code.
        
        Args:
            tool_name: Name of the tool to update
            code: New Python code
            description: Optional new description
        """
        registry = get_registry()
        result = registry.update_tool(tool_name, code, description)
        return json.dumps(result)
    
    @mcp.tool()
    async def delete_dynamic_tool(tool_name: str) -> str:
        """Delete a dynamic tool from the registry.
        
        Args:
            tool_name: Name of the tool to delete
        """
        registry = get_registry()
        result = registry.delete_tool(tool_name)
        return json.dumps(result)
    
    @mcp.tool()
    async def export_tool(tool_name: str) -> str:
        """Export a tool's code and metadata for sharing.
        
        Args:
            tool_name: Name of the tool to export
        """
        registry = get_registry()
        data = registry.export_tool(tool_name)
        
        if data is None:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})
        
        return json.dumps(data)
    
    @mcp.tool()
    async def import_tool(tool_data: str) -> str:
        """Import a tool from exported JSON data.
        
        Args:
            tool_data: JSON string from export_tool
        """
        registry = get_registry()
        
        try:
            data = json.loads(tool_data)
        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid JSON: {e}"})
        
        result = registry.import_tool(data)
        return json.dumps(result)
    
    # Register resource for listing tools
    @mcp.resource("tools://registry")
    async def get_tool_registry() -> str:
        """Get the complete tool registry as a resource."""
        registry = get_registry()
        tools = registry.list_tools()
        
        return json.dumps({
            "builtin_count": len([t for t in tools if t.tool_type == ToolType.BUILTIN]),
            "dynamic_count": len([t for t in tools if t.tool_type == ToolType.DYNAMIC]),
            "tools": [t.to_dict() for t in tools],
        })
    
    # Register a prompt template for tool creation
    @mcp.prompt()
    async def create_tool_template(task_description: str) -> str:
        """Generate a template for creating a new dynamic tool.
        
        Args:
            task_description: What the tool should do
        """
        return f"""# Create a Dynamic Tool

Based on the task: "{task_description}"

## Step 1: Check if similar tool exists
Use `check_tool_exists` to see if this capability already exists.

## Step 2: Design the tool
Think about:
- What inputs does it need?
- What should it return?
- What libraries might be needed?

## Step 3: Write the code
```python
def run(param1: str, param2: int = 10) -> str:
    \"\"\"
    Description of what this function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Result description
    \"\"\"
    # Your implementation here
    result = do_something(param1, param2)
    return f"Result: {{result}}"
```

## Step 4: Create the tool
Use `create_dynamic_tool` with:
- name: A short, descriptive name (e.g., "data_processor")
- code: The Python code above
- description: What the tool does
- dependencies: Any pip packages needed (comma-separated)
- tags: Categories for organization (comma-separated)

## Step 5: Test the tool
Use `run_dynamic_tool` to test your new tool with sample inputs.
"""