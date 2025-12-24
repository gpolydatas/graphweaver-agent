"""
# /home/abdul/graphweaver-agent/src/graphweaver_agent/dynamic_tools/agent_tools.py
#
Dynamic tool management with STREAMING support.
Uses get_stream_writer() to emit progress during tool creation/execution.
"""
import os
import json
import importlib.util
from typing import Optional, Dict, Any, List
from pathlib import Path

from langchain_core.tools import tool

# Try to import streaming support
try:
    from langgraph.config import get_stream_writer
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    def get_stream_writer():
        """Dummy writer when streaming not available."""
        class DummyWriter:
            def __call__(self, data):
                pass
        return DummyWriter()

from .tool_registry import ToolRegistry

# Initialize registry
DYNAMIC_TOOLS_DIR = os.environ.get(
    "DYNAMIC_TOOLS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "dynamic_tools")
)
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get or create the tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry(DYNAMIC_TOOLS_DIR)
    return _registry


def emit(data: dict):
    """Emit streaming data if available."""
    try:
        writer = get_stream_writer()
        writer(data)
    except Exception:
        pass  # Streaming not available or not in streaming context


# =============================================================================
# Dynamic Tool Management Tools (with Streaming)
# =============================================================================

@tool
def check_tool_exists(tool_name: str) -> str:
    """Check if a dynamic tool already exists.
    
    ALWAYS call this before creating a new tool to avoid duplicates.
    
    Args:
        tool_name: Name of the tool to check
    
    Returns:
        Information about whether the tool exists and its details if it does
    """
    emit({"status": "checking", "tool": tool_name})
    
    registry = get_registry()
    
    # Check dynamic tools
    if registry.tool_exists(tool_name):
        tool_info = registry.get_tool_info(tool_name)
        emit({"status": "found", "tool": tool_name, "type": "dynamic"})
        return f"✓ Dynamic tool '{tool_name}' EXISTS\n  Description: {tool_info.get('description', 'N/A')}\n  File: {tool_info.get('file_path', 'N/A')}"
    
    # Check builtin tools
    builtin_tools = [
        "configure_database", "test_database_connection", "list_database_tables",
        "get_table_schema", "get_column_stats", "run_fk_discovery",
        "analyze_potential_fk", "validate_fk_with_data", "clear_neo4j_graph",
        "add_fk_to_graph", "get_graph_stats", "analyze_graph_centrality",
        "find_table_communities", "predict_missing_fks", "run_cypher",
        "generate_text_embeddings", "generate_kg_embeddings", "semantic_search_tables",
        "semantic_search_columns", "find_similar_tables", "find_similar_columns",
        "show_sample_business_rules", "load_business_rules", "execute_business_rule",
        "execute_all_business_rules", "import_lineage_to_graph", "analyze_data_flow",
        "find_impact_analysis", "test_rdf_connection", "sync_graph_to_rdf",
        "run_sparql", "learn_rules_with_ltn", "generate_business_rules_from_ltn",
    ]
    
    if tool_name in builtin_tools:
        emit({"status": "found", "tool": tool_name, "type": "builtin"})
        return f"✓ Builtin tool '{tool_name}' EXISTS (cannot be modified)"
    
    emit({"status": "not_found", "tool": tool_name})
    return f"✗ Tool '{tool_name}' does NOT exist. You can create it with create_dynamic_tool."


@tool
def list_available_tools() -> str:
    """List all available tools (builtin and dynamic).
    
    Returns:
        Categorized list of all available tools
    """
    emit({"status": "listing", "message": "Gathering tool list..."})
    
    registry = get_registry()
    
    output = "## Available Tools\n\n"
    
    # Builtin tools by category
    output += "### Builtin Tools\n\n"
    
    categories = {
        "Database": ["configure_database", "test_database_connection", "list_database_tables", "get_table_schema", "get_column_stats"],
        "FK Discovery": ["run_fk_discovery", "analyze_potential_fk", "validate_fk_with_data"],
        "Graph": ["clear_neo4j_graph", "add_fk_to_graph", "get_graph_stats", "analyze_graph_centrality", "find_table_communities", "predict_missing_fks", "run_cypher"],
        "Embeddings": ["generate_text_embeddings", "generate_kg_embeddings", "semantic_search_tables", "semantic_search_columns", "find_similar_tables", "find_similar_columns"],
        "Business Rules": ["show_sample_business_rules", "load_business_rules", "execute_business_rule", "execute_all_business_rules", "import_lineage_to_graph", "analyze_data_flow", "find_impact_analysis"],
        "RDF": ["test_rdf_connection", "sync_graph_to_rdf", "run_sparql"],
        "LTN": ["learn_rules_with_ltn", "generate_business_rules_from_ltn"],
        "Dynamic Tools": ["check_tool_exists", "list_available_tools", "create_dynamic_tool", "run_dynamic_tool", "get_tool_source", "update_dynamic_tool", "delete_dynamic_tool"],
    }
    
    for category, tools in categories.items():
        output += f"**{category}**: {', '.join(tools)}\n"
    
    # Dynamic tools
    dynamic_tools = registry.list_tools()
    
    if dynamic_tools:
        output += "\n### Dynamic Tools (User Created)\n\n"
        for tool_info in dynamic_tools:
            emit({"dynamic_tool": tool_info['name']})
            output += f"- **{tool_info['name']}**: {tool_info.get('description', 'No description')}\n"
    else:
        output += "\n### Dynamic Tools\n\nNo dynamic tools created yet.\n"
    
    emit({"status": "complete", "dynamic_count": len(dynamic_tools)})
    return output


@tool
def create_dynamic_tool(name: str, description: str, code: str) -> str:
    """Create a new dynamic tool from Python code.
    
    The code MUST define a function called `run()` that will be the tool's entry point.
    The `run()` function should have type hints and a docstring.
    
    IMPORTANT: Always call check_tool_exists first to avoid duplicates!
    
    Args:
        name: Unique name for the tool (snake_case, e.g., 'json_parser')
        description: Brief description of what the tool does
        code: Python source code defining a run() function
    
    Example code:
        def run(data: str, format: str = "json") -> str:
            \"\"\"Process data in the specified format.\"\"\"
            import json
            parsed = json.loads(data)
            return f"Processed {len(parsed)} items"
    
    Returns:
        Success message or error details
    """
    emit({"status": "creating", "tool": name})
    emit({"status": "validating", "message": "Checking tool name and code..."})
    
    # Validate name
    if not name.replace("_", "").isalnum():
        emit({"status": "error", "message": "Invalid tool name"})
        return f"ERROR: Tool name must be alphanumeric with underscores. Got: '{name}'"
    
    # Check if exists
    registry = get_registry()
    if registry.tool_exists(name):
        emit({"status": "error", "message": "Tool already exists"})
        return f"ERROR: Tool '{name}' already exists. Use update_dynamic_tool to modify it."
    
    # Validate code has run() function
    emit({"status": "validating", "message": "Checking for run() function..."})
    if "def run(" not in code:
        emit({"status": "error", "message": "Missing run() function"})
        return "ERROR: Code must define a `run()` function. Example:\n\ndef run(arg: str) -> str:\n    \"\"\"Description.\"\"\"\n    return result"
    
    # Try to compile the code
    emit({"status": "compiling", "message": "Compiling Python code..."})
    try:
        compile(code, f"<tool:{name}>", "exec")
    except SyntaxError as e:
        emit({"status": "error", "message": f"Syntax error: {e}"})
        return f"ERROR: Syntax error in code:\n{e}"
    
    # Save the tool
    emit({"status": "saving", "message": "Writing tool file..."})
    try:
        file_path = registry.create_tool(name, description, code)
        emit({"status": "complete", "tool": name, "file": file_path})
        return f"✓ Created tool '{name}'\n  File: {file_path}\n  Description: {description}\n\nUse run_dynamic_tool('{name}', ...) to execute it."
    except Exception as e:
        emit({"status": "error", "message": str(e)})
        return f"ERROR creating tool: {type(e).__name__}: {e}"


@tool
def run_dynamic_tool(tool_name: str, **kwargs) -> str:
    """Execute a dynamic tool by name.
    
    Args:
        tool_name: Name of the dynamic tool to run
        **kwargs: Arguments to pass to the tool's run() function
    
    Returns:
        The tool's output or error message
    """
    emit({"status": "loading", "tool": tool_name})
    
    registry = get_registry()
    
    if not registry.tool_exists(tool_name):
        emit({"status": "error", "message": "Tool not found"})
        return f"ERROR: Tool '{tool_name}' not found. Use list_available_tools to see available tools."
    
    emit({"status": "executing", "tool": tool_name, "args": list(kwargs.keys())})
    
    try:
        result = registry.execute_tool(tool_name, **kwargs)
        emit({"status": "complete", "tool": tool_name})
        return f"## Tool Output: {tool_name}\n\n{result}"
    except Exception as e:
        import traceback
        emit({"status": "error", "tool": tool_name, "error": str(e)})
        return f"ERROR executing '{tool_name}':\n{type(e).__name__}: {e}\n\n{traceback.format_exc()}"


@tool
def get_tool_source(tool_name: str) -> str:
    """Get the source code of a dynamic tool.
    
    Args:
        tool_name: Name of the tool
    
    Returns:
        The tool's source code or error message
    """
    emit({"status": "fetching", "tool": tool_name})
    
    registry = get_registry()
    
    if not registry.tool_exists(tool_name):
        emit({"status": "error", "message": "Tool not found"})
        return f"ERROR: Tool '{tool_name}' not found."
    
    try:
        source = registry.get_tool_source(tool_name)
        emit({"status": "complete", "tool": tool_name})
        return f"## Source: {tool_name}\n\n```python\n{source}\n```"
    except Exception as e:
        emit({"status": "error", "message": str(e)})
        return f"ERROR: {e}"


@tool
def update_dynamic_tool(tool_name: str, code: str, description: str = None) -> str:
    """Update an existing dynamic tool's code.
    
    Args:
        tool_name: Name of the tool to update
        code: New Python source code (must define run() function)
        description: New description (optional, keeps existing if not provided)
    
    Returns:
        Success message or error details
    """
    emit({"status": "updating", "tool": tool_name})
    
    registry = get_registry()
    
    if not registry.tool_exists(tool_name):
        emit({"status": "error", "message": "Tool not found"})
        return f"ERROR: Tool '{tool_name}' not found. Use create_dynamic_tool to create it."
    
    # Validate code
    emit({"status": "validating", "message": "Checking code..."})
    if "def run(" not in code:
        emit({"status": "error", "message": "Missing run() function"})
        return "ERROR: Code must define a `run()` function."
    
    try:
        compile(code, f"<tool:{tool_name}>", "exec")
    except SyntaxError as e:
        emit({"status": "error", "message": f"Syntax error: {e}"})
        return f"ERROR: Syntax error:\n{e}"
    
    emit({"status": "saving", "message": "Writing updated code..."})
    
    try:
        registry.update_tool(tool_name, code, description)
        emit({"status": "complete", "tool": tool_name})
        return f"✓ Updated tool '{tool_name}'"
    except Exception as e:
        emit({"status": "error", "message": str(e)})
        return f"ERROR: {e}"


@tool
def delete_dynamic_tool(tool_name: str) -> str:
    """Delete a dynamic tool.
    
    Args:
        tool_name: Name of the tool to delete
    
    Returns:
        Success message or error
    """
    emit({"status": "deleting", "tool": tool_name})
    
    registry = get_registry()
    
    if not registry.tool_exists(tool_name):
        emit({"status": "error", "message": "Tool not found"})
        return f"ERROR: Tool '{tool_name}' not found."
    
    try:
        registry.delete_tool(tool_name)
        emit({"status": "complete", "tool": tool_name})
        return f"✓ Deleted tool '{tool_name}'"
    except Exception as e:
        emit({"status": "error", "message": str(e)})
        return f"ERROR: {e}"


# Export the tools for use in agent
DYNAMIC_TOOL_MANAGEMENT_TOOLS = [
    check_tool_exists,
    list_available_tools,
    create_dynamic_tool,
    run_dynamic_tool,
    get_tool_source,
    update_dynamic_tool,
    delete_dynamic_tool,
]