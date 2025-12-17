"""
ToolRegistry - Dynamic tool creation, storage, and execution for GraphWeaver Agent.

This module enables the agent to:
1. Create new tools at runtime from Python code
2. Test tools before persisting them
3. Save tools to the codebase for persistence
4. Load tools from files on startup
5. Execute dynamic tools with arguments
6. Track tool metadata and dependencies
"""

import os
import sys
import json
import importlib.util
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum


class ToolType(str, Enum):
    """Type of tool."""
    BUILTIN = "builtin"      # Core tools defined in agent.py or mcp server
    DYNAMIC = "dynamic"      # Runtime-created tools
    EXTERNAL = "external"    # Loaded from external packages


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    tool_type: ToolType
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_file: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_type: str = "str"
    version: str = "1.0.0"
    author: str = "agent"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type.value if isinstance(self.tool_type, ToolType) else self.tool_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_file": self.source_file,
            "dependencies": self.dependencies,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolMetadata":
        tool_type = data.get("tool_type", "dynamic")
        if isinstance(tool_type, str):
            tool_type = ToolType(tool_type)
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            tool_type=tool_type,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            source_file=data.get("source_file"),
            dependencies=data.get("dependencies", []),
            parameters=data.get("parameters", {}),
            return_type=data.get("return_type", "str"),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "agent"),
            tags=data.get("tags", []),
        )


class ToolRegistry:
    """
    Registry for managing dynamic tools.
    
    Handles:
    - Registration of builtin and dynamic tools
    - Persistence of tool code to files
    - Loading of tools from files on startup
    - Execution of dynamic tools
    - Dependency management
    """
    
    def __init__(self, tools_dir: Optional[str] = None):
        """
        Initialize the tool registry.
        
        Args:
            tools_dir: Directory to store dynamic tool files.
                      Defaults to ./dynamic_tools/ in the current working directory.
        """
        self._tools: Dict[str, ToolMetadata] = {}
        self._functions: Dict[str, Callable] = {}
        self._tools_dir = Path(tools_dir) if tools_dir else self._get_default_tools_dir()
        self._registry_file = self._tools_dir / "tool_registry.json"
        
        # Ensure tools directory exists
        self._tools_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing tools
        self._load_registry()
        self._load_dynamic_tools()
    
    def _get_default_tools_dir(self) -> Path:
        """Get the default tools directory."""
        return Path(os.getcwd()) / "dynamic_tools"
    
    def _load_registry(self) -> None:
        """Load tool registry from JSON file."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, "r") as f:
                    data = json.load(f)
                    for tool_data in data.get("tools", []):
                        metadata = ToolMetadata.from_dict(tool_data)
                        self._tools[metadata.name] = metadata
            except Exception as e:
                print(f"Warning: Failed to load tool registry: {e}")
    
    def _save_registry(self) -> None:
        """Save tool registry to JSON file."""
        try:
            data = {
                "version": "1.0.0",
                "updated_at": datetime.now().isoformat(),
                "tools": [t.to_dict() for t in self._tools.values() if t.tool_type == ToolType.DYNAMIC]
            }
            with open(self._registry_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save tool registry: {e}")
    
    def _load_dynamic_tools(self) -> None:
        """Load all dynamic tools from the tools directory."""
        if not self._tools_dir.exists():
            return
        
        for tool_file in self._tools_dir.glob("tool_*.py"):
            tool_name = tool_file.stem.replace("tool_", "")
            if tool_name in self._tools:
                try:
                    self._load_tool_from_file(tool_name, tool_file)
                except Exception as e:
                    print(f"Warning: Failed to load tool {tool_name}: {e}")
    
    def _load_tool_from_file(self, tool_name: str, tool_file: Path) -> None:
        """Load a tool from a Python file."""
        spec = importlib.util.spec_from_file_location(f"dynamic_tool_{tool_name}", tool_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load tool from {tool_file}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"dynamic_tool_{tool_name}"] = module
        spec.loader.exec_module(module)
        
        # Look for the main function
        if hasattr(module, "run"):
            self._functions[tool_name] = module.run
        elif hasattr(module, tool_name):
            self._functions[tool_name] = getattr(module, tool_name)
        elif hasattr(module, "main"):
            self._functions[tool_name] = module.main
        else:
            # Find any callable that's not a builtin
            for name, obj in vars(module).items():
                if callable(obj) and not name.startswith("_"):
                    self._functions[tool_name] = obj
                    break
    
    def register_builtin(self, name: str, func: Callable, description: str = "",
                         parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a builtin tool (from agent.py or mcp server).
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            parameters: Parameter schema
        """
        self._tools[name] = ToolMetadata(
            name=name,
            description=description or func.__doc__ or "",
            tool_type=ToolType.BUILTIN,
            parameters=parameters or {},
        )
        self._functions[name] = func
    
    def tool_exists(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name."""
        return self._tools.get(name)
    
    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[ToolMetadata]:
        """
        List all registered tools.
        
        Args:
            tool_type: Filter by tool type (None for all)
        """
        if tool_type is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.tool_type == tool_type]
    
    def create_tool(
        self,
        name: str,
        code: str,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        test_first: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new dynamic tool from Python code.
        
        Args:
            name: Tool name (alphanumeric and underscores only)
            code: Python code defining the tool (must have a 'run' function)
            description: Tool description
            dependencies: List of pip packages to install
            parameters: Parameter schema
            test_first: Whether to test the code before saving
            tags: Optional tags for categorization
            
        Returns:
            Dict with success status and any error messages
        """
        # Validate name
        if not name.replace("_", "").isalnum():
            return {"success": False, "error": "Tool name must be alphanumeric with underscores only"}
        
        if self.tool_exists(name) and self._tools[name].tool_type == ToolType.BUILTIN:
            return {"success": False, "error": f"Cannot override builtin tool: {name}"}
        
        # Install dependencies if any
        if dependencies:
            for dep in dependencies:
                try:
                    self._install_dependency(dep)
                except Exception as e:
                    return {"success": False, "error": f"Failed to install dependency {dep}: {e}"}
        
        # Test the code first
        if test_first:
            test_result = self._test_code(code, name)
            if not test_result["success"]:
                return test_result
        
        # Save the tool file
        tool_file = self._tools_dir / f"tool_{name}.py"
        try:
            # Add header comment
            header = f'''"""
Dynamic Tool: {name}
Created: {datetime.now().isoformat()}
Description: {description}

Dependencies: {dependencies or []}
"""

'''
            with open(tool_file, "w") as f:
                f.write(header + code)
        except Exception as e:
            return {"success": False, "error": f"Failed to save tool file: {e}"}
        
        # Register the tool
        self._tools[name] = ToolMetadata(
            name=name,
            description=description,
            tool_type=ToolType.DYNAMIC,
            source_file=str(tool_file),
            dependencies=dependencies or [],
            parameters=parameters or {},
            tags=tags or [],
        )
        
        # Load the function
        try:
            self._load_tool_from_file(name, tool_file)
        except Exception as e:
            return {"success": False, "error": f"Failed to load tool: {e}"}
        
        # Save registry
        self._save_registry()
        
        return {
            "success": True,
            "message": f"Tool '{name}' created successfully",
            "file": str(tool_file),
        }
    
    def _install_dependency(self, package: str) -> None:
        """Install a pip package."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"pip install failed: {result.stderr}")
    
    def _test_code(self, code: str, name: str) -> Dict[str, Any]:
        """Test tool code by compiling and checking for run function."""
        try:
            # Compile the code
            compiled = compile(code, f"<tool_{name}>", "exec")
            
            # Execute in isolated namespace
            namespace: Dict[str, Any] = {}
            exec(compiled, namespace)
            
            # Check for required function
            has_run = "run" in namespace and callable(namespace["run"])
            has_named = name in namespace and callable(namespace[name])
            has_main = "main" in namespace and callable(namespace["main"])
            
            if not (has_run or has_named or has_main):
                return {
                    "success": False,
                    "error": f"Tool code must define a callable function named 'run', '{name}', or 'main'"
                }
            
            return {"success": True}
            
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Code test failed: {e}\n{traceback.format_exc()}"}
    
    def run_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a dynamic tool.
        
        Args:
            name: Tool name
            arguments: Arguments to pass to the tool function
            
        Returns:
            Dict with result or error
        """
        if name not in self._functions:
            return {"success": False, "error": f"Tool '{name}' not found or not loaded"}
        
        try:
            func = self._functions[name]
            args = arguments or {}
            result = func(**args)
            return {"success": True, "result": result}
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {e}",
                "traceback": traceback.format_exc()
            }
    
    def get_tool_code(self, name: str) -> Optional[str]:
        """Get the source code of a dynamic tool."""
        metadata = self._tools.get(name)
        if metadata is None or metadata.source_file is None:
            return None
        
        try:
            with open(metadata.source_file, "r") as f:
                return f.read()
        except Exception:
            return None
    
    def update_tool(self, name: str, code: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Update an existing dynamic tool.
        
        Args:
            name: Tool name
            code: New Python code
            description: Optional new description
            
        Returns:
            Dict with success status
        """
        if name not in self._tools:
            return {"success": False, "error": f"Tool '{name}' not found"}
        
        metadata = self._tools[name]
        if metadata.tool_type == ToolType.BUILTIN:
            return {"success": False, "error": "Cannot update builtin tools"}
        
        # Test the new code
        test_result = self._test_code(code, name)
        if not test_result["success"]:
            return test_result
        
        # Update the file
        tool_file = self._tools_dir / f"tool_{name}.py"
        try:
            header = f'''"""
Dynamic Tool: {name}
Created: {metadata.created_at}
Updated: {datetime.now().isoformat()}
Description: {description or metadata.description}

Dependencies: {metadata.dependencies}
"""

'''
            with open(tool_file, "w") as f:
                f.write(header + code)
        except Exception as e:
            return {"success": False, "error": f"Failed to update tool file: {e}"}
        
        # Update metadata
        metadata.updated_at = datetime.now().isoformat()
        if description:
            metadata.description = description
        metadata.version = self._increment_version(metadata.version)
        
        # Reload the function
        try:
            self._load_tool_from_file(name, tool_file)
        except Exception as e:
            return {"success": False, "error": f"Failed to reload tool: {e}"}
        
        # Save registry
        self._save_registry()
        
        return {"success": True, "message": f"Tool '{name}' updated to version {metadata.version}"}
    
    def _increment_version(self, version: str) -> str:
        """Increment the patch version."""
        parts = version.split(".")
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
        return ".".join(parts)
    
    def delete_tool(self, name: str) -> Dict[str, Any]:
        """
        Delete a dynamic tool.
        
        Args:
            name: Tool name
            
        Returns:
            Dict with success status
        """
        if name not in self._tools:
            return {"success": False, "error": f"Tool '{name}' not found"}
        
        metadata = self._tools[name]
        if metadata.tool_type == ToolType.BUILTIN:
            return {"success": False, "error": "Cannot delete builtin tools"}
        
        # Delete the file
        if metadata.source_file and os.path.exists(metadata.source_file):
            try:
                os.remove(metadata.source_file)
            except Exception as e:
                return {"success": False, "error": f"Failed to delete tool file: {e}"}
        
        # Remove from registry
        del self._tools[name]
        if name in self._functions:
            del self._functions[name]
        
        # Save registry
        self._save_registry()
        
        return {"success": True, "message": f"Tool '{name}' deleted"}
    
    def export_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Export a tool's metadata and code for sharing.
        
        Args:
            name: Tool name
            
        Returns:
            Dict with metadata and code, or None if not found
        """
        metadata = self._tools.get(name)
        if metadata is None:
            return None
        
        return {
            "metadata": metadata.to_dict(),
            "code": self.get_tool_code(name),
        }
    
    def import_tool(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a tool from exported data.
        
        Args:
            data: Dict with 'metadata' and 'code' keys
            
        Returns:
            Dict with success status
        """
        metadata = data.get("metadata", {})
        code = data.get("code")
        
        if not metadata.get("name") or not code:
            return {"success": False, "error": "Invalid tool data: missing name or code"}
        
        return self.create_tool(
            name=metadata["name"],
            code=code,
            description=metadata.get("description", ""),
            dependencies=metadata.get("dependencies", []),
            parameters=metadata.get("parameters", {}),
            tags=metadata.get("tags", []),
        )


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry(tools_dir: Optional[str] = None) -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry(tools_dir=tools_dir)
    return _registry