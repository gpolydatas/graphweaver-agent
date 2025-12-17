"""
Dynamic Tool: json_analyzer
Created: 2024-01-01T00:00:00
Description: Analyze JSON structure and provide statistics

Dependencies: []
"""

import json
from typing import Any, Dict


def _analyze_value(value: Any, path: str = "$") -> Dict[str, Any]:
    """Recursively analyze a JSON value."""
    result = {
        "type": type(value).__name__,
        "path": path,
    }
    
    if isinstance(value, dict):
        result["keys"] = list(value.keys())
        result["key_count"] = len(value)
        result["children"] = {
            k: _analyze_value(v, f"{path}.{k}")
            for k, v in value.items()
        }
    elif isinstance(value, list):
        result["length"] = len(value)
        if value:
            result["element_type"] = type(value[0]).__name__
            if isinstance(value[0], dict):
                result["element_keys"] = list(value[0].keys()) if value[0] else []
    elif isinstance(value, str):
        result["length"] = len(value)
    elif isinstance(value, (int, float)):
        result["value"] = value
    
    return result


def run(json_data: str, max_depth: int = 3) -> str:
    """
    Analyze JSON structure and provide statistics.
    
    Args:
        json_data: JSON string to analyze
        max_depth: Maximum depth to analyze (default: 3)
    
    Returns:
        Analysis report as markdown
    """
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as e:
        return f"## Error\n\nInvalid JSON: {e}"
    
    analysis = _analyze_value(data)
    
    output = "## JSON Analysis\n\n"
    output += f"**Root Type:** {analysis['type']}\n\n"
    
    if analysis['type'] == 'dict':
        output += f"**Keys ({analysis['key_count']}):** {', '.join(analysis['keys'])}\n\n"
        output += "### Structure\n\n"
        for key, child in analysis.get('children', {}).items():
            output += f"- `{key}`: {child['type']}"
            if child['type'] == 'list':
                output += f" (length: {child.get('length', 0)})"
            elif child['type'] == 'str':
                output += f" (length: {child.get('length', 0)})"
            output += "\n"
    
    elif analysis['type'] == 'list':
        output += f"**Length:** {analysis['length']}\n"
        output += f"**Element Type:** {analysis.get('element_type', 'mixed')}\n"
        if analysis.get('element_keys'):
            output += f"**Element Keys:** {', '.join(analysis['element_keys'])}\n"
    
    return output