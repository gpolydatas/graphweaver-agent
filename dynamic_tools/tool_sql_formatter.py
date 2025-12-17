"""
Dynamic Tool: sql_formatter
Created: 2024-01-01T00:00:00
Description: Format SQL queries for readability

Dependencies: ['sqlparse']
"""

import sqlparse


def run(query: str, indent_width: int = 2, keyword_case: str = "upper") -> str:
    """
    Format a SQL query for better readability.
    
    Args:
        query: SQL query to format
        indent_width: Number of spaces for indentation (default: 2)
        keyword_case: Case for keywords: 'upper', 'lower', 'capitalize' (default: 'upper')
    
    Returns:
        Formatted SQL query
    """
    formatted = sqlparse.format(
        query,
        reindent=True,
        indent_width=indent_width,
        keyword_case=keyword_case,
        identifier_case='lower',
        strip_comments=False,
        use_space_around_operators=True,
    )
    
    return f"## Formatted SQL:\n\n```sql\n{formatted}\n```"