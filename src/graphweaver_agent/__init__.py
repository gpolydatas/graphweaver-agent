"""
GraphWeaver Agent - Claude-powered autonomous FK discovery and knowledge graph building.

This package provides:
- Database connectors for PostgreSQL
- Neo4j graph operations  
- FK discovery algorithms
- Business rules engine with lineage tracking
- Dynamic tool creation and management
"""

from .models.schemas import (
    DataType,
    ColumnDataType,
    ColumnMetadata,
    TableMetadata,
    ForeignKeyCandidate,
    FKCandidate,
    DataSourceConfig,
    Neo4jConfig,
    ColumnStatistics,
    PipelineConfig,
    RelationshipCardinality,
    DiscoveryResult,
    ExistingForeignKey,
    DatabaseType,
)
from .connectors.postgresql import PostgreSQLConnector
from .graph.neo4j_ops import Neo4jClient, GraphBuilder, GraphAnalyzer

# Dynamic Tools
from .dynamic_tools import (
    ToolRegistry,
    ToolMetadata,
    ToolType,
    get_registry,
)

__all__ = [
    # Models
    "DataType",
    "ColumnDataType",
    "ColumnMetadata",
    "TableMetadata",
    "ForeignKeyCandidate",
    "FKCandidate",
    "DataSourceConfig",
    "Neo4jConfig",
    "ColumnStatistics",
    "PipelineConfig",
    "RelationshipCardinality",
    "DiscoveryResult",
    "ExistingForeignKey",
    "DatabaseType",
    
    # Connectors
    "PostgreSQLConnector",
    
    # Graph
    "Neo4jClient",
    "GraphBuilder",
    "GraphAnalyzer",
    
    # Dynamic Tools
    "ToolRegistry",
    "ToolMetadata",
    "ToolType",
    "get_registry",
]