"""
graphweaver-agent/streamlit_app.py

Streamlit Chat Interface for GraphWeaver Agent with Real-Time Streaming
"""
import os
import sys
import streamlit as st
from typing import Optional, Generator, Dict, Any, List
import anthropic

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_servers"))

from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from graphweaver_agent import (
    DataSourceConfig, Neo4jConfig, PostgreSQLConnector,
    Neo4jClient, GraphBuilder, GraphAnalyzer,
)
from graphweaver_agent.discovery.pipeline import run_discovery, FKDetectionPipeline, PipelineConfig
from graphweaver_agent.business_rules import (
    BusinessRulesExecutor, BusinessRulesConfig, BusinessRule, MarquezClient,
    import_lineage_to_neo4j, generate_sample_rules,
)

# RDF imports
from graphweaver_agent.rdf import (
    FusekiClient, RDFSyncManager, sync_neo4j_to_rdf,
    GraphWeaverOntology, SPARQLQueryBuilder, PREFIXES_SPARQL
)

# LTN imports
try:
    from graphweaver_agent.ltn import (
        LTNRuleLearner,
        BusinessRuleGenerator,
        LTNKnowledgeBase,
        LearnedRule,
        GeneratedRule,
        RuleLearningConfig,
    )
    LTN_AVAILABLE = True
except ImportError:
    LTN_AVAILABLE = False

# Dynamic Tools imports
from graphweaver_agent.dynamic_tools.agent_tools import (
    check_tool_exists,
    list_available_tools,
    create_dynamic_tool,
    run_dynamic_tool,
    get_tool_source,
    update_dynamic_tool,
    delete_dynamic_tool,
    DYNAMIC_TOOL_MANAGEMENT_TOOLS,
)


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are GraphWeaver Agent, an intelligent AI assistant for database exploration, FK discovery, knowledge graph building, and data lineage analysis.

You have access to many tools to help users:
- Explore PostgreSQL databases (test connections, list tables, get schemas)
- Discover foreign key relationships using a 5-stage pipeline
- Build and analyze Neo4j knowledge graphs
- Generate and use text/KG embeddings for semantic search
- Execute business rules and capture lineage to Marquez
- Work with RDF/SPARQL via Apache Jena Fuseki
- Learn rules with Logic Tensor Networks (LTN)
- Create and manage dynamic tools at runtime

Be helpful, thorough, and explain what you're doing. When discovering FKs, validate with actual data. When building graphs, explain the structure."""


# =============================================================================
# Streaming Tool Definitions (JSON schema format for Anthropic API)
# =============================================================================

STREAMING_TOOLS = [
    # Dynamic tool management
    {"name": "check_tool_exists", "description": "Check if a dynamic tool exists in the registry",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string", "description": "Name of the tool to check"}}, "required": ["tool_name"]}},
    {"name": "list_available_tools", "description": "List all available tools - both builtin and dynamic",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "create_dynamic_tool", "description": "Create a new dynamic tool. Code must define a run() function.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "code": {"type": "string"}}, "required": ["name", "description", "code"]}},
    {"name": "run_dynamic_tool", "description": "Execute a dynamic tool by name",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string"}}, "required": ["tool_name"]}},
    {"name": "get_tool_source", "description": "Get the source code of a dynamic tool",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string"}}, "required": ["tool_name"]}},
    {"name": "update_dynamic_tool", "description": "Update a dynamic tool's code",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string"}, "code": {"type": "string"}, "description": {"type": "string"}}, "required": ["tool_name", "code"]}},
    {"name": "delete_dynamic_tool", "description": "Delete a dynamic tool",
     "input_schema": {"type": "object", "properties": {"tool_name": {"type": "string"}}, "required": ["tool_name"]}},

    # Database
    {"name": "configure_database", "description": "Configure which PostgreSQL database to connect to",
     "input_schema": {"type": "object", "properties": {"host": {"type": "string"}, "port": {"type": "integer"}, "database": {"type": "string"}, "username": {"type": "string"}, "password": {"type": "string"}}, "required": ["host", "port", "database", "username", "password"]}},
    {"name": "test_database_connection", "description": "Test the PostgreSQL database connection",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_database_tables", "description": "List all tables in the database with column counts",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_table_schema", "description": "Get schema details for a table (columns, types, PKs)",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "get_column_stats", "description": "Get statistics for a column (uniqueness, nulls, samples)",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}, "column_name": {"type": "string"}}, "required": ["table_name", "column_name"]}},

    # FK Discovery
    {"name": "run_fk_discovery", "description": "Run the full 5-stage FK discovery pipeline on the database",
     "input_schema": {"type": "object", "properties": {"min_match_rate": {"type": "number", "default": 0.95}, "min_score": {"type": "number", "default": 0.5}}}},
    {"name": "analyze_potential_fk", "description": "Analyze a potential FK relationship and get a score",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},
    {"name": "validate_fk_with_data", "description": "Validate a FK by checking actual data integrity",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},

    # Graph
    {"name": "clear_neo4j_graph", "description": "Clear all nodes and relationships from Neo4j",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "add_fk_to_graph", "description": "Add a FK relationship to the Neo4j graph",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "source_column": {"type": "string"}, "target_table": {"type": "string"}, "target_column": {"type": "string"}, "score": {"type": "number", "default": 1.0}, "cardinality": {"type": "string", "default": "1:N"}}, "required": ["source_table", "source_column", "target_table", "target_column"]}},
    {"name": "get_graph_stats", "description": "Get statistics about the Neo4j graph",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_graph_centrality", "description": "Find hub and authority tables in the graph",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "find_table_communities", "description": "Find clusters of related tables",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "predict_missing_fks", "description": "Predict missing FKs based on column naming patterns",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_cypher", "description": "Execute a Cypher query on Neo4j",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "connect_datasets_to_tables", "description": "Connect Dataset nodes to their matching Table nodes",
     "input_schema": {"type": "object", "properties": {}}},

    # Embeddings
    {"name": "generate_text_embeddings", "description": "Generate text embeddings for all tables, columns, jobs, and datasets",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "generate_kg_embeddings", "description": "Generate knowledge graph embeddings using Neo4j GDS FastRP",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "create_vector_indexes", "description": "Create Neo4j vector indexes for fast similarity search",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "semantic_search_tables", "description": "Search for tables using natural language",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}, "required": ["query"]}},
    {"name": "semantic_search_columns", "description": "Search for columns using natural language",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}}, "required": ["query"]}},
    {"name": "find_similar_tables", "description": "Find tables similar to a given table",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}, "required": ["table_name"]}},
    {"name": "find_similar_columns", "description": "Find columns similar to a given column",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}, "column_name": {"type": "string"}, "top_k": {"type": "integer", "default": 10}}, "required": ["table_name", "column_name"]}},
    {"name": "predict_fks_from_embeddings", "description": "Predict potential FK relationships using KG embeddings",
     "input_schema": {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.7}, "top_k": {"type": "integer", "default": 20}}}},
    {"name": "semantic_fk_discovery", "description": "Discover FK relationships using semantic similarity",
     "input_schema": {"type": "object", "properties": {"source_table": {"type": "string"}, "min_score": {"type": "number", "default": 0.6}}}},

    # Business Rules
    {"name": "show_sample_business_rules", "description": "Show sample business rules YAML format",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "load_business_rules", "description": "Load business rules from YAML content",
     "input_schema": {"type": "object", "properties": {"yaml_content": {"type": "string"}}, "required": ["yaml_content"]}},
    {"name": "load_business_rules_from_file", "description": "Load business rules from a YAML file",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string", "default": "business_rules.yaml"}}}},
    {"name": "list_business_rules", "description": "List all loaded business rules",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "execute_business_rule", "description": "Execute a single business rule",
     "input_schema": {"type": "object", "properties": {"rule_name": {"type": "string"}, "capture_lineage": {"type": "boolean", "default": True}}, "required": ["rule_name"]}},
    {"name": "execute_all_business_rules", "description": "Execute all loaded business rules",
     "input_schema": {"type": "object", "properties": {"capture_lineage": {"type": "boolean", "default": True}}}},
    {"name": "get_marquez_lineage", "description": "Get lineage graph for a dataset from Marquez",
     "input_schema": {"type": "object", "properties": {"dataset_name": {"type": "string"}, "depth": {"type": "integer", "default": 3}}, "required": ["dataset_name"]}},
    {"name": "list_marquez_jobs", "description": "List all jobs tracked in Marquez",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "import_lineage_to_graph", "description": "Import lineage from Marquez into Neo4j",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_data_flow", "description": "Analyze complete data flow for a table",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "find_impact_analysis", "description": "Find all downstream impacts if a table changes",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},

    # RDF
    {"name": "test_rdf_connection", "description": "Test connection to Apache Jena Fuseki",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sync_graph_to_rdf", "description": "Sync Neo4j graph to RDF/Fuseki",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_sparql", "description": "Execute a SPARQL query on Fuseki",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "sparql_list_tables", "description": "List all tables in the RDF store",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sparql_get_foreign_keys", "description": "Get foreign keys from RDF store",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}}},
    {"name": "sparql_table_lineage", "description": "Get lineage for a table from RDF store",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "sparql_downstream_impact", "description": "Get downstream impact analysis from RDF",
     "input_schema": {"type": "object", "properties": {"table_name": {"type": "string"}}, "required": ["table_name"]}},
    {"name": "sparql_hub_tables", "description": "Find hub tables from RDF store",
     "input_schema": {"type": "object", "properties": {"min_connections": {"type": "integer", "default": 3}}}},
    {"name": "sparql_orphan_tables", "description": "Find orphan tables from RDF store",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "sparql_search", "description": "Search RDF store by label",
     "input_schema": {"type": "object", "properties": {"search_term": {"type": "string"}}, "required": ["search_term"]}},
    {"name": "get_rdf_statistics", "description": "Get RDF store statistics",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "export_rdf_turtle", "description": "Export graph as RDF Turtle format",
     "input_schema": {"type": "object", "properties": {}}},

    # LTN
    {"name": "learn_rules_with_ltn", "description": "Use Logic Tensor Networks to learn rules from data",
     "input_schema": {"type": "object", "properties": {"epochs": {"type": "integer", "default": 100}}}},
    {"name": "generate_business_rules_from_ltn", "description": "Generate business rules from learned LTN rules",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "generate_all_validation_rules", "description": "Generate validation rules for all discovered FKs",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "export_generated_rules_yaml", "description": "Export generated rules as YAML",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string", "default": "business_rules_generated.yaml"}}}},
    {"name": "export_generated_rules_sql", "description": "Export generated rules as SQL",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "show_ltn_knowledge_base", "description": "Show the current LTN knowledge base state",
     "input_schema": {"type": "object", "properties": {}}},
]


# =============================================================================
# Global Connection Getters (Cached via Session State)
# =============================================================================

def get_pg_config() -> DataSourceConfig:
    """Get PostgreSQL config from session state or environment."""
    return DataSourceConfig(
        host=st.session_state.get("pg_host", os.environ.get("POSTGRES_HOST", "localhost")),
        port=int(st.session_state.get("pg_port", os.environ.get("POSTGRES_PORT", "5432"))),
        database=st.session_state.get("pg_database", os.environ.get("POSTGRES_DB", "orders")),
        username=st.session_state.get("pg_username", os.environ.get("POSTGRES_USER", "saphenia")),
        password=st.session_state.get("pg_password", os.environ.get("POSTGRES_PASSWORD", "secret")),
    )


def get_pg() -> PostgreSQLConnector:
    """Get or create PostgreSQL connector."""
    if "pg_connector" not in st.session_state:
        st.session_state.pg_connector = PostgreSQLConnector(get_pg_config())
    return st.session_state.pg_connector


def get_neo4j() -> Neo4jClient:
    """Get or create Neo4j client."""
    if "neo4j_client" not in st.session_state:
        st.session_state.neo4j_client = Neo4jClient(Neo4jConfig(
            uri=st.session_state.get("neo4j_uri", os.environ.get("NEO4J_URI", "bolt://localhost:7687")),
            username=st.session_state.get("neo4j_user", os.environ.get("NEO4J_USER", "neo4j")),
            password=st.session_state.get("neo4j_password", os.environ.get("NEO4J_PASSWORD", "password")),
        ))
    return st.session_state.neo4j_client


def get_text_embedder():
    """Get or create text embedder."""
    if "text_embedder" not in st.session_state:
        from graphweaver_agent.embeddings.text_embeddings import TextEmbedder
        st.session_state.text_embedder = TextEmbedder()
    return st.session_state.text_embedder


def get_kg_embedder():
    """Get or create KG embedder."""
    if "kg_embedder" not in st.session_state:
        from graphweaver_agent.embeddings.kg_embeddings import KGEmbedder
        st.session_state.kg_embedder = KGEmbedder(get_neo4j())
    return st.session_state.kg_embedder


def get_fuseki() -> FusekiClient:
    """Get or create Fuseki client."""
    if "fuseki_client" not in st.session_state:
        st.session_state.fuseki_client = FusekiClient()
    return st.session_state.fuseki_client


def get_sparql() -> SPARQLQueryBuilder:
    """Get or create SPARQL query builder."""
    if "sparql_builder" not in st.session_state:
        st.session_state.sparql_builder = SPARQLQueryBuilder(get_fuseki())
    return st.session_state.sparql_builder


def get_marquez() -> MarquezClient:
    """Get or create Marquez client."""
    if "marquez_client" not in st.session_state:
        st.session_state.marquez_client = MarquezClient(
            base_url=os.environ.get("MARQUEZ_URL", "http://localhost:5000")
        )
    return st.session_state.marquez_client


def get_rule_learner():
    """Get or create LTN rule learner."""
    if "rule_learner" not in st.session_state:
        if not LTN_AVAILABLE:
            return None
        config = RuleLearningConfig(
            embedding_dim=384,
            use_text_embeddings=True,
            use_kg_embeddings=True,
        )
        st.session_state.rule_learner = LTNRuleLearner(get_neo4j(), config)
    return st.session_state.rule_learner


def get_rule_generator():
    """Get or create business rule generator."""
    if "rule_generator" not in st.session_state:
        if not LTN_AVAILABLE:
            return None
        st.session_state.rule_generator = BusinessRuleGenerator(get_neo4j())
    return st.session_state.rule_generator


def get_registry():
    """Get the dynamic tool registry."""
    if "tool_registry" not in st.session_state:
        from graphweaver_agent.dynamic_tools.tool_registry import ToolRegistry
        st.session_state.tool_registry = ToolRegistry(
            os.environ.get("DYNAMIC_TOOLS_DIR",
                          os.path.join(os.path.dirname(__file__), "dynamic_tools"))
        )
    return st.session_state.tool_registry


# =============================================================================
# Tool Implementation Functions
# =============================================================================

def impl_configure_database(host: str, port: int, database: str, username: str, password: str) -> str:
    st.session_state.pg_host = host
    st.session_state.pg_port = port
    st.session_state.pg_database = database
    st.session_state.pg_username = username
    st.session_state.pg_password = password
    if "pg_connector" in st.session_state:
        del st.session_state.pg_connector
    return f"✓ Configured database: {username}@{host}:{port}/{database}"


def impl_test_database_connection() -> str:
    result = get_pg().test_connection()
    if result["success"]:
        return f"✓ Connected to database '{result['database']}' as '{result['user']}'"
    return f"✗ Failed: {result['error']}"


def impl_list_database_tables() -> str:
    tables = get_pg().get_tables_with_info()
    output = "Tables:\n"
    for t in tables:
        output += f"  - {t['table_name']}: {t['column_count']} columns, ~{t['row_estimate']} rows\n"
    return output


def impl_get_table_schema(table_name: str) -> str:
    schema = get_pg().get_table_schema(table_name)
    if not schema:
        return f"Table '{table_name}' not found"
    
    output = f"Table: {table_name}\n"
    output += f"Primary Key: {', '.join(schema.get('primary_keys', []))}\n"
    output += "Columns:\n"
    for col in schema.get('columns', []):
        pk_marker = " [PK]" if col['name'] in schema.get('primary_keys', []) else ""
        output += f"  - {col['name']}: {col['data_type']}{pk_marker}\n"
    return output


def impl_get_column_stats(table_name: str, column_name: str) -> str:
    stats = get_pg().get_column_stats(table_name, column_name)
    output = f"Stats for {table_name}.{column_name}:\n"
    output += f"  Type: {stats.get('data_type', 'unknown')}\n"
    output += f"  Distinct: {stats.get('distinct_count', 'N/A')}\n"
    output += f"  Nulls: {stats.get('null_count', 'N/A')}\n"
    output += f"  Samples: {stats.get('sample_values', [])}\n"
    return output


def impl_run_fk_discovery(min_match_rate: float = 0.95, min_score: float = 0.5) -> str:
    try:
        pg_config = get_pg_config()
        result = run_discovery(
            host=pg_config.host,
            port=pg_config.port,
            database=pg_config.database,
            username=pg_config.username,
            password=pg_config.password,
            schema=pg_config.schema_name,
            min_match_rate=min_match_rate,
            min_score=min_score,
        )
        
        summary = result["summary"]
        output = "## FK Discovery Results\n\n"
        output += f"- Tables scanned: {summary['tables_scanned']}\n"
        output += f"- Total columns: {summary['total_columns']}\n"
        output += f"- Initial candidates: {summary['initial_candidates']}\n"
        output += f"- **Final FKs discovered: {summary['final_fks_discovered']}**\n"
        output += f"- Duration: {summary['duration_seconds']}s\n\n"
        
        if result.get("fks"):
            output += "### Discovered FKs:\n"
            for fk in result["fks"]:
                output += f"  - {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']} (score: {fk['score']:.2f})\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def impl_analyze_potential_fk(source_table: str, source_column: str, target_table: str, target_column: str) -> str:
    from graphweaver_agent.discovery.pipeline import FKDetectionPipeline
    
    pg = get_pg()
    pipeline = FKDetectionPipeline(pg)
    score = pipeline.analyze_candidate(source_table, source_column, target_table, target_column)
    
    output = f"Analysis: {source_table}.{source_column} → {target_table}.{target_column}\n"
    output += f"  Score: {score:.3f}\n"
    if score >= 0.8:
        output += "  Recommendation: LIKELY FK - Should validate with data\n"
    elif score >= 0.5:
        output += "  Recommendation: POSSIBLE FK - Needs investigation\n"
    else:
        output += "  Recommendation: UNLIKELY FK\n"
    return output


def impl_validate_fk_with_data(source_table: str, source_column: str, target_table: str, target_column: str) -> str:
    pg = get_pg()
    result = pg.validate_fk(source_table, source_column, target_table, target_column)
    
    output = f"Validation: {source_table}.{source_column} → {target_table}.{target_column}\n"
    if result['is_valid']:
        output += f"  ✓ CONFIRMED FK\n"
        output += f"  Match rate: {result['match_rate']:.1%}\n"
        output += f"  Cardinality: {result.get('cardinality', 'unknown')}\n"
    else:
        output += f"  ✗ NOT A VALID FK\n"
        output += f"  Match rate: {result['match_rate']:.1%}\n"
        if result.get('orphan_count'):
            output += f"  Orphan values: {result['orphan_count']}\n"
    return output


def impl_clear_neo4j_graph() -> str:
    neo4j = get_neo4j()
    neo4j.run_write("MATCH (n) DETACH DELETE n")
    return "✓ Neo4j graph cleared"


def impl_add_fk_to_graph(source_table: str, source_column: str, target_table: str, target_column: str, score: float = 1.0, cardinality: str = "1:N") -> str:
    builder = GraphBuilder(get_neo4j())
    builder.add_fk_relationship(source_table, source_column, target_table, target_column, score, cardinality)
    return f"✓ Added: {source_table}.{source_column} → {target_table}.{target_column}"


def impl_get_graph_stats() -> str:
    stats = GraphAnalyzer(get_neo4j()).get_statistics()
    return f"Graph: {stats['tables']} tables, {stats['columns']} columns, {stats['fks']} FKs"


def impl_analyze_graph_centrality() -> str:
    result = GraphAnalyzer(get_neo4j()).centrality_analysis()
    output = "Centrality Analysis:\n"
    output += f"  Hub tables (fact/transaction): {result['hub_tables']}\n"
    output += f"  Authority tables (dimension): {result['authority_tables']}\n"
    output += f"  Isolated tables: {result['isolated_tables']}\n"
    return output


def impl_find_table_communities() -> str:
    communities = GraphAnalyzer(get_neo4j()).community_detection()
    if not communities:
        return "No communities found."
    output = "Communities:\n"
    for i, c in enumerate(communities):
        output += f"  {i+1}. {', '.join(c['tables'])}\n"
    return output


def impl_predict_missing_fks() -> str:
    predictions = GraphAnalyzer(get_neo4j()).predict_missing_fks()
    if not predictions:
        return "No predictions - graph appears complete."
    output = "Predicted missing FKs:\n"
    for p in predictions:
        output += f"  - {p['source_table']}.{p['source_column']} → {p['target_table']}\n"
    return output


def impl_run_cypher(query: str) -> str:
    neo4j = get_neo4j()
    try:
        results = neo4j.run_query(query)
        if not results:
            return "Query executed successfully. No results returned."
        output = f"Results ({len(results)} rows):\n"
        for row in results[:50]:
            output += f"  {dict(row)}\n"
        if len(results) > 50:
            output += f"  ... and {len(results) - 50} more rows"
        return output
    except Exception as e:
        try:
            neo4j.run_write(query)
            return "Write query executed successfully."
        except Exception as e2:
            return f"Error: {e2}"


def impl_connect_datasets_to_tables() -> str:
    neo4j = get_neo4j()
    result = neo4j.run_query("""
        MATCH (d:Dataset)
        MATCH (t:Table)
        WHERE d.name = t.name
        MERGE (d)-[:REPRESENTS]->(t)
        RETURN d.name as dataset, t.name as table
    """)
    if not result:
        return "No matching Dataset-Table pairs found."
    output = f"Connected {len(result)} Datasets to Tables\n"
    for row in result:
        output += f"  Dataset '{row['dataset']}' → Table '{row['table']}'\n"
    return output


def impl_generate_text_embeddings() -> str:
    try:
        from graphweaver_agent.embeddings.text_embeddings import embed_all_metadata
        stats = embed_all_metadata(
            neo4j_client=get_neo4j(),
            pg_connector=get_pg(),
            embedder=get_text_embedder(),
        )
        output = "## Text Embeddings Generated\n"
        output += f"- Tables embedded: {stats['tables']}\n"
        output += f"- Columns embedded: {stats['columns']}\n"
        output += f"- Jobs embedded: {stats['jobs']}\n"
        output += f"- Datasets embedded: {stats['datasets']}\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def impl_generate_kg_embeddings() -> str:
    try:
        from graphweaver_agent.embeddings.kg_embeddings import generate_fastrp_embeddings
        stats = generate_fastrp_embeddings(get_neo4j())
        output = "## KG Embeddings Generated\n"
        output += f"- Nodes embedded: {stats.get('nodes_embedded', 'unknown')}\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def impl_create_vector_indexes() -> str:
    try:
        from graphweaver_agent.embeddings.vector_indexes import create_all_indexes
        stats = create_all_indexes(get_neo4j())
        return f"Vector indexes created: {stats.get('indexes_created', 0)}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_semantic_search_tables(query: str, top_k: int = 5) -> str:
    try:
        from graphweaver_agent.embeddings.text_embeddings import search_tables
        results = search_tables(get_neo4j(), get_text_embedder(), query, top_k)
        if not results:
            return f"No tables found matching '{query}'"
        output = f"## Tables matching '{query}'\n"
        for r in results:
            output += f"- **{r['name']}** (score: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_semantic_search_columns(query: str, top_k: int = 10) -> str:
    try:
        from graphweaver_agent.embeddings.text_embeddings import search_columns
        results = search_columns(get_neo4j(), get_text_embedder(), query, top_k)
        if not results:
            return f"No columns found matching '{query}'"
        output = f"## Columns matching '{query}'\n"
        for r in results:
            output += f"- **{r['table']}.{r['name']}** (score: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_find_similar_tables(table_name: str, top_k: int = 5) -> str:
    try:
        from graphweaver_agent.embeddings.text_embeddings import find_similar_tables
        results = find_similar_tables(get_neo4j(), table_name, top_k)
        if not results:
            return f"No similar tables found for '{table_name}'"
        output = f"## Tables similar to '{table_name}'\n"
        for r in results:
            output += f"- **{r['name']}** (similarity: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_find_similar_columns(table_name: str, column_name: str, top_k: int = 10) -> str:
    try:
        from graphweaver_agent.embeddings.text_embeddings import find_similar_columns
        results = find_similar_columns(get_neo4j(), table_name, column_name, top_k)
        if not results:
            return f"No similar columns found for '{table_name}.{column_name}'"
        output = f"## Columns similar to '{table_name}.{column_name}'\n"
        for r in results:
            output += f"- **{r['table']}.{r['name']}** (similarity: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_predict_fks_from_embeddings(threshold: float = 0.7, top_k: int = 20) -> str:
    try:
        from graphweaver_agent.embeddings.semantic_fk import predict_fks_from_embeddings
        predictions = predict_fks_from_embeddings(get_neo4j(), threshold, top_k)
        if not predictions:
            return "No FK predictions found."
        output = "## FK Predictions from Embeddings\n"
        for p in predictions:
            output += f"- {p['source_table']}.{p['source_column']} → {p['target_table']}.{p['target_column']} (score: {p['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_semantic_fk_discovery(source_table: str = None, min_score: float = 0.6) -> str:
    try:
        from graphweaver_agent.embeddings.semantic_fk import SemanticFKDiscovery
        discovery = SemanticFKDiscovery(get_neo4j(), get_text_embedder())
        results = discovery.discover(source_table=source_table, min_score=min_score)
        if not results:
            return "No semantic FK relationships found."
        output = "## Semantic FK Discovery Results\n"
        for r in results:
            output += f"- {r['source_table']}.{r['source_column']} → {r['target_table']}.{r['target_column']} (score: {r['score']:.3f})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_show_sample_business_rules() -> str:
    return generate_sample_rules()


def impl_load_business_rules(yaml_content: str) -> str:
    import yaml
    try:
        data = yaml.safe_load(yaml_content)
        if "rules_config" not in st.session_state:
            st.session_state.rules_config = BusinessRulesConfig()
        
        config = st.session_state.rules_config
        config.namespace = data.get("namespace", "default")
        config.rules = []
        
        for r in data.get("rules", []):
            config.rules.append(BusinessRule(
                name=r["name"],
                description=r.get("description", ""),
                sql=r["sql"],
                inputs=r.get("inputs", []),
                outputs=r.get("outputs", []),
                rule_type=r.get("type", "query"),
                tags=r.get("tags", []),
            ))
        
        return f"✓ Loaded {len(config.rules)} business rules in namespace '{config.namespace}'"
    except Exception as e:
        return f"ERROR loading rules: {type(e).__name__}: {e}"


def impl_load_business_rules_from_file(file_path: str = "business_rules.yaml") -> str:
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return impl_load_business_rules(content)
    except FileNotFoundError:
        return f"ERROR: File not found: {file_path}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_list_business_rules() -> str:
    if "rules_config" not in st.session_state or not st.session_state.rules_config.rules:
        return "No business rules loaded. Use load_business_rules or load_business_rules_from_file first."
    
    config = st.session_state.rules_config
    output = f"## Business Rules (namespace: {config.namespace})\n\n"
    for r in config.rules:
        output += f"- **{r.name}** ({r.rule_type}): {r.description}\n"
    return output


def impl_execute_business_rule(rule_name: str, capture_lineage: bool = True) -> str:
    if "rules_config" not in st.session_state:
        return "No business rules loaded."
    
    config = st.session_state.rules_config
    rule = next((r for r in config.rules if r.name == rule_name), None)
    
    if not rule:
        return f"Rule '{rule_name}' not found."
    
    try:
        executor = BusinessRulesExecutor(get_pg(), get_marquez() if capture_lineage else None)
        result = executor.execute_rule(rule, capture_lineage=capture_lineage)
        
        output = f"## Executed: {rule_name}\n"
        output += f"- Rows returned: {result.get('row_count', 0)}\n"
        if capture_lineage:
            output += f"- Lineage captured: ✓\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_execute_all_business_rules(capture_lineage: bool = True) -> str:
    if "rules_config" not in st.session_state or not st.session_state.rules_config.rules:
        return "No business rules loaded."
    
    config = st.session_state.rules_config
    executor = BusinessRulesExecutor(get_pg(), get_marquez() if capture_lineage else None)
    
    output = f"## Executing {len(config.rules)} rules\n\n"
    success = 0
    for rule in config.rules:
        try:
            result = executor.execute_rule(rule, capture_lineage=capture_lineage)
            output += f"✓ {rule.name}: {result.get('row_count', 0)} rows\n"
            success += 1
        except Exception as e:
            output += f"✗ {rule.name}: {type(e).__name__}: {e}\n"
    
    output += f"\n**{success}/{len(config.rules)} rules executed successfully**"
    return output


def impl_get_marquez_lineage(dataset_name: str, depth: int = 3) -> str:
    try:
        lineage = get_marquez().get_lineage(dataset_name, depth)
        return f"Lineage for {dataset_name}:\n{lineage}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_list_marquez_jobs() -> str:
    try:
        jobs = get_marquez().list_jobs()
        if not jobs:
            return "No jobs found in Marquez."
        output = "## Marquez Jobs\n"
        for j in jobs:
            output += f"- {j['name']} ({j['namespace']})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_import_lineage_to_graph() -> str:
    try:
        stats = import_lineage_to_neo4j(get_marquez(), get_neo4j())
        output = "## Lineage Imported to Neo4j\n"
        output += f"- Jobs imported: {stats.get('jobs', 0)}\n"
        output += f"- Datasets imported: {stats.get('datasets', 0)}\n"
        output += f"- Relationships: {stats.get('relationships', 0)}\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def impl_analyze_data_flow(table_name: str) -> str:
    try:
        analyzer = GraphAnalyzer(get_neo4j())
        flow = analyzer.analyze_data_flow(table_name)
        output = f"## Data Flow for '{table_name}'\n\n"
        output += f"### Upstream (data sources):\n"
        for u in flow.get('upstream', []):
            output += f"  - {u}\n"
        output += f"\n### Downstream (data consumers):\n"
        for d in flow.get('downstream', []):
            output += f"  - {d}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_find_impact_analysis(table_name: str) -> str:
    try:
        analyzer = GraphAnalyzer(get_neo4j())
        impact = analyzer.impact_analysis(table_name)
        output = f"## Impact Analysis for '{table_name}'\n\n"
        output += f"If this table changes, the following will be affected:\n\n"
        for item in impact.get('affected', []):
            output += f"  - {item['type']}: {item['name']} (via {item.get('relationship', 'unknown')})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_test_rdf_connection() -> str:
    try:
        fuseki = get_fuseki()
        if fuseki.test_connection():
            return "✓ Connected to Apache Jena Fuseki"
        return "✗ Failed to connect to Fuseki"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_sync_graph_to_rdf() -> str:
    try:
        stats = sync_neo4j_to_rdf(get_neo4j(), get_fuseki())
        output = "## Graph Synced to RDF\n"
        output += f"- Triples created: {stats.get('triples', 0)}\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def impl_run_sparql(query: str) -> str:
    try:
        results = get_fuseki().sparql_query(query)
        if not results:
            return "Query executed. No results."
        output = f"Results ({len(results)} rows):\n"
        for row in results[:50]:
            output += f"  {row}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_sparql_list_tables() -> str:
    try:
        results = get_sparql().list_tables()
        if not results:
            return "No tables found in RDF store."
        output = "## Tables in RDF Store\n"
        for r in results:
            output += f"- {r.get('label', 'unknown')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_sparql_get_foreign_keys(table_name: str = None) -> str:
    try:
        results = get_sparql().get_foreign_keys(table_name)
        if not results:
            return "No foreign keys found."
        output = "## Foreign Keys\n"
        for r in results:
            output += f"- {r.get('sourceTableLabel')}.{r.get('sourceColLabel')} → {r.get('targetTableLabel')}.{r.get('targetColLabel')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_sparql_table_lineage(table_name: str) -> str:
    try:
        results = get_sparql().get_table_lineage(table_name)
        if not results:
            return f"No lineage found for {table_name}"
        output = f"## Lineage for {table_name}\n"
        for r in results:
            output += f"- {r.get('jobLabel')} {r.get('direction')} {r.get('datasetLabel')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_sparql_downstream_impact(table_name: str) -> str:
    try:
        results = get_sparql().get_downstream_impact(table_name)
        if not results:
            return f"No downstream impact found for {table_name}"
        output = f"## Downstream Impact for {table_name}\n"
        for r in results:
            output += f"- {r.get('dependentTableLabel')} ({r.get('relationshipType')})\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_sparql_hub_tables(min_connections: int = 3) -> str:
    try:
        results = get_sparql().get_hub_tables(min_connections)
        if not results:
            return f"No hub tables found with {min_connections}+ connections."
        output = f"## Hub Tables ({min_connections}+ connections)\n"
        for r in results:
            output += f"- {r.get('label')}: {r.get('totalConnections')} connections\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_sparql_orphan_tables() -> str:
    try:
        results = get_sparql().find_orphan_tables()
        if not results:
            return "No orphan tables found."
        output = "## Orphan Tables\n"
        for r in results:
            output += f"- {r.get('label')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_sparql_search(search_term: str) -> str:
    try:
        results = get_sparql().search_by_label(search_term)
        if not results:
            return f"No results for '{search_term}'"
        output = f"## Search Results for '{search_term}'\n"
        for r in results:
            output += f"- {r.get('type', 'unknown')}: {r.get('label')}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_get_rdf_statistics() -> str:
    try:
        stats = get_sparql().get_statistics()
        output = "## RDF Statistics\n"
        for k, v in stats.items():
            output += f"- {k}: {v}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_export_rdf_turtle() -> str:
    try:
        turtle = get_fuseki().export_turtle()
        return f"## RDF Turtle Export\n```turtle\n{turtle[:5000]}{'...' if len(turtle) > 5000 else ''}\n```"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_learn_rules_with_ltn(epochs: int = 100) -> str:
    if not LTN_AVAILABLE:
        return "LTN not available. Please install ltn package."
    try:
        learner = get_rule_learner()
        results = learner.learn(epochs=epochs)
        output = "## LTN Rule Learning Results\n"
        output += f"- Epochs: {epochs}\n"
        output += f"- Rules learned: {len(results.get('rules', []))}\n"
        for r in results.get('rules', []):
            output += f"  - {r['name']}: confidence {r['confidence']:.3f}\n"
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def impl_generate_business_rules_from_ltn() -> str:
    if not LTN_AVAILABLE:
        return "LTN not available."
    try:
        generator = get_rule_generator()
        rules = generator.generate_from_ltn()
        if "generated_rules" not in st.session_state:
            st.session_state.generated_rules = []
        st.session_state.generated_rules = rules
        
        output = f"## Generated {len(rules)} Business Rules from LTN\n"
        for r in rules:
            output += f"- {r.name}: {r.description}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_generate_all_validation_rules() -> str:
    if not LTN_AVAILABLE:
        return "LTN not available."
    try:
        generator = get_rule_generator()
        rules = generator.generate_all_validation_rules(get_neo4j())
        if "generated_rules" not in st.session_state:
            st.session_state.generated_rules = []
        st.session_state.generated_rules.extend(rules)
        
        output = f"## Generated {len(rules)} Validation Rules\n"
        for r in rules:
            output += f"- {r.name}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_export_generated_rules_yaml(file_path: str = "business_rules_generated.yaml") -> str:
    if "generated_rules" not in st.session_state or not st.session_state.generated_rules:
        return "No generated rules to export."
    try:
        import yaml
        rules_data = {
            "version": "1.0",
            "namespace": "generated",
            "rules": [
                {
                    "name": r.name,
                    "description": r.description,
                    "type": r.rule_type,
                    "sql": r.sql,
                    "inputs": r.inputs,
                    "outputs": r.outputs,
                    "tags": r.tags,
                }
                for r in st.session_state.generated_rules
            ]
        }
        with open(file_path, 'w') as f:
            yaml.dump(rules_data, f, default_flow_style=False)
        return f"✓ Exported {len(st.session_state.generated_rules)} rules to {file_path}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_export_generated_rules_sql() -> str:
    if "generated_rules" not in st.session_state or not st.session_state.generated_rules:
        return "No generated rules to export."
    output = "## Generated Rules SQL\n\n"
    for r in st.session_state.generated_rules:
        output += f"-- {r.name}: {r.description}\n"
        output += f"{r.sql}\n\n"
    return output


def impl_show_ltn_knowledge_base() -> str:
    if not LTN_AVAILABLE:
        return "LTN not available."
    try:
        kb = LTNKnowledgeBase(get_neo4j())
        state = kb.get_state()
        output = "## LTN Knowledge Base\n"
        output += f"- Tables: {state.get('tables', 0)}\n"
        output += f"- Columns: {state.get('columns', 0)}\n"
        output += f"- FKs: {state.get('fks', 0)}\n"
        output += f"- Predicates: {state.get('predicates', [])}\n"
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_check_tool_exists(tool_name: str) -> str:
    return "✓ EXISTS" if get_registry().tool_exists(tool_name) else "✗ NOT FOUND"


def impl_list_available_tools() -> str:
    dynamic = get_registry().list_tools()
    output = "## Available Tools\n\n"
    output += "### Builtin Tools:\n"
    output += "- **Database**: configure_database, test_database_connection, list_database_tables, get_table_schema, get_column_stats\n"
    output += "- **FK Discovery**: run_fk_discovery, analyze_potential_fk, validate_fk_with_data\n"
    output += "- **Graph**: clear_neo4j_graph, add_fk_to_graph, get_graph_stats, analyze_graph_centrality, find_table_communities, predict_missing_fks, run_cypher\n"
    output += "- **Embeddings**: generate_text_embeddings, generate_kg_embeddings, create_vector_indexes, semantic_search_tables, semantic_search_columns\n"
    output += "- **Business Rules**: load_business_rules, execute_business_rule, execute_all_business_rules\n"
    output += "- **RDF/SPARQL**: test_rdf_connection, sync_graph_to_rdf, run_sparql\n"
    output += "- **LTN**: learn_rules_with_ltn, generate_business_rules_from_ltn\n"
    output += "\n### Dynamic Tools:\n"
    if dynamic:
        for t in dynamic:
            output += f"- **{t['name']}**: {t.get('description', 'No description')}\n"
    else:
        output += "- None created yet\n"
    return output


def impl_create_dynamic_tool(name: str, description: str, code: str) -> str:
    r = get_registry()
    if r.tool_exists(name):
        return f"ERROR: Tool '{name}' already exists."
    if "def run(" not in code:
        return "ERROR: Code must define a run() function."
    try:
        compile(code, name, "exec")
        path = r.create_tool(name, description, code)
        return f"✓ Created tool '{name}'"
    except SyntaxError as e:
        return f"ERROR: Syntax error: {e}"


def impl_run_dynamic_tool(tool_name: str, **kwargs) -> str:
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found."
    try:
        return str(r.execute_tool(tool_name, **kwargs))
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_get_tool_source(tool_name: str) -> str:
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found"
    return r.get_tool_source(tool_name)


def impl_update_dynamic_tool(tool_name: str, code: str, description: str = None) -> str:
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found."
    try:
        r.update_tool(tool_name, code, description)
        return f"✓ Updated tool '{tool_name}'"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def impl_delete_dynamic_tool(tool_name: str) -> str:
    r = get_registry()
    if not r.tool_exists(tool_name):
        return f"ERROR: Tool '{tool_name}' not found"
    r.delete_tool(tool_name)
    return f"✓ Deleted tool '{tool_name}'"


# =============================================================================
# Tool Function Mapping
# =============================================================================

STREAMING_TOOL_FUNCTIONS = {
    # Dynamic tools
    "check_tool_exists": lambda **kw: impl_check_tool_exists(**kw),
    "list_available_tools": lambda **kw: impl_list_available_tools(),
    "create_dynamic_tool": lambda **kw: impl_create_dynamic_tool(**kw),
    "run_dynamic_tool": lambda **kw: impl_run_dynamic_tool(**kw),
    "get_tool_source": lambda **kw: impl_get_tool_source(**kw),
    "update_dynamic_tool": lambda **kw: impl_update_dynamic_tool(**kw),
    "delete_dynamic_tool": lambda **kw: impl_delete_dynamic_tool(**kw),
    # Database
    "configure_database": lambda **kw: impl_configure_database(**kw),
    "test_database_connection": lambda **kw: impl_test_database_connection(),
    "list_database_tables": lambda **kw: impl_list_database_tables(),
    "get_table_schema": lambda **kw: impl_get_table_schema(**kw),
    "get_column_stats": lambda **kw: impl_get_column_stats(**kw),
    # FK Discovery
    "run_fk_discovery": lambda **kw: impl_run_fk_discovery(**kw),
    "analyze_potential_fk": lambda **kw: impl_analyze_potential_fk(**kw),
    "validate_fk_with_data": lambda **kw: impl_validate_fk_with_data(**kw),
    # Graph
    "clear_neo4j_graph": lambda **kw: impl_clear_neo4j_graph(),
    "add_fk_to_graph": lambda **kw: impl_add_fk_to_graph(**kw),
    "get_graph_stats": lambda **kw: impl_get_graph_stats(),
    "analyze_graph_centrality": lambda **kw: impl_analyze_graph_centrality(),
    "find_table_communities": lambda **kw: impl_find_table_communities(),
    "predict_missing_fks": lambda **kw: impl_predict_missing_fks(),
    "run_cypher": lambda **kw: impl_run_cypher(**kw),
    "connect_datasets_to_tables": lambda **kw: impl_connect_datasets_to_tables(),
    # Embeddings
    "generate_text_embeddings": lambda **kw: impl_generate_text_embeddings(),
    "generate_kg_embeddings": lambda **kw: impl_generate_kg_embeddings(),
    "create_vector_indexes": lambda **kw: impl_create_vector_indexes(),
    "semantic_search_tables": lambda **kw: impl_semantic_search_tables(**kw),
    "semantic_search_columns": lambda **kw: impl_semantic_search_columns(**kw),
    "find_similar_tables": lambda **kw: impl_find_similar_tables(**kw),
    "find_similar_columns": lambda **kw: impl_find_similar_columns(**kw),
    "predict_fks_from_embeddings": lambda **kw: impl_predict_fks_from_embeddings(**kw),
    "semantic_fk_discovery": lambda **kw: impl_semantic_fk_discovery(**kw),
    # Business Rules
    "show_sample_business_rules": lambda **kw: impl_show_sample_business_rules(),
    "load_business_rules": lambda **kw: impl_load_business_rules(**kw),
    "load_business_rules_from_file": lambda **kw: impl_load_business_rules_from_file(**kw),
    "list_business_rules": lambda **kw: impl_list_business_rules(),
    "execute_business_rule": lambda **kw: impl_execute_business_rule(**kw),
    "execute_all_business_rules": lambda **kw: impl_execute_all_business_rules(**kw),
    "get_marquez_lineage": lambda **kw: impl_get_marquez_lineage(**kw),
    "list_marquez_jobs": lambda **kw: impl_list_marquez_jobs(),
    "import_lineage_to_graph": lambda **kw: impl_import_lineage_to_graph(),
    "analyze_data_flow": lambda **kw: impl_analyze_data_flow(**kw),
    "find_impact_analysis": lambda **kw: impl_find_impact_analysis(**kw),
    # RDF
    "test_rdf_connection": lambda **kw: impl_test_rdf_connection(),
    "sync_graph_to_rdf": lambda **kw: impl_sync_graph_to_rdf(),
    "run_sparql": lambda **kw: impl_run_sparql(**kw),
    "sparql_list_tables": lambda **kw: impl_sparql_list_tables(),
    "sparql_get_foreign_keys": lambda **kw: impl_sparql_get_foreign_keys(**kw),
    "sparql_table_lineage": lambda **kw: impl_sparql_table_lineage(**kw),
    "sparql_downstream_impact": lambda **kw: impl_sparql_downstream_impact(**kw),
    "sparql_hub_tables": lambda **kw: impl_sparql_hub_tables(**kw),
    "sparql_orphan_tables": lambda **kw: impl_sparql_orphan_tables(),
    "sparql_search": lambda **kw: impl_sparql_search(**kw),
    "get_rdf_statistics": lambda **kw: impl_get_rdf_statistics(),
    "export_rdf_turtle": lambda **kw: impl_export_rdf_turtle(),
    # LTN
    "learn_rules_with_ltn": lambda **kw: impl_learn_rules_with_ltn(**kw),
    "generate_business_rules_from_ltn": lambda **kw: impl_generate_business_rules_from_ltn(),
    "generate_all_validation_rules": lambda **kw: impl_generate_all_validation_rules(),
    "export_generated_rules_yaml": lambda **kw: impl_export_generated_rules_yaml(**kw),
    "export_generated_rules_sql": lambda **kw: impl_export_generated_rules_sql(),
    "show_ltn_knowledge_base": lambda **kw: impl_show_ltn_knowledge_base(),
}


# =============================================================================
# Streaming Chat Implementation
# =============================================================================

def stream_agent_response(client: anthropic.Anthropic, messages: List[Dict], message_placeholder) -> str:
    """
    Stream response from Claude with tool execution, updating the Streamlit placeholder in real-time.
    
    Returns the final complete response text.
    """
    full_response = ""
    tool_results_log = []
    
    # Agentic loop - keep going until no more tool calls
    while True:
        tool_inputs = {}
        current_block_id = None
        current_response_content = []
        
        # Stream the response
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=STREAMING_TOOLS,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_block_id = event.content_block.id
                        tool_inputs[current_block_id] = {
                            "name": event.content_block.name,
                            "input": ""
                        }
                        # Show tool being called
                        full_response += f"\n\n🔧 **Calling: {event.content_block.name}**\n"
                        message_placeholder.markdown(full_response + "▌")
                        
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        full_response += event.delta.text
                        message_placeholder.markdown(full_response + "▌")
                    elif event.delta.type == "input_json_delta":
                        if current_block_id:
                            tool_inputs[current_block_id]["input"] += event.delta.partial_json
                            
                elif event.type == "content_block_stop":
                    current_block_id = None
            
            # Get the final message
            response = stream.get_final_message()
            current_response_content = response.content
        
        # Add assistant message to history
        messages.append({"role": "assistant", "content": current_response_content})
        
        # Check if we need to execute tools
        if response.stop_reason != "tool_use":
            break
        
        # Execute tool calls
        tool_results = []
        for block in current_response_content:
            if block.type == "tool_use":
                full_response += f"\n⏳ Executing...\n"
                message_placeholder.markdown(full_response + "▌")
                
                fn = STREAMING_TOOL_FUNCTIONS.get(block.name)
                if fn:
                    try:
                        result = fn(**block.input)
                    except Exception as e:
                        result = f"Error: {type(e).__name__}: {e}"
                else:
                    result = f"Unknown tool: {block.name}"
                
                # Show tool result
                full_response += f"\n```\n{result}\n```\n"
                message_placeholder.markdown(full_response + "▌")
                tool_results_log.append({"tool": block.name, "result": result})
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        # Add tool results to messages for next iteration
        messages.append({"role": "user", "content": tool_results})
    
    # Remove cursor and return final response
    message_placeholder.markdown(full_response)
    return full_response


def get_anthropic_client():
    """Get or create Anthropic client."""
    api_key = st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="GraphWeaver Agent",
        page_icon="🕸️",
        layout="wide",
    )
    
    st.title("🕸️ GraphWeaver Agent")
    st.caption("Chat with Claude to discover FK relationships, build knowledge graphs, and analyze data lineage")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.get("anthropic_api_key", os.environ.get("ANTHROPIC_API_KEY", "")),
            help="Enter your Anthropic API key"
        )
        if api_key:
            st.session_state.anthropic_api_key = api_key
        
        # Streaming toggle
        st.session_state.use_streaming = st.toggle(
            "🔴 Enable Streaming",
            value=st.session_state.get("use_streaming", True),
            help="Stream responses in real-time"
        )
        
        st.divider()
        
        # Database Configuration
        st.subheader("🗄️ PostgreSQL")
        pg_host = st.text_input("Host", value=st.session_state.get("pg_host", os.environ.get("POSTGRES_HOST", "localhost")))
        pg_port = st.number_input("Port", value=int(st.session_state.get("pg_port", os.environ.get("POSTGRES_PORT", "5432"))), min_value=1, max_value=65535)
        pg_database = st.text_input("Database", value=st.session_state.get("pg_database", os.environ.get("POSTGRES_DB", "orders")))
        pg_username = st.text_input("Username", value=st.session_state.get("pg_username", os.environ.get("POSTGRES_USER", "saphenia")))
        pg_password = st.text_input("Password", type="password", value=st.session_state.get("pg_password", os.environ.get("POSTGRES_PASSWORD", "secret")))
        
        if st.button("Update DB Config"):
            st.session_state.pg_host = pg_host
            st.session_state.pg_port = pg_port
            st.session_state.pg_database = pg_database
            st.session_state.pg_username = pg_username
            st.session_state.pg_password = pg_password
            if "pg_connector" in st.session_state:
                del st.session_state.pg_connector
            st.success("Database configuration updated!")
        
        st.divider()
        
        # Neo4j Configuration
        st.subheader("🔵 Neo4j")
        neo4j_uri = st.text_input("URI", value=st.session_state.get("neo4j_uri", os.environ.get("NEO4J_URI", "bolt://localhost:7687")))
        neo4j_user = st.text_input("Neo4j User", value=st.session_state.get("neo4j_user", os.environ.get("NEO4J_USER", "neo4j")))
        neo4j_password = st.text_input("Neo4j Password", type="password", value=st.session_state.get("neo4j_password", os.environ.get("NEO4J_PASSWORD", "password")))
        
        if st.button("Update Neo4j Config"):
            st.session_state.neo4j_uri = neo4j_uri
            st.session_state.neo4j_user = neo4j_user
            st.session_state.neo4j_password = neo4j_password
            if "neo4j_client" in st.session_state:
                del st.session_state.neo4j_client
            st.success("Neo4j configuration updated!")
        
        st.divider()
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        if st.button("🔍 Discover FKs", use_container_width=True):
            st.session_state.quick_action = "Discover all foreign key relationships in the database"
        if st.button("📊 List Tables", use_container_width=True):
            st.session_state.quick_action = "List all database tables"
        if st.button("🧠 Generate Embeddings", use_container_width=True):
            st.session_state.quick_action = "Generate text embeddings for semantic search"
        if st.button("📈 Analyze Graph", use_container_width=True):
            st.session_state.quick_action = "Analyze graph centrality and find communities"
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.streaming_messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "streaming_messages" not in st.session_state:
        st.session_state.streaming_messages = []
    
    # Check for API key
    if not st.session_state.get("anthropic_api_key") and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to start chatting.")
        st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle quick actions
    if "quick_action" in st.session_state and st.session_state.quick_action:
        prompt = st.session_state.quick_action
        st.session_state.quick_action = None
    else:
        prompt = st.chat_input("Ask me about your database...")
    
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            try:
                if st.session_state.get("use_streaming", True):
                    # STREAMING MODE
                    client = get_anthropic_client()
                    if client is None:
                        st.error("Failed to create Anthropic client. Please check your API key.")
                        st.stop()
                    
                    # Build messages for API
                    st.session_state.streaming_messages.append({"role": "user", "content": prompt})
                    
                    # Create placeholder for streaming
                    message_placeholder = st.empty()
                    
                    # Stream response
                    response = stream_agent_response(
                        client, 
                        st.session_state.streaming_messages.copy(),
                        message_placeholder
                    )
                    
                    # Update chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.streaming_messages.append({"role": "assistant", "content": response})
                    
                    # Keep history manageable
                    if len(st.session_state.streaming_messages) > 20:
                        st.session_state.streaming_messages = st.session_state.streaming_messages[-20:]
                else:
                    # NON-STREAMING MODE (fallback to LangChain)
                    with st.spinner("Thinking..."):
                        from langgraph.prebuilt import create_react_agent
                        
                        llm = ChatAnthropic(
                            model="claude-sonnet-4-20250514",
                            api_key=st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY"),
                            max_tokens=8192,
                        )
                        
                        # Create tools list (simplified for non-streaming)
                        tools = [
                            test_database_connection_tool,
                            list_database_tables_tool,
                            run_fk_discovery_tool,
                            get_graph_stats_tool,
                        ]
                        
                        agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
                        result = agent.invoke(
                            {"messages": [HumanMessage(content=prompt)]},
                            config={"recursion_limit": 100}
                        )
                        
                        # Extract response
                        messages = result.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            response = getattr(last_msg, 'content', str(last_msg))
                        else:
                            response = str(result)
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
            except Exception as e:
                import traceback
                error_msg = f"Error: {type(e).__name__}: {e}\n\n```\n{traceback.format_exc()}\n```"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})


# Tool wrappers for non-streaming mode
@tool
def test_database_connection_tool() -> str:
    """Test connection to PostgreSQL database."""
    return impl_test_database_connection()

@tool
def list_database_tables_tool() -> str:
    """List all tables with row counts."""
    return impl_list_database_tables()

@tool
def run_fk_discovery_tool(min_match_rate: float = 0.95, min_score: float = 0.5) -> str:
    """Run complete 5-stage FK discovery pipeline."""
    return impl_run_fk_discovery(min_match_rate, min_score)

@tool
def get_graph_stats_tool() -> str:
    """Get current graph statistics."""
    return impl_get_graph_stats()


if __name__ == "__main__":
    main()
