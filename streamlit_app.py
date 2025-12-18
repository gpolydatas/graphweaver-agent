"""
graphweaver-agent/streamlit_app.py

Streamlit Chat Interface for GraphWeaver Agent
"""
import os
import sys
import streamlit as st
from typing import Optional

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


# =============================================================================
# Tool Definitions
# =============================================================================

@tool
def configure_database(host: str, port: int, database: str, username: str, password: str) -> str:
    """Configure which PostgreSQL database to connect to."""
    st.session_state.pg_host = host
    st.session_state.pg_port = port
    st.session_state.pg_database = database
    st.session_state.pg_username = username
    st.session_state.pg_password = password
    if "pg_connector" in st.session_state:
        del st.session_state.pg_connector
    return f"✓ Configured database: {username}@{host}:{port}/{database}"


@tool
def test_database_connection() -> str:
    """Test connection to PostgreSQL database."""
    result = get_pg().test_connection()
    if result["success"]:
        return f"✓ Connected to database '{result['database']}' as '{result['user']}'"
    return f"✗ Failed: {result['error']}"


@tool
def list_database_tables() -> str:
    """List all tables with row counts."""
    tables = get_pg().get_tables_with_info()
    output = "Tables:\n"
    for t in tables:
        output += f"  - {t['table_name']}: {t['column_count']} columns, ~{t['row_estimate']} rows\n"
    return output


@tool
def run_fk_discovery(min_match_rate: float = 0.95, min_score: float = 0.5) -> str:
    """Run complete 5-stage FK discovery pipeline on the database."""
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
        output += "### Pipeline Summary\n"
        output += f"- Tables scanned: {summary['tables_scanned']}\n"
        output += f"- Total columns: {summary['total_columns']}\n"
        output += f"- Initial candidates: {summary['initial_candidates']}\n"
        output += f"- After Stage 1 (Statistical): {summary['stage1_statistical_passed']}\n"
        output += f"- After Stage 2 (Mathematical): {summary['stage2_mathematical_passed']}\n"
        output += f"- After Stage 3 (Sampling): {summary['stage3_sampling_passed']}\n"
        output += f"- **Final FKs discovered: {summary['final_fks_discovered']}**\n"
        output += f"- Duration: {summary['duration_seconds']}s\n\n"
        
        output += "### Discovered Foreign Keys\n\n"
        if result["discovered_fks"]:
            for fk in result["discovered_fks"]:
                scores = fk["scores"]
                output += f"**{fk['relationship']}**\n"
                output += f"  - Confidence: {fk['confidence']:.1%}\n"
                output += f"  - Cardinality: {fk['cardinality']}\n"
                output += f"  - Scores: name={scores['name_similarity']:.2f}, "
                output += f"type={scores['type_score']:.2f}, "
                output += f"uniqueness={scores['uniqueness']:.2f}, "
                output += f"geometric_mean={scores['geometric_mean']:.2f}, "
                output += f"match_rate={scores['match_rate']:.1%}\n\n"
        else:
            output += "No foreign keys discovered.\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR in FK discovery: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def get_table_schema(table_name: str) -> str:
    """Get columns and primary keys for a table."""
    meta = get_pg().get_table_metadata(table_name)
    output = f"Table: {table_name} ({meta.row_count} rows)\n"
    output += f"Primary Key: {', '.join(meta.primary_key_columns) or 'None'}\n"
    output += "Columns:\n"
    for col in meta.columns:
        pk = " [PK]" if col.is_primary_key else ""
        output += f"  - {col.column_name}: {col.data_type.value}{pk}\n"
    return output


@tool
def get_column_stats(table_name: str, column_name: str) -> str:
    """Get statistics for a column - uniqueness, nulls, samples."""
    stats = get_pg().get_column_statistics(table_name, column_name)
    return (f"{table_name}.{column_name}:\n"
            f"  Distinct: {stats.distinct_count}/{stats.total_count} ({stats.uniqueness_ratio:.1%})\n"
            f"  Nulls: {stats.null_count} ({stats.null_ratio:.1%})\n"
            f"  Samples: {stats.sample_values[:5]}")


@tool
def analyze_potential_fk(source_table: str, source_column: str, 
                         target_table: str, target_column: str) -> str:
    """Analyze if a column pair could be a FK."""
    pg = get_pg()
    source_stats = pg.get_column_statistics(source_table, source_column)
    target_stats = pg.get_column_statistics(target_table, target_column)
    source_meta = pg.get_table_metadata(source_table)
    target_meta = pg.get_table_metadata(target_table)
    
    source_col = next((c for c in source_meta.columns if c.column_name == source_column), None)
    target_col = next((c for c in target_meta.columns if c.column_name == target_column), None)
    
    if not source_col or not target_col:
        return "Error: Column not found"
    
    type_compatible = source_col.data_type == target_col.data_type
    target_unique = target_stats.uniqueness_ratio > 0.95
    
    output = f"Analysis: {source_table}.{source_column} → {target_table}.{target_column}\n"
    output += f"  Type compatible: {type_compatible}\n"
    output += f"  Target uniqueness: {target_stats.uniqueness_ratio:.1%}\n"
    output += f"  Source distinct: {source_stats.distinct_count}\n"
    output += f"  Target distinct: {target_stats.distinct_count}\n"
    
    if type_compatible and target_unique:
        output += f"  Recommendation: LIKELY FK - validate with data\n"
    elif type_compatible:
        output += f"  Recommendation: POSSIBLE - target not unique enough\n"
    else:
        output += f"  Recommendation: UNLIKELY - type mismatch\n"
    
    return output


@tool
def validate_fk_with_data(source_table: str, source_column: str,
                          target_table: str, target_column: str) -> str:
    """Validate FK by checking if values actually exist."""
    pg = get_pg()
    result = pg.check_referential_integrity(source_table, source_column, target_table, target_column)
    
    if result["match_rate"] >= 0.95:
        verdict = "✓ CONFIRMED FK"
    elif result["match_rate"] >= 0.8:
        verdict = "⚠ LIKELY FK (some orphans)"
    else:
        verdict = "✗ NOT A FK"
    
    return (f"Validation: {source_table}.{source_column} → {target_table}.{target_column}\n"
            f"  {verdict}\n"
            f"  Match rate: {result['match_rate']:.1%} ({result['matches']}/{result['sample_size']})")


@tool
def clear_neo4j_graph() -> str:
    """Clear all data from Neo4j graph."""
    try:
        GraphBuilder(get_neo4j()).clear_graph()
        return "✓ Graph cleared"
    except Exception as e:
        return f"ERROR clearing graph: {type(e).__name__}: {e}"


@tool
def add_fk_to_graph(source_table: str, source_column: str,
                    target_table: str, target_column: str,
                    score: float, cardinality: str = "1:N") -> str:
    """Add a confirmed FK relationship to the Neo4j graph."""
    try:
        builder = GraphBuilder(get_neo4j())
        builder.add_table(source_table)
        builder.add_table(target_table)
        builder.add_fk_relationship(source_table, source_column, target_table, target_column, score, cardinality)
        return f"✓ Added: {source_table}.{source_column} → {target_table}.{target_column}"
    except Exception as e:
        return f"ERROR adding FK: {type(e).__name__}: {e}"


@tool
def get_graph_stats() -> str:
    """Get current graph statistics."""
    try:
        stats = GraphAnalyzer(get_neo4j()).get_statistics()
        return f"Graph: {stats['tables']} tables, {stats['columns']} columns, {stats['fks']} FKs"
    except Exception as e:
        return f"ERROR getting stats: {type(e).__name__}: {e}"


@tool
def analyze_graph_centrality() -> str:
    """Find hub tables (fact tables) and authority tables (dimensions)."""
    try:
        result = GraphAnalyzer(get_neo4j()).centrality_analysis()
        output = "Centrality Analysis:\n"
        output += f"  Hub tables (fact/transaction): {result['hub_tables']}\n"
        output += f"  Authority tables (dimension): {result['authority_tables']}\n"
        output += f"  Isolated tables: {result['isolated_tables']}\n"
        return output
    except Exception as e:
        return f"ERROR analyzing centrality: {type(e).__name__}: {e}"


@tool
def find_table_communities() -> str:
    """Find clusters of related tables."""
    try:
        communities = GraphAnalyzer(get_neo4j()).community_detection()
        if not communities:
            return "No communities found."
        output = "Communities:\n"
        for i, c in enumerate(communities):
            output += f"  {i+1}. {', '.join(c['tables'])}\n"
        return output
    except Exception as e:
        return f"ERROR finding communities: {type(e).__name__}: {e}"


@tool
def predict_missing_fks() -> str:
    """Predict missing FKs based on column naming patterns."""
    try:
        predictions = GraphAnalyzer(get_neo4j()).predict_missing_fks()
        if not predictions:
            return "No predictions - graph appears complete."
        output = "Predicted missing FKs:\n"
        for p in predictions:
            output += f"  - {p['source_table']}.{p['source_column']} → {p['target_table']}\n"
        return output
    except Exception as e:
        return f"ERROR predicting FKs: {type(e).__name__}: {e}"


@tool
def run_cypher(query: str) -> str:
    """Run a custom Cypher query on the Neo4j graph database."""
    try:
        neo4j = get_neo4j()
    except Exception as e:
        return f"Error: Not connected to Neo4j: {e}"
    
    try:
        results = neo4j.run_query(query)
        if results is None:
            results = []
        
        if not results:
            return "Query executed successfully. No results returned (0 rows)."
        
        output = f"Results ({len(results)} rows):\n"
        for i, row in enumerate(results[:50]):
            output += f"  {dict(row)}\n"
        if len(results) > 50:
            output += f"  ... and {len(results) - 50} more rows"
        return output
    except Exception as e:
        try:
            neo4j.run_write(query)
            return "Write query executed successfully."
        except Exception as e2:
            return f"Error executing query: {e2}"


@tool
def connect_datasets_to_tables() -> str:
    """Connect Dataset nodes to their matching Table nodes in the graph."""
    try:
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
        
        output = f"## Connected {len(result)} Datasets to Tables\n\n"
        output += "Created REPRESENTS relationships:\n"
        for row in result:
            output += f"  Dataset '{row['dataset']}' → Table '{row['table']}'\n"
        output += "\nThe FK graph and lineage graph are now connected!"
        
        return output
    except Exception as e:
        return f"ERROR connecting datasets to tables: {type(e).__name__}: {e}"


@tool
def generate_text_embeddings() -> str:
    """Generate text embeddings for all tables, columns, jobs, and datasets."""
    try:
        from graphweaver_agent.embeddings.text_embeddings import embed_all_metadata, TextEmbedder
        
        neo4j = get_neo4j()
        pg = get_pg()
        embedder = get_text_embedder()
        
        stats = embed_all_metadata(
            neo4j_client=neo4j,
            pg_connector=pg,
            embedder=embedder,
        )
        
        output = "## Text Embeddings Generated\n\n"
        output += f"- Tables embedded: {stats['tables']}\n"
        output += f"- Columns embedded: {stats['columns']}\n"
        output += f"- Jobs embedded: {stats['jobs']}\n"
        output += f"- Datasets embedded: {stats['datasets']}\n"
        output += "\nText embeddings are now stored on Neo4j nodes."
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR generating text embeddings: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def generate_kg_embeddings() -> str:
    """Generate knowledge graph embeddings using Neo4j GDS FastRP algorithm."""
    try:
        from graphweaver_agent.embeddings.kg_embeddings import generate_fastrp_embeddings
        
        stats = generate_fastrp_embeddings(get_neo4j())
        
        output = "## KG Embeddings Generated\n\n"
        output += f"- Nodes embedded: {stats.get('nodes_embedded', 'unknown')}\n"
        output += f"- Embedding dimension: 128\n"
        output += "\nKG embeddings capture graph structure/topology."
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR generating KG embeddings: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def create_vector_indexes() -> str:
    """Create Neo4j vector indexes for fast similarity search."""
    try:
        from graphweaver_agent.embeddings.vector_indexes import create_all_indexes
        
        stats = create_all_indexes(get_neo4j())
        
        output = "## Vector Indexes Created\n\n"
        output += f"- Indexes created: {stats.get('indexes_created', 0)}\n"
        output += "\nYou can now use semantic search tools."
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR creating indexes: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def semantic_search_tables(query: str, limit: int = 5) -> str:
    """Search tables using natural language."""
    try:
        from graphweaver_agent.embeddings.text_embeddings import search_tables
        
        results = search_tables(get_neo4j(), get_text_embedder(), query, limit)
        
        if not results:
            return f"No tables found matching '{query}'"
        
        output = f"## Tables matching '{query}'\n\n"
        for r in results:
            output += f"- **{r['name']}** (score: {r['score']:.3f})\n"
        
        return output
    except Exception as e:
        return f"ERROR searching: {type(e).__name__}: {e}"


@tool
def semantic_search_columns(query: str, limit: int = 10) -> str:
    """Search columns using natural language."""
    try:
        from graphweaver_agent.embeddings.text_embeddings import search_columns
        
        results = search_columns(get_neo4j(), get_text_embedder(), query, limit)
        
        if not results:
            return f"No columns found matching '{query}'"
        
        output = f"## Columns matching '{query}'\n\n"
        for r in results:
            output += f"- **{r['table']}.{r['name']}** (score: {r['score']:.3f})\n"
        
        return output
    except Exception as e:
        return f"ERROR searching: {type(e).__name__}: {e}"


@tool
def find_similar_tables(table_name: str, limit: int = 5) -> str:
    """Find structurally/semantically similar tables."""
    try:
        from graphweaver_agent.embeddings.text_embeddings import find_similar_tables as _find_similar
        
        results = _find_similar(get_neo4j(), table_name, limit)
        
        if not results:
            return f"No similar tables found for '{table_name}'"
        
        output = f"## Tables similar to '{table_name}'\n\n"
        for r in results:
            output += f"- **{r['name']}** (score: {r['score']:.3f})\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def find_similar_columns(table_name: str, column_name: str, limit: int = 10) -> str:
    """Find similar columns across all tables."""
    try:
        from graphweaver_agent.embeddings.text_embeddings import find_similar_columns as _find_similar
        
        results = _find_similar(get_neo4j(), table_name, column_name, limit)
        
        if not results:
            return f"No similar columns found for '{table_name}.{column_name}'"
        
        output = f"## Columns similar to '{table_name}.{column_name}'\n\n"
        for r in results:
            output += f"- **{r['table']}.{r['name']}** (score: {r['score']:.3f})\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def predict_fks_from_embeddings(threshold: float = 0.8) -> str:
    """Predict FKs using graph structure embeddings."""
    try:
        from graphweaver_agent.embeddings.semantic_fk import predict_fks_from_embeddings as _predict
        
        predictions = _predict(get_neo4j(), threshold)
        
        if not predictions:
            return "No FK predictions from embeddings."
        
        output = "## Predicted FKs from Graph Embeddings\n\n"
        for p in predictions:
            output += f"- {p['source_table']}.{p['source_column']} → "
            output += f"{p['target_table']}.{p['target_column']} "
            output += f"(score: {p['score']:.3f})\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def semantic_fk_discovery() -> str:
    """Discover FKs using semantic similarity (for FKs with non-standard names)."""
    try:
        from graphweaver_agent.embeddings.semantic_fk import semantic_fk_discovery as _discover
        
        results = _discover(get_neo4j(), get_pg(), get_text_embedder())
        
        if not results:
            return "No additional FKs found via semantic discovery."
        
        output = "## Semantic FK Discovery Results\n\n"
        for r in results:
            output += f"- {r['relationship']} (semantic score: {r['semantic_score']:.3f})\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# Business Rules Tools

@tool
def show_sample_business_rules() -> str:
    """Show example YAML format for business rules."""
    return generate_sample_rules()


@tool
def load_business_rules(yaml_content: str) -> str:
    """Load business rules from YAML string."""
    try:
        import yaml
        data = yaml.safe_load(yaml_content)
        
        rules = []
        for rule_data in data.get('rules', []):
            rules.append(BusinessRule(**rule_data))
        
        st.session_state.rules_config = BusinessRulesConfig(
            version=data.get('version', '1.0'),
            namespace=data.get('namespace', 'default'),
            rules=rules,
        )
        
        output = f"✓ Loaded {len(st.session_state.rules_config.rules)} business rules:\n"
        for rule in st.session_state.rules_config.rules:
            output += f"  - {rule.name}: {rule.description} [{rule.type.value}]\n"
        return output
    except Exception as e:
        return f"ERROR loading rules: {type(e).__name__}: {e}"


@tool
def load_business_rules_from_file(file_path: str = "business_rules.yaml") -> str:
    """Load business rules from a YAML file on disk."""
    try:
        import yaml
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        rules = []
        for rule_data in data.get('rules', []):
            rules.append(BusinessRule(**rule_data))
        
        st.session_state.rules_config = BusinessRulesConfig(
            version=data.get('version', '1.0'),
            namespace=data.get('namespace', 'default'),
            rules=rules,
        )
        
        output = f"✓ Loaded {len(st.session_state.rules_config.rules)} business rules from {file_path}:\n"
        for rule in st.session_state.rules_config.rules:
            output += f"  - {rule.name}: {rule.description} [{rule.type.value}]\n"
        return output
    except FileNotFoundError:
        return f"ERROR: File '{file_path}' not found"
    except Exception as e:
        return f"ERROR loading rules: {type(e).__name__}: {e}"


@tool
def list_business_rules() -> str:
    """List all loaded business rules."""
    if "rules_config" not in st.session_state or not st.session_state.rules_config.rules:
        return "No business rules loaded. Use load_business_rules() first."
    
    rules_config = st.session_state.rules_config
    output = f"## Business Rules (namespace: {rules_config.namespace})\n\n"
    for rule in rules_config.rules:
        output += f"**{rule.name}** [{rule.type.value}]\n"
        output += f"  {rule.description}\n"
        output += f"  Inputs: {', '.join(rule.inputs)}\n"
        output += f"  Outputs: {', '.join(rule.outputs) if rule.outputs else 'query result'}\n"
        if rule.tags:
            output += f"  Tags: {', '.join(rule.tags)}\n"
        output += "\n"
    return output


@tool
def execute_business_rule(rule_name: str, capture_lineage: bool = True) -> str:
    """Execute a single business rule and optionally capture lineage."""
    if "rules_config" not in st.session_state:
        return "No business rules loaded. Use load_business_rules() first."
    
    rules_config = st.session_state.rules_config
    rule = next((r for r in rules_config.rules if r.name == rule_name), None)
    if not rule:
        return f"Rule '{rule_name}' not found. Available: {[r.name for r in rules_config.rules]}"
    
    try:
        executor = BusinessRulesExecutor(
            connector=get_pg(),
            marquez_url=os.environ.get("MARQUEZ_URL", "http://localhost:5000"),
            namespace=rules_config.namespace,
        )
        
        result = executor.execute_rule(rule, emit_lineage=capture_lineage)
        
        output = f"## Executed: {rule_name}\n\n"
        output += f"**Status:** {result['status']}\n"
        output += f"**Duration:** {result['duration_seconds']:.2f}s\n"
        output += f"**Rows returned:** {result['rows']}\n"
        
        if result.get('error'):
            output += f"**Error:** {result['error']}\n"
        
        if capture_lineage:
            output += f"**Lineage captured:** Run ID {result['run_id']}\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR executing rule: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def execute_all_business_rules(capture_lineage: bool = True) -> str:
    """Execute all loaded business rules and capture lineage."""
    if "rules_config" not in st.session_state or not st.session_state.rules_config.rules:
        return "No business rules loaded. Use load_business_rules() first."
    
    try:
        rules_config = st.session_state.rules_config
        executor = BusinessRulesExecutor(
            connector=get_pg(),
            marquez_url=os.environ.get("MARQUEZ_URL", "http://localhost:5000"),
            namespace=rules_config.namespace,
        )
        
        results = executor.execute_all_rules(rules_config, emit_lineage=capture_lineage)
        
        output = f"## Executed {len(results)} Business Rules\n\n"
        
        success = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - success
        output += f"**Results:** {success} succeeded, {failed} failed\n\n"
        
        for result in results:
            status_icon = "✓" if result['status'] == 'success' else "✗"
            output += f"{status_icon} **{result['rule_name']}**: "
            output += f"{result['rows']} rows, {result['duration_seconds']:.2f}s"
            if result.get('error'):
                output += f" - ERROR: {result['error']}"
            output += "\n"
        
        if capture_lineage:
            output += f"\n**Lineage captured in Marquez** (namespace: {rules_config.namespace})"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def import_lineage_to_graph() -> str:
    """Import lineage from Marquez into the Neo4j graph."""
    try:
        rules_config = st.session_state.get("rules_config")
        namespace = rules_config.namespace if rules_config else "default"
        
        stats = import_lineage_to_neo4j(
            marquez_url=os.environ.get("MARQUEZ_URL", "http://localhost:5000"),
            neo4j_client=get_neo4j(),
            namespace=namespace,
        )
        
        output = "## Lineage Imported to Neo4j\n\n"
        output += f"- Jobs imported: {stats.get('jobs', 0)}\n"
        output += f"- Datasets imported: {stats.get('datasets', 0)}\n"
        output += f"- READS relationships: {stats.get('reads', 0)}\n"
        output += f"- WRITES relationships: {stats.get('writes', 0)}\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR importing lineage: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def analyze_data_flow(table_name: str) -> str:
    """Analyze all data flow for a table - FKs and lineage."""
    try:
        neo4j = get_neo4j()
        
        output = f"## Data Flow Analysis: {table_name}\n\n"
        
        # FK relationships - outgoing
        fk_out = neo4j.run_query("""
            MATCH (t:Table {name: $name})<-[:BELONGS_TO]-(c:Column)-[fk:FK_TO]->(tc:Column)-[:BELONGS_TO]->(tt:Table)
            RETURN c.name as column, tt.name as references_table, tc.name as references_column, fk.score as score
        """, {"name": table_name})
        
        if fk_out:
            output += "### Foreign Keys (FK →)\n"
            for row in fk_out:
                output += f"  {row['column']} → {row['references_table']}.{row['references_column']}"
                if row.get('score'):
                    output += f" (score: {row['score']:.2f})"
                output += "\n"
            output += "\n"
        
        # FK relationships - incoming
        fk_in = neo4j.run_query("""
            MATCH (st:Table)<-[:BELONGS_TO]-(sc:Column)-[fk:FK_TO]->(tc:Column)-[:BELONGS_TO]->(t:Table {name: $name})
            RETURN st.name as source_table, sc.name as source_column, tc.name as column, fk.score as score
        """, {"name": table_name})
        
        if fk_in:
            output += "### Referenced By (FK ←)\n"
            for row in fk_in:
                output += f"  {row['source_table']}.{row['source_column']} → {row['column']}"
                if row.get('score'):
                    output += f" (score: {row['score']:.2f})"
                output += "\n"
            output += "\n"
        
        # Jobs that read this table
        readers = neo4j.run_query("""
            MATCH (j:Job)-[:READS]->(d:Dataset {name: $name})
            RETURN j.name as job_name, j.description as description
        """, {"name": table_name})
        
        if readers:
            output += "### Jobs Reading This Table\n"
            for row in readers:
                output += f"  ⚙️ {row['job_name']}"
                if row.get('description'):
                    output += f" - {row['description']}"
                output += "\n"
            output += "\n"
        
        # Jobs that write this table
        writers = neo4j.run_query("""
            MATCH (j:Job)-[:WRITES]->(d:Dataset {name: $name})
            RETURN j.name as job_name, j.description as description
        """, {"name": table_name})
        
        if writers:
            output += "### Jobs Writing This Table\n"
            for row in writers:
                output += f"  ⚙️ {row['job_name']}"
                if row.get('description'):
                    output += f" - {row['description']}"
                output += "\n"
            output += "\n"
        
        if not (fk_out or fk_in or readers or writers):
            output += "No relationships found. Run FK discovery and/or execute business rules first."
        
        return output
    except Exception as e:
        return f"ERROR analyzing data flow: {type(e).__name__}: {e}"


@tool
def find_impact_analysis(table_name: str) -> str:
    """Find all downstream impacts if a table changes."""
    try:
        neo4j = get_neo4j()
        
        output = f"## Impact Analysis: What breaks if '{table_name}' changes?\n\n"
        
        # Tables that depend on this via FK
        dependent_tables = neo4j.run_query("""
            MATCH (t:Table {name: $name})<-[:BELONGS_TO]-(c:Column)<-[:FK_TO]-(fc:Column)-[:BELONGS_TO]->(ft:Table)
            RETURN DISTINCT ft.name as table_name
        """, {"name": table_name})
        
        if dependent_tables:
            output += "### Dependent Tables (via FK)\n"
            for row in dependent_tables:
                output += f"  📊 {row['table_name']}\n"
            output += "\n"
        
        # Jobs that read this table
        dependent_jobs = neo4j.run_query("""
            MATCH (j:Job)-[:READS]->(d:Dataset {name: $name})
            RETURN j.name as job_name
        """, {"name": table_name})
        
        if dependent_jobs:
            output += "### Jobs That Read This Table\n"
            for row in dependent_jobs:
                output += f"  ⚙️ {row['job_name']}\n"
            output += "\n"
        
        # Downstream datasets (via jobs)
        downstream = neo4j.run_query("""
            MATCH (d1:Dataset {name: $name})<-[:READS]-(j:Job)-[:WRITES]->(d2:Dataset)
            RETURN DISTINCT j.name as job_name, d2.name as output_dataset
        """, {"name": table_name})
        
        if downstream:
            output += "### Downstream Datasets (via Jobs)\n"
            for row in downstream:
                output += f"  {row['job_name']} → {row['output_dataset']}\n"
            output += "\n"
        
        total = len(dependent_tables or []) + len(dependent_jobs or []) + len(downstream or [])
        output += f"**Total potential impacts: {total}**"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# RDF Tools

@tool
def test_rdf_connection() -> str:
    """Test connection to the RDF triple store (Apache Jena Fuseki)."""
    try:
        fuseki = get_fuseki()
        result = fuseki.test_connection()
        
        if result["success"]:
            count = fuseki.get_triple_count()
            return f"✓ Connected to Fuseki RDF store\n  Dataset: {fuseki.config.dataset}\n  Triples: {count}"
        else:
            return f"✗ Connection failed: {result.get('error')}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sync_graph_to_rdf() -> str:
    """Synchronize the entire Neo4j graph to the RDF triple store."""
    try:
        fuseki = get_fuseki()
        neo4j = get_neo4j()
        
        fuseki.ensure_dataset_exists()
        
        stats = sync_neo4j_to_rdf(neo4j, fuseki)
        
        if "error" in stats:
            return f"ERROR: {stats['error']}"
        
        output = "## RDF Sync Complete\n\n"
        output += f"- Tables synced: {stats.get('tables', 0)}\n"
        output += f"- Columns synced: {stats.get('columns', 0)}\n"
        output += f"- Foreign keys synced: {stats.get('fks', 0)}\n"
        output += f"- Jobs synced: {stats.get('jobs', 0)}\n"
        output += f"- Datasets synced: {stats.get('datasets', 0)}\n"
        output += f"- **Total triples: {stats.get('total_triples', 0)}**\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR syncing to RDF: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def run_sparql(query: str) -> str:
    """Run a custom SPARQL query on the RDF store."""
    try:
        sparql = get_sparql()
        results = sparql.custom_query(query)
        
        if not results:
            return "Query executed. No results returned."
        
        output = f"Results ({len(results)} rows):\n"
        for row in results[:50]:
            output += f"  {row}\n"
        if len(results) > 50:
            output += f"  ... and {len(results) - 50} more rows"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# LTN Tools (if available)

@tool
def learn_rules_with_ltn() -> str:
    """Learn logical rules from the graph using Logic Tensor Networks."""
    if not LTN_AVAILABLE:
        return "LTN not available. Install with: pip install ltn tensorflow"
    
    try:
        learner = get_rule_learner()
        if learner is None:
            return "LTN not available."
        
        learned_rules = learner.learn_rules()
        
        if not learned_rules:
            return "No rules learned. Make sure the graph has sufficient data."
        
        output = f"## Learned {len(learned_rules)} Rules with LTN\n\n"
        
        by_type = {}
        for rule in learned_rules:
            rtype = rule.rule_type
            if rtype not in by_type:
                by_type[rtype] = []
            by_type[rtype].append(rule)
        
        for rtype, rules in by_type.items():
            output += f"### {rtype.title()} Rules:\n"
            for rule in rules:
                output += f"- **{rule.name}**: `{rule.formula}`\n"
                output += f"  Confidence: {rule.confidence:.2f}, Support: {rule.support}\n"
            output += "\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR learning rules: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def generate_business_rules_from_ltn() -> str:
    """Generate executable business rules from learned LTN patterns."""
    if not LTN_AVAILABLE:
        return "LTN not available. Install with: pip install ltn tensorflow"
    
    try:
        learner = get_rule_learner()
        generator = get_rule_generator()
        
        if learner is None or generator is None:
            return "LTN not available."
        
        learned_rules = learner.get_learned_rules()
        
        if not learned_rules:
            learned_rules = learner.learn_rules()
        
        if not learned_rules:
            return "No learned rules available. Run learn_rules_with_ltn first."
        
        generated_rules = generator.generate_from_learned_rules(learned_rules)
        
        if not generated_rules:
            return "No business rules generated."
        
        output = f"## Generated {len(generated_rules)} Business Rules\n\n"
        
        by_type = {}
        for rule in generated_rules:
            rtype = rule.rule_type
            if rtype not in by_type:
                by_type[rtype] = []
            by_type[rtype].append(rule)
        
        for rtype, rules in by_type.items():
            output += f"### {rtype.title()} Rules ({len(rules)}):\n"
            for rule in rules[:5]:
                output += f"- **{rule.name}**\n"
                output += f"  {rule.description}\n"
            if len(rules) > 5:
                output += f"  ... and {len(rules) - 5} more\n"
            output += "\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR generating rules: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are GraphWeaver Agent, a powerful assistant for discovering database relationships, 
building knowledge graphs, and analyzing data lineage.

You have access to tools for:
- Database exploration and FK discovery
- Neo4j graph building and analysis
- Semantic search with embeddings
- Business rules execution with lineage capture
- RDF/SPARQL querying
- LTN rule learning

Be helpful, thorough, and explain your reasoning. When asked to discover FKs or analyze data, 
use the appropriate tools and provide clear insights.

Common workflows:
1. FK Discovery: test_database_connection → list_database_tables → run_fk_discovery
2. Embeddings: generate_text_embeddings → create_vector_indexes → semantic_search_*
3. Business Rules: load_business_rules_from_file → execute_all_business_rules → import_lineage_to_graph
4. Analysis: analyze_data_flow, find_impact_analysis, analyze_graph_centrality
"""


# =============================================================================
# Agent Creation
# =============================================================================

@st.cache_resource
def create_agent():
    """Create the LangGraph agent with Claude."""
    
    api_key = st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        return None
    
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.1,
        max_tokens=4096,
        api_key=api_key,
    )
    
    tools = [
        # Database
        configure_database,
        test_database_connection,
        list_database_tables,
        get_table_schema,
        get_column_stats,
        
        # FK Discovery
        run_fk_discovery,
        analyze_potential_fk,
        validate_fk_with_data,
        
        # Neo4j Graph
        clear_neo4j_graph,
        add_fk_to_graph,
        get_graph_stats,
        analyze_graph_centrality,
        find_table_communities,
        predict_missing_fks,
        run_cypher,
        connect_datasets_to_tables,
        
        # Embeddings & Semantic Search
        generate_text_embeddings,
        generate_kg_embeddings,
        create_vector_indexes,
        semantic_search_tables,
        semantic_search_columns,
        find_similar_tables,
        find_similar_columns,
        predict_fks_from_embeddings,
        semantic_fk_discovery,
        
        # Business Rules & Lineage
        show_sample_business_rules,
        load_business_rules,
        load_business_rules_from_file,
        list_business_rules,
        execute_business_rule,
        execute_all_business_rules,
        import_lineage_to_graph,
        analyze_data_flow,
        find_impact_analysis,
        
        # RDF Tools
        test_rdf_connection,
        sync_graph_to_rdf,
        run_sparql,
        
        # LTN Tools
        learn_rules_with_ltn,
        generate_business_rules_from_ltn,
        
        # Dynamic Tools
        check_tool_exists,
        list_available_tools,
        create_dynamic_tool,
        run_dynamic_tool,
        get_tool_source,
        update_dynamic_tool,
        delete_dynamic_tool,
    ]
    
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    
    return agent


def extract_response(result) -> str:
    """Extract text response from LangGraph result."""
    if not isinstance(result, dict):
        return str(result)
    
    messages = result.get("messages", [])
    
    if not messages:
        return str(result)
    
    last_msg = messages[-1]
    content = getattr(last_msg, 'content', None)
    
    if content is None:
        return str(last_msg)
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
            elif hasattr(block, 'text'):
                text_parts.append(block.text)
        return '\n'.join(text_parts) if text_parts else str(content)
    
    return str(content)


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
            st.session_state.chat_history = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
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
            with st.spinner("Thinking..."):
                try:
                    agent = create_agent()
                    
                    if agent is None:
                        st.error("Failed to create agent. Please check your API key.")
                        st.stop()
                    
                    # Build messages for agent
                    messages = st.session_state.chat_history + [HumanMessage(content=prompt)]
                    
                    # Invoke agent
                    result = agent.invoke(
                        {"messages": messages},
                        config={"recursion_limit": 100}
                    )
                    
                    # Extract response
                    response = extract_response(result)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Update history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=response))
                    
                    # Keep history manageable
                    if len(st.session_state.chat_history) > 20:
                        st.session_state.chat_history = st.session_state.chat_history[-20:]
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error: {type(e).__name__}: {e}\n\n```\n{traceback.format_exc()}\n```"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})


if __name__ == "__main__":
    main()