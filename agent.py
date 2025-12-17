"""
GraphWeaver LangChain Agent - Claude-powered autonomous FK discovery.

This agent uses Claude to:
1. Reason about what to explore
2. Decide which tables/columns to analyze
3. Call tools to discover FK relationships
4. Build knowledge graph
5. Generate insights
6. Use text and KG embeddings for semantic search

The agent THINKS using Claude and makes decisions, not just runs a script.
"""
import os
import sys
from typing import Optional

from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mcp_servers"))

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
# Global Connections
# =============================================================================

_pg: Optional[PostgreSQLConnector] = None
_neo4j: Optional[Neo4jClient] = None
_pg_config: Optional[DataSourceConfig] = None
_text_embedder = None
_kg_embedder = None
_fuseki: Optional[FusekiClient] = None
_sparql: Optional[SPARQLQueryBuilder] = None
_rule_learner = None
_rule_generator = None


def get_pg() -> PostgreSQLConnector:
    global _pg, _pg_config
    if _pg is None:
        if _pg_config is None:
            _pg_config = DataSourceConfig(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                database=os.environ.get("POSTGRES_DB", "orders"),
                username=os.environ.get("POSTGRES_USER", "saphenia"),
                password=os.environ.get("POSTGRES_PASSWORD", "secret"),
            )
        _pg = PostgreSQLConnector(_pg_config)
    return _pg


def get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient(Neo4jConfig(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
        ))
    return _neo4j


def get_text_embedder():
    global _text_embedder
    if _text_embedder is None:
        from graphweaver_agent.embeddings.text_embeddings import TextEmbedder
        _text_embedder = TextEmbedder()
    return _text_embedder


def get_kg_embedder():
    global _kg_embedder
    if _kg_embedder is None:
        from graphweaver_agent.embeddings.kg_embeddings import KGEmbedder
        _kg_embedder = KGEmbedder(get_neo4j())
    return _kg_embedder


def get_fuseki() -> FusekiClient:
    global _fuseki
    if _fuseki is None:
        _fuseki = FusekiClient()
    return _fuseki


def get_sparql() -> SPARQLQueryBuilder:
    global _sparql
    if _sparql is None:
        _sparql = SPARQLQueryBuilder(get_fuseki())
    return _sparql


def get_rule_learner():
    global _rule_learner
    if _rule_learner is None:
        if not LTN_AVAILABLE:
            return None
        config = RuleLearningConfig(
            embedding_dim=384,  # Match text embedding dim
            use_text_embeddings=True,
            use_kg_embeddings=True,
        )
        _rule_learner = LTNRuleLearner(get_neo4j(), config)
    return _rule_learner


def get_rule_generator():
    global _rule_generator
    if _rule_generator is None:
        if not LTN_AVAILABLE:
            return None
        _rule_generator = BusinessRuleGenerator(get_neo4j())
    return _rule_generator


# =============================================================================
# Tools for Claude to Call
# =============================================================================

@tool
def configure_database(host: str, port: int, database: str, username: str, password: str) -> str:
    """Configure which PostgreSQL database to connect to. Call this to point at a different database.
    
    Args:
        host: Database host (e.g., localhost, db.example.com)
        port: Port number (usually 5432)
        database: Database name
        username: Username
        password: Password
    """
    global _pg, _pg_config
    _pg_config = DataSourceConfig(
        host=host, port=port, database=database,
        username=username, password=password,
    )
    _pg = None  # Reset connection
    return f"✓ Configured database: {username}@{host}:{port}/{database}"


@tool
def test_database_connection() -> str:
    """Test connection to PostgreSQL database. Call this first."""
    result = get_pg().test_connection()
    if result["success"]:
        return f"✓ Connected to database '{result['database']}' as '{result['user']}'"
    return f"✗ Failed: {result['error']}"


@tool
def list_database_tables() -> str:
    """List all tables with row counts. Use to see what tables exist."""
    tables = get_pg().get_tables_with_info()
    output = "Tables:\n"
    for t in tables:
        output += f"  - {t['table_name']}: {t['column_count']} columns, ~{t['row_estimate']} rows\n"
    return output


@tool
def run_fk_discovery(min_match_rate: float = 0.95, min_score: float = 0.5) -> str:
    """
    Run complete 5-stage FK discovery pipeline on the database.
    
    This is the main discovery tool that:
    1. Stage 1 (Statistical): Filters by type compatibility, cardinality, uniqueness
    2. Stage 2 (Mathematical): Scores using geometric mean of features
    3. Stage 3 (Sampling): Validates with actual data (checks referential integrity)
    4. Stage 4 (Graph): Removes cycles, determines cardinality (1:1, 1:N)
    5. Stage 5 (Semantic): Filters out value columns, validates name patterns
    
    Args:
        min_match_rate: Minimum data match rate to confirm FK (default 0.95 = 95%)
        min_score: Minimum score threshold (default 0.5)
    
    Returns detailed results including all scores and statistics.
    """
    try:
        global _pg_config
        if _pg_config is None:
            _pg_config = DataSourceConfig(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                database=os.environ.get("POSTGRES_DB", "orders"),
                username=os.environ.get("POSTGRES_USER", "saphenia"),
                password=os.environ.get("POSTGRES_PASSWORD", "secret"),
            )
        
        result = run_discovery(
            host=_pg_config.host,
            port=_pg_config.port,
            database=_pg_config.database,
            username=_pg_config.username,
            password=_pg_config.password,
            schema=_pg_config.schema_name,
            min_match_rate=min_match_rate,
            min_score=min_score,
        )
        
        # Format output
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
    """Get columns and primary keys for a table.
    
    Args:
        table_name: Name of table to inspect
    """
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
    """Get statistics for a column - uniqueness, nulls, samples.
    
    Args:
        table_name: Table name
        column_name: Column name
    """
    stats = get_pg().get_column_statistics(table_name, column_name)
    return (f"{table_name}.{column_name}:\n"
            f"  Distinct: {stats.distinct_count}/{stats.total_count} ({stats.uniqueness_ratio:.1%})\n"
            f"  Nulls: {stats.null_count} ({stats.null_ratio:.1%})\n"
            f"  Samples: {stats.sample_values[:5]}")


@tool
def analyze_potential_fk(source_table: str, source_column: str, 
                         target_table: str, target_column: str) -> str:
    """Analyze if a column pair could be a FK. Checks types, names, cardinality.
    
    Args:
        source_table: Table with potential FK
        source_column: Column that might be FK
        target_table: Table being referenced
        target_column: Column being referenced (usually PK)
    """
    pg = get_pg()
    source_stats = pg.get_column_statistics(source_table, source_column)
    target_stats = pg.get_column_statistics(target_table, target_column)
    source_meta = pg.get_table_metadata(source_table)
    target_meta = pg.get_table_metadata(target_table)
    
    source_col = next((c for c in source_meta.columns if c.column_name == source_column), None)
    target_col = next((c for c in target_meta.columns if c.column_name == target_column), None)
    
    if not source_col or not target_col:
        return "Error: Column not found"
    
    # Basic analysis
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
    """Validate FK by checking if values actually exist. The definitive test.
    
    Args:
        source_table: Table with FK
        source_column: FK column
        target_table: Referenced table  
        target_column: Referenced column
    """
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
    """Clear all data from Neo4j graph. Use before rebuilding."""
    try:
        GraphBuilder(get_neo4j()).clear_graph()
        return "✓ Graph cleared"
    except Exception as e:
        return f"ERROR clearing graph: {type(e).__name__}: {e}"


@tool
def add_fk_to_graph(source_table: str, source_column: str,
                    target_table: str, target_column: str,
                    score: float, cardinality: str = "1:N") -> str:
    """Add a confirmed FK relationship to the Neo4j graph.
    
    Args:
        source_table: Table with FK
        source_column: FK column
        target_table: Referenced table
        target_column: Referenced column
        score: Confidence score 0-1
        cardinality: 1:1, 1:N, or N:M
    """
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
    """Run a custom Cypher query on the Neo4j graph database.
    
    Use this to execute any Cypher query for reading or writing data.
    Examples:
    - MATCH (n) RETURN n LIMIT 10
    - MATCH (d:Dataset) MATCH (t:Table) WHERE d.name = t.name MERGE (d)-[:REPRESENTS]->(t) RETURN d.name, t.name
    - CREATE (n:Label {property: 'value'})
    
    Args:
        query: The Cypher query to execute
        
    Returns:
        Query results as formatted string
    """
    try:
        neo4j = get_neo4j()
    except Exception as e:
        return f"Error: Not connected to Neo4j: {e}"
    
    try:
        # Try as read query first
        results = neo4j.run_query(query)
        if results is None:
            results = []
        
        if not results:
            return "Query executed successfully. No results returned (0 rows)."
        
        # Format results
        output = f"Results ({len(results)} rows):\n"
        for i, row in enumerate(results[:50]):  # Limit to 50 rows
            output += f"  {dict(row)}\n"
        if len(results) > 50:
            output += f"  ... and {len(results) - 50} more rows"
        return output
    except Exception as e:
        # Try as write query
        try:
            neo4j.run_write(query)
            return "Write query executed successfully."
        except Exception as e2:
            return f"Error executing query: {e2}"


@tool
def connect_datasets_to_tables() -> str:
    """Connect Dataset nodes to their matching Table nodes in the graph.
    
    This creates REPRESENTS relationships between Datasets (from lineage)
    and Tables (from FK discovery) that have the same name, unifying the graph.
    """
    try:
        neo4j = get_neo4j()
        
        # Run the merge query
        result = neo4j.run_query("""
            MATCH (d:Dataset)
            MATCH (t:Table)
            WHERE d.name = t.name
            MERGE (d)-[:REPRESENTS]->(t)
            RETURN d.name as dataset, t.name as table
        """)
        
        if not result:
            return "No matching Dataset-Table pairs found. Make sure you have both FK discovery results and lineage data in the graph."
        
        output = f"## Connected {len(result)} Datasets to Tables\n\n"
        output += "Created REPRESENTS relationships:\n"
        for row in result:
            output += f"  Dataset '{row['dataset']}' → Table '{row['table']}'\n"
        output += "\nThe FK graph and lineage graph are now connected!"
        
        return output
    except Exception as e:
        return f"ERROR connecting datasets to tables: {type(e).__name__}: {e}"


# =============================================================================
# Embedding Tools - FIXED TO ACTUALLY WORK
# =============================================================================

@tool
def generate_text_embeddings() -> str:
    """Generate text embeddings for all tables, columns, jobs, and datasets in the graph.
    
    This uses the all-MiniLM-L6-v2 model (384 dimensions) to create semantic
    embeddings based on names, types, and metadata. Embeddings are stored
    directly on Neo4j nodes.
    
    Use this to enable semantic search capabilities.
    """
    try:
        # Import here to ensure it's loaded
        from graphweaver_agent.embeddings.text_embeddings import embed_all_metadata, TextEmbedder
        
        print("[generate_text_embeddings] Starting...")
        
        # Get connections
        neo4j = get_neo4j()
        pg = get_pg()
        embedder = get_text_embedder()
        
        print("[generate_text_embeddings] Calling embed_all_metadata...")
        
        # Actually call the function
        stats = embed_all_metadata(
            neo4j_client=neo4j,
            pg_connector=pg,
            embedder=embedder,
        )
        
        print(f"[generate_text_embeddings] Done: {stats}")
        
        output = "## Text Embeddings Generated\n\n"
        output += f"- Tables embedded: {stats['tables']}\n"
        output += f"- Columns embedded: {stats['columns']}\n"
        output += f"- Jobs embedded: {stats['jobs']}\n"
        output += f"- Datasets embedded: {stats['datasets']}\n"
        output += "\nText embeddings are now stored on Neo4j nodes."
        output += "\nYou can now use semantic_search_tables and semantic_search_columns."
        
        return output
    except Exception as e:
        import traceback
        error_msg = f"ERROR generating text embeddings: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


@tool
def generate_kg_embeddings() -> str:
    """Generate knowledge graph embeddings using Neo4j GDS FastRP algorithm.
    
    This creates structural embeddings (128 dimensions) that capture the
    graph topology - nodes with similar neighborhoods get similar embeddings.
    Useful for link prediction and finding structurally similar entities.
    
    Requires Neo4j GDS plugin (included in docker-compose).
    """
    try:
        # Import here to ensure it's loaded
        from graphweaver_agent.embeddings.kg_embeddings import generate_all_kg_embeddings
        
        print("[generate_kg_embeddings] Starting...")
        
        neo4j = get_neo4j()
        
        print("[generate_kg_embeddings] Calling generate_all_kg_embeddings...")
        
        # Actually call the function
        result = generate_all_kg_embeddings(neo4j)
        
        print(f"[generate_kg_embeddings] Done: {result}")
        
        if "error" in result:
            return f"ERROR: {result['error']}"
        
        output = "## KG Embeddings Generated\n\n"
        output += f"- Graph projected: {result['projection']['nodeCount']} nodes, {result['projection']['relationshipCount']} relationships\n"
        output += f"- Embeddings written: {result['embeddings']['properties_written']} properties\n"
        output += "\nKG embeddings are now stored on Neo4j nodes."
        output += "\nYou can now use find_similar_tables and predict_fks_from_embeddings."
        
        return output
    except Exception as e:
        import traceback
        error_msg = f"ERROR generating KG embeddings: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


@tool
def create_vector_indexes() -> str:
    """Create Neo4j vector indexes for fast similarity search.
    
    Creates HNSW indexes for:
    - Text embeddings (384 dimensions)
    - KG embeddings (128 dimensions)
    
    Run this after generating embeddings to enable fast vector search.
    """
    try:
        from graphweaver_agent.embeddings.vector_indexes import VectorIndexManager
        
        print("[create_vector_indexes] Starting...")
        
        manager = VectorIndexManager(get_neo4j())
        
        # Check if vector indexes are supported
        if not manager.check_vector_support():
            return "WARNING: Your Neo4j version may not support vector indexes. Similarity search will still work but may be slower."
        
        result = manager.create_all_indexes()
        
        print(f"[create_vector_indexes] Done: {result}")
        
        output = "## Vector Indexes Created\n\n"
        output += "### Text Embedding Indexes:\n"
        for name, success in result["text_indexes"].items():
            status = "✓" if success else "✗"
            output += f"  {status} {name}\n"
        
        output += "\n### KG Embedding Indexes:\n"
        for name, success in result["kg_indexes"].items():
            status = "✓" if success else "✗"
            output += f"  {status} {name}\n"
        
        return output
    except Exception as e:
        import traceback
        error_msg = f"ERROR creating indexes: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


@tool
def semantic_search_tables(query: str, top_k: int = 5) -> str:
    """Search for tables using natural language.
    
    Examples:
    - "customer data" -> finds customers, users, accounts tables
    - "financial transactions" -> finds orders, payments, invoices
    - "product inventory" -> finds products, stock, inventory tables
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
    """
    try:
        print(f"[semantic_search_tables] Query: {query}")
        
        embedder = get_text_embedder()
        neo4j = get_neo4j()
        
        # Embed the query
        query_emb = embedder.embed_text(query)
        
        print(f"[semantic_search_tables] Query embedded, searching...")
        
        # Search using cosine similarity
        result = neo4j.run_query("""
            MATCH (t:Table)
            WHERE t.text_embedding IS NOT NULL
            WITH t, gds.similarity.cosine(t.text_embedding, $embedding) AS score
            WHERE score > 0.3
            RETURN t.name AS table_name, score
            ORDER BY score DESC
            LIMIT $top_k
        """, {"embedding": query_emb.embedding, "top_k": top_k})
        
        if not result:
            return f"No tables found matching '{query}'. Make sure text embeddings are generated (run generate_text_embeddings first)."
        
        output = f"## Tables matching '{query}':\n\n"
        for r in result:
            output += f"  - **{r['table_name']}** (similarity: {r['score']:.2f})\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR in semantic search: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def semantic_search_columns(query: str, top_k: int = 10) -> str:
    """Search for columns using natural language.
    
    Examples:
    - "customer identifier" -> finds customer_id, buyer_id, client_code
    - "monetary amount" -> finds price, cost, total_amount
    - "creation timestamp" -> finds created_at, timestamp, date_created
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
    """
    try:
        print(f"[semantic_search_columns] Query: {query}")
        
        embedder = get_text_embedder()
        neo4j = get_neo4j()
        
        # Embed the query
        query_emb = embedder.embed_text(query)
        
        print(f"[semantic_search_columns] Query embedded, searching...")
        
        # Search using cosine similarity
        result = neo4j.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE c.text_embedding IS NOT NULL
            WITH t, c, gds.similarity.cosine(c.text_embedding, $embedding) AS score
            WHERE score > 0.3
            RETURN t.name AS table_name, c.name AS column_name, score
            ORDER BY score DESC
            LIMIT $top_k
        """, {"embedding": query_emb.embedding, "top_k": top_k})
        
        if not result:
            return f"No columns found matching '{query}'. Make sure text embeddings are generated (run generate_text_embeddings first)."
        
        output = f"## Columns matching '{query}':\n\n"
        for r in result:
            output += f"  - **{r['table_name']}.{r['column_name']}** (similarity: {r['score']:.2f})\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR in semantic search: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def find_similar_tables(table_name: str, top_k: int = 5) -> str:
    """Find tables similar to a given table using both text and KG embeddings.
    
    Uses combined similarity from:
    - Text embeddings (semantic similarity of names/metadata)
    - KG embeddings (structural similarity in the graph)
    
    Args:
        table_name: Name of the source table
        top_k: Number of similar tables to find
    """
    try:
        print(f"[find_similar_tables] Finding tables similar to: {table_name}")
        
        neo4j = get_neo4j()
        
        # Try combined similarity first
        result = neo4j.run_query("""
            MATCH (source:Table {name: $name})
            MATCH (other:Table)
            WHERE other <> source 
              AND other.text_embedding IS NOT NULL
            WITH source, other,
                 gds.similarity.cosine(source.text_embedding, other.text_embedding) AS text_sim,
                 CASE 
                   WHEN source.kg_embedding IS NOT NULL AND other.kg_embedding IS NOT NULL
                   THEN gds.similarity.cosine(source.kg_embedding, other.kg_embedding)
                   ELSE null
                 END AS kg_sim
            WITH other.name AS table_name,
                 text_sim,
                 kg_sim,
                 CASE 
                   WHEN kg_sim IS NOT NULL THEN (text_sim + kg_sim) / 2
                   ELSE text_sim
                 END AS combined_score
            RETURN table_name, text_sim, kg_sim, combined_score
            ORDER BY combined_score DESC
            LIMIT $top_k
        """, {"name": table_name, "top_k": top_k})
        
        if not result:
            return f"No similar tables found for '{table_name}'. Run generate_text_embeddings first."
        
        output = f"## Tables similar to '{table_name}':\n\n"
        for r in result:
            output += f"  - **{r['table_name']}**\n"
            if r.get('text_sim'):
                output += f"    Text similarity: {r['text_sim']:.2f}\n"
            if r.get('kg_sim'):
                output += f"    Graph similarity: {r['kg_sim']:.2f}\n"
            output += f"    Combined score: {r['combined_score']:.2f}\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR finding similar tables: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def find_similar_columns(table_name: str, column_name: str, top_k: int = 10) -> str:
    """Find columns similar to a given column using text embeddings.
    
    Useful for finding:
    - Alternative names for the same concept
    - Potential FK targets with non-standard naming
    - Related columns across tables
    
    Args:
        table_name: Table containing the source column
        column_name: Name of the source column
        top_k: Number of similar columns to find
    """
    try:
        print(f"[find_similar_columns] Finding columns similar to: {table_name}.{column_name}")
        
        neo4j = get_neo4j()
        
        result = neo4j.run_query("""
            MATCH (source:Column {name: $column_name})-[:BELONGS_TO]->(st:Table {name: $table_name})
            MATCH (other:Column)-[:BELONGS_TO]->(t:Table)
            WHERE other <> source 
              AND other.text_embedding IS NOT NULL
              AND t.name <> $table_name
            WITH t.name AS table_name,
                 other.name AS column_name,
                 gds.similarity.cosine(source.text_embedding, other.text_embedding) AS score
            RETURN table_name, column_name, score
            ORDER BY score DESC
            LIMIT $top_k
        """, {"table_name": table_name, "column_name": column_name, "top_k": top_k})
        
        if not result:
            return f"No similar columns found for '{table_name}.{column_name}'. Run generate_text_embeddings first."
        
        output = f"## Columns similar to '{table_name}.{column_name}':\n\n"
        for r in result:
            output += f"  - **{r['table_name']}.{r['column_name']}** (similarity: {r['score']:.2f})\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR finding similar columns: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def predict_fks_from_embeddings(threshold: float = 0.7, top_k: int = 20) -> str:
    """Predict potential FK relationships using KG embeddings.
    
    Finds column pairs that are structurally similar in the graph
    but don't have FK relationships yet. This can discover FKs that
    statistical methods miss.
    
    Args:
        threshold: Minimum similarity threshold (0-1)
        top_k: Maximum predictions to return
    """
    try:
        print(f"[predict_fks_from_embeddings] Threshold: {threshold}, top_k: {top_k}")
        
        kg_embedder = get_kg_embedder()
        predictions = kg_embedder.predict_missing_links(
            threshold=threshold,
            top_k=top_k,
        )
        
        if not predictions:
            return "No FK predictions found. The graph may be complete or try lowering the threshold."
        
        output = f"## Predicted FKs from Graph Structure:\n\n"
        for p in predictions:
            output += f"  - **{p['source_table']}.{p['source_column']}** → **{p['target_table']}.{p['target_column']}**\n"
            output += f"    Similarity: {p['similarity']:.2f}\n"
        
        output += "\nValidate these with validate_fk_with_data before adding to the graph."
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR predicting FKs: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def semantic_fk_discovery(source_table: str = None, min_score: float = 0.6) -> str:
    """Discover FK relationships using semantic similarity of column names.
    
    Finds FKs even when names don't match patterns, like:
    - customer_id → buyer_identifier
    - prod_code → item_sku
    
    Args:
        source_table: Limit search to this table (optional)
        min_score: Minimum combined score threshold
    """
    try:
        print(f"[semantic_fk_discovery] source_table: {source_table}, min_score: {min_score}")
        
        from graphweaver_agent.embeddings.semantic_fk import SemanticFKDiscovery
        
        discovery = SemanticFKDiscovery(
            neo4j_client=get_neo4j(),
            text_embedder=get_text_embedder(),
            min_combined_score=min_score,
        )
        
        candidates = discovery.find_semantic_fk_candidates(
            source_table=source_table,
            top_k=30,
        )
        
        if not candidates:
            return "No semantic FK candidates found. Run generate_text_embeddings first."
        
        output = "## Semantic FK Candidates:\n\n"
        for c in candidates:
            output += f"**{c.source_table}.{c.source_column}** → **{c.target_table}.{c.target_column}**\n"
            output += f"  Semantic similarity: {c.semantic_similarity:.2f}\n"
            if c.kg_similarity:
                output += f"  Graph similarity: {c.kg_similarity:.2f}\n"
            output += f"  Combined score: {c.combined_score:.2f}\n"
            output += f"  Recommendation: {c.recommendation}\n\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR in semantic FK discovery: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# =============================================================================
# Business Rules & Lineage Tools
# =============================================================================

_rules_config: Optional[BusinessRulesConfig] = None
_marquez_client: Optional[MarquezClient] = None


def get_marquez() -> MarquezClient:
    global _marquez_client
    if _marquez_client is None:
        _marquez_client = MarquezClient(
            url=os.environ.get("MARQUEZ_URL", "http://localhost:5000")
        )
    return _marquez_client


@tool
def show_sample_business_rules() -> str:
    """Show a sample business rules YAML file that you can customize."""
    return generate_sample_rules()


@tool
def load_business_rules(yaml_content: str) -> str:
    """Load business rules from YAML content.
    
    Args:
        yaml_content: YAML string defining business rules
    """
    global _rules_config
    try:
        import yaml
        data = yaml.safe_load(yaml_content)
        
        rules = []
        for rule_data in data.get('rules', []):
            rules.append(BusinessRule(**rule_data))
        
        _rules_config = BusinessRulesConfig(
            version=data.get('version', '1.0'),
            namespace=data.get('namespace', 'default'),
            rules=rules,
        )
        
        output = f"✓ Loaded {len(_rules_config.rules)} business rules:\n"
        for rule in _rules_config.rules:
            output += f"  - {rule.name}: {rule.description} [{rule.type.value}]\n"
            output += f"    Inputs: {', '.join(rule.inputs)}\n"
        return output
    except Exception as e:
        return f"ERROR loading rules: {type(e).__name__}: {e}"


@tool
def load_business_rules_from_file(file_path: str = "business_rules.yaml") -> str:
    """Load business rules from a YAML file on disk.
    
    Args:
        file_path: Path to the YAML file (default: business_rules.yaml)
    """
    global _rules_config
    try:
        import yaml
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        rules = []
        for rule_data in data.get('rules', []):
            rules.append(BusinessRule(**rule_data))
        
        _rules_config = BusinessRulesConfig(
            version=data.get('version', '1.0'),
            namespace=data.get('namespace', 'default'),
            rules=rules,
        )
        
        output = f"✓ Loaded {len(_rules_config.rules)} business rules from {file_path}:\n"
        for rule in _rules_config.rules:
            output += f"  - {rule.name}: {rule.description} [{rule.type.value}]\n"
        return output
    except FileNotFoundError:
        return f"ERROR: File '{file_path}' not found"
    except Exception as e:
        return f"ERROR loading rules: {type(e).__name__}: {e}"


@tool
def list_business_rules() -> str:
    """List all loaded business rules."""
    if _rules_config is None or not _rules_config.rules:
        return "No business rules loaded. Use load_business_rules() first."
    
    output = f"## Business Rules (namespace: {_rules_config.namespace})\n\n"
    for rule in _rules_config.rules:
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
    """Execute a single business rule and optionally capture lineage.
    
    Args:
        rule_name: Name of the rule to execute
        capture_lineage: Whether to send lineage to Marquez (default True)
    """
    if _rules_config is None:
        return "No business rules loaded. Use load_business_rules() first."
    
    rule = next((r for r in _rules_config.rules if r.name == rule_name), None)
    if not rule:
        return f"Rule '{rule_name}' not found. Available: {[r.name for r in _rules_config.rules]}"
    
    try:
        executor = BusinessRulesExecutor(
            connector=get_pg(),
            marquez_url=os.environ.get("MARQUEZ_URL", "http://localhost:5000"),
            namespace=_rules_config.namespace,
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
        
        if result.get('columns'):
            output += f"**Columns:** {', '.join(result['columns'])}\n"
        
        if result.get('metrics'):
            output += "\n### Metrics:\n"
            for col, metrics in result['metrics'].items():
                output += f"  {col}: sum={metrics['sum']:.2f}, avg={metrics['avg']:.2f}, "
                output += f"min={metrics['min']:.2f}, max={metrics['max']:.2f}\n"
        
        if result.get('data') and len(result['data']) > 0:
            output += f"\n### Sample Data (first {min(5, len(result['data']))} rows):\n"
            for row in result['data'][:5]:
                output += f"  {row}\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR executing rule: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def execute_all_business_rules(capture_lineage: bool = True) -> str:
    """Execute all loaded business rules and capture lineage.
    
    Args:
        capture_lineage: Whether to send lineage to Marquez
    """
    if _rules_config is None or not _rules_config.rules:
        return "No business rules loaded. Use load_business_rules() first."
    
    try:
        executor = BusinessRulesExecutor(
            connector=get_pg(),
            marquez_url=os.environ.get("MARQUEZ_URL", "http://localhost:5000"),
            namespace=_rules_config.namespace,
        )
        
        results = executor.execute_all_rules(_rules_config, emit_lineage=capture_lineage)
        
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
            output += f"\n**Lineage captured in Marquez** (namespace: {_rules_config.namespace})"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def get_marquez_lineage(dataset_name: str, depth: int = 3) -> str:
    """Get lineage graph for a dataset from Marquez.
    
    Args:
        dataset_name: Name of the dataset/table
        depth: How many levels of lineage to retrieve
    """
    try:
        marquez = get_marquez()
        namespace = _rules_config.namespace if _rules_config else "default"
        
        lineage = marquez.get_dataset_lineage(namespace, dataset_name, depth)
        
        if not lineage:
            return f"No lineage found for {dataset_name}"
        
        output = f"## Lineage for {dataset_name}\n\n"
        
        graph = lineage.get("graph", [])
        for node in graph:
            node_type = node.get("type", "")
            node_id = node.get("id", "")
            
            if node_type == "DATASET":
                output += f"📊 Dataset: {node.get('data', {}).get('name', node_id)}\n"
            elif node_type == "JOB":
                output += f"⚙️ Job: {node.get('data', {}).get('name', node_id)}\n"
            
            # Show edges
            in_edges = node.get("inEdges", [])
            out_edges = node.get("outEdges", [])
            if in_edges:
                output += f"   ← reads from: {len(in_edges)} sources\n"
            if out_edges:
                output += f"   → writes to: {len(out_edges)} targets\n"
        
        return output
    except Exception as e:
        return f"ERROR getting lineage: {type(e).__name__}: {e}"


@tool
def list_marquez_jobs() -> str:
    """List all jobs tracked in Marquez."""
    try:
        marquez = get_marquez()
        namespace = _rules_config.namespace if _rules_config else "default"
        
        jobs = marquez.get_jobs(namespace)
        
        if not jobs:
            return "No jobs found in Marquez."
        
        output = f"## Jobs in Marquez (namespace: {namespace})\n\n"
        for job in jobs:
            output += f"**{job.get('name')}**\n"
            output += f"  Inputs: {[i.get('name') for i in job.get('inputs', [])]}\n"
            output += f"  Outputs: {[o.get('name') for o in job.get('outputs', [])]}\n"
            if job.get('latestRun'):
                run = job['latestRun']
                output += f"  Last run: {run.get('state')} at {run.get('startedAt')}\n"
            output += "\n"
        
        return output
    except Exception as e:
        return f"ERROR listing jobs: {type(e).__name__}: {e}"


@tool
def import_lineage_to_graph() -> str:
    """Import lineage data from Marquez into Neo4j graph.
    
    This creates Job nodes and READS/WRITES relationships,
    linking them to existing Table nodes.
    """
    try:
        marquez = get_marquez()
        neo4j = get_neo4j()
        namespace = _rules_config.namespace if _rules_config else "default"
        
        stats = import_lineage_to_neo4j(marquez, neo4j, namespace)
        
        output = "## Imported Lineage to Neo4j\n\n"
        output += f"- Jobs created: {stats['jobs']}\n"
        output += f"- Datasets linked: {stats['datasets']}\n"
        output += f"- READS relationships: {stats['reads']}\n"
        output += f"- WRITES relationships: {stats['writes']}\n"
        output += "\nThe graph now contains both FK relationships AND data lineage!"
        
        return output
    except Exception as e:
        return f"ERROR importing lineage: {type(e).__name__}: {e}"


@tool
def analyze_data_flow(table_name: str) -> str:
    """Analyze complete data flow for a table - both FKs and lineage.
    
    Shows:
    - What tables this table references (FK relationships)
    - What tables reference this table (FK relationships)  
    - What jobs read from this table
    - What jobs write to this table
    
    Args:
        table_name: Name of the table to analyze
    """
    try:
        neo4j = get_neo4j()
        
        output = f"## Data Flow Analysis: {table_name}\n\n"
        
        # FK relationships - outgoing
        fk_out = neo4j.run_query("""
            MATCH (t:Table {name: $name})<-[:BELONGS_TO]-(c:Column)-[fk:FK_TO]->(tc:Column)-[:BELONGS_TO]->(tt:Table)
            RETURN c.name as column, tt.name as references_table, tc.name as references_column, fk.score as score
        """, {"name": table_name})
        
        if fk_out:
            output += "### References (FK →)\n"
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
    """Find all downstream impacts if a table changes.
    
    Shows what tables and jobs would be affected.
    
    Args:
        table_name: Table to analyze impact for
    """
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


# =============================================================================
# RDF Tools
# =============================================================================

@tool
def test_rdf_connection() -> str:
    """Test connection to the RDF triple store (Apache Jena Fuseki).
    
    Returns connection status and dataset info.
    """
    try:
        fuseki = get_fuseki()
        result = fuseki.test_connection()
        
        if result["success"]:
            # Get triple count
            count = fuseki.get_triple_count()
            return f"✓ Connected to Fuseki RDF store\n  Dataset: {fuseki.config.dataset}\n  Triples: {count}"
        else:
            return f"✗ Connection failed: {result.get('error')}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sync_graph_to_rdf() -> str:
    """Synchronize the entire Neo4j graph to the RDF triple store.
    
    This exports all:
    - Tables and columns
    - Foreign key relationships
    - Jobs and datasets (lineage)
    - Dataset-table links
    
    The RDF store uses standard ontologies (DCAT, PROV-O, Dublin Core)
    for interoperability with other data catalog systems.
    """
    try:
        print("[sync_graph_to_rdf] Starting sync...")
        
        fuseki = get_fuseki()
        neo4j = get_neo4j()
        
        # Ensure dataset exists
        fuseki.ensure_dataset_exists()
        
        # Run sync
        stats = sync_neo4j_to_rdf(neo4j, fuseki)
        
        print(f"[sync_graph_to_rdf] Done: {stats}")
        
        if "error" in stats:
            return f"ERROR: {stats['error']}"
        
        output = "## RDF Sync Complete\n\n"
        output += f"- Tables synced: {stats.get('tables', 0)}\n"
        output += f"- Columns synced: {stats.get('columns', 0)}\n"
        output += f"- Foreign keys synced: {stats.get('fks', 0)}\n"
        output += f"- Jobs synced: {stats.get('jobs', 0)}\n"
        output += f"- Datasets synced: {stats.get('datasets', 0)}\n"
        output += f"- Dataset-table links: {stats.get('links', 0)}\n"
        output += f"- **Total triples: {stats.get('total_triples', 0)}**\n"
        output += "\nYou can now query the RDF store with SPARQL or access it at http://localhost:3030"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR syncing to RDF: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def run_sparql(query: str) -> str:
    """Run a custom SPARQL query on the RDF store.
    
    The graph uses these prefixes:
    - gw: <http://graphweaver.io/ontology#> (GraphWeaver vocabulary)
    - gwdata: <http://graphweaver.io/data#> (instance data)
    - dcat: <http://www.w3.org/ns/dcat#> (Data Catalog)
    - prov: <http://www.w3.org/ns/prov#> (Provenance)
    - dct: <http://purl.org/dc/terms/> (Dublin Core)
    
    Example queries:
    - SELECT ?table ?label WHERE { ?table a gw:Table ; rdfs:label ?label }
    - SELECT ?col WHERE { ?col gw:belongsToTable ?table . ?table rdfs:label "orders" }
    
    Args:
        query: SPARQL SELECT query
    """
    try:
        print(f"[run_sparql] Executing query...")
        
        sparql = get_sparql()
        results = sparql.custom_query(query)
        
        if not results:
            return "Query executed. No results returned."
        
        output = f"Results ({len(results)} rows):\n"
        for i, row in enumerate(results[:50]):
            output += f"  {row}\n"
        if len(results) > 50:
            output += f"  ... and {len(results) - 50} more rows"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR executing SPARQL: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def sparql_list_tables() -> str:
    """List all tables in the RDF store with column counts."""
    try:
        sparql = get_sparql()
        results = sparql.list_tables()
        
        if not results:
            return "No tables found. Run sync_graph_to_rdf first."
        
        output = "## Tables in RDF Store\n\n"
        for r in results:
            label = r.get("label", "?")
            cols = r.get("columnCount", 0)
            rows = r.get("rowCount", "?")
            output += f"- **{label}**: {cols} columns, {rows} rows\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_get_foreign_keys(table_name: str = None) -> str:
    """Get foreign key relationships from RDF store.
    
    Args:
        table_name: Optional - filter by table name
    """
    try:
        sparql = get_sparql()
        results = sparql.get_foreign_keys(table_name)
        
        if not results:
            return f"No foreign keys found{' for ' + table_name if table_name else ''}."
        
        output = f"## Foreign Keys{' for ' + table_name if table_name else ''}\n\n"
        for r in results:
            src = f"{r.get('sourceTableLabel')}.{r.get('sourceColLabel')}"
            tgt = f"{r.get('targetTableLabel')}.{r.get('targetColLabel')}"
            score = r.get("score", "?")
            card = r.get("cardinality", "?")
            output += f"- **{src}** → **{tgt}** (score: {score}, cardinality: {card})\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_table_lineage(table_name: str) -> str:
    """Get lineage for a table from RDF store - what jobs read/write it.
    
    Args:
        table_name: Name of the table
    """
    try:
        sparql = get_sparql()
        results = sparql.get_table_lineage(table_name)
        
        if not results:
            return f"No lineage found for '{table_name}'."
        
        output = f"## Lineage for {table_name}\n\n"
        
        reads = [r for r in results if r.get("direction") == "reads"]
        writes = [r for r in results if r.get("direction") == "writes"]
        
        if reads:
            output += "### Jobs that READ this table:\n"
            for r in reads:
                output += f"  - {r.get('jobLabel')}\n"
        
        if writes:
            output += "\n### Jobs that WRITE to this table:\n"
            for r in writes:
                output += f"  - {r.get('jobLabel')}\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_downstream_impact(table_name: str) -> str:
    """Find downstream impact - what depends on this table (via RDF/SPARQL).
    
    Uses SPARQL to traverse both FK relationships and lineage.
    
    Args:
        table_name: Name of the table to analyze
    """
    try:
        sparql = get_sparql()
        results = sparql.get_downstream_impact(table_name)
        
        if not results:
            return f"No downstream dependencies found for '{table_name}'."
        
        output = f"## Downstream Impact: {table_name}\n\n"
        
        fk_deps = [r for r in results if r.get("relationshipType") == "FK_REFERENCE"]
        lineage_deps = [r for r in results if r.get("relationshipType") == "LINEAGE"]
        
        if fk_deps:
            output += "### Tables referencing via FK:\n"
            for r in fk_deps:
                output += f"  - {r.get('dependentTableLabel')}\n"
        
        if lineage_deps:
            output += "\n### Downstream via data lineage:\n"
            for r in lineage_deps:
                output += f"  - {r.get('dependentTableLabel')}\n"
        
        output += f"\n**Total dependencies: {len(results)}**"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_hub_tables(min_connections: int = 2) -> str:
    """Find hub tables with many connections (via RDF/SPARQL).
    
    Args:
        min_connections: Minimum connections to be considered a hub
    """
    try:
        sparql = get_sparql()
        results = sparql.get_hub_tables(min_connections)
        
        if not results:
            return f"No hub tables found with >= {min_connections} connections."
        
        output = f"## Hub Tables (>= {min_connections} connections)\n\n"
        for r in results:
            label = r.get("label")
            in_fks = r.get("incomingFKs", 0)
            out_fks = r.get("outgoingFKs", 0)
            reads = r.get("readByJobs", 0)
            total = r.get("totalConnections", 0)
            output += f"- **{label}**: {total} total (in:{in_fks}, out:{out_fks}, reads:{reads})\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_orphan_tables() -> str:
    """Find tables with no FK relationships (via RDF/SPARQL)."""
    try:
        sparql = get_sparql()
        results = sparql.find_orphan_tables()
        
        if not results:
            return "No orphan tables found - all tables have relationships."
        
        output = "## Orphan Tables (no FK relationships)\n\n"
        for r in results:
            output += f"- {r.get('label')}\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def sparql_search(search_term: str) -> str:
    """Search the RDF graph by label.
    
    Searches across tables, columns, jobs, and datasets.
    
    Args:
        search_term: Text to search for in labels
    """
    try:
        sparql = get_sparql()
        results = sparql.search_by_label(search_term)
        
        if not results:
            return f"No results found for '{search_term}'."
        
        output = f"## Search results for '{search_term}'\n\n"
        
        # Group by type
        by_type = {}
        for r in results:
            rtype = r.get("type", "").split("#")[-1]
            if rtype not in by_type:
                by_type[rtype] = []
            by_type[rtype].append(r.get("label"))
        
        for rtype, labels in by_type.items():
            output += f"### {rtype}s:\n"
            for label in labels:
                output += f"  - {label}\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def get_rdf_statistics() -> str:
    """Get statistics from the RDF store."""
    try:
        fuseki = get_fuseki()
        sparql = get_sparql()
        
        # Get triple count
        triple_count = fuseki.get_triple_count("http://graphweaver.io/graph/main")
        
        # Get entity counts
        stats = sparql.get_statistics()
        
        output = "## RDF Store Statistics\n\n"
        output += f"- Total triples: {triple_count}\n"
        output += f"- Tables: {stats.get('tables', 0)}\n"
        output += f"- Columns: {stats.get('columns', 0)}\n"
        output += f"- Foreign Keys: {stats.get('foreignKeys', 0)}\n"
        output += f"- Jobs: {stats.get('jobs', 0)}\n"
        output += f"- Datasets: {stats.get('datasets', 0)}\n"
        output += f"\nFuseki UI: http://localhost:3030"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def export_rdf_turtle() -> str:
    """Export the graph ontology in Turtle format.
    
    Returns the GraphWeaver ontology definition.
    """
    try:
        ontology = GraphWeaverOntology.get_ontology_turtle()
        
        output = "## GraphWeaver Ontology (Turtle format)\n\n"
        output += "```turtle\n"
        output += ontology[:3000]  # Truncate for display
        if len(ontology) > 3000:
            output += "\n... (truncated)"
        output += "\n```"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# =============================================================================
# LTN Tools - Logic Tensor Networks for Rule Learning
# =============================================================================

@tool
def learn_rules_with_ltn() -> str:
    """Learn logical rules from the knowledge graph using LTN.
    
    This uses Logic Tensor Networks to:
    - Learn FK patterns from graph structure
    - Identify table classifications (fact, dimension, junction)
    - Discover column naming patterns
    - Extract logical constraints
    
    Requires text embeddings to be generated first.
    """
    try:
        if not LTN_AVAILABLE:
            return "LTN not available. Install with: pip install ltn tensorflow"
        
        print("[learn_rules_with_ltn] Starting rule learning...")
        
        learner = get_rule_learner()
        if learner is None:
            return "LTN not available. Install with: pip install ltn tensorflow"
        
        learned_rules = learner.learn_rules()
        
        if not learned_rules:
            return "No rules learned. Make sure you have FK discovery results and embeddings generated."
        
        output = f"## Learned {len(learned_rules)} Rules with LTN\n\n"
        
        # Group by type
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
                if rule.description:
                    output += f"  {rule.description}\n"
            output += "\n"
        
        output += "\nUse `generate_business_rules_from_ltn` to convert these to executable SQL rules."
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR learning rules: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def generate_business_rules_from_ltn() -> str:
    """Generate executable business rules from learned LTN patterns.
    
    Converts learned rules into:
    - SQL validation queries
    - Data quality checks
    - Aggregation rules
    
    Run learn_rules_with_ltn first to learn patterns.
    """
    try:
        if not LTN_AVAILABLE:
            return "LTN not available. Install with: pip install ltn tensorflow"
        
        print("[generate_business_rules_from_ltn] Generating rules...")
        
        learner = get_rule_learner()
        generator = get_rule_generator()
        
        if learner is None or generator is None:
            return "LTN not available. Install with: pip install ltn tensorflow"
        
        # Get learned rules
        learned_rules = learner.get_learned_rules()
        
        if not learned_rules:
            # Learn rules first
            learned_rules = learner.learn_rules()
        
        if not learned_rules:
            return "No learned rules available. Run learn_rules_with_ltn first."
        
        # Generate business rules
        generated_rules = generator.generate_from_learned_rules(learned_rules)
        
        if not generated_rules:
            return "No business rules generated."
        
        output = f"## Generated {len(generated_rules)} Business Rules\n\n"
        
        # Group by type
        by_type = {}
        for rule in generated_rules:
            rtype = rule.rule_type
            if rtype not in by_type:
                by_type[rtype] = []
            by_type[rtype].append(rule)
        
        for rtype, rules in by_type.items():
            output += f"### {rtype.title()} Rules ({len(rules)}):\n"
            for rule in rules[:5]:  # Show first 5 per type
                output += f"- **{rule.name}**\n"
                output += f"  {rule.description}\n"
                output += f"  Inputs: {', '.join(rule.inputs)}\n"
            if len(rules) > 5:
                output += f"  ... and {len(rules) - 5} more\n"
            output += "\n"
        
        output += "\nUse `export_generated_rules_yaml` to export as YAML file."
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR generating rules: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def generate_all_validation_rules() -> str:
    """Generate all possible validation rules from the Neo4j graph.
    
    Creates rules for:
    - FK referential integrity checks
    - PK uniqueness validation
    - Row count metrics
    
    Does not require LTN - uses graph structure directly.
    """
    try:
        if not LTN_AVAILABLE:
            return "LTN module not available. Install with: pip install ltn tensorflow"
        
        print("[generate_all_validation_rules] Generating all rules...")
        
        generator = get_rule_generator()
        if generator is None:
            return "Rule generator not available."
        
        all_rules = generator.generate_all_rules()
        
        if not all_rules:
            return "No rules generated. Make sure FK discovery has been run."
        
        output = f"## Generated {len(all_rules)} Validation Rules\n\n"
        
        # Group by type
        by_type = {}
        for rule in all_rules:
            rtype = rule.rule_type
            if rtype not in by_type:
                by_type[rtype] = []
            by_type[rtype].append(rule)
        
        for rtype, rules in by_type.items():
            output += f"### {rtype.title()} Rules ({len(rules)}):\n"
            for rule in rules:
                output += f"- **{rule.name}**: {rule.description}\n"
            output += "\n"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def export_generated_rules_yaml() -> str:
    """Export generated business rules as YAML.
    
    Creates a business_rules_generated.yaml file that can be used with
    load_business_rules_from_file and execute_all_business_rules.
    """
    try:
        if not LTN_AVAILABLE:
            return "LTN module not available."
        
        generator = get_rule_generator()
        if generator is None:
            return "Rule generator not available."
        
        if not generator.generated_rules:
            # Generate rules first
            generator.generate_all_rules()
        
        if not generator.generated_rules:
            return "No rules to export. Run generate_business_rules_from_ltn first."
        
        yaml_content = generator.export_yaml()
        
        # Save to file
        filename = "business_rules_generated.yaml"
        with open(filename, "w") as f:
            f.write(yaml_content)
        
        output = f"## Exported {len(generator.generated_rules)} Rules to {filename}\n\n"
        output += "```yaml\n"
        output += yaml_content[:2000]
        if len(yaml_content) > 2000:
            output += "\n... (truncated)"
        output += "\n```\n"
        output += f"\nLoad with: `load_business_rules_from_file business_rules_generated.yaml`"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def export_generated_rules_sql() -> str:
    """Export generated rules as SQL script.
    
    Creates a validation_rules.sql file with all validation queries.
    """
    try:
        if not LTN_AVAILABLE:
            return "LTN module not available."
        
        generator = get_rule_generator()
        if generator is None:
            return "Rule generator not available."
        
        if not generator.generated_rules:
            generator.generate_all_rules()
        
        if not generator.generated_rules:
            return "No rules to export."
        
        sql_content = generator.export_sql()
        
        # Save to file
        filename = "validation_rules.sql"
        with open(filename, "w") as f:
            f.write(sql_content)
        
        output = f"## Exported SQL to {filename}\n\n"
        output += "```sql\n"
        output += sql_content[:2000]
        if len(sql_content) > 2000:
            output += "\n... (truncated)"
        output += "\n```"
        
        return output
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"


@tool
def show_ltn_knowledge_base() -> str:
    """Show the LTN knowledge base with axioms and constraints.
    
    Displays the logical rules and constraints used for reasoning.
    """
    try:
        if not LTN_AVAILABLE:
            return "LTN not available. Install with: pip install ltn tensorflow"
        
        kb = LTNKnowledgeBase.create_default()
        
        output = "## LTN Knowledge Base\n\n"
        
        output += "### Axioms (Logical Rules):\n"
        for axiom in kb.get_all_axioms():
            output += f"- **{axiom.name}**: `{axiom.formula}`\n"
            output += f"  Type: {axiom.axiom_type.value}, Weight: {axiom.weight}\n"
            if axiom.description:
                output += f"  {axiom.description}\n"
        
        output += "\n### Available Predicates:\n"
        output += "- `IsPK(c)` - Column is primary key\n"
        output += "- `IsFK(c)` - Column is foreign key\n"
        output += "- `FK(c1, c2)` - FK relationship between columns\n"
        output += "- `IsFact(t)` - Table is a fact table\n"
        output += "- `IsDimension(t)` - Table is a dimension table\n"
        output += "- `IsJunction(t)` - Table is a junction table\n"
        output += "- `BelongsTo(c, t)` - Column belongs to table\n"
        output += "- `SameType(c1, c2)` - Columns have same data type\n"
        
        return output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


SYSTEM_PROMPT = """You are GraphWeaver Agent - an AI assistant that helps users discover foreign key relationships, execute business rules, capture data lineage, and perform semantic search on database metadata.

## Your Capabilities:

### FK Discovery
- `run_fk_discovery` - Complete 5-stage FK discovery pipeline
- `analyze_potential_fk` - Score a single column pair
- `validate_fk_with_data` - Test FK with actual data
- `semantic_fk_discovery` - Find FKs using semantic similarity (even with non-matching names)

### Database Exploration
- `configure_database` - Connect to a PostgreSQL database
- `test_database_connection` - Verify connectivity
- `list_database_tables` - See all tables
- `get_table_schema` - See columns, types, PKs
- `get_column_stats` - Get column statistics

### Knowledge Graph (Neo4j)
- `clear_neo4j_graph` - Reset the graph
- `add_fk_to_graph` - Add FK relationship
- `get_graph_stats` - Graph statistics
- `analyze_graph_centrality` - Hub/authority analysis
- `find_table_communities` - Find related clusters
- `predict_missing_fks` - Suggest missing FKs (name-based)
- `run_cypher` - Execute any custom Cypher query
- `connect_datasets_to_tables` - Link Dataset nodes to Table nodes

### Embeddings & Semantic Search
- `generate_text_embeddings` - Create semantic embeddings for all metadata
- `generate_kg_embeddings` - Create graph structure embeddings (FastRP)
- `create_vector_indexes` - Create Neo4j vector indexes for fast search
- `semantic_search_tables` - Search tables using natural language
- `semantic_search_columns` - Search columns using natural language
- `find_similar_tables` - Find structurally/semantically similar tables
- `find_similar_columns` - Find similar columns across tables
- `predict_fks_from_embeddings` - Predict FKs using graph structure

### Business Rules & Lineage
- `show_sample_business_rules` - Show example YAML format
- `load_business_rules` - Load rules from YAML string
- `load_business_rules_from_file` - Load rules from a YAML file
- `list_business_rules` - Show loaded rules
- `execute_business_rule` - Run single rule with lineage capture
- `execute_all_business_rules` - Run all rules
- `get_marquez_lineage` - Get lineage for a dataset
- `list_marquez_jobs` - Show tracked jobs
- `import_lineage_to_graph` - Import lineage to Neo4j
- `analyze_data_flow` - Full analysis (FKs + lineage)
- `find_impact_analysis` - What breaks if table changes?

### RDF / SPARQL
- `test_rdf_connection` - Test connection to Fuseki triple store
- `sync_graph_to_rdf` - Sync Neo4j graph to RDF store
- `run_sparql` - Execute custom SPARQL queries
- `sparql_list_tables` - List tables via SPARQL
- `sparql_get_foreign_keys` - Get FKs via SPARQL
- `sparql_table_lineage` - Get lineage via SPARQL
- `sparql_downstream_impact` - Find downstream deps via SPARQL
- `sparql_hub_tables` - Find hub tables via SPARQL
- `sparql_orphan_tables` - Find orphan tables
- `sparql_search` - Search graph by label
- `get_rdf_statistics` - RDF store stats
- `export_rdf_turtle` - Export ontology

### LTN / Rule Learning
- `learn_rules_with_ltn` - Learn logical rules from graph using LTN
- `generate_business_rules_from_ltn` - Generate executable rules from learned patterns
- `generate_all_validation_rules` - Generate all validation rules from graph
- `export_generated_rules_yaml` - Export rules as YAML file
- `export_generated_rules_sql` - Export rules as SQL script
- `show_ltn_knowledge_base` - Show LTN axioms and predicates

### Dynamic Tools
- `check_tool_exists` - Check if a tool exists before creating
- `list_available_tools` - List all builtin and dynamic tools
- `create_dynamic_tool` - Create a new tool from Python code
- `run_dynamic_tool` - Execute a dynamic tool
- `get_tool_source` - Get source code of a dynamic tool
- `update_dynamic_tool` - Update existing tool code
- `delete_dynamic_tool` - Remove a dynamic tool

## Dynamic Tool Workflow:

When asked to create new functionality:
1. **Check**: Use `check_tool_exists` to see if capability exists
2. **Create**: Use `create_dynamic_tool` with Python code defining a `run()` function
3. **Test**: Use `run_dynamic_tool` to verify it works
4. **Persist**: Tools are saved to `dynamic_tools/` directory for future use

Example tool code:
```python
def run(data: str, format: str = "json") -> str:
    \"\"\"Process data in the specified format.\"\"\"
    result = do_something(data)
    return f"Processed: {result}"
```

## Typical Workflow:

1. **Discover FKs**: `run_fk_discovery`
2. **Generate embeddings**: `generate_text_embeddings`, `generate_kg_embeddings`
3. **Create indexes**: `create_vector_indexes`
4. **Semantic FK discovery**: `semantic_fk_discovery` (finds FKs missed by name matching)
5. **Learn rules with LTN**: `learn_rules_with_ltn`
6. **Generate validation rules**: `generate_all_validation_rules`
7. **Export rules**: `export_generated_rules_yaml`
8. **Load business rules**: `load_business_rules_from_file`
9. **Execute rules**: `execute_all_business_rules` (captures lineage)
10. **Import lineage**: `import_lineage_to_graph`
11. **Connect graphs**: `connect_datasets_to_tables`
12. **Sync to RDF**: `sync_graph_to_rdf`
13. **Analyze**: `semantic_search_tables`, `find_similar_tables`, `find_impact_analysis`

## Embedding-Powered Features:

**Text Embeddings** (384 dims, all-MiniLM-L6-v2):
- Enable natural language search: "find customer-related columns"
- Find semantically similar names: customer_id ↔ buyer_identifier
- Search by concept: "monetary amounts" → price, cost, total

**KG Embeddings** (128 dims, FastRP):
- Find structurally similar nodes in the graph
- Predict missing FKs based on graph topology
- Discover hidden relationships

## RDF/SPARQL Features:

- Standard ontologies: DCAT, PROV-O, Dublin Core
- Interoperability with other data catalogs
- SPARQL queries for complex traversals
- Fuseki UI at http://localhost:3030

## LTN Features:

- Learn logical rules from graph patterns
- Generate SQL validation queries automatically
- Discover FK naming patterns
- Classify tables as fact/dimension/junction
- Export learned rules as YAML or SQL

Be helpful and thorough!"""


def create_agent(verbose: bool = True):
    """Create the LangGraph agent with Claude."""
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable required")
    
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.1,
        max_tokens=4096,
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
        get_marquez_lineage,
        list_marquez_jobs,
        import_lineage_to_graph,
        analyze_data_flow,
        find_impact_analysis,
        
        # RDF Tools
        test_rdf_connection,
        sync_graph_to_rdf,
        run_sparql,
        sparql_list_tables,
        sparql_get_foreign_keys,
        sparql_table_lineage,
        sparql_downstream_impact,
        sparql_hub_tables,
        sparql_orphan_tables,
        sparql_search,
        get_rdf_statistics,
        export_rdf_turtle,
        
        # LTN Tools
        learn_rules_with_ltn,
        generate_business_rules_from_ltn,
        generate_all_validation_rules,
        export_generated_rules_yaml,
        export_generated_rules_sql,
        show_ltn_knowledge_base,
        
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


# =============================================================================
# Main Entry Points
# =============================================================================

def run_autonomous_discovery(verbose: bool = True) -> dict:
    """Run fully autonomous FK discovery using Claude."""
    
    print("\n" + "="*60)
    print("  GraphWeaver Agent - Claude-Powered FK Discovery")
    print("="*60 + "\n")
    
    agent = create_agent(verbose=verbose)
    
    instruction = """Discover all foreign key relationships in this database.

Work autonomously:
1. Connect and explore all tables
2. Identify and validate FK candidates  
3. Build the Neo4j graph with confirmed FKs
4. Generate embeddings for semantic search
5. Analyze and report insights

Go!"""
    
    result = agent.invoke(
        {"messages": [HumanMessage(content=instruction)]},
        config={"recursion_limit": 100}
    )
    response = extract_response(result)
    
    print("\n" + "="*60)
    print("  FINAL REPORT")
    print("="*60 + "\n")
    print(response)
    
    return {"output": response}


def run_interactive():
    """Run agent in interactive mode - chat with Claude."""
    import sys
    
    agent = create_agent(verbose=True)
    history = []
    
    print("\n" + "="*60)
    print("  GraphWeaver Agent - Chat with Claude")
    print("="*60)
    print("\nI can help you discover FK relationships in your database.")
    print("\nTry saying:")
    print("  • 'connect and show me the tables'")
    print("  • 'find all foreign keys'")
    print("  • 'is orders.customer_id a FK to customers?'")
    print("  • 'build the graph and analyze it'")
    print("  • 'load business rules from file and execute them'")
    print("  • 'generate embeddings and search for customer columns'")
    print("  • 'find tables similar to orders'")
    print("  • 'sync graph to RDF and run SPARQL queries'")
    print("  • 'learn rules with LTN and generate validation rules'")
    print("  • 'create a tool to format SQL queries'")
    print("\nType 'quit' to exit.\n")
    sys.stdout.flush()
    
    while True:
        try:
            sys.stdout.write("You: ")
            sys.stdout.flush()
            user_input = sys.stdin.readline()
            
            if not user_input:
                print("\nEnd of input.")
                break
            
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\nThinking...", flush=True)
            
            messages = history + [HumanMessage(content=user_input)]
            
            print("[TRACE] Calling agent.invoke...", flush=True)
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 100}
            )
            print("[TRACE] agent.invoke returned", flush=True)
            
            # Extract text response from the result
            print("[TRACE] Calling extract_response...", flush=True)
            response_text = extract_response(result)
            print(f"[TRACE] extract_response returned: type={type(response_text)}, len={len(str(response_text)) if response_text else 0}", flush=True)
            print(f"[TRACE] response_text repr: {repr(response_text)[:200]}", flush=True)
            
            print(f"\nAgent: {response_text}\n")
            sys.stdout.flush()
            
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=response_text))
            
            if len(history) > 20:
                history = history[-20:]
                
        except EOFError:
            print("\nEnd of input.")
            break
        except KeyboardInterrupt:
            print("\n")
            break
        except BaseException as e:
            import traceback
            import sys
            print("\n" + "="*60, file=sys.stderr)
            print("EXCEPTION CAUGHT", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(f"Type: {type(e)}", file=sys.stderr)
            print(f"Name: {type(e).__name__}", file=sys.stderr)
            print(f"Args: {e.args}", file=sys.stderr)
            print(f"Str:  '{str(e)}'", file=sys.stderr)
            print(f"Repr: {repr(e)}", file=sys.stderr)
            if hasattr(e, 'code'):
                print(f"Code: {e.code}", file=sys.stderr)
            if hasattr(e, 'returncode'):
                print(f"Returncode: {e.returncode}", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print("TRACEBACK:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("="*60, file=sys.stderr)
            sys.stderr.flush()
            
            # Also print to stdout
            print(f"\n[EXCEPTION] Type: {type(e).__name__}", flush=True)
            print(f"[EXCEPTION] Message: '{str(e)}'", flush=True)
            print(f"[EXCEPTION] Args: {e.args}", flush=True)
    
    print("Goodbye!")


def extract_response(result) -> str:
    """Extract text response from LangGraph result."""
    import sys
    print(f"[EXTRACT] result type: {type(result)}", file=sys.stderr, flush=True)
    
    if not isinstance(result, dict):
        print(f"[EXTRACT] result is not dict, returning str(result)", file=sys.stderr, flush=True)
        return str(result)
    
    messages = result.get("messages", [])
    print(f"[EXTRACT] messages count: {len(messages)}", file=sys.stderr, flush=True)
    
    if not messages:
        print(f"[EXTRACT] no messages, returning str(result)", file=sys.stderr, flush=True)
        return str(result)
    
    # Get the last message
    last_msg = messages[-1]
    print(f"[EXTRACT] last_msg type: {type(last_msg)}", file=sys.stderr, flush=True)
    
    # Handle different content types
    content = getattr(last_msg, 'content', None)
    print(f"[EXTRACT] content type: {type(content)}", file=sys.stderr, flush=True)
    print(f"[EXTRACT] content repr: {repr(content)[:500] if content else 'None'}", file=sys.stderr, flush=True)
    
    if content is None:
        print(f"[EXTRACT] content is None, returning str(last_msg)", file=sys.stderr, flush=True)
        return str(last_msg)
    
    # Content can be a string or a list of content blocks
    if isinstance(content, str):
        print(f"[EXTRACT] content is str, returning it", file=sys.stderr, flush=True)
        return content
    
    if isinstance(content, list):
        print(f"[EXTRACT] content is list with {len(content)} items", file=sys.stderr, flush=True)
        # Extract text from content blocks
        text_parts = []
        for i, block in enumerate(content):
            print(f"[EXTRACT] block[{i}] type: {type(block)}", file=sys.stderr, flush=True)
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                print(f"[EXTRACT] block[{i}] keys: {block.keys()}", file=sys.stderr, flush=True)
                if block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
                elif block.get('type') == 'tool_use':
                    print(f"[EXTRACT] block[{i}] is tool_use, skipping", file=sys.stderr, flush=True)
            elif hasattr(block, 'text'):
                text_parts.append(block.text)
        result_text = '\n'.join(text_parts) if text_parts else str(content)
        print(f"[EXTRACT] returning joined text, len={len(result_text)}", file=sys.stderr, flush=True)
        return result_text
    
    print(f"[EXTRACT] content is unknown type, returning str(content)", file=sys.stderr, flush=True)
    return str(content)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphWeaver Agent")
    parser.add_argument("--auto", "-a", action="store_true", help="Run autonomous discovery then exit")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    if args.auto:
        run_autonomous_discovery(verbose=not args.quiet)
    else:
        # Default is interactive
        run_interactive()


if __name__ == "__main__":
    main()