"""Neo4j Graph Operations."""
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from neo4j import GraphDatabase
from graphweaver_agent.models import Neo4jConfig


class Neo4jClient:
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._driver = None
    
    def connect(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )
        return self._driver
    
    def disconnect(self):
        if self._driver:
            self._driver.close()
            self._driver = None
    
    @contextmanager
    def session(self):
        driver = self.connect()
        session = driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()
    
    def run_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        with self.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def run_write(self, query: str, params: Optional[Dict] = None):
        with self.session() as session:
            session.run(query, params or {})
    
    def test_connection(self) -> Dict[str, Any]:
        try:
            self.run_query("RETURN 1")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


class GraphBuilder:
    def __init__(self, client: Neo4jClient):
        self.client = client
    
    def clear_graph(self):
        self.client.run_write("MATCH (n) DETACH DELETE n")
    
    def add_table(self, table_name: str, datasource_id: str = "default"):
        self.client.run_write("""
            MERGE (d:DataSource {id: $ds_id})
            MERGE (t:Table {name: $name})
            MERGE (t)-[:BELONGS_TO]->(d)
        """, {"ds_id": datasource_id, "name": table_name})
    
    def add_fk_relationship(self, source_table: str, source_col: str,
                           target_table: str, target_col: str,
                           score: float, cardinality: str):
        # First ensure both tables exist
        self.client.run_write("""
            MERGE (st:Table {name: $src_table})
            MERGE (tt:Table {name: $tgt_table})
        """, {"src_table": source_table, "tgt_table": target_table})
        
        # Then create columns and FK relationship
        self.client.run_write("""
            MATCH (st:Table {name: $src_table})
            MATCH (tt:Table {name: $tgt_table})
            MERGE (sc:Column {name: $src_col, table: $src_table})
            MERGE (tc:Column {name: $tgt_col, table: $tgt_table})
            MERGE (sc)-[:BELONGS_TO]->(st)
            MERGE (tc)-[:BELONGS_TO]->(tt)
            MERGE (sc)-[r:FK_TO]->(tc)
            SET r.score = $score, r.cardinality = $cardinality
        """, {
            "src_table": source_table, "src_col": source_col,
            "tgt_table": target_table, "tgt_col": target_col,
            "score": score, "cardinality": cardinality,
        })


class GraphAnalyzer:
    def __init__(self, client: Neo4jClient):
        self.client = client
    
    def get_statistics(self) -> Dict[str, int]:
        result = self.client.run_query("""
            MATCH (t:Table) WITH count(t) as tables
            MATCH (c:Column) WITH tables, count(c) as columns
            MATCH ()-[r:FK_TO]->() WITH tables, columns, count(r) as fks
            RETURN tables, columns, fks
        """)
        return result[0] if result else {"tables": 0, "columns": 0, "fks": 0}
    
    def centrality_analysis(self) -> Dict[str, Any]:
        results = self.client.run_query("""
            MATCH (t:Table)
            OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)-[:FK_TO]->()
            WITH t, count(DISTINCT c) as out_degree
            OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)<-[:FK_TO]-()
            WITH t, out_degree, count(DISTINCT c) as in_degree
            RETURN t.name as table_name, in_degree, out_degree,
                   in_degree + out_degree as total_degree
            ORDER BY total_degree DESC
        """)
        return {
            "centrality": results,
            "hub_tables": [r["table_name"] for r in results if r["out_degree"] > 1],
            "authority_tables": [r["table_name"] for r in results if r["in_degree"] > 1],
            "isolated_tables": [r["table_name"] for r in results if r["total_degree"] == 0],
        }
    
    def community_detection(self) -> List[Dict]:
        results = self.client.run_query("""
            MATCH (t1:Table)<-[:BELONGS_TO]-(c1:Column)-[:FK_TO]->(c2:Column)-[:BELONGS_TO]->(t2:Table)
            WITH t1.name as source, collect(DISTINCT t2.name) as targets
            RETURN source, targets ORDER BY size(targets) DESC
        """)
        communities = []
        visited = set()
        for row in results:
            if row["source"] in visited:
                continue
            community = {row["source"]}
            community.update(row["targets"])
            visited.update(community)
            communities.append({"tables": list(community), "size": len(community)})
        return communities
    
    def predict_missing_fks(self) -> List[Dict]:
        return self.client.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE c.name ENDS WITH '_id' AND NOT (c)-[:FK_TO]->() AND c.name <> 'id'
            WITH c, t
            MATCH (target:Table)
            WHERE toLower(replace(c.name, '_id', '')) = toLower(target.name)
               OR toLower(replace(c.name, '_id', '')) + 's' = toLower(target.name)
            RETURN t.name as source_table, c.name as source_column,
                   target.name as target_table, 'name_pattern' as reason
        """)
