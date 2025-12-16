"""
RDF Sync Manager - Synchronize Neo4j graph to RDF triple store.

Keeps both stores in sync:
- Neo4j for analytics, embeddings, fast graph traversal
- RDF/Fuseki for standards compliance, SPARQL, ontology reasoning
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from .fuseki_client import FusekiClient
from .ontology import (
    PREFIXES, GraphWeaverOntology,
    make_table_uri, make_column_uri, make_job_uri, make_dataset_uri, make_fk_uri
)


class RDFSyncManager:
    """Manage synchronization between Neo4j and RDF store."""
    
    def __init__(self, fuseki_client: FusekiClient, neo4j_client=None):
        """
        Initialize sync manager.
        
        Args:
            fuseki_client: FusekiClient instance
            neo4j_client: Neo4jClient instance (optional)
        """
        self.fuseki = fuseki_client
        self.neo4j = neo4j_client
        self.ontology = GraphWeaverOntology()
        self.data_ns = "http://graphweaver.io/data#"
        self.graph_uri = "http://graphweaver.io/graph/main"
        
    def initialize_store(self) -> Dict[str, Any]:
        """
        Initialize the RDF store with ontology.
        
        Returns:
            Status dict
        """
        print("[RDFSync] Initializing RDF store...")
        
        # Ensure dataset exists
        if not self.fuseki.ensure_dataset_exists():
            return {"success": False, "error": "Failed to create dataset"}
        
        # Clear existing data
        self.fuseki.clear_graph(self.graph_uri)
        
        # Load ontology
        ontology_turtle = GraphWeaverOntology.get_ontology_turtle()
        if not self.fuseki.insert_turtle(ontology_turtle, graph=f"{self.graph_uri}/ontology"):
            return {"success": False, "error": "Failed to load ontology"}
        
        print("[RDFSync] Ontology loaded")
        return {"success": True, "message": "RDF store initialized"}
    
    def sync_tables(self, tables: List[Dict[str, Any]]) -> int:
        """
        Sync table metadata to RDF.
        
        Args:
            tables: List of table dicts with name, columns, etc.
            
        Returns:
            Number of tables synced
        """
        print(f"[RDFSync] Syncing {len(tables)} tables...")
        
        turtle_lines = [PREFIXES]
        
        for table in tables:
            table_name = table.get("name") or table.get("table_name")
            if not table_name:
                continue
                
            table_uri = make_table_uri(table_name)
            
            # Table triples
            turtle_lines.append(f"""
<{table_uri}> a gw:Table, dcat:Dataset ;
    rdfs:label "{table_name}" ;
    dct:title "{table_name}" ;
    gw:rowCount {table.get('row_count', 0)} .
""")
            
            # Column triples
            columns = table.get("columns", [])
            for col in columns:
                col_name = col.get("name") or col.get("column_name")
                if not col_name:
                    continue
                    
                col_uri = make_column_uri(table_name, col_name)
                data_type = col.get("data_type", "unknown")
                is_pk = "true" if col.get("is_primary_key") else "false"
                is_nullable = "true" if col.get("is_nullable", True) else "false"
                
                turtle_lines.append(f"""
<{col_uri}> a gw:Column ;
    rdfs:label "{col_name}" ;
    gw:belongsToTable <{table_uri}> ;
    gw:hasDataType "{data_type}" ;
    gw:isPrimaryKey {is_pk} ;
    gw:isNullable {is_nullable} .

<{table_uri}> gw:hasColumn <{col_uri}> .
""")
        
        turtle_content = "\n".join(turtle_lines)
        
        if self.fuseki.insert_turtle(turtle_content, graph=self.graph_uri):
            print(f"[RDFSync] Synced {len(tables)} tables")
            return len(tables)
        return 0
    
    def sync_foreign_keys(self, fks: List[Dict[str, Any]]) -> int:
        """
        Sync foreign key relationships to RDF.
        
        Args:
            fks: List of FK dicts
            
        Returns:
            Number of FKs synced
        """
        print(f"[RDFSync] Syncing {len(fks)} foreign keys...")
        
        turtle_lines = [PREFIXES]
        
        for fk in fks:
            source_table = fk.get("source_table")
            source_column = fk.get("source_column")
            target_table = fk.get("target_table")
            target_column = fk.get("target_column")
            
            if not all([source_table, source_column, target_table, target_column]):
                continue
            
            source_col_uri = make_column_uri(source_table, source_column)
            target_col_uri = make_column_uri(target_table, target_column)
            fk_uri = make_fk_uri(source_table, source_column, target_table, target_column)
            
            score = fk.get("score", fk.get("confidence", 0.0))
            cardinality = fk.get("cardinality", "1:N")
            
            turtle_lines.append(f"""
<{fk_uri}> a gw:ForeignKey ;
    rdfs:label "{source_table}.{source_column} -> {target_table}.{target_column}" ;
    gw:confidenceScore {score} ;
    gw:cardinality "{cardinality}" .

<{source_col_uri}> gw:references <{target_col_uri}> .
""")
        
        turtle_content = "\n".join(turtle_lines)
        
        if self.fuseki.insert_turtle(turtle_content, graph=self.graph_uri):
            print(f"[RDFSync] Synced {len(fks)} foreign keys")
            return len(fks)
        return 0
    
    def sync_jobs(self, jobs: List[Dict[str, Any]]) -> int:
        """
        Sync job/lineage data to RDF.
        
        Args:
            jobs: List of job dicts with inputs/outputs
            
        Returns:
            Number of jobs synced
        """
        print(f"[RDFSync] Syncing {len(jobs)} jobs...")
        
        turtle_lines = [PREFIXES]
        
        for job in jobs:
            job_name = job.get("name")
            if not job_name:
                continue
                
            job_uri = make_job_uri(job_name)
            description = job.get("description", "").replace('"', '\\"')
            
            turtle_lines.append(f"""
<{job_uri}> a gw:Job, prov:Activity ;
    rdfs:label "{job_name}" ;
    dct:description "{description}" .
""")
            
            # Input datasets
            for input_ds in job.get("inputs", []):
                ds_name = input_ds.get("name") if isinstance(input_ds, dict) else input_ds
                if ds_name:
                    ds_uri = make_dataset_uri(ds_name)
                    turtle_lines.append(f"""
<{ds_uri}> a gw:Dataset, dcat:Dataset ;
    rdfs:label "{ds_name}" .
    
<{job_uri}> gw:readsFrom <{ds_uri}> ;
    prov:used <{ds_uri}> .
""")
            
            # Output datasets
            for output_ds in job.get("outputs", []):
                ds_name = output_ds.get("name") if isinstance(output_ds, dict) else output_ds
                if ds_name:
                    ds_uri = make_dataset_uri(ds_name)
                    turtle_lines.append(f"""
<{ds_uri}> a gw:Dataset, dcat:Dataset ;
    rdfs:label "{ds_name}" .
    
<{job_uri}> gw:writesTo <{ds_uri}> ;
    prov:generated <{ds_uri}> .
""")
        
        turtle_content = "\n".join(turtle_lines)
        
        if self.fuseki.insert_turtle(turtle_content, graph=self.graph_uri):
            print(f"[RDFSync] Synced {len(jobs)} jobs")
            return len(jobs)
        return 0
    
    def sync_dataset_table_links(self, links: List[Dict[str, str]]) -> int:
        """
        Sync Dataset-Table REPRESENTS relationships.
        
        Args:
            links: List of {dataset, table} dicts
            
        Returns:
            Number of links synced
        """
        print(f"[RDFSync] Syncing {len(links)} dataset-table links...")
        
        turtle_lines = [PREFIXES]
        
        for link in links:
            ds_name = link.get("dataset")
            table_name = link.get("table")
            
            if ds_name and table_name:
                ds_uri = make_dataset_uri(ds_name)
                table_uri = make_table_uri(table_name)
                
                turtle_lines.append(f"""
<{ds_uri}> gw:represents <{table_uri}> .
""")
        
        turtle_content = "\n".join(turtle_lines)
        
        if self.fuseki.insert_turtle(turtle_content, graph=self.graph_uri):
            print(f"[RDFSync] Synced {len(links)} links")
            return len(links)
        return 0
    
    def sync_from_neo4j(self) -> Dict[str, int]:
        """
        Full sync from Neo4j to RDF.
        
        Returns:
            Stats dict with counts
        """
        if self.neo4j is None:
            return {"error": "Neo4j client not configured"}
        
        print("[RDFSync] Starting full sync from Neo4j...")
        stats = {"tables": 0, "columns": 0, "fks": 0, "jobs": 0, "datasets": 0, "links": 0}
        
        # Initialize store
        self.initialize_store()
        
        # Get and sync tables with columns
        tables_result = self.neo4j.run_query("""
            MATCH (t:Table)
            OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)
            WITH t, collect({
                name: c.name,
                data_type: c.data_type,
                is_primary_key: c.is_primary_key,
                is_nullable: c.is_nullable
            }) as columns
            RETURN t.name as name, t.row_count as row_count, columns
        """)
        
        if tables_result:
            tables = [dict(r) for r in tables_result]
            stats["tables"] = self.sync_tables(tables)
            for t in tables:
                stats["columns"] += len(t.get("columns", []))
        
        # Get and sync FKs
        fks_result = self.neo4j.run_query("""
            MATCH (sc:Column)-[fk:FK_TO]->(tc:Column)
            MATCH (sc)-[:BELONGS_TO]->(st:Table)
            MATCH (tc)-[:BELONGS_TO]->(tt:Table)
            RETURN st.name as source_table, sc.name as source_column,
                   tt.name as target_table, tc.name as target_column,
                   fk.score as score, fk.cardinality as cardinality
        """)
        
        if fks_result:
            fks = [dict(r) for r in fks_result]
            stats["fks"] = self.sync_foreign_keys(fks)
        
        # Get and sync jobs
        jobs_result = self.neo4j.run_query("""
            MATCH (j:Job)
            OPTIONAL MATCH (j)-[:READS]->(input:Dataset)
            OPTIONAL MATCH (j)-[:WRITES]->(output:Dataset)
            WITH j, collect(DISTINCT input.name) as inputs, collect(DISTINCT output.name) as outputs
            RETURN j.name as name, j.description as description, inputs, outputs
        """)
        
        if jobs_result:
            jobs = [dict(r) for r in jobs_result]
            stats["jobs"] = self.sync_jobs(jobs)
            
            # Count unique datasets
            all_datasets = set()
            for job in jobs:
                all_datasets.update(job.get("inputs", []))
                all_datasets.update(job.get("outputs", []))
            stats["datasets"] = len([d for d in all_datasets if d])
        
        # Get and sync dataset-table links
        links_result = self.neo4j.run_query("""
            MATCH (d:Dataset)-[:REPRESENTS]->(t:Table)
            RETURN d.name as dataset, t.name as table
        """)
        
        if links_result:
            links = [dict(r) for r in links_result]
            stats["links"] = self.sync_dataset_table_links(links)
        
        # Get final triple count
        stats["total_triples"] = self.fuseki.get_triple_count(self.graph_uri)
        
        print(f"[RDFSync] Sync complete: {stats}")
        return stats
    
    def export_turtle(self) -> str:
        """
        Export entire graph as Turtle.
        
        Returns:
            Turtle format string
        """
        query = f"""
        CONSTRUCT {{ ?s ?p ?o }}
        WHERE {{
            GRAPH <{self.graph_uri}> {{ ?s ?p ?o }}
        }}
        """
        
        try:
            response = self.fuseki.fuseki_client.sparql_query(query)
            # Note: This returns bindings, need different approach for CONSTRUCT
            # For now, return a simple representation
            return f"# Export from {self.graph_uri}\n# Use SPARQL CONSTRUCT for full export"
        except:
            return ""


def sync_neo4j_to_rdf(neo4j_client, fuseki_client: Optional[FusekiClient] = None) -> Dict[str, int]:
    """
    Convenience function to sync Neo4j graph to RDF.
    
    Args:
        neo4j_client: Neo4j client
        fuseki_client: Fuseki client (creates new if None)
        
    Returns:
        Sync statistics
    """
    if fuseki_client is None:
        fuseki_client = FusekiClient()
    
    sync_manager = RDFSyncManager(fuseki_client, neo4j_client)
    return sync_manager.sync_from_neo4j()