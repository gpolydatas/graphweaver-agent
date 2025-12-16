"""RDF Sync Manager - Synchronize Neo4j graph to RDF triple store."""
from typing import Dict, List, Any, Optional
from .fuseki_client import FusekiClient
from .ontology import PREFIXES, GraphWeaverOntology, make_table_uri, make_column_uri, make_job_uri, make_dataset_uri


class RDFSyncManager:
    def __init__(self, fuseki_client: FusekiClient, neo4j_client=None):
        self.fuseki = fuseki_client
        self.neo4j = neo4j_client
        self.graph_uri = "http://graphweaver.io/graph/main"

    def initialize_store(self) -> Dict[str, Any]:
        if not self.fuseki.ensure_dataset_exists():
            return {"success": False, "error": "Failed to create dataset"}
        self.fuseki.clear_graph(self.graph_uri)
        ontology_turtle = GraphWeaverOntology.get_ontology_turtle()
        self.fuseki.insert_turtle(ontology_turtle, graph=f"{self.graph_uri}/ontology")
        return {"success": True}

    def sync_tables(self, tables: List[Dict[str, Any]]) -> int:
        turtle_lines = [PREFIXES]
        for table in tables:
            table_name = table.get("name") or table.get("table_name")
            if not table_name:
                continue
            table_uri = make_table_uri(table_name)
            turtle_lines.append(f'<{table_uri}> a gw:Table ; rdfs:label "{table_name}" .')
            for col in table.get("columns", []):
                col_name = col.get("name") or col.get("column_name")
                if col_name:
                    col_uri = make_column_uri(table_name, col_name)
                    turtle_lines.append(f'<{col_uri}> a gw:Column ; rdfs:label "{col_name}" ; gw:belongsToTable <{table_uri}> .')
                    turtle_lines.append(f'<{table_uri}> gw:hasColumn <{col_uri}> .')
        self.fuseki.insert_turtle("\n".join(turtle_lines), graph=self.graph_uri)
        return len(tables)

    def sync_foreign_keys(self, fks: List[Dict[str, Any]]) -> int:
        turtle_lines = [PREFIXES]
        for fk in fks:
            src_col_uri = make_column_uri(fk.get("source_table", ""), fk.get("source_column", ""))
            tgt_col_uri = make_column_uri(fk.get("target_table", ""), fk.get("target_column", ""))
            turtle_lines.append(f'<{src_col_uri}> gw:references <{tgt_col_uri}> .')
        self.fuseki.insert_turtle("\n".join(turtle_lines), graph=self.graph_uri)
        return len(fks)

    def sync_jobs(self, jobs: List[Dict[str, Any]]) -> int:
        turtle_lines = [PREFIXES]
        for job in jobs:
            job_name = job.get("name")
            if not job_name:
                continue
            job_uri = make_job_uri(job_name)
            turtle_lines.append(f'<{job_uri}> a gw:Job ; rdfs:label "{job_name}" .')
            for inp in job.get("inputs", []):
                ds_name = inp.get("name") if isinstance(inp, dict) else inp
                if ds_name:
                    ds_uri = make_dataset_uri(ds_name)
                    turtle_lines.append(f'<{ds_uri}> a gw:Dataset ; rdfs:label "{ds_name}" .')
                    turtle_lines.append(f'<{job_uri}> gw:readsFrom <{ds_uri}> .')
            for out in job.get("outputs", []):
                ds_name = out.get("name") if isinstance(out, dict) else out
                if ds_name:
                    ds_uri = make_dataset_uri(ds_name)
                    turtle_lines.append(f'<{ds_uri}> a gw:Dataset ; rdfs:label "{ds_name}" .')
                    turtle_lines.append(f'<{job_uri}> gw:writesTo <{ds_uri}> .')
        self.fuseki.insert_turtle("\n".join(turtle_lines), graph=self.graph_uri)
        return len(jobs)

    def sync_from_neo4j(self) -> Dict[str, int]:
        if self.neo4j is None:
            return {"error": "Neo4j client not configured"}
        stats = {"tables": 0, "columns": 0, "fks": 0, "jobs": 0, "datasets": 0, "links": 0}
        self.initialize_store()

        tables_result = self.neo4j.run_query("""
            MATCH (t:Table)
            OPTIONAL MATCH (t)<-[:BELONGS_TO]-(c:Column)
            WITH t, collect({name: c.name, data_type: c.data_type}) as columns
            RETURN t.name as name, columns
        """)
        if tables_result:
            tables = [dict(r) for r in tables_result]
            stats["tables"] = self.sync_tables(tables)
            for t in tables:
                stats["columns"] += len(t.get("columns", []))

        fks_result = self.neo4j.run_query("""
            MATCH (sc:Column)-[fk:FK_TO]->(tc:Column)
            MATCH (sc)-[:BELONGS_TO]->(st:Table)
            MATCH (tc)-[:BELONGS_TO]->(tt:Table)
            RETURN st.name as source_table, sc.name as source_column,
                   tt.name as target_table, tc.name as target_column
        """)
        if fks_result:
            fks = [dict(r) for r in fks_result]
            stats["fks"] = self.sync_foreign_keys(fks)

        jobs_result = self.neo4j.run_query("""
            MATCH (j:Job)
            OPTIONAL MATCH (j)-[:READS]->(input:Dataset)
            OPTIONAL MATCH (j)-[:WRITES]->(output:Dataset)
            WITH j, collect(DISTINCT input.name) as inputs, collect(DISTINCT output.name) as outputs
            RETURN j.name as name, inputs, outputs
        """)
        if jobs_result:
            jobs = [dict(r) for r in jobs_result]
            stats["jobs"] = self.sync_jobs(jobs)

        stats["total_triples"] = self.fuseki.get_triple_count(self.graph_uri)
        return stats


def sync_neo4j_to_rdf(neo4j_client, fuseki_client: Optional[FusekiClient] = None) -> Dict[str, int]:
    if fuseki_client is None:
        fuseki_client = FusekiClient()
    sync_manager = RDFSyncManager(fuseki_client, neo4j_client)
    return sync_manager.sync_from_neo4j()
