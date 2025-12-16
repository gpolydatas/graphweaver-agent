"""
SPARQL Query Builder - Pre-built queries for common operations.

Provides easy access to common data catalog queries via SPARQL.
"""
from typing import List, Dict, Any, Optional
from .ontology import PREFIXES_SPARQL


class SPARQLQueryBuilder:
    """Build and execute common SPARQL queries."""
    
    def __init__(self, fuseki_client):
        """
        Initialize query builder.
        
        Args:
            fuseki_client: FusekiClient instance
        """
        self.fuseki = fuseki_client
        self.graph_uri = "http://graphweaver.io/graph/main"
    
    def _run(self, query: str) -> List[Dict[str, Any]]:
        """Run a query with prefixes."""
        full_query = PREFIXES_SPARQL + "\n" + query
        return self.fuseki.sparql_query(full_query)
    
    def list_tables(self) -> List[Dict[str, Any]]:
        """List all tables with column counts."""
        query = f"""
        SELECT ?table ?label (COUNT(?col) as ?columnCount) ?rowCount
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?table a gw:Table ;
                       rdfs:label ?label .
                OPTIONAL {{ ?table gw:rowCount ?rowCount }}
                OPTIONAL {{ ?table gw:hasColumn ?col }}
            }}
        }}
        GROUP BY ?table ?label ?rowCount
        ORDER BY ?label
        """
        return self._run(query)
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get all columns for a table."""
        query = f"""
        SELECT ?column ?label ?dataType ?isPK ?isNullable
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?table a gw:Table ;
                       rdfs:label "{table_name}" ;
                       gw:hasColumn ?column .
                ?column rdfs:label ?label .
                OPTIONAL {{ ?column gw:hasDataType ?dataType }}
                OPTIONAL {{ ?column gw:isPrimaryKey ?isPK }}
                OPTIONAL {{ ?column gw:isNullable ?isNullable }}
            }}
        }}
        ORDER BY ?label
        """
        return self._run(query)
    
    def get_foreign_keys(self, table_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get foreign key relationships."""
        filter_clause = ""
        if table_name:
            filter_clause = f'FILTER(?sourceTableLabel = "{table_name}" || ?targetTableLabel = "{table_name}")'
        
        query = f"""
        SELECT ?sourceTable ?sourceTableLabel ?sourceCol ?sourceColLabel 
               ?targetTable ?targetTableLabel ?targetCol ?targetColLabel
               ?score ?cardinality
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?sourceCol gw:references ?targetCol .
                ?sourceCol rdfs:label ?sourceColLabel ;
                           gw:belongsToTable ?sourceTable .
                ?sourceTable rdfs:label ?sourceTableLabel .
                ?targetCol rdfs:label ?targetColLabel ;
                           gw:belongsToTable ?targetTable .
                ?targetTable rdfs:label ?targetTableLabel .
                
                OPTIONAL {{
                    ?fk a gw:ForeignKey ;
                        gw:confidenceScore ?score ;
                        gw:cardinality ?cardinality .
                }}
                {filter_clause}
            }}
        }}
        ORDER BY ?sourceTableLabel ?sourceColLabel
        """
        return self._run(query)
    
    def get_table_lineage(self, table_name: str) -> List[Dict[str, Any]]:
        """Get lineage for a table (jobs that read/write)."""
        query = f"""
        SELECT ?job ?jobLabel ?direction ?dataset ?datasetLabel
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?dataset a gw:Dataset ;
                         rdfs:label "{table_name}" .
                
                {{
                    ?job gw:readsFrom ?dataset .
                    BIND("reads" as ?direction)
                }}
                UNION
                {{
                    ?job gw:writesTo ?dataset .
                    BIND("writes" as ?direction)
                }}
                
                ?job rdfs:label ?jobLabel .
                ?dataset rdfs:label ?datasetLabel .
            }}
        }}
        ORDER BY ?direction ?jobLabel
        """
        return self._run(query)
    
    def get_downstream_impact(self, table_name: str) -> List[Dict[str, Any]]:
        """Get downstream impact - what depends on this table."""
        query = f"""
        SELECT DISTINCT ?dependentTable ?dependentTableLabel ?relationshipType
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                # Tables that reference this via FK
                {{
                    ?table a gw:Table ;
                           rdfs:label "{table_name}" .
                    ?table gw:hasColumn ?col .
                    ?depCol gw:references ?col .
                    ?depCol gw:belongsToTable ?dependentTable .
                    ?dependentTable rdfs:label ?dependentTableLabel .
                    BIND("FK_REFERENCE" as ?relationshipType)
                }}
                UNION
                # Datasets that are written by jobs reading this table
                {{
                    ?dataset a gw:Dataset ;
                             rdfs:label "{table_name}" .
                    ?job gw:readsFrom ?dataset ;
                         gw:writesTo ?outputDataset .
                    ?outputDataset rdfs:label ?dependentTableLabel .
                    BIND(?outputDataset as ?dependentTable)
                    BIND("LINEAGE" as ?relationshipType)
                }}
            }}
        }}
        ORDER BY ?relationshipType ?dependentTableLabel
        """
        return self._run(query)
    
    def get_hub_tables(self, min_connections: int = 3) -> List[Dict[str, Any]]:
        """Find hub tables with many connections."""
        query = f"""
        SELECT ?table ?label 
               (COUNT(DISTINCT ?inRef) as ?incomingFKs)
               (COUNT(DISTINCT ?outRef) as ?outgoingFKs)
               (COUNT(DISTINCT ?readsJob) as ?readByJobs)
               ((COUNT(DISTINCT ?inRef) + COUNT(DISTINCT ?outRef) + COUNT(DISTINCT ?readsJob)) as ?totalConnections)
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?table a gw:Table ;
                       rdfs:label ?label .
                
                # Incoming FK references
                OPTIONAL {{
                    ?table gw:hasColumn ?col .
                    ?inRef gw:references ?col .
                }}
                
                # Outgoing FK references
                OPTIONAL {{
                    ?table gw:hasColumn ?outCol .
                    ?outCol gw:references ?outRef .
                }}
                
                # Jobs that read this table
                OPTIONAL {{
                    ?dataset gw:represents ?table .
                    ?readsJob gw:readsFrom ?dataset .
                }}
            }}
        }}
        GROUP BY ?table ?label
        HAVING ((COUNT(DISTINCT ?inRef) + COUNT(DISTINCT ?outRef) + COUNT(DISTINCT ?readsJob)) >= {min_connections})
        ORDER BY DESC(?totalConnections)
        """
        return self._run(query)
    
    def find_orphan_tables(self) -> List[Dict[str, Any]]:
        """Find tables with no FK relationships."""
        query = f"""
        SELECT ?table ?label
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?table a gw:Table ;
                       rdfs:label ?label .
                
                FILTER NOT EXISTS {{
                    ?table gw:hasColumn ?col .
                    {{ ?col gw:references ?other }}
                    UNION
                    {{ ?other gw:references ?col }}
                }}
            }}
        }}
        ORDER BY ?label
        """
        return self._run(query)
    
    def search_by_label(self, search_term: str) -> List[Dict[str, Any]]:
        """Search all resources by label."""
        query = f"""
        SELECT ?resource ?type ?label
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?resource rdfs:label ?label .
                ?resource a ?type .
                FILTER(CONTAINS(LCASE(?label), LCASE("{search_term}")))
                FILTER(?type IN (gw:Table, gw:Column, gw:Job, gw:Dataset))
            }}
        }}
        ORDER BY ?type ?label
        LIMIT 50
        """
        return self._run(query)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get overall graph statistics."""
        query = f"""
        SELECT 
            (COUNT(DISTINCT ?table) as ?tables)
            (COUNT(DISTINCT ?column) as ?columns)
            (COUNT(DISTINCT ?fk) as ?foreignKeys)
            (COUNT(DISTINCT ?job) as ?jobs)
            (COUNT(DISTINCT ?dataset) as ?datasets)
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                OPTIONAL {{ ?table a gw:Table }}
                OPTIONAL {{ ?column a gw:Column }}
                OPTIONAL {{ ?fk a gw:ForeignKey }}
                OPTIONAL {{ ?job a gw:Job }}
                OPTIONAL {{ ?dataset a gw:Dataset }}
            }}
        }}
        """
        results = self._run(query)
        if results:
            return results[0]
        return {}
    
    def custom_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a custom SPARQL query."""
        return self._run(query)