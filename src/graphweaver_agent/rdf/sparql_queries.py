"""SPARQL Query Builder."""
from typing import List, Dict, Any, Optional
from .ontology import PREFIXES_SPARQL


class SPARQLQueryBuilder:
    def __init__(self, fuseki_client):
        self.fuseki = fuseki_client
        self.graph_uri = "http://graphweaver.io/graph/main"

    def _run(self, query: str) -> List[Dict[str, Any]]:
        full_query = PREFIXES_SPARQL + "\n" + query
        return self.fuseki.sparql_query(full_query)

    def list_tables(self) -> List[Dict[str, Any]]:
        query = f"""
        SELECT ?table ?label (COUNT(?col) as ?columnCount)
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?table a gw:Table ; rdfs:label ?label .
                OPTIONAL {{ ?table gw:hasColumn ?col }}
            }}
        }}
        GROUP BY ?table ?label
        ORDER BY ?label
        """
        return self._run(query)

    def get_foreign_keys(self, table_name: Optional[str] = None) -> List[Dict[str, Any]]:
        filter_clause = f'FILTER(?sourceTableLabel = "{table_name}" || ?targetTableLabel = "{table_name}")' if table_name else ""
        query = f"""
        SELECT ?sourceTableLabel ?sourceColLabel ?targetTableLabel ?targetColLabel
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?sourceCol gw:references ?targetCol .
                ?sourceCol rdfs:label ?sourceColLabel ; gw:belongsToTable ?sourceTable .
                ?sourceTable rdfs:label ?sourceTableLabel .
                ?targetCol rdfs:label ?targetColLabel ; gw:belongsToTable ?targetTable .
                ?targetTable rdfs:label ?targetTableLabel .
                {filter_clause}
            }}
        }}
        """
        return self._run(query)

    def get_table_lineage(self, table_name: str) -> List[Dict[str, Any]]:
        query = f"""
        SELECT ?job ?jobLabel ?direction
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?dataset a gw:Dataset ; rdfs:label "{table_name}" .
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
            }}
        }}
        """
        return self._run(query)

    def get_downstream_impact(self, table_name: str) -> List[Dict[str, Any]]:
        query = f"""
        SELECT DISTINCT ?dependentTableLabel ?relationshipType
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                {{
                    ?table a gw:Table ; rdfs:label "{table_name}" .
                    ?table gw:hasColumn ?col .
                    ?depCol gw:references ?col .
                    ?depCol gw:belongsToTable ?dependentTable .
                    ?dependentTable rdfs:label ?dependentTableLabel .
                    BIND("FK_REFERENCE" as ?relationshipType)
                }}
                UNION
                {{
                    ?dataset a gw:Dataset ; rdfs:label "{table_name}" .
                    ?job gw:readsFrom ?dataset ; gw:writesTo ?outputDataset .
                    ?outputDataset rdfs:label ?dependentTableLabel .
                    BIND("LINEAGE" as ?relationshipType)
                }}
            }}
        }}
        """
        return self._run(query)

    def get_hub_tables(self, min_connections: int = 2) -> List[Dict[str, Any]]:
        query = f"""
        SELECT ?label (COUNT(DISTINCT ?ref) as ?totalConnections)
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?table a gw:Table ; rdfs:label ?label .
                OPTIONAL {{ ?table gw:hasColumn ?col . ?col gw:references ?ref }}
                OPTIONAL {{ ?table gw:hasColumn ?col2 . ?ref2 gw:references ?col2 }}
            }}
        }}
        GROUP BY ?table ?label
        HAVING (COUNT(DISTINCT ?ref) >= {min_connections})
        ORDER BY DESC(?totalConnections)
        """
        return self._run(query)

    def find_orphan_tables(self) -> List[Dict[str, Any]]:
        query = f"""
        SELECT ?label
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?table a gw:Table ; rdfs:label ?label .
                FILTER NOT EXISTS {{
                    ?table gw:hasColumn ?col .
                    {{ ?col gw:references ?other }} UNION {{ ?other gw:references ?col }}
                }}
            }}
        }}
        """
        return self._run(query)

    def search_by_label(self, search_term: str) -> List[Dict[str, Any]]:
        query = f"""
        SELECT ?resource ?type ?label
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                ?resource rdfs:label ?label ; a ?type .
                FILTER(CONTAINS(LCASE(?label), LCASE("{search_term}")))
                FILTER(?type IN (gw:Table, gw:Column, gw:Job, gw:Dataset))
            }}
        }}
        LIMIT 50
        """
        return self._run(query)

    def get_statistics(self) -> Dict[str, int]:
        query = f"""
        SELECT 
            (COUNT(DISTINCT ?table) as ?tables)
            (COUNT(DISTINCT ?column) as ?columns)
            (COUNT(DISTINCT ?job) as ?jobs)
            (COUNT(DISTINCT ?dataset) as ?datasets)
        WHERE {{
            GRAPH <{self.graph_uri}> {{
                OPTIONAL {{ ?table a gw:Table }}
                OPTIONAL {{ ?column a gw:Column }}
                OPTIONAL {{ ?job a gw:Job }}
                OPTIONAL {{ ?dataset a gw:Dataset }}
            }}
        }}
        """
        results = self._run(query)
        return results[0] if results else {}

    def custom_query(self, query: str) -> List[Dict[str, Any]]:
        return self._run(query)
