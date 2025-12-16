"""
Apache Jena Fuseki Client - Connect to RDF triple store.

Provides SPARQL query and update capabilities.
"""
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests


@dataclass
class FusekiConfig:
    """Configuration for Fuseki connection."""
    url: str = "http://localhost:3030"
    dataset: str = "graphweaver"
    username: str = "admin"
    password: str = "admin"


class FusekiClient:
    """Client for Apache Jena Fuseki triple store."""
    
    def __init__(self, config: Optional[FusekiConfig] = None):
        """
        Initialize Fuseki client.
        
        Args:
            config: FusekiConfig instance (uses env vars if None)
        """
        if config is None:
            config = FusekiConfig(
                url=os.environ.get("FUSEKI_URL", "http://localhost:3030"),
                dataset=os.environ.get("FUSEKI_DATASET", "graphweaver"),
                username=os.environ.get("FUSEKI_USER", "admin"),
                password=os.environ.get("FUSEKI_PASSWORD", "admin"),
            )
        self.config = config
        self.base_url = f"{config.url}/{config.dataset}"
        self.auth = (config.username, config.password)
        
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Fuseki server."""
        try:
            response = requests.get(
                f"{self.config.url}/$/ping",
                timeout=5
            )
            if response.status_code == 200:
                return {"success": True, "message": "Connected to Fuseki"}
            return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def ensure_dataset_exists(self) -> bool:
        """Ensure the dataset exists, create if not."""
        try:
            # Check if dataset exists
            response = requests.get(
                f"{self.config.url}/$/datasets/{self.config.dataset}",
                auth=self.auth,
                timeout=5
            )
            if response.status_code == 200:
                print(f"[FusekiClient] Dataset '{self.config.dataset}' exists")
                return True
            
            # Create dataset
            print(f"[FusekiClient] Creating dataset '{self.config.dataset}'...")
            response = requests.post(
                f"{self.config.url}/$/datasets",
                auth=self.auth,
                data={
                    "dbName": self.config.dataset,
                    "dbType": "tdb2"
                },
                timeout=10
            )
            if response.status_code in [200, 201]:
                print(f"[FusekiClient] Dataset created successfully")
                return True
            else:
                print(f"[FusekiClient] Failed to create dataset: {response.status_code}")
                return False
        except Exception as e:
            print(f"[FusekiClient] Error: {e}")
            return False
    
    def sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL SELECT query.
        
        Args:
            query: SPARQL query string
            
        Returns:
            List of result bindings
        """
        try:
            response = requests.post(
                f"{self.base_url}/sparql",
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"[FusekiClient] Query error: {response.status_code}")
                print(f"[FusekiClient] Response: {response.text}")
                return []
            
            result = response.json()
            bindings = result.get("results", {}).get("bindings", [])
            
            # Simplify bindings
            simplified = []
            for binding in bindings:
                row = {}
                for var, value in binding.items():
                    row[var] = value.get("value")
                simplified.append(row)
            
            return simplified
        except Exception as e:
            print(f"[FusekiClient] Query exception: {e}")
            return []
    
    def sparql_update(self, update: str) -> bool:
        """
        Execute a SPARQL UPDATE query.
        
        Args:
            update: SPARQL UPDATE string
            
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/update",
                data={"update": update},
                auth=self.auth,
                timeout=30
            )
            
            if response.status_code in [200, 204]:
                return True
            else:
                print(f"[FusekiClient] Update error: {response.status_code}")
                print(f"[FusekiClient] Response: {response.text}")
                return False
        except Exception as e:
            print(f"[FusekiClient] Update exception: {e}")
            return False
    
    def insert_triples(self, triples: List[str], graph: Optional[str] = None) -> bool:
        """
        Insert triples into the store.
        
        Args:
            triples: List of triple strings (N-Triples format)
            graph: Optional named graph URI
            
        Returns:
            True if successful
        """
        if not triples:
            return True
        
        # Build INSERT DATA query
        triples_str = "\n".join(triples)
        
        if graph:
            update = f"""
            INSERT DATA {{
                GRAPH <{graph}> {{
                    {triples_str}
                }}
            }}
            """
        else:
            update = f"""
            INSERT DATA {{
                {triples_str}
            }}
            """
        
        return self.sparql_update(update)
    
    def insert_turtle(self, turtle_content: str, graph: Optional[str] = None) -> bool:
        """
        Insert Turtle format RDF data.
        
        Args:
            turtle_content: RDF data in Turtle format
            graph: Optional named graph URI
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/data"
            if graph:
                url += f"?graph={graph}"
            
            response = requests.post(
                url,
                data=turtle_content.encode('utf-8'),
                headers={"Content-Type": "text/turtle; charset=utf-8"},
                auth=self.auth,
                timeout=30
            )
            
            if response.status_code in [200, 201, 204]:
                return True
            else:
                print(f"[FusekiClient] Insert error: {response.status_code}")
                print(f"[FusekiClient] Response: {response.text}")
                return False
        except Exception as e:
            print(f"[FusekiClient] Insert exception: {e}")
            return False
    
    def clear_graph(self, graph: Optional[str] = None) -> bool:
        """
        Clear all triples from a graph.
        
        Args:
            graph: Named graph URI (None for default graph)
            
        Returns:
            True if successful
        """
        if graph:
            update = f"CLEAR GRAPH <{graph}>"
        else:
            update = "CLEAR DEFAULT"
        
        return self.sparql_update(update)
    
    def get_triple_count(self, graph: Optional[str] = None) -> int:
        """Get count of triples in a graph."""
        if graph:
            query = f"SELECT (COUNT(*) as ?count) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
        else:
            query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
        
        result = self.sparql_query(query)
        if result:
            return int(result[0].get("count", 0))
        return 0
    
    def describe(self, uri: str) -> List[Dict[str, Any]]:
        """
        Get all triples about a resource.
        
        Args:
            uri: Resource URI
            
        Returns:
            List of predicate-object pairs
        """
        query = f"""
        SELECT ?p ?o WHERE {{
            <{uri}> ?p ?o
        }}
        """
        return self.sparql_query(query)