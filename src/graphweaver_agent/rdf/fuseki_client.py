"""
Apache Jena Fuseki Client - Connect to RDF triple store.
"""
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests


@dataclass
class FusekiConfig:
    url: str = "http://localhost:3030"
    dataset: str = "graphweaver"
    username: str = "admin"
    password: str = "admin"


class FusekiClient:
    def __init__(self, config: Optional[FusekiConfig] = None):
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
        try:
            response = requests.get(f"{self.config.url}/$/ping", timeout=5)
            if response.status_code == 200:
                return {"success": True, "message": "Connected to Fuseki"}
            return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def ensure_dataset_exists(self) -> bool:
        try:
            response = requests.get(
                f"{self.config.url}/$/datasets/{self.config.dataset}",
                auth=self.auth, timeout=5
            )
            if response.status_code == 200:
                return True
            response = requests.post(
                f"{self.config.url}/$/datasets",
                auth=self.auth,
                data={"dbName": self.config.dataset, "dbType": "tdb2"},
                timeout=10
            )
            return response.status_code in [200, 201]
        except Exception as e:
            print(f"[FusekiClient] Error: {e}")
            return False

    def sparql_query(self, query: str) -> List[Dict[str, Any]]:
        try:
            response = requests.post(
                f"{self.base_url}/sparql",
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30
            )
            if response.status_code != 200:
                return []
            result = response.json()
            bindings = result.get("results", {}).get("bindings", [])
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
        try:
            response = requests.post(
                f"{self.base_url}/update",
                data={"update": update},
                auth=self.auth,
                timeout=30
            )
            return response.status_code in [200, 204]
        except Exception as e:
            print(f"[FusekiClient] Update exception: {e}")
            return False

    def insert_turtle(self, turtle_content: str, graph: Optional[str] = None) -> bool:
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
            return response.status_code in [200, 201, 204]
        except Exception as e:
            print(f"[FusekiClient] Insert exception: {e}")
            return False

    def clear_graph(self, graph: Optional[str] = None) -> bool:
        if graph:
            update = f"CLEAR GRAPH <{graph}>"
        else:
            update = "CLEAR DEFAULT"
        return self.sparql_update(update)

    def get_triple_count(self, graph: Optional[str] = None) -> int:
        if graph:
            query = f"SELECT (COUNT(*) as ?count) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
        else:
            query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
        result = self.sparql_query(query)
        if result:
            return int(result[0].get("count", 0))
        return 0
