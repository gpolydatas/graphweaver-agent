"""RDF module for GraphWeaver Agent - RDF/SPARQL support with Apache Jena Fuseki."""

from .fuseki_client import FusekiClient
from .rdf_sync import RDFSyncManager, sync_neo4j_to_rdf
from .ontology import GraphWeaverOntology, PREFIXES
from .sparql_queries import SPARQLQueryBuilder

__all__ = [
    "FusekiClient",
    "RDFSyncManager",
    "sync_neo4j_to_rdf",
    "GraphWeaverOntology",
    "PREFIXES",
    "SPARQLQueryBuilder",
]