"""
GraphWeaver Ontology - RDF/OWL vocabulary for data catalog.

Maps GraphWeaver concepts to standard ontologies:
- DCAT (Data Catalog Vocabulary)
- RDFS (RDF Schema)
- Dublin Core
- PROV-O (Provenance)
- Custom GraphWeaver vocabulary
"""
from typing import Dict, List
from dataclasses import dataclass


# Standard prefixes
PREFIXES = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dcat: <http://www.w3.org/ns/dcat#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix gw: <http://graphweaver.io/ontology#> .
@prefix gwdata: <http://graphweaver.io/data#> .
"""

PREFIXES_SPARQL = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX gw: <http://graphweaver.io/ontology#>
PREFIX gwdata: <http://graphweaver.io/data#>
"""


@dataclass
class GraphWeaverOntology:
    """GraphWeaver ontology definitions."""
    
    # Base URIs
    ONTOLOGY_NS = "http://graphweaver.io/ontology#"
    DATA_NS = "http://graphweaver.io/data#"
    
    # Classes
    TABLE = f"{ONTOLOGY_NS}Table"
    COLUMN = f"{ONTOLOGY_NS}Column"
    FOREIGN_KEY = f"{ONTOLOGY_NS}ForeignKey"
    JOB = f"{ONTOLOGY_NS}Job"
    DATASET = f"{ONTOLOGY_NS}Dataset"
    DATA_SOURCE = f"{ONTOLOGY_NS}DataSource"
    
    # Properties
    HAS_COLUMN = f"{ONTOLOGY_NS}hasColumn"
    BELONGS_TO_TABLE = f"{ONTOLOGY_NS}belongsToTable"
    REFERENCES = f"{ONTOLOGY_NS}references"
    HAS_DATA_TYPE = f"{ONTOLOGY_NS}hasDataType"
    IS_PRIMARY_KEY = f"{ONTOLOGY_NS}isPrimaryKey"
    IS_NULLABLE = f"{ONTOLOGY_NS}isNullable"
    CONFIDENCE_SCORE = f"{ONTOLOGY_NS}confidenceScore"
    CARDINALITY = f"{ONTOLOGY_NS}cardinality"
    READS_FROM = f"{ONTOLOGY_NS}readsFrom"
    WRITES_TO = f"{ONTOLOGY_NS}writesTo"
    REPRESENTS = f"{ONTOLOGY_NS}represents"
    HAS_EMBEDDING = f"{ONTOLOGY_NS}hasEmbedding"
    EMBEDDING_MODEL = f"{ONTOLOGY_NS}embeddingModel"
    
    @classmethod
    def get_ontology_turtle(cls) -> str:
        """Get the full ontology in Turtle format."""
        return f"""{PREFIXES}

# =============================================================================
# GraphWeaver Ontology
# =============================================================================

gw: a owl:Ontology ;
    rdfs:label "GraphWeaver Ontology" ;
    rdfs:comment "Vocabulary for describing database schemas, foreign keys, and data lineage" ;
    owl:versionInfo "1.0" .

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

gw:Table a owl:Class ;
    rdfs:subClassOf dcat:Dataset ;
    rdfs:label "Database Table" ;
    rdfs:comment "A table in a relational database" .

gw:Column a owl:Class ;
    rdfs:subClassOf rdf:Property ;
    rdfs:label "Table Column" ;
    rdfs:comment "A column in a database table" .

gw:ForeignKey a owl:Class ;
    rdfs:label "Foreign Key Relationship" ;
    rdfs:comment "A foreign key constraint between columns" .

gw:Job a owl:Class ;
    rdfs:subClassOf prov:Activity ;
    rdfs:label "Data Processing Job" ;
    rdfs:comment "A data transformation or processing job" .

gw:Dataset a owl:Class ;
    rdfs:subClassOf dcat:Dataset ;
    rdfs:label "Dataset" ;
    rdfs:comment "A dataset in the data lineage graph" .

gw:DataSource a owl:Class ;
    rdfs:subClassOf dcat:Catalog ;
    rdfs:label "Data Source" ;
    rdfs:comment "A database or data source" .

# -----------------------------------------------------------------------------
# Object Properties
# -----------------------------------------------------------------------------

gw:hasColumn a owl:ObjectProperty ;
    rdfs:domain gw:Table ;
    rdfs:range gw:Column ;
    rdfs:label "has column" ;
    rdfs:comment "Links a table to its columns" .

gw:belongsToTable a owl:ObjectProperty ;
    rdfs:domain gw:Column ;
    rdfs:range gw:Table ;
    owl:inverseOf gw:hasColumn ;
    rdfs:label "belongs to table" .

gw:references a owl:ObjectProperty ;
    rdfs:domain gw:Column ;
    rdfs:range gw:Column ;
    rdfs:label "references" ;
    rdfs:comment "Foreign key reference from one column to another" .

gw:readsFrom a owl:ObjectProperty ;
    rdfs:domain gw:Job ;
    rdfs:range gw:Dataset ;
    rdfs:subPropertyOf prov:used ;
    rdfs:label "reads from" .

gw:writesTo a owl:ObjectProperty ;
    rdfs:domain gw:Job ;
    rdfs:range gw:Dataset ;
    rdfs:subPropertyOf prov:generated ;
    rdfs:label "writes to" .

gw:represents a owl:ObjectProperty ;
    rdfs:domain gw:Dataset ;
    rdfs:range gw:Table ;
    rdfs:label "represents" ;
    rdfs:comment "Links a lineage dataset to its physical table" .

# -----------------------------------------------------------------------------
# Data Properties
# -----------------------------------------------------------------------------

gw:hasDataType a owl:DatatypeProperty ;
    rdfs:domain gw:Column ;
    rdfs:range xsd:string ;
    rdfs:label "data type" .

gw:isPrimaryKey a owl:DatatypeProperty ;
    rdfs:domain gw:Column ;
    rdfs:range xsd:boolean ;
    rdfs:label "is primary key" .

gw:isNullable a owl:DatatypeProperty ;
    rdfs:domain gw:Column ;
    rdfs:range xsd:boolean ;
    rdfs:label "is nullable" .

gw:confidenceScore a owl:DatatypeProperty ;
    rdfs:domain gw:ForeignKey ;
    rdfs:range xsd:decimal ;
    rdfs:label "confidence score" .

gw:cardinality a owl:DatatypeProperty ;
    rdfs:domain gw:ForeignKey ;
    rdfs:range xsd:string ;
    rdfs:label "cardinality" ;
    rdfs:comment "Relationship cardinality: 1:1, 1:N, N:M" .

gw:hasEmbedding a owl:DatatypeProperty ;
    rdfs:range xsd:string ;
    rdfs:label "has embedding" ;
    rdfs:comment "Vector embedding as JSON array" .

gw:embeddingModel a owl:DatatypeProperty ;
    rdfs:range xsd:string ;
    rdfs:label "embedding model" .

gw:rowCount a owl:DatatypeProperty ;
    rdfs:domain gw:Table ;
    rdfs:range xsd:integer ;
    rdfs:label "row count" .

"""


def uri_safe(name: str) -> str:
    """Make a string safe for use in URIs."""
    return name.replace(" ", "_").replace("-", "_").replace(".", "_")


def make_table_uri(table_name: str, namespace: str = "http://graphweaver.io/data#") -> str:
    """Create URI for a table."""
    return f"{namespace}table_{uri_safe(table_name)}"


def make_column_uri(table_name: str, column_name: str, namespace: str = "http://graphweaver.io/data#") -> str:
    """Create URI for a column."""
    return f"{namespace}column_{uri_safe(table_name)}_{uri_safe(column_name)}"


def make_job_uri(job_name: str, namespace: str = "http://graphweaver.io/data#") -> str:
    """Create URI for a job."""
    return f"{namespace}job_{uri_safe(job_name)}"


def make_dataset_uri(dataset_name: str, namespace: str = "http://graphweaver.io/data#") -> str:
    """Create URI for a dataset."""
    return f"{namespace}dataset_{uri_safe(dataset_name)}"


def make_fk_uri(source_table: str, source_col: str, target_table: str, target_col: str, 
                namespace: str = "http://graphweaver.io/data#") -> str:
    """Create URI for a foreign key relationship."""
    safe_name = f"{uri_safe(source_table)}_{uri_safe(source_col)}_to_{uri_safe(target_table)}_{uri_safe(target_col)}"
    return f"{namespace}fk_{safe_name}"