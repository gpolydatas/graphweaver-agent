"""GraphWeaver Ontology - RDF/OWL vocabulary."""

PREFIXES_SPARQL = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX gw: <http://graphweaver.io/ontology#>
PREFIX gwdata: <http://graphweaver.io/data#>
"""

PREFIXES = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dcat: <http://www.w3.org/ns/dcat#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix gw: <http://graphweaver.io/ontology#> .
@prefix gwdata: <http://graphweaver.io/data#> .
"""


class GraphWeaverOntology:
    ONTOLOGY_NS = "http://graphweaver.io/ontology#"
    DATA_NS = "http://graphweaver.io/data#"

    @classmethod
    def get_ontology_turtle(cls) -> str:
        return f"""{PREFIXES}
gw: a owl:Ontology ;
    rdfs:label "GraphWeaver Ontology" .

gw:Table a owl:Class ;
    rdfs:subClassOf dcat:Dataset ;
    rdfs:label "Database Table" .

gw:Column a owl:Class ;
    rdfs:label "Table Column" .

gw:Job a owl:Class ;
    rdfs:subClassOf prov:Activity ;
    rdfs:label "Data Processing Job" .

gw:Dataset a owl:Class ;
    rdfs:subClassOf dcat:Dataset ;
    rdfs:label "Dataset" .

gw:hasColumn a owl:ObjectProperty ;
    rdfs:domain gw:Table ;
    rdfs:range gw:Column .

gw:belongsToTable a owl:ObjectProperty ;
    rdfs:domain gw:Column ;
    rdfs:range gw:Table .

gw:references a owl:ObjectProperty ;
    rdfs:domain gw:Column ;
    rdfs:range gw:Column .

gw:readsFrom a owl:ObjectProperty ;
    rdfs:domain gw:Job ;
    rdfs:range gw:Dataset .

gw:writesTo a owl:ObjectProperty ;
    rdfs:domain gw:Job ;
    rdfs:range gw:Dataset .

gw:represents a owl:ObjectProperty ;
    rdfs:domain gw:Dataset ;
    rdfs:range gw:Table .
"""


def uri_safe(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace(".", "_")


def make_table_uri(table_name: str, namespace: str = "http://graphweaver.io/data#") -> str:
    return f"{namespace}table_{uri_safe(table_name)}"


def make_column_uri(table_name: str, column_name: str, namespace: str = "http://graphweaver.io/data#") -> str:
    return f"{namespace}column_{uri_safe(table_name)}_{uri_safe(column_name)}"


def make_job_uri(job_name: str, namespace: str = "http://graphweaver.io/data#") -> str:
    return f"{namespace}job_{uri_safe(job_name)}"


def make_dataset_uri(dataset_name: str, namespace: str = "http://graphweaver.io/data#") -> str:
    return f"{namespace}dataset_{uri_safe(dataset_name)}"
