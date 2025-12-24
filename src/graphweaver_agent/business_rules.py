"""
Business Rules Engine - Define and execute business operations with lineage tracking.

Users define business rules in YAML. The agent executes them and captures
lineage via OpenLineage/Marquez. Results feed back into knowledge graph.
"""
import os
import yaml
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# Business Rule Models
# =============================================================================

class RuleType(str, Enum):
    QUERY = "query"              # SELECT - read only
    AGGREGATION = "aggregation"  # GROUP BY with metrics
    TRANSFORMATION = "transformation"  # Creates derived data
    VALIDATION = "validation"    # Data quality checks
    METRIC = "metric"           # KPI calculation


class BusinessRule(BaseModel):
    """A single business rule definition."""
    name: str
    description: str
    type: RuleType = RuleType.QUERY
    sql: str
    inputs: List[str] = Field(default_factory=list)  # Source tables
    outputs: List[str] = Field(default_factory=list)  # Output tables/views
    schedule: Optional[str] = None  # cron expression if scheduled
    owner: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Column-level lineage hints
    column_mappings: Dict[str, str] = Field(default_factory=dict)  # output_col -> input_table.input_col


class BusinessRulesConfig(BaseModel):
    """Complete business rules configuration."""
    version: str = "1.0"
    namespace: str = "default"
    rules: List[BusinessRule] = Field(default_factory=list)


# =============================================================================
# OpenLineage Event Models
# =============================================================================

@dataclass
class OpenLineageDataset:
    namespace: str
    name: str
    facets: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class OpenLineageJob:
    namespace: str
    name: str
    facets: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenLineageRun:
    runId: str
    facets: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenLineageRunEvent:
    eventType: str  # START, COMPLETE, FAIL
    eventTime: str
    run: OpenLineageRun
    job: OpenLineageJob
    inputs: List[OpenLineageDataset]
    outputs: List[OpenLineageDataset]
    producer: str = "https://github.com/graphweaver/agent"
    schemaURL: str = "https://openlineage.io/spec/1-0-5/OpenLineage.json#/$defs/RunEvent"


# =============================================================================
# Marquez Client
# =============================================================================

class MarquezClient:
    """Client for Marquez lineage server."""
    
    def __init__(self, url: str = None, base_url: str = None, **kwargs):
        # Accept 'url', 'base_url', or any other parameter for maximum compatibility
        # Priority: url > base_url > kwargs.get('host') > default
        effective_url = url or base_url or kwargs.get('host') or "http://localhost:5000"
        self.url = effective_url.rstrip("/")
        self.base_url = self.url  # Alias for compatibility
        self.api_url = f"{self.url}/api/v1"
        print(f"[MarquezClient] Initialized with URL: {self.url}")
    
    def emit_event(self, event: OpenLineageRunEvent) -> bool:
        """Emit an OpenLineage event to Marquez."""
        import requests
        
        payload = {
            "eventType": event.eventType,
            "eventTime": event.eventTime,
            "producer": event.producer,
            "schemaURL": event.schemaURL,
            "run": {
                "runId": event.run.runId,
                "facets": event.run.facets if event.run.facets else {}
            },
            "job": {
                "namespace": event.job.namespace,
                "name": event.job.name,
                "facets": event.job.facets if event.job.facets else {}
            },
            "inputs": [
                {
                    "namespace": d.namespace,
                    "name": d.name,
                    "facets": d.facets if d.facets else {}
                }
                for d in event.inputs
            ],
            "outputs": [
                {
                    "namespace": d.namespace,
                    "name": d.name,
                    "facets": d.facets if d.facets else {}
                }
                for d in event.outputs
            ],
        }
        
        endpoint = f"{self.url}/api/v1/lineage"
        print(f"[MarquezClient] Sending {event.eventType} event to {endpoint}")
        print(f"[MarquezClient] Job: {event.job.namespace}/{event.job.name}")
        print(f"[MarquezClient] Run ID: {event.run.runId}")
        print(f"[MarquezClient] Inputs: {[d.name for d in event.inputs]}")
        print(f"[MarquezClient] Outputs: {[d.name for d in event.outputs]}")
        
        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            print(f"[MarquezClient] Response: {resp.status_code}")
            if resp.status_code not in (200, 201):
                print(f"[MarquezClient] ERROR Response body: {resp.text}")
                return False
            print(f"[MarquezClient] SUCCESS - Event recorded")
            return True
        except requests.exceptions.ConnectionError as e:
            print(f"[MarquezClient] CONNECTION ERROR: Cannot reach {endpoint}")
            print(f"[MarquezClient] Details: {e}")
            return False
        except Exception as e:
            print(f"[MarquezClient] ERROR: {type(e).__name__}: {e}")
            return False
    
    def get_datasets(self, namespace: str = "default") -> List[Dict]:
        """Get all datasets in a namespace."""
        import requests
        try:
            resp = requests.get(f"{self.api_url}/namespaces/{namespace}/datasets", timeout=10)
            print(f"[MarquezClient] get_datasets({namespace}): {resp.status_code}")
            return resp.json().get("datasets", [])
        except Exception as e:
            print(f"[MarquezClient] get_datasets ERROR: {e}")
            return []
    
    def get_dataset_lineage(self, namespace: str, dataset: str, depth: int = 5) -> Dict:
        """Get lineage graph for a dataset."""
        import requests
        try:
            resp = requests.get(f"{self.api_url}/lineage", params={
                "nodeId": f"dataset:{namespace}:{dataset}",
                "depth": depth,
            }, timeout=10)
            print(f"[MarquezClient] get_dataset_lineage({namespace}/{dataset}): {resp.status_code}")
            return resp.json()
        except Exception as e:
            print(f"[MarquezClient] get_dataset_lineage ERROR: {e}")
            return {}
    
    def get_jobs(self, namespace: str = "default") -> List[Dict]:
        """Get all jobs in a namespace."""
        import requests
        try:
            resp = requests.get(f"{self.api_url}/namespaces/{namespace}/jobs", timeout=10)
            print(f"[MarquezClient] get_jobs({namespace}): {resp.status_code}")
            if resp.status_code == 200:
                jobs = resp.json().get("jobs", [])
                print(f"[MarquezClient] Found {len(jobs)} jobs")
                return jobs
            return []
        except Exception as e:
            print(f"[MarquezClient] get_jobs ERROR: {e}")
            return []
    
    def get_job_runs(self, namespace: str, job: str) -> List[Dict]:
        """Get runs for a job."""
        import requests
        try:
            resp = requests.get(f"{self.api_url}/namespaces/{namespace}/jobs/{job}/runs", timeout=10)
            return resp.json().get("runs", [])
        except Exception as e:
            print(f"[MarquezClient] get_job_runs ERROR: {e}")
            return []
    
    def list_jobs(self, namespace: str = "default") -> List[Dict]:
        """Alias for get_jobs for compatibility."""
        return self.get_jobs(namespace)
    
    def get_lineage(self, dataset_name: str, depth: int = 5) -> Dict:
        """Alias for get_dataset_lineage for compatibility."""
        return self.get_dataset_lineage("default", dataset_name, depth)
    
    def test_connection(self) -> bool:
        """Test if Marquez is reachable."""
        import requests
        try:
            resp = requests.get(f"{self.api_url}/namespaces", timeout=5)
            print(f"[MarquezClient] Connection test: {resp.status_code}")
            return resp.status_code == 200
        except Exception as e:
            print(f"[MarquezClient] Connection test FAILED: {e}")
            return False


# =============================================================================
# Business Rules Executor
# =============================================================================

class BusinessRulesExecutor:
    """Execute business rules and capture lineage."""
    
    def __init__(
        self,
        connector,  # PostgreSQLConnector
        marquez_url: str = "http://localhost:5000",
        namespace: str = "default",
        db_namespace: str = "postgres",
    ):
        self.connector = connector
        self.marquez = MarquezClient(marquez_url)
        self.namespace = namespace
        self.db_namespace = db_namespace
        
        # Test connection on init
        print(f"[BusinessRulesExecutor] Testing Marquez connection...")
        if self.marquez.test_connection():
            print(f"[BusinessRulesExecutor] Marquez connection OK")
        else:
            print(f"[BusinessRulesExecutor] WARNING: Marquez not reachable at {marquez_url}")
    
    def load_rules(self, yaml_path: str) -> BusinessRulesConfig:
        """Load business rules from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        rules = []
        for rule_data in data.get('rules', []):
            rules.append(BusinessRule(**rule_data))
        
        return BusinessRulesConfig(
            version=data.get('version', '1.0'),
            namespace=data.get('namespace', 'default'),
            rules=rules,
        )
    
    def execute_rule(self, rule: BusinessRule, emit_lineage: bool = True) -> Dict[str, Any]:
        """Execute a single business rule and capture lineage."""
        run_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        print(f"\n[Executor] Running rule: {rule.name}")
        print(f"[Executor] Run ID: {run_id}")
        print(f"[Executor] Lineage capture: {emit_lineage}")
        
        # Build input/output datasets
        inputs = [
            OpenLineageDataset(namespace=self.db_namespace, name=table)
            for table in rule.inputs
        ]
        outputs = [
            OpenLineageDataset(namespace=self.db_namespace, name=table)
            for table in rule.outputs
        ] if rule.outputs else [
            OpenLineageDataset(
                namespace=self.db_namespace, 
                name=f"{rule.name}_result",
            )
        ]
        
        job = OpenLineageJob(
            namespace=self.namespace,
            name=rule.name,
            facets={
                "documentation": {
                    "_producer": "https://github.com/graphweaver/agent",
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                    "description": rule.description,
                },
                "sql": {
                    "_producer": "https://github.com/graphweaver/agent",
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SqlJobFacet.json",
                    "query": rule.sql,
                },
            }
        )
        
        run = OpenLineageRun(
            runId=run_id,
            facets={}
        )
        
        # Emit START event
        if emit_lineage:
            start_event = OpenLineageRunEvent(
                eventType="START",
                eventTime=start_time.isoformat(),
                run=run,
                job=job,
                inputs=inputs,
                outputs=outputs,
            )
            success = self.marquez.emit_event(start_event)
            if not success:
                print(f"[Executor] WARNING: Failed to emit START event")
        
        # Execute the SQL
        result = {
            "rule_name": rule.name,
            "run_id": run_id,
            "start_time": start_time.isoformat(),
            "status": "unknown",
            "rows": 0,
            "data": [],
            "error": None,
            "lineage_captured": False,
        }
        
        try:
            data = self.connector.execute_query(rule.sql, limit=10000)
            result["status"] = "success"
            result["rows"] = len(data)
            result["data"] = data[:100]  # Return first 100 rows
            result["columns"] = list(data[0].keys()) if data else []
            
            # Calculate basic stats for metrics
            if rule.type == RuleType.METRIC and data:
                numeric_cols = [k for k, v in data[0].items() if isinstance(v, (int, float))]
                result["metrics"] = {}
                for col in numeric_cols:
                    values = [row[col] for row in data if row[col] is not None]
                    if values:
                        result["metrics"][col] = {
                            "sum": sum(values),
                            "avg": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                        }
            
            event_type = "COMPLETE"
            print(f"[Executor] Rule {rule.name} completed: {result['rows']} rows")
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            event_type = "FAIL"
            print(f"[Executor] Rule {rule.name} FAILED: {e}")
        
        end_time = datetime.now(timezone.utc)
        result["end_time"] = end_time.isoformat()
        result["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # Emit COMPLETE/FAIL event
        if emit_lineage:
            run.facets = {
                "nominalTime": {
                    "_producer": "https://github.com/graphweaver/agent",
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/NominalTimeRunFacet.json",
                    "nominalStartTime": start_time.isoformat(),
                    "nominalEndTime": end_time.isoformat(),
                }
            }
            
            end_event = OpenLineageRunEvent(
                eventType=event_type,
                eventTime=end_time.isoformat(),
                run=run,
                job=job,
                inputs=inputs,
                outputs=outputs,
            )
            success = self.marquez.emit_event(end_event)
            result["lineage_captured"] = success
            if success:
                print(f"[Executor] Lineage captured for {rule.name}")
            else:
                print(f"[Executor] WARNING: Failed to capture lineage for {rule.name}")
        
        return result
    
    def execute_all_rules(
        self, 
        config: BusinessRulesConfig,
        emit_lineage: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute all business rules."""
        print(f"\n{'='*60}")
        print(f"Executing {len(config.rules)} business rules")
        print(f"Namespace: {config.namespace}")
        print(f"Lineage capture: {emit_lineage}")
        print(f"{'='*60}\n")
        
        # Update namespace from config
        self.namespace = config.namespace
        
        results = []
        for rule in config.rules:
            result = self.execute_rule(rule, emit_lineage=emit_lineage)
            results.append(result)
        
        # Summary
        success_count = sum(1 for r in results if r['status'] == 'success')
        lineage_count = sum(1 for r in results if r.get('lineage_captured'))
        print(f"\n{'='*60}")
        print(f"Execution complete: {success_count}/{len(results)} succeeded")
        print(f"Lineage captured: {lineage_count}/{len(results)}")
        print(f"{'='*60}\n")
        
        return results


# =============================================================================
# Lineage to Neo4j Integration
# =============================================================================

def import_lineage_to_neo4j(
    marquez_client: MarquezClient,
    neo4j_client,  # Neo4jClient
    namespace: str = "default",
) -> Dict[str, int]:
    """Import Marquez lineage data into Neo4j graph."""
    
    print(f"[import_lineage] Fetching jobs from namespace: {namespace}")
    
    stats = {"jobs": 0, "datasets": 0, "reads": 0, "writes": 0}
    
    # Get all jobs
    jobs = marquez_client.get_jobs(namespace)
    print(f"[import_lineage] Found {len(jobs)} jobs")
    
    for job in jobs:
        job_name = job.get("name")
        print(f"[import_lineage] Processing job: {job_name}")
        
        # Create Job node
        neo4j_client.run_write("""
            MERGE (j:Job {name: $name, namespace: $namespace})
            SET j.description = $desc,
                j.updated_at = datetime()
        """, {
            "name": job_name,
            "namespace": namespace,
            "desc": job.get("description", ""),
        })
        stats["jobs"] += 1
        
        # Get job's inputs/outputs
        for input_ds in job.get("inputs", []):
            ds_name = input_ds.get("name")
            print(f"[import_lineage]   Input: {ds_name}")
            
            # Create Dataset node and link to Table if exists
            neo4j_client.run_write("""
                MERGE (d:Dataset {name: $name, namespace: $namespace})
                WITH d
                OPTIONAL MATCH (t:Table {name: $name})
                FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (d)-[:REPRESENTS]->(t)
                )
                WITH d
                MATCH (j:Job {name: $job_name, namespace: $job_namespace})
                MERGE (j)-[:READS]->(d)
            """, {
                "name": ds_name,
                "namespace": input_ds.get("namespace", namespace),
                "job_name": job_name,
                "job_namespace": namespace,
            })
            stats["datasets"] += 1
            stats["reads"] += 1
        
        for output_ds in job.get("outputs", []):
            ds_name = output_ds.get("name")
            print(f"[import_lineage]   Output: {ds_name}")
            
            neo4j_client.run_write("""
                MERGE (d:Dataset {name: $name, namespace: $namespace})
                WITH d
                OPTIONAL MATCH (t:Table {name: $name})
                FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (d)-[:REPRESENTS]->(t)
                )
                WITH d
                MATCH (j:Job {name: $job_name, namespace: $job_namespace})
                MERGE (j)-[:WRITES]->(d)
            """, {
                "name": ds_name,
                "namespace": output_ds.get("namespace", namespace),
                "job_name": job_name,
                "job_namespace": namespace,
            })
            stats["datasets"] += 1
            stats["writes"] += 1
    
    print(f"[import_lineage] Import complete: {stats}")
    return stats


# =============================================================================
# Helper Functions
# =============================================================================

def generate_sample_rules() -> str:
    """Generate a sample business rules YAML."""
    return '''# Business Rules Configuration
# Define your business operations here. The agent will execute them
# and capture lineage via OpenLineage/Marquez.

version: "1.0"
namespace: "ecommerce"

rules:
  # Revenue Metrics
  - name: daily_revenue
    description: Calculate daily revenue by category
    type: aggregation
    sql: |
      SELECT 
        c.name as category,
        DATE(o.order_date) as date,
        COUNT(DISTINCT o.id) as order_count,
        SUM(oi.quantity) as items_sold,
        SUM(oi.quantity * oi.unit_price) as revenue
      FROM order_items oi
      JOIN orders o ON oi.order_id = o.id
      JOIN products p ON oi.product_id = p.id
      JOIN categories c ON p.category_id = c.id
      GROUP BY c.name, DATE(o.order_date)
      ORDER BY date DESC, revenue DESC
    inputs:
      - order_items
      - orders
      - products
      - categories
    outputs:
      - daily_revenue_report
    tags:
      - revenue
      - daily
    owner: analytics_team

  # Customer Analytics
  - name: customer_lifetime_value
    description: Calculate lifetime value per customer
    type: metric
    sql: |
      SELECT 
        c.id as customer_id,
        c.name as customer_name,
        c.email,
        COUNT(DISTINCT o.id) as total_orders,
        SUM(o.total_amount) as lifetime_value,
        MIN(o.order_date) as first_order,
        MAX(o.order_date) as last_order,
        AVG(o.total_amount) as avg_order_value
      FROM customers c
      LEFT JOIN orders o ON c.id = o.customer_id
      GROUP BY c.id, c.name, c.email
      ORDER BY lifetime_value DESC NULLS LAST
    inputs:
      - customers
      - orders
    outputs:
      - customer_clv
    tags:
      - customer
      - clv

  # Product Performance
  - name: product_performance
    description: Analyze product sales performance
    type: aggregation
    sql: |
      SELECT 
        p.id as product_id,
        p.name as product_name,
        c.name as category,
        p.price as unit_price,
        COUNT(DISTINCT oi.order_id) as times_ordered,
        SUM(oi.quantity) as total_quantity,
        SUM(oi.quantity * oi.unit_price) as total_revenue,
        AVG(oi.quantity) as avg_quantity_per_order
      FROM products p
      JOIN categories c ON p.category_id = c.id
      LEFT JOIN order_items oi ON p.id = oi.product_id
      GROUP BY p.id, p.name, c.name, p.price
      ORDER BY total_revenue DESC NULLS LAST
    inputs:
      - products
      - categories
      - order_items
    outputs:
      - product_performance_report
    tags:
      - product
      - sales

  # Supplier Analysis
  - name: supplier_product_count
    description: Count products per supplier
    type: query
    sql: |
      SELECT 
        s.id as supplier_id,
        s.name as supplier_name,
        s.contact_email,
        COUNT(ps.product_id) as product_count,
        COUNT(CASE WHEN ps.is_primary THEN 1 END) as primary_products
      FROM suppliers s
      LEFT JOIN product_suppliers ps ON s.id = ps.supplier_id
      GROUP BY s.id, s.name, s.contact_email
      ORDER BY product_count DESC
    inputs:
      - suppliers
      - product_suppliers
    outputs:
      - supplier_summary
    tags:
      - supplier

  # Data Quality Check
  - name: orphan_orders_check
    description: Find orders without valid customers
    type: validation
    sql: |
      SELECT o.*
      FROM orders o
      LEFT JOIN customers c ON o.customer_id = c.id
      WHERE c.id IS NULL
    inputs:
      - orders
      - customers
    outputs:
      - data_quality_issues
    tags:
      - validation
      - data_quality

  # Order Fulfillment
  - name: orders_with_details
    description: Full order details with customer and items
    type: query
    sql: |
      SELECT 
        o.id as order_id,
        o.order_date,
        c.name as customer_name,
        c.email as customer_email,
        COUNT(oi.id) as item_count,
        SUM(oi.quantity) as total_items,
        o.total_amount
      FROM orders o
      JOIN customers c ON o.customer_id = c.id
      JOIN order_items oi ON o.id = oi.order_id
      GROUP BY o.id, o.order_date, c.name, c.email, o.total_amount
      ORDER BY o.order_date DESC
    inputs:
      - orders
      - customers
      - order_items
    outputs:
      - order_details
    tags:
      - orders
      - fulfillment
'''
