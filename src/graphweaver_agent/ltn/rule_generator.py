"""
Business Rule Generator - Generate executable business rules from learned patterns.

Converts LTN rules into SQL constraints, validation queries, and YAML business rules.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .rule_learner import LearnedRule


class RuleTemplateType(Enum):
    """Types of rule templates."""
    VALIDATION = "validation"
    CONSTRAINT = "constraint"
    AGGREGATION = "aggregation"
    TRANSFORMATION = "transformation"
    ALERT = "alert"


@dataclass
class RuleTemplate:
    """Template for generating business rules."""
    name: str
    template_type: RuleTemplateType
    sql_template: str
    description: str
    parameters: List[str] = field(default_factory=list)
    
    def render(self, **kwargs) -> str:
        """Render template with parameters."""
        sql = self.sql_template
        for key, value in kwargs.items():
            sql = sql.replace(f"{{{key}}}", str(value))
        return sql


@dataclass
class GeneratedRule:
    """A generated business rule."""
    name: str
    description: str
    rule_type: str
    sql_query: str
    inputs: List[str]
    outputs: List[str]
    source_rule: Optional[LearnedRule] = None
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    
    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to YAML-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.rule_type,
            "sql": self.sql_query,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tags": self.tags,
        }


class BusinessRuleGenerator:
    """Generate business rules from learned patterns."""
    
    # Standard rule templates
    TEMPLATES = {
        "fk_validation": RuleTemplate(
            name="fk_validation",
            template_type=RuleTemplateType.VALIDATION,
            sql_template="""
SELECT {source_table}.{source_column}
FROM {source_table}
LEFT JOIN {target_table} ON {source_table}.{source_column} = {target_table}.{target_column}
WHERE {target_table}.{target_column} IS NULL
  AND {source_table}.{source_column} IS NOT NULL
""",
            description="Validate FK referential integrity",
            parameters=["source_table", "source_column", "target_table", "target_column"],
        ),
        
        "pk_uniqueness": RuleTemplate(
            name="pk_uniqueness",
            template_type=RuleTemplateType.VALIDATION,
            sql_template="""
SELECT {column}, COUNT(*) as cnt
FROM {table}
GROUP BY {column}
HAVING COUNT(*) > 1
""",
            description="Validate PK uniqueness",
            parameters=["table", "column"],
        ),
        
        "not_null_check": RuleTemplate(
            name="not_null_check",
            template_type=RuleTemplateType.VALIDATION,
            sql_template="""
SELECT COUNT(*) as null_count
FROM {table}
WHERE {column} IS NULL
""",
            description="Check for NULL values",
            parameters=["table", "column"],
        ),
        
        "table_row_count": RuleTemplate(
            name="table_row_count",
            template_type=RuleTemplateType.AGGREGATION,
            sql_template="""
SELECT '{table}' as table_name, COUNT(*) as row_count
FROM {table}
""",
            description="Count rows in table",
            parameters=["table"],
        ),
        
        "fk_coverage": RuleTemplate(
            name="fk_coverage",
            template_type=RuleTemplateType.AGGREGATION,
            sql_template="""
SELECT 
    COUNT(DISTINCT {source_table}.{source_column}) as source_values,
    COUNT(DISTINCT {target_table}.{target_column}) as target_values,
    CAST(COUNT(DISTINCT {source_table}.{source_column}) AS FLOAT) / 
        NULLIF(COUNT(DISTINCT {target_table}.{target_column}), 0) as coverage_ratio
FROM {source_table}
CROSS JOIN {target_table}
""",
            description="Calculate FK coverage ratio",
            parameters=["source_table", "source_column", "target_table", "target_column"],
        ),
        
        "dimension_completeness": RuleTemplate(
            name="dimension_completeness",
            template_type=RuleTemplateType.VALIDATION,
            sql_template="""
SELECT 
    '{table}' as dimension_table,
    SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) as null_count,
    COUNT(*) as total_count,
    ROUND(100.0 * SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as null_percentage
FROM {table}
""",
            description="Check dimension table completeness",
            parameters=["table", "column"],
        ),
    }
    
    def __init__(self, neo4j_client=None):
        self.neo4j = neo4j_client
        self.generated_rules: List[GeneratedRule] = []
        self.templates = dict(self.TEMPLATES)
        
    def add_template(self, template: RuleTemplate):
        """Add a custom template."""
        self.templates[template.name] = template
    
    def generate_from_learned_rules(
        self,
        learned_rules: List[LearnedRule],
    ) -> List[GeneratedRule]:
        """Generate business rules from learned rules."""
        print(f"[RuleGenerator] Generating from {len(learned_rules)} learned rules...")
        
        generated = []
        
        for rule in learned_rules:
            gen_rules = self._generate_from_rule(rule)
            generated.extend(gen_rules)
        
        self.generated_rules.extend(generated)
        print(f"[RuleGenerator] Generated {len(generated)} business rules")
        
        return generated
    
    def _generate_from_rule(self, rule: LearnedRule) -> List[GeneratedRule]:
        """Generate rules from a single learned rule."""
        generated = []
        
        if "FK" in rule.formula or rule.source_predicate == "FK":
            # Generate FK validation rules
            fk_rules = self._generate_fk_rules(rule)
            generated.extend(fk_rules)
        
        if "IsPK" in rule.formula or rule.source_predicate == "IsPK":
            # Generate PK validation rules
            pk_rules = self._generate_pk_rules(rule)
            generated.extend(pk_rules)
        
        if "IsFact" in rule.formula or "IsDimension" in rule.formula:
            # Generate table classification rules
            table_rules = self._generate_table_rules(rule)
            generated.extend(table_rules)
        
        return generated
    
    def _generate_fk_rules(self, rule: LearnedRule) -> List[GeneratedRule]:
        """Generate FK-related business rules."""
        rules = []
        
        if self.neo4j is None:
            return rules
        
        # Get FK relationships from Neo4j
        fks = self.neo4j.run_query("""
            MATCH (sc:Column)-[:FK_TO]->(tc:Column)
            MATCH (sc)-[:BELONGS_TO]->(st:Table)
            MATCH (tc)-[:BELONGS_TO]->(tt:Table)
            RETURN st.name as source_table, sc.name as source_column,
                   tt.name as target_table, tc.name as target_column
        """)
        
        if not fks:
            return rules
        
        for fk in fks:
            # FK validation rule
            template = self.templates["fk_validation"]
            sql = template.render(
                source_table=fk["source_table"],
                source_column=fk["source_column"],
                target_table=fk["target_table"],
                target_column=fk["target_column"],
            )
            
            rules.append(GeneratedRule(
                name=f"validate_fk_{fk['source_table']}_{fk['source_column']}",
                description=f"Validate {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}",
                rule_type="validation",
                sql_query=sql.strip(),
                inputs=[fk["source_table"], fk["target_table"]],
                outputs=[],
                source_rule=rule,
                confidence=rule.confidence,
                tags=["fk", "validation", "referential_integrity"],
            ))
        
        return rules
    
    def _generate_pk_rules(self, rule: LearnedRule) -> List[GeneratedRule]:
        """Generate PK-related business rules."""
        rules = []
        
        if self.neo4j is None:
            return rules
        
        # Get PK columns from Neo4j
        pks = self.neo4j.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE c.is_primary_key = true
            RETURN t.name as table_name, c.name as column_name
        """)
        
        if not pks:
            return rules
        
        for pk in pks:
            # PK uniqueness rule
            template = self.templates["pk_uniqueness"]
            sql = template.render(
                table=pk["table_name"],
                column=pk["column_name"],
            )
            
            rules.append(GeneratedRule(
                name=f"validate_pk_unique_{pk['table_name']}",
                description=f"Validate {pk['table_name']}.{pk['column_name']} is unique",
                rule_type="validation",
                sql_query=sql.strip(),
                inputs=[pk["table_name"]],
                outputs=[],
                source_rule=rule,
                tags=["pk", "validation", "uniqueness"],
            ))
            
            # PK not null rule
            template = self.templates["not_null_check"]
            sql = template.render(
                table=pk["table_name"],
                column=pk["column_name"],
            )
            
            rules.append(GeneratedRule(
                name=f"validate_pk_not_null_{pk['table_name']}",
                description=f"Validate {pk['table_name']}.{pk['column_name']} is not null",
                rule_type="validation",
                sql_query=sql.strip(),
                inputs=[pk["table_name"]],
                outputs=[],
                source_rule=rule,
                tags=["pk", "validation", "not_null"],
            ))
        
        return rules
    
    def _generate_table_rules(self, rule: LearnedRule) -> List[GeneratedRule]:
        """Generate table-level business rules."""
        rules = []
        
        if self.neo4j is None:
            return rules
        
        # Get all tables
        tables = self.neo4j.run_query("MATCH (t:Table) RETURN t.name as name")
        
        if not tables:
            return rules
        
        for table in tables:
            # Row count rule
            template = self.templates["table_row_count"]
            sql = template.render(table=table["name"])
            
            rules.append(GeneratedRule(
                name=f"count_rows_{table['name']}",
                description=f"Count rows in {table['name']}",
                rule_type="aggregation",
                sql_query=sql.strip(),
                inputs=[table["name"]],
                outputs=[f"{table['name']}_count"],
                source_rule=rule,
                tags=["aggregation", "row_count"],
            ))
        
        return rules
    
    def generate_all_rules(self) -> List[GeneratedRule]:
        """Generate all possible business rules from Neo4j graph."""
        print("[RuleGenerator] Generating all business rules...")
        
        all_rules = []
        
        # Generate FK validation rules
        all_rules.extend(self._generate_all_fk_validations())
        
        # Generate PK validation rules
        all_rules.extend(self._generate_all_pk_validations())
        
        # Generate row count rules
        all_rules.extend(self._generate_all_row_counts())
        
        self.generated_rules = all_rules
        print(f"[RuleGenerator] Generated {len(all_rules)} total rules")
        
        return all_rules
    
    def _generate_all_fk_validations(self) -> List[GeneratedRule]:
        """Generate FK validation rules for all FKs."""
        if self.neo4j is None:
            return []
        
        fks = self.neo4j.run_query("""
            MATCH (sc:Column)-[:FK_TO]->(tc:Column)
            MATCH (sc)-[:BELONGS_TO]->(st:Table)
            MATCH (tc)-[:BELONGS_TO]->(tt:Table)
            RETURN st.name as source_table, sc.name as source_column,
                   tt.name as target_table, tc.name as target_column
        """)
        
        if not fks:
            return []
        
        rules = []
        for fk in fks:
            template = self.templates["fk_validation"]
            sql = template.render(
                source_table=fk["source_table"],
                source_column=fk["source_column"],
                target_table=fk["target_table"],
                target_column=fk["target_column"],
            )
            
            rules.append(GeneratedRule(
                name=f"fk_integrity_{fk['source_table']}_{fk['source_column']}",
                description=f"Check {fk['source_table']}.{fk['source_column']} references valid {fk['target_table']}.{fk['target_column']}",
                rule_type="validation",
                sql_query=sql.strip(),
                inputs=[fk["source_table"], fk["target_table"]],
                outputs=[],
                tags=["fk", "integrity"],
            ))
        
        return rules
    
    def _generate_all_pk_validations(self) -> List[GeneratedRule]:
        """Generate PK validation rules."""
        if self.neo4j is None:
            return []
        
        pks = self.neo4j.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE c.is_primary_key = true OR c.name = 'id'
            RETURN DISTINCT t.name as table_name, c.name as column_name
        """)
        
        if not pks:
            return []
        
        rules = []
        for pk in pks:
            # Uniqueness check
            template = self.templates["pk_uniqueness"]
            sql = template.render(table=pk["table_name"], column=pk["column_name"])
            
            rules.append(GeneratedRule(
                name=f"pk_unique_{pk['table_name']}",
                description=f"Ensure {pk['table_name']}.{pk['column_name']} values are unique",
                rule_type="validation",
                sql_query=sql.strip(),
                inputs=[pk["table_name"]],
                outputs=[],
                tags=["pk", "uniqueness"],
            ))
        
        return rules
    
    def _generate_all_row_counts(self) -> List[GeneratedRule]:
        """Generate row count rules for all tables."""
        if self.neo4j is None:
            return []
        
        tables = self.neo4j.run_query("MATCH (t:Table) RETURN t.name as name")
        
        if not tables:
            return []
        
        rules = []
        for table in tables:
            template = self.templates["table_row_count"]
            sql = template.render(table=table["name"])
            
            rules.append(GeneratedRule(
                name=f"row_count_{table['name']}",
                description=f"Get row count for {table['name']}",
                rule_type="aggregation",
                sql_query=sql.strip(),
                inputs=[table["name"]],
                outputs=[f"{table['name']}_row_count"],
                tags=["aggregation"],
            ))
        
        return rules
    
    def export_yaml(self, rules: Optional[List[GeneratedRule]] = None) -> str:
        """Export rules as YAML."""
        import yaml
        
        rules = rules or self.generated_rules
        
        rules_data = {
            "version": "1.0",
            "namespace": "generated",
            "rules": [r.to_yaml_dict() for r in rules],
        }
        
        return yaml.dump(rules_data, default_flow_style=False, sort_keys=False)
    
    def export_sql(self, rules: Optional[List[GeneratedRule]] = None) -> str:
        """Export rules as SQL script."""
        rules = rules or self.generated_rules
        
        lines = [
            "-- Generated Business Rules",
            "-- ========================",
            "",
        ]
        
        for rule in rules:
            lines.append(f"-- Rule: {rule.name}")
            lines.append(f"-- Description: {rule.description}")
            lines.append(f"-- Type: {rule.rule_type}")
            lines.append(rule.sql_query)
            lines.append("")
            lines.append("-- ---")
            lines.append("")
        
        return "\n".join(lines)