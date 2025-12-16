"""
LTN Rule Learner - Learn logical rules from Neo4j knowledge graph.

Uses graph structure and embeddings to learn predicate groundings.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    import ltn
    import tensorflow as tf
    LTN_AVAILABLE = True
except ImportError:
    LTN_AVAILABLE = False
    tf = None
    ltn = None

from .predicates import LTNPredicateFactory
from .knowledge_base import LTNKnowledgeBase, Axiom, AxiomType
from .trainer import LTNTrainer, TrainingConfig, TrainingResult


@dataclass
class RuleLearningConfig:
    """Configuration for rule learning."""
    embedding_dim: int = 128
    use_text_embeddings: bool = True
    use_kg_embeddings: bool = True
    min_confidence: float = 0.7
    max_rules: int = 50
    training_epochs: int = 100
    learning_rate: float = 0.001


@dataclass
class LearnedRule:
    """A rule learned from data."""
    name: str
    formula: str
    confidence: float
    support: int  # Number of examples supporting the rule
    rule_type: str
    source_predicate: str
    target_predicate: Optional[str] = None
    description: str = ""
    
    def to_sql(self) -> Optional[str]:
        """Convert rule to SQL constraint if possible."""
        # Basic conversion for common patterns
        if "FK" in self.formula and "References" in self.formula:
            return f"-- FK constraint: {self.formula}"
        return None
    
    def to_yaml(self) -> Dict[str, Any]:
        """Convert to YAML-compatible dict for business rules."""
        return {
            "name": self.name,
            "description": self.description or self.formula,
            "type": "validation",
            "formula": self.formula,
            "confidence": self.confidence,
        }


class LTNRuleLearner:
    """Learn logical rules from Neo4j knowledge graph."""
    
    def __init__(
        self,
        neo4j_client,
        config: Optional[RuleLearningConfig] = None,
    ):
        self.neo4j = neo4j_client
        self.config = config or RuleLearningConfig()
        self.predicate_factory = LTNPredicateFactory(self.config.embedding_dim)
        self.kb = LTNKnowledgeBase.create_default()
        self.learned_rules: List[LearnedRule] = []
        
        # Entity embeddings
        self.table_embeddings: Dict[str, np.ndarray] = {}
        self.column_embeddings: Dict[str, np.ndarray] = {}
        
    def load_embeddings_from_neo4j(self) -> Dict[str, int]:
        """Load embeddings from Neo4j graph."""
        print("[LTNRuleLearner] Loading embeddings from Neo4j...")
        stats = {"tables": 0, "columns": 0}
        
        # Load table embeddings
        if self.config.use_text_embeddings:
            tables = self.neo4j.run_query("""
                MATCH (t:Table)
                WHERE t.text_embedding IS NOT NULL
                RETURN t.name as name, t.text_embedding as embedding
            """)
            
            if tables:
                for row in tables:
                    name = row["name"]
                    emb = row["embedding"]
                    if emb:
                        self.table_embeddings[name] = np.array(emb, dtype=np.float32)
                        stats["tables"] += 1
        
        # Load column embeddings
        columns = self.neo4j.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            WHERE c.text_embedding IS NOT NULL
            RETURN t.name + '.' + c.name as name, c.text_embedding as embedding
        """)
        
        if columns:
            for row in columns:
                name = row["name"]
                emb = row["embedding"]
                if emb:
                    self.column_embeddings[name] = np.array(emb, dtype=np.float32)
                    stats["columns"] += 1
        
        print(f"[LTNRuleLearner] Loaded {stats['tables']} table, {stats['columns']} column embeddings")
        return stats
    
    def extract_training_data(self) -> Dict[str, Any]:
        """Extract training data from Neo4j."""
        print("[LTNRuleLearner] Extracting training data...")
        
        data = {
            "tables": [],
            "columns": [],
            "fk_pairs": [],
            "pk_columns": [],
            "table_column_pairs": [],
        }
        
        # Get all tables
        tables = self.neo4j.run_query("MATCH (t:Table) RETURN t.name as name")
        if tables:
            data["tables"] = [r["name"] for r in tables]
        
        # Get all columns with their tables
        columns = self.neo4j.run_query("""
            MATCH (c:Column)-[:BELONGS_TO]->(t:Table)
            RETURN t.name as table, c.name as column, c.is_primary_key as is_pk
        """)
        if columns:
            for row in columns:
                col_name = f"{row['table']}.{row['column']}"
                data["columns"].append(col_name)
                data["table_column_pairs"].append((row["table"], col_name))
                if row.get("is_pk"):
                    data["pk_columns"].append(col_name)
        
        # Get FK relationships
        fks = self.neo4j.run_query("""
            MATCH (sc:Column)-[:FK_TO]->(tc:Column)
            MATCH (sc)-[:BELONGS_TO]->(st:Table)
            MATCH (tc)-[:BELONGS_TO]->(tt:Table)
            RETURN st.name + '.' + sc.name as source, tt.name + '.' + tc.name as target
        """)
        if fks:
            data["fk_pairs"] = [(r["source"], r["target"]) for r in fks]
        
        print(f"[LTNRuleLearner] Extracted: {len(data['tables'])} tables, "
              f"{len(data['columns'])} columns, {len(data['fk_pairs'])} FKs")
        
        return data
    
    def prepare_ltn_data(self, training_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert training data to LTN format."""
        if not LTN_AVAILABLE:
            return {}
        
        ltn_data = {}
        
        # Table embeddings
        table_embs = []
        for table in training_data["tables"]:
            if table in self.table_embeddings:
                table_embs.append(self.table_embeddings[table])
            else:
                # Random embedding if not found
                table_embs.append(np.random.randn(self.config.embedding_dim).astype(np.float32))
        
        if table_embs:
            ltn_data["tables"] = np.stack(table_embs)
        
        # Column embeddings
        col_embs = []
        for col in training_data["columns"]:
            if col in self.column_embeddings:
                col_embs.append(self.column_embeddings[col])
            else:
                col_embs.append(np.random.randn(self.config.embedding_dim).astype(np.float32))
        
        if col_embs:
            ltn_data["columns"] = np.stack(col_embs)
        
        # FK labels (for supervised learning)
        if training_data["fk_pairs"]:
            col_to_idx = {c: i for i, c in enumerate(training_data["columns"])}
            fk_matrix = np.zeros((len(training_data["columns"]), len(training_data["columns"])))
            
            for src, tgt in training_data["fk_pairs"]:
                if src in col_to_idx and tgt in col_to_idx:
                    fk_matrix[col_to_idx[src], col_to_idx[tgt]] = 1.0
            
            ltn_data["fk_labels"] = fk_matrix
        
        return ltn_data
    
    def learn_rules(self) -> List[LearnedRule]:
        """Learn rules from the knowledge graph."""
        print("[LTNRuleLearner] Starting rule learning...")
        
        # Load embeddings
        self.load_embeddings_from_neo4j()
        
        # Extract training data
        training_data = self.extract_training_data()
        
        # Learn FK rules
        fk_rules = self._learn_fk_rules(training_data)
        self.learned_rules.extend(fk_rules)
        
        # Learn table classification rules
        table_rules = self._learn_table_rules(training_data)
        self.learned_rules.extend(table_rules)
        
        # Learn column pattern rules
        column_rules = self._learn_column_rules(training_data)
        self.learned_rules.extend(column_rules)
        
        print(f"[LTNRuleLearner] Learned {len(self.learned_rules)} rules")
        
        return self.learned_rules
    
    def _learn_fk_rules(self, training_data: Dict[str, Any]) -> List[LearnedRule]:
        """Learn FK-related rules."""
        rules = []
        
        fk_pairs = training_data.get("fk_pairs", [])
        pk_columns = training_data.get("pk_columns", [])
        
        if fk_pairs:
            # Rule: FK columns reference PK columns
            fk_to_pk_count = sum(1 for _, tgt in fk_pairs if tgt in pk_columns)
            confidence = fk_to_pk_count / len(fk_pairs) if fk_pairs else 0
            
            if confidence >= self.config.min_confidence:
                rules.append(LearnedRule(
                    name="fk_references_pk",
                    formula="∀x,y(FK(x,y) → IsPK(y))",
                    confidence=confidence,
                    support=fk_to_pk_count,
                    rule_type="implication",
                    source_predicate="FK",
                    target_predicate="IsPK",
                    description="Foreign keys reference primary keys",
                ))
            
            # Analyze FK naming patterns
            id_suffix_count = sum(1 for src, _ in fk_pairs if src.endswith("_id") or src.endswith(".id"))
            if id_suffix_count > 0:
                confidence = id_suffix_count / len(fk_pairs)
                if confidence >= self.config.min_confidence:
                    rules.append(LearnedRule(
                        name="fk_naming_pattern",
                        formula="∀x(IsFK(x) → HasIdSuffix(x))",
                        confidence=confidence,
                        support=id_suffix_count,
                        rule_type="pattern",
                        source_predicate="IsFK",
                        description="FK columns typically end with '_id'",
                    ))
        
        return rules
    
    def _learn_table_rules(self, training_data: Dict[str, Any]) -> List[LearnedRule]:
        """Learn table classification rules."""
        rules = []
        
        tables = training_data.get("tables", [])
        fk_pairs = training_data.get("fk_pairs", [])
        
        if not tables:
            return rules
        
        # Count FKs per table
        fk_counts = {}
        referenced_counts = {}
        
        for src, tgt in fk_pairs:
            src_table = src.split(".")[0]
            tgt_table = tgt.split(".")[0]
            fk_counts[src_table] = fk_counts.get(src_table, 0) + 1
            referenced_counts[tgt_table] = referenced_counts.get(tgt_table, 0) + 1
        
        # Identify fact tables (many FKs, few references)
        fact_tables = [t for t in tables if fk_counts.get(t, 0) >= 2 and referenced_counts.get(t, 0) <= 1]
        
        if fact_tables:
            rules.append(LearnedRule(
                name="fact_table_pattern",
                formula="∀t(IsFact(t) ↔ (FKCount(t) ≥ 2 ∧ RefCount(t) ≤ 1))",
                confidence=0.9,
                support=len(fact_tables),
                rule_type="definition",
                source_predicate="IsFact",
                description=f"Fact tables have multiple FKs: {fact_tables}",
            ))
        
        # Identify dimension tables (few FKs, many references)
        dim_tables = [t for t in tables if fk_counts.get(t, 0) <= 1 and referenced_counts.get(t, 0) >= 1]
        
        if dim_tables:
            rules.append(LearnedRule(
                name="dimension_table_pattern",
                formula="∀t(IsDimension(t) ↔ (FKCount(t) ≤ 1 ∧ RefCount(t) ≥ 1))",
                confidence=0.9,
                support=len(dim_tables),
                rule_type="definition",
                source_predicate="IsDimension",
                description=f"Dimension tables are referenced: {dim_tables}",
            ))
        
        # Identify junction tables (2+ FKs, forms composite PK)
        junction_tables = [t for t in tables if fk_counts.get(t, 0) >= 2]
        
        if junction_tables:
            rules.append(LearnedRule(
                name="junction_table_pattern",
                formula="∀t(IsJunction(t) ↔ FKCount(t) ≥ 2)",
                confidence=0.85,
                support=len(junction_tables),
                rule_type="definition",
                source_predicate="IsJunction",
                description=f"Junction tables have multiple FKs: {junction_tables}",
            ))
        
        return rules
    
    def _learn_column_rules(self, training_data: Dict[str, Any]) -> List[LearnedRule]:
        """Learn column classification rules."""
        rules = []
        
        columns = training_data.get("columns", [])
        pk_columns = training_data.get("pk_columns", [])
        
        if not columns:
            return rules
        
        # ID column pattern
        id_columns = [c for c in columns if c.endswith(".id") or "_id" in c]
        if id_columns:
            rules.append(LearnedRule(
                name="id_column_pattern",
                formula="∀c(HasIdName(c) → (IsPK(c) ∨ IsFK(c)))",
                confidence=0.95,
                support=len(id_columns),
                rule_type="pattern",
                source_predicate="HasIdName",
                description="Columns with 'id' are usually PK or FK",
            ))
        
        # Timestamp column pattern
        ts_columns = [c for c in columns if any(x in c.lower() for x in ["created", "updated", "timestamp", "date"])]
        if ts_columns:
            rules.append(LearnedRule(
                name="timestamp_column_pattern",
                formula="∀c(HasTimestampName(c) → IsTemporal(c))",
                confidence=0.9,
                support=len(ts_columns),
                rule_type="pattern",
                source_predicate="HasTimestampName",
                description="Columns with timestamp names are temporal",
            ))
        
        return rules
    
    def get_learned_rules(self) -> List[LearnedRule]:
        """Get all learned rules."""
        return self.learned_rules
    
    def export_rules_yaml(self) -> str:
        """Export learned rules as YAML business rules."""
        import yaml
        
        rules_data = {
            "version": "1.0",
            "namespace": "ltn_learned",
            "rules": [rule.to_yaml() for rule in self.learned_rules],
        }
        
        return yaml.dump(rules_data, default_flow_style=False)