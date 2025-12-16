"""
LTN Knowledge Base - Store and manage logical axioms and constraints.

The knowledge base contains:
- Axioms: Facts that are known to be true
- Constraints: Rules that should be satisfied
- Queries: Logical questions to answer
"""
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import ltn
    import tensorflow as tf
    LTN_AVAILABLE = True
except ImportError:
    LTN_AVAILABLE = False
    tf = None
    ltn = None


class AxiomType(Enum):
    """Types of axioms."""
    FACT = "fact"  # Ground truth from data
    RULE = "rule"  # Logical implication
    CONSTRAINT = "constraint"  # Must be satisfied
    SOFT_CONSTRAINT = "soft_constraint"  # Should be satisfied


@dataclass
class Axiom:
    """A logical axiom in the knowledge base."""
    name: str
    formula: str  # Human-readable formula
    axiom_type: AxiomType
    weight: float = 1.0
    description: str = ""
    ltn_formula: Any = None  # LTN formula object
    
    def __str__(self):
        return f"{self.name}: {self.formula}"


@dataclass
class Constraint:
    """A constraint that must be satisfied."""
    name: str
    formula: str
    penalty: float = 1.0
    is_hard: bool = False
    description: str = ""


class LTNKnowledgeBase:
    """Knowledge base for LTN reasoning."""
    
    def __init__(self):
        self.axioms: Dict[str, Axiom] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.constants: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.predicates: Dict[str, Any] = {}
        self.functions: Dict[str, Any] = {}
        
        # Fuzzy operators
        if LTN_AVAILABLE:
            self.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
            self.And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
            self.Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
            self.Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
            self.Equiv = ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv())
            self.Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
            self.Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=2), semantics="exists")
    
    def add_constant(self, name: str, value: np.ndarray) -> Any:
        """Add a constant (grounded entity)."""
        if not LTN_AVAILABLE:
            raise ImportError("LTN not installed")
        const = ltn.Constant(value, trainable=False)
        self.constants[name] = const
        return const
    
    def add_variable(self, name: str, values: np.ndarray) -> Any:
        """Add a variable (set of entities)."""
        if not LTN_AVAILABLE:
            raise ImportError("LTN not installed")
        var = ltn.Variable(name, values)
        self.variables[name] = var
        return var
    
    def add_predicate(self, name: str, predicate: Any):
        """Add a predicate."""
        self.predicates[name] = predicate
    
    def add_axiom(self, axiom: Axiom):
        """Add an axiom to the knowledge base."""
        self.axioms[axiom.name] = axiom
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint."""
        self.constraints[constraint.name] = constraint
    
    def add_fk_axioms(self):
        """Add standard FK-related axioms."""
        axioms = [
            Axiom(
                name="pk_unique",
                formula="∀x(IsPK(x) → IsUnique(x))",
                axiom_type=AxiomType.RULE,
                description="Primary keys are always unique",
            ),
            Axiom(
                name="fk_references_pk",
                formula="∀x,y(FK(x,y) → IsPK(y))",
                axiom_type=AxiomType.RULE,
                description="Foreign keys reference primary keys",
            ),
            Axiom(
                name="fk_same_type",
                formula="∀x,y(FK(x,y) → SameType(x,y))",
                axiom_type=AxiomType.RULE,
                description="FK and PK must have compatible types",
            ),
            Axiom(
                name="dimension_has_pk",
                formula="∀t(IsDimension(t) → ∃c(BelongsTo(c,t) ∧ IsPK(c)))",
                axiom_type=AxiomType.RULE,
                description="Dimension tables have primary keys",
            ),
            Axiom(
                name="fact_has_fks",
                formula="∀t(IsFact(t) → ∃c(BelongsTo(c,t) ∧ IsFK(c)))",
                axiom_type=AxiomType.RULE,
                description="Fact tables have foreign keys",
            ),
            Axiom(
                name="junction_two_fks",
                formula="∀t(IsJunction(t) → ∃c1,c2(BelongsTo(c1,t) ∧ BelongsTo(c2,t) ∧ IsFK(c1) ∧ IsFK(c2) ∧ c1≠c2))",
                axiom_type=AxiomType.RULE,
                description="Junction tables have at least two foreign keys",
            ),
        ]
        
        for axiom in axioms:
            self.add_axiom(axiom)
    
    def add_data_quality_axioms(self):
        """Add data quality axioms."""
        axioms = [
            Axiom(
                name="pk_not_null",
                formula="∀x(IsPK(x) → ¬IsNullable(x))",
                axiom_type=AxiomType.CONSTRAINT,
                description="Primary keys cannot be null",
            ),
            Axiom(
                name="fk_referential_integrity",
                formula="∀x,y(FK(x,y) → References(x,y))",
                axiom_type=AxiomType.CONSTRAINT,
                description="FK values must exist in referenced table",
            ),
        ]
        
        for axiom in axioms:
            self.add_axiom(axiom)
    
    def get_all_axioms(self) -> List[Axiom]:
        """Get all axioms."""
        return list(self.axioms.values())
    
    def get_axioms_by_type(self, axiom_type: AxiomType) -> List[Axiom]:
        """Get axioms of a specific type."""
        return [a for a in self.axioms.values() if a.axiom_type == axiom_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export knowledge base to dictionary."""
        return {
            "axioms": [
                {
                    "name": a.name,
                    "formula": a.formula,
                    "type": a.axiom_type.value,
                    "weight": a.weight,
                    "description": a.description,
                }
                for a in self.axioms.values()
            ],
            "constraints": [
                {
                    "name": c.name,
                    "formula": c.formula,
                    "penalty": c.penalty,
                    "is_hard": c.is_hard,
                }
                for c in self.constraints.values()
            ],
            "predicates": list(self.predicates.keys()),
        }
    
    @classmethod
    def create_default(cls) -> "LTNKnowledgeBase":
        """Create knowledge base with default axioms."""
        kb = cls()
        kb.add_fk_axioms()
        kb.add_data_quality_axioms()
        return kb