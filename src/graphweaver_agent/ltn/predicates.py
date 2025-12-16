"""
LTN Predicates for GraphWeaver - Define learnable predicates for database schema.

Predicates represent properties and relationships that can be learned from data.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    import ltn
    import tensorflow as tf
    LTN_AVAILABLE = True
except ImportError:
    LTN_AVAILABLE = False
    tf = None
    ltn = None


class PredicateType(Enum):
    """Types of predicates."""
    UNARY = "unary"  # P(x) - property of single entity
    BINARY = "binary"  # R(x,y) - relationship between two entities
    TERNARY = "ternary"  # R(x,y,z) - three-way relationship


@dataclass
class PredicateDefinition:
    """Definition of a predicate."""
    name: str
    arity: int
    predicate_type: PredicateType
    description: str
    input_dims: List[int]
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = "elu"


class BasePredicate:
    """Base class for LTN predicates."""
    
    def __init__(self, definition: PredicateDefinition):
        self.definition = definition
        self.name = definition.name
        self.arity = definition.arity
        self._predicate = None
        self._model = None
        
    def build(self) -> Any:
        """Build the LTN predicate."""
        if not LTN_AVAILABLE:
            raise ImportError("LTN not installed. Run: pip install ltn")
        
        # Create neural network for predicate
        layers = []
        input_dim = sum(self.definition.input_dims)
        
        for hidden_dim in self.definition.hidden_dims:
            layers.append(tf.keras.layers.Dense(hidden_dim, activation=self.definition.activation))
        
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        self._model = tf.keras.Sequential(layers)
        self._predicate = ltn.Predicate(self._model)
        
        return self._predicate
    
    @property
    def predicate(self):
        if self._predicate is None:
            self.build()
        return self._predicate
    
    @property
    def model(self):
        return self._model
    
    def __call__(self, *args):
        """Call the predicate with arguments."""
        return self.predicate(*args)


class TablePredicate(BasePredicate):
    """Predicate for table properties: IsTable(x), IsFact(x), IsDimension(x)."""
    
    def __init__(self, name: str, embedding_dim: int = 128):
        definition = PredicateDefinition(
            name=name,
            arity=1,
            predicate_type=PredicateType.UNARY,
            description=f"Unary predicate {name} for table classification",
            input_dims=[embedding_dim],
            hidden_dims=[64, 32],
        )
        super().__init__(definition)


class ColumnPredicate(BasePredicate):
    """Predicate for column properties: IsPK(x), IsFK(x), IsNullable(x)."""
    
    def __init__(self, name: str, embedding_dim: int = 128):
        definition = PredicateDefinition(
            name=name,
            arity=1,
            predicate_type=PredicateType.UNARY,
            description=f"Unary predicate {name} for column classification",
            input_dims=[embedding_dim],
            hidden_dims=[64, 32],
        )
        super().__init__(definition)


class FKPredicate(BasePredicate):
    """Predicate for FK relationships: FK(source_col, target_col)."""
    
    def __init__(self, embedding_dim: int = 128):
        definition = PredicateDefinition(
            name="FK",
            arity=2,
            predicate_type=PredicateType.BINARY,
            description="Binary predicate for foreign key relationships",
            input_dims=[embedding_dim, embedding_dim],
            hidden_dims=[128, 64, 32],
        )
        super().__init__(definition)


class RelationshipPredicate(BasePredicate):
    """Generic relationship predicate: Rel(x, y)."""
    
    def __init__(self, name: str, embedding_dim: int = 128):
        definition = PredicateDefinition(
            name=name,
            arity=2,
            predicate_type=PredicateType.BINARY,
            description=f"Binary predicate {name} for entity relationships",
            input_dims=[embedding_dim, embedding_dim],
            hidden_dims=[128, 64, 32],
        )
        super().__init__(definition)


class LTNPredicateFactory:
    """Factory for creating LTN predicates."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.predicates: Dict[str, BasePredicate] = {}
        
    def create_table_predicates(self) -> Dict[str, TablePredicate]:
        """Create standard table predicates."""
        table_preds = {
            "IsTable": TablePredicate("IsTable", self.embedding_dim),
            "IsFact": TablePredicate("IsFact", self.embedding_dim),
            "IsDimension": TablePredicate("IsDimension", self.embedding_dim),
            "IsJunction": TablePredicate("IsJunction", self.embedding_dim),
            "HasPK": TablePredicate("HasPK", self.embedding_dim),
        }
        self.predicates.update(table_preds)
        return table_preds
    
    def create_column_predicates(self) -> Dict[str, ColumnPredicate]:
        """Create standard column predicates."""
        col_preds = {
            "IsPK": ColumnPredicate("IsPK", self.embedding_dim),
            "IsFK": ColumnPredicate("IsFK", self.embedding_dim),
            "IsNullable": ColumnPredicate("IsNullable", self.embedding_dim),
            "IsNumeric": ColumnPredicate("IsNumeric", self.embedding_dim),
            "IsIdentifier": ColumnPredicate("IsIdentifier", self.embedding_dim),
            "IsTemporal": ColumnPredicate("IsTemporal", self.embedding_dim),
        }
        self.predicates.update(col_preds)
        return col_preds
    
    def create_relationship_predicates(self) -> Dict[str, RelationshipPredicate]:
        """Create standard relationship predicates."""
        rel_preds = {
            "FK": FKPredicate(self.embedding_dim),
            "BelongsTo": RelationshipPredicate("BelongsTo", self.embedding_dim),
            "References": RelationshipPredicate("References", self.embedding_dim),
            "DependsOn": RelationshipPredicate("DependsOn", self.embedding_dim),
            "SameType": RelationshipPredicate("SameType", self.embedding_dim),
            "SimilarTo": RelationshipPredicate("SimilarTo", self.embedding_dim),
        }
        self.predicates.update(rel_preds)
        return rel_preds
    
    def create_all_predicates(self) -> Dict[str, BasePredicate]:
        """Create all standard predicates."""
        self.create_table_predicates()
        self.create_column_predicates()
        self.create_relationship_predicates()
        return self.predicates
    
    def get_predicate(self, name: str) -> Optional[BasePredicate]:
        """Get predicate by name."""
        return self.predicates.get(name)
    
    def build_all(self):
        """Build all predicates."""
        for pred in self.predicates.values():
            pred.build()