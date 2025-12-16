"""
LTN Trainer - Train predicates to satisfy logical constraints.

Uses satisfiability of formulas as training objective.
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

from .knowledge_base import LTNKnowledgeBase, Axiom, AxiomType


@dataclass
class TrainingConfig:
    """Configuration for LTN training."""
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    p_schedule: str = "linear"  # How to increase p for pMean
    p_start: float = 1.0
    p_end: float = 5.0
    early_stopping: bool = True
    patience: int = 10
    min_sat: float = 0.95  # Target satisfiability
    verbose: bool = True


@dataclass
class TrainingResult:
    """Result of LTN training."""
    final_sat: float
    epochs_trained: int
    sat_history: List[float]
    loss_history: List[float]
    predicate_accuracies: Dict[str, float]
    learned_rules: List[Dict[str, Any]]


class LTNTrainer:
    """Trainer for LTN predicates."""
    
    def __init__(
        self,
        knowledge_base: LTNKnowledgeBase,
        config: Optional[TrainingConfig] = None,
    ):
        if not LTN_AVAILABLE:
            raise ImportError("LTN not installed. Run: pip install ltn")
        
        self.kb = knowledge_base
        self.config = config or TrainingConfig()
        self.optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)
        self.sat_history = []
        self.loss_history = []
        
    def compute_satisfiability(self, formulas: List[Any]) -> tf.Tensor:
        """Compute overall satisfiability of formulas."""
        if not formulas:
            return tf.constant(1.0)
        
        sat_values = [f for f in formulas if f is not None]
        if not sat_values:
            return tf.constant(1.0)
        
        # Use pMean aggregation
        return ltn.fuzzy_ops.Aggreg_pMeanError(p=2)(sat_values)
    
    def build_formula(self, axiom: Axiom) -> Optional[Any]:
        """Build LTN formula from axiom."""
        # This is a simplified version - real implementation would parse the formula
        # For now, we'll build formulas programmatically
        return axiom.ltn_formula
    
    def train_step(self, training_data: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Single training step."""
        # Update variables with current batch
        for name, values in training_data.items():
            if name in self.kb.variables:
                self.kb.variables[name] = ltn.Variable(name, values)
        
        with tf.GradientTape() as tape:
            # Compute satisfiability of all axioms
            formulas = []
            for axiom in self.kb.get_all_axioms():
                formula = self.build_formula(axiom)
                if formula is not None:
                    formulas.append(formula * axiom.weight)
            
            sat = self.compute_satisfiability(formulas)
            loss = 1.0 - sat
        
        # Get trainable variables from all predicates
        trainable_vars = []
        for pred in self.kb.predicates.values():
            if hasattr(pred, 'model') and pred.model is not None:
                trainable_vars.extend(pred.model.trainable_variables)
        
        if trainable_vars:
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return float(sat), float(loss)
    
    def train(
        self,
        training_data: Dict[str, np.ndarray],
        validation_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> TrainingResult:
        """Train the LTN model."""
        print("[LTNTrainer] Starting training...")
        
        best_sat = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            sat, loss = self.train_step(training_data)
            self.sat_history.append(sat)
            self.loss_history.append(loss)
            
            if self.config.verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}: SAT={sat:.4f}, Loss={loss:.4f}")
            
            # Early stopping
            if sat > best_sat:
                best_sat = sat
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.config.early_stopping and patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch}")
                break
            
            if sat >= self.config.min_sat:
                print(f"  Target satisfiability reached at epoch {epoch}")
                break
        
        # Compute predicate accuracies
        pred_accuracies = self._compute_predicate_accuracies(validation_data or training_data)
        
        # Extract learned rules
        learned_rules = self._extract_learned_rules()
        
        return TrainingResult(
            final_sat=self.sat_history[-1] if self.sat_history else 0.0,
            epochs_trained=len(self.sat_history),
            sat_history=self.sat_history,
            loss_history=self.loss_history,
            predicate_accuracies=pred_accuracies,
            learned_rules=learned_rules,
        )
    
    def _compute_predicate_accuracies(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute accuracy for each predicate."""
        accuracies = {}
        
        for name, pred in self.kb.predicates.items():
            # This would need ground truth labels to compute actual accuracy
            # For now, return satisfiability as proxy
            accuracies[name] = 0.0
        
        return accuracies
    
    def _extract_learned_rules(self) -> List[Dict[str, Any]]:
        """Extract rules learned by the model."""
        rules = []
        
        for axiom in self.kb.get_all_axioms():
            rules.append({
                "name": axiom.name,
                "formula": axiom.formula,
                "type": axiom.axiom_type.value,
                "weight": axiom.weight,
            })
        
        return rules