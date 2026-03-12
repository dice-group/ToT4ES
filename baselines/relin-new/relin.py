"""
RELIN: Relatedness and Informativeness-Based Centrality for Entity Summarization

This module implements the RELIN algorithm as described in:
"Relatedness and Informativeness-Based Centrality for Entity Summarization"
by G. Cheng, T. Tran, and Y. Qu
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import math


class Feature:
    """Represents a property-value pair (feature) of an entity."""
    
    def __init__(self, prop: str, value: str):
        """
        Initialize a feature.
        
        Args:
            prop: Property name
            value: Value (can be entity or literal)
        """
        self.prop = prop
        self.value = value
    
    def __repr__(self):
        return f"Feature({self.prop}, {self.value})"
    
    def __eq__(self, other):
        if not isinstance(other, Feature):
            return False
        return self.prop == other.prop and self.value == other.value
    
    def __hash__(self):
        return hash((self.prop, self.value))


class Entity:
    """Represents an entity in the data graph."""
    
    def __init__(self, entity_id: str):
        """
        Initialize an entity.
        
        Args:
            entity_id: Unique identifier for the entity
        """
        self.entity_id = entity_id
        self.features: Set[Feature] = set()
    
    def add_feature(self, prop: str, value: str):
        """Add a feature (property-value pair) to the entity."""
        self.features.add(Feature(prop, value))
    
    def get_features(self) -> List[Feature]:
        """Get all features of the entity."""
        return list(self.features)
    
    def __repr__(self):
        return f"Entity({self.entity_id})"


class RelatednessMeasure:
    """Computes relatedness between phrases using co-occurrence statistics."""
    
    def __init__(self):
        """Initialize the relatedness measure with phrase statistics."""
        self.phrase_stats = defaultdict(lambda: {"count": 0, "co_occurrences": {}})
        self.total_docs = int(1e7)  # Default normalizing constant N
    
    def add_occurrence(self, phrase: str, count: int = 1):
        """Record occurrence of a phrase."""
        self.phrase_stats[phrase]["count"] += count
    
    def add_co_occurrence(self, phrase1: str, phrase2: str, count: int = 1):
        """Record co-occurrence of two phrases (stored bidirectionally)."""
        if phrase2 not in self.phrase_stats[phrase1]["co_occurrences"]:
            self.phrase_stats[phrase1]["co_occurrences"][phrase2] = 0
        self.phrase_stats[phrase1]["co_occurrences"][phrase2] += count
        # Store reverse direction for symmetric PMI lookup
        if phrase1 not in self.phrase_stats[phrase2]["co_occurrences"]:
            self.phrase_stats[phrase2]["co_occurrences"][phrase1] = 0
        self.phrase_stats[phrase2]["co_occurrences"][phrase1] += count
    
    def pmi(self, phrase1: str, phrase2: str, total_docs: int = None) -> float:
        """
        Compute Pointwise Mutual Information (PMI) between two phrases.
        
        PMI(s1, s2) = log(P(s1, s2) / (P(s1) * P(s2)))
        
        Args:
            phrase1: First phrase
            phrase2: Second phrase
            total_docs: Total number of documents (normalization constant).
                       If None, uses the stored value from training.
        
        Returns:
            PMI value (non-negative)
        """
        if total_docs is None:
            total_docs = self.total_docs
        
        # Estimate probabilities: P(si) = Hits(si) / N
        p_phrase1 = self.phrase_stats[phrase1]["count"] / total_docs
        p_phrase2 = self.phrase_stats[phrase2]["count"] / total_docs
        
        if phrase1 == phrase2:
            # Self-PMI: PMI(si, si) = log(P(si,si) / P(si)^2)
            # P(si,si) = P(si) (a doc containing si also contains si)
            # So self-PMI = log(1/P(si)) = -log(P(si))
            if p_phrase1 == 0:
                return 0.0
            return max(0, -math.log(p_phrase1))
        
        # Co-occurrence: P(si, sj) = Hits(si, sj) / N
        co_occur_count = self.phrase_stats[phrase1]["co_occurrences"].get(phrase2, 0)
        p_both = co_occur_count / total_docs
        
        # Avoid log(0)
        if p_phrase1 == 0 or p_phrase2 == 0 or p_both == 0:
            return 0.0
        
        # Paper Eq. 6: PMI(si, sj) = log(P(si,sj) / (P(si) * P(sj)))
        pmi_value = math.log(p_both / (p_phrase1 * p_phrase2))
        return max(0, pmi_value)  # Use non-negative PMI


class InformativenessCalculator:
    """Computes informativeness of features based on self-information."""
    
    def __init__(self):
        """Initialize the informativeness calculator."""
        self.entity_features: Dict[str, Set[Feature]] = defaultdict(set)
    
    def record_feature_set(self, entity_id: str, features: List[Feature]):
        """Record the feature set of an entity."""
        self.entity_features[entity_id] = set(features)
    
    def compute_conditional_probability(self, feature_target: Feature, 
                                       feature_condition: Feature) -> float:
        """
        Compute P(fp | fq) - probability that fp occurs given fq occurs.
        
        P(fp | fq) = |{e ∈ E | fp, fq ∈ FS(e)}| / |{e ∈ E | fq ∈ FS(e)}|
        
        Args:
            feature_target: Target feature fp
            feature_condition: Conditioning feature fq
        
        Returns:
            Conditional probability
        """
        # Count entities containing both features
        both_count = 0
        condition_count = 0
        
        for entity_id, features in self.entity_features.items():
            if feature_condition in features:
                condition_count += 1
                if feature_target in features:
                    both_count += 1
        
        if condition_count == 0:
            return 0.0
        
        return both_count / condition_count
    
    def compute_self_information(self, feature_target: Feature, 
                                feature_condition: Optional[Feature] = None) -> float:
        """
        Compute self-information of a feature.
        
        SelfInfo(fp | fq) = -log(P(fp | fq))
        or
        SelfInfo(fp) = -log(P(fp))
        
        Args:
            feature_target: Target feature
            feature_condition: Optional conditioning feature (if None, uses marginal probability)
        
        Returns:
            Self-information value (always non-negative)
        """
        if feature_condition is None:
            # Marginal probability
            count = sum(1 for features in self.entity_features.values() 
                       if feature_target in features)
            total = len(self.entity_features)
            if total == 0 or count == 0:
                return 0.0
            prob = count / total
        else:
            prob = self.compute_conditional_probability(feature_target, feature_condition)
        
        if prob == 0 or prob == 1:
            return 0.0
        
        return -math.log(prob)


class RELIN:
    """
    RELIN (Relatedness and Informativeness-Based Centrality) for Entity Summarization.
    
    A variant of the random surfer model that measures centrality based on both
    relatedness and informativeness of features.
    """
    
    def __init__(self, lambda_param: float = 0.85, iterations: int = 10,
                 use_conditional_informativeness: bool = True):
        """
        Initialize RELIN.
        
        Args:
            lambda_param: Parameter λ ∈ [0,1] controlling the balance between
                         relational moves (1-λ) and informational jumps (λ)
            iterations: Number of iterations for convergence
            use_conditional_informativeness: If True, use conditional informativeness
                                            P(fp|fq), else use marginal P(fp)
        """
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.use_conditional = use_conditional_informativeness
        self.relatedness_measure = RelatednessMeasure()
        self.informativeness_calc = InformativenessCalculator()
    
    def train_relatedness(self, corpus_stats: Dict[str, int], 
                         co_occurrence_stats: Dict[Tuple[str, str], int],
                         total_docs: int = int(1e7)):
        """
        Train the relatedness measure with corpus statistics.
        
        Args:
            corpus_stats: Dictionary mapping phrases to occurrence counts
            co_occurrence_stats: Dictionary mapping (phrase1, phrase2) to co-occurrence counts
            total_docs: Total number of documents in corpus
        """
        self.relatedness_measure.total_docs = total_docs
        for phrase, count in corpus_stats.items():
            self.relatedness_measure.add_occurrence(phrase, count)
        
        for (phrase1, phrase2), count in co_occurrence_stats.items():
            self.relatedness_measure.add_co_occurrence(phrase1, phrase2, count)
    
    def prepare_informativeness(self, entities: List[Entity]):
        """
        Prepare informativeness data from entities.
        
        Args:
            entities: List of entities with their feature sets
        """
        for entity in entities:
            features = entity.get_features()
            self.informativeness_calc.record_feature_set(entity.entity_id, features)
    
    def compute_relatedness_matrix(self, features: List[Feature]) -> np.ndarray:
        """
        Compute the relatedness matrix M between features.
        
        Mp,q = sqrt(Rel(Prop(fp), Prop(fq)) * Rel(Val(fp), Val(fq)))  (Paper Eq. 5)
        
        Args:
            features: List of features to compute relatedness for
        
        Returns:
            Relatedness matrix (normalized for stochasticity)
        """
        n = len(features)
        M = np.zeros((n, n))
        
        for i, fi in enumerate(features):
            for j, fj in enumerate(features):
                # Compute relatedness between properties
                prop_rel = self.relatedness_measure.pmi(fi.prop, fj.prop)
                # Compute relatedness between values
                val_rel = self.relatedness_measure.pmi(fi.value, fj.value)
                
                # Paper Eq. 5: M_p,q = sqrt(Rel(Prop(fp),Prop(fq)) * Rel(Val(fp),Val(fq)))
                M[i, j] = math.sqrt(prop_rel * val_rel)
        
        # Normalize each column to make it stochastic
        col_sums = M.sum(axis=0)
        zero_cols = col_sums == 0
        col_sums[zero_cols] = 1  # Avoid division by zero
        M = M / col_sums
        
        # For columns that were all zero, use uniform distribution
        # (analogous to dangling nodes in PageRank)
        if np.any(zero_cols):
            M[:, zero_cols] = 1.0 / n
        
        return M
    
    def compute_informativeness_matrix(self, features: List[Feature]) -> np.ndarray:
        """
        Compute the informativeness matrix J for features.
        
        Jp,q = SelfInfo(fp | fq) = -log(P(fp | fq))
        
        Args:
            features: List of features to compute informativeness for
        
        Returns:
            Informativeness matrix (normalized for stochasticity)
        """
        n = len(features)
        J = np.zeros((n, n))
        
        for i, fi in enumerate(features):
            for j, fj in enumerate(features):
                if self.use_conditional:
                    info = self.informativeness_calc.compute_self_information(fi, fj)
                else:
                    info = self.informativeness_calc.compute_self_information(fi)
                
                # Paper Eq. 9: Jp,q = SelfInfo(fp|fq) = -log(P(fp|fq))
                # When p==q, P(fp|fp) = 1 so SelfInfo = 0 (surfer shouldn't jump to itself)
                J[i, j] = info
        
        # Normalize each column to make it stochastic (uniform if all zeros)
        col_sums = J.sum(axis=0)
        zero_cols = col_sums == 0
        col_sums[zero_cols] = 1
        J = J / col_sums
        
        # For columns that were all zero, use uniform distribution
        if np.any(zero_cols):
            J[:, zero_cols] = 1.0 / n
        
        return J
    
    def summarize(self, entity: Entity, k: int) -> List[Tuple[Feature, float]]:
        """
        Summarize an entity by selecting top-k features.
        
        Args:
            entity: Entity to summarize
            k: Number of features to select
        
        Returns:
            List of (feature, score) tuples sorted by score (descending)
        """
        features = entity.get_features()
        
        if k >= len(features):
            # Return all features with equal scores
            score = 1.0 / len(features)
            return [(f, score) for f in features]
        
        n = len(features)
        
        # Compute matrices
        M = self.compute_relatedness_matrix(features)
        J = self.compute_informativeness_matrix(features)
        
        # Create diagonal matrices
        delta = np.eye(n) * (1 - self.lambda_param)
        lambda_diag = np.eye(n) * self.lambda_param
        
        # Compute transition matrix
        transition_matrix = M @ delta + J @ lambda_diag
        
        # Initialize uniform distribution
        x = np.ones(n) / n
        
        # Paper Eq. 2-3: Power iteration x(t+1) = T * x(t), converges to x*
        # T = (M·Δ + J·Λ) is column-stochastic, so x remains a probability vector.
        for _ in range(self.iterations):
            x_new = transition_matrix @ x
            # Numerical safety only — should not be needed if matrices are correct
            x_new = np.nan_to_num(x_new, nan=0.0, posinf=0.0, neginf=0.0)
            x_sum = x_new.sum()
            if x_sum > 0:
                x = x_new / x_sum
            else:
                x = np.ones(n) / n
        
        # Rank features and return top-k
        ranked = sorted(zip(features, x), key=lambda item: item[1], reverse=True)
        return ranked[:k]


# Example usage and testing
def example_usage():
    """Demonstrate RELIN usage."""
    print("=" * 80)
    print("RELIN: Relatedness and Informativeness-Based Centrality")
    print("=" * 80)
    
    # Create some entities with features
    entity1 = Entity("ex:Rudi_Studer")
    entity1.add_feature("foaf:givenName", "Rudi")
    entity1.add_feature("foaf:familyName", "Studer")
    entity1.add_feature("swrc:publication", "ex:Semantic-Wikipedia")
    entity1.add_feature("swrc:year", "2009")
    entity1.add_feature("foaf:based_near", "Karlsruhe")
    entity1.add_feature("foaf:workplaceHomepage", "http://aifb.kit.edu")
    
    entity2 = Entity("ex:Another_Person")
    entity2.add_feature("foaf:givenName", "John")
    entity2.add_feature("foaf:familyName", "Doe")
    entity2.add_feature("swrc:publication", "ex:Another-Paper")
    entity2.add_feature("swrc:year", "2010")
    
    # Initialize RELIN
    relin = RELIN(lambda_param=1.0, iterations=10)
    
    # Simulate corpus statistics (in practice, these come from a real corpus)
    corpus_stats = {
        "foaf:givenName": 1000000,
        "foaf:familyName": 1000000,
        "swrc:publication": 500000,
        "swrc:year": 800000,
        "foaf:based_near": 300000,
        "foaf:workplaceHomepage": 100000,
        "Rudi": 50000,
        "Studer": 10000,
        "Semantic-Wikipedia": 5000,
        "John": 500000,
        "Doe": 50000,
    }
    
    # Co-occurrence statistics
    co_occurrence_stats = {
        ("foaf:givenName", "foaf:familyName"): 900000,  # High co-occurrence
        ("swrc:publication", "swrc:year"): 400000,      # High co-occurrence
        ("foaf:givenName", "swrc:publication"): 100000, # Lower co-occurrence
        ("Rudi", "Studer"): 40000,                      # High co-occurrence
        ("Semantic-Wikipedia", "swrc:publication"): 4000, # High co-occurrence
    }
    
    relin.train_relatedness(corpus_stats, co_occurrence_stats)
    
    # Prepare informativeness data
    entities = [entity1, entity2]
    relin.prepare_informativeness(entities)
    
    # Summarize entity1 with k=3
    print(f"\nEntity: {entity1.entity_id}")
    print(f"Total features: {len(entity1.get_features())}")
    print("\nAll features:")
    for feature in entity1.get_features():
        print(f"  - {feature}")
    
    summary = relin.summarize(entity1, k=3)
    print(f"\nTop-3 features (with RELIN scores):")
    for i, (feature, score) in enumerate(summary, 1):
        print(f"  {i}. {feature.prop} = {feature.value} (score: {score:.4f})")
    
    # Try different lambda values
    print("\n" + "=" * 80)
    print("Testing different lambda values:")
    print("=" * 80)
    
    for lambda_val in [0.0, 0.5, 0.85, 1.0]:
        relin_test = RELIN(lambda_param=lambda_val, iterations=10)
        relin_test.train_relatedness(corpus_stats, co_occurrence_stats)
        relin_test.prepare_informativeness(entities)
        
        summary = relin_test.summarize(entity1, k=3)
        print(f"\nλ = {lambda_val}:")
        labels = ["Relatedness only", "Mixed", "Balanced", "Informativeness only"]
        label_map = {0.0: 0, 0.5: 1, 0.85: 2, 1.0: 3}
        print(f"  ({labels[label_map[lambda_val]]})")
        
        for i, (feature, score) in enumerate(summary, 1):
            print(f"    {i}. {feature.prop} = {feature.value} (score: {score:.4f})")


if __name__ == "__main__":
    example_usage()
