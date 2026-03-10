import os
import math
from rdflib import Graph
from collections import Counter, defaultdict


class DatasetStats:
    """
    Pre-computed dataset-level statistics for RELIN.

    Provides:
    - PMI between predicates (Eq. 6 of the paper, using dataset as corpus)
    - PMI between values (Eq. 6 of the paper, using dataset as corpus)
    - Feature probability P(f) across all entities (Eq. 7-8)
    - Conditional feature probability P(f_p | f_q) (Eq. 8)
    """

    def __init__(self):
        self.entity_count = 0
        self.pred_entities = defaultdict(set)       # pred_uri_str -> {entity_ids}
        self.val_entities = defaultdict(set)         # val_str -> {entity_ids}
        self.feature_entities = defaultdict(set)     # (pred_str, val_n3) -> {entity_ids}

    # ---------- Relatedness (PMI) ----------

    def pred_pmi(self, pred1: str, pred2: str) -> float:
        """
        PMI between two predicates based on entity co-occurrence.
        Paper Eq. 6: PMI(s_i, s_j) = log(P(s_i, s_j) / (P(s_i) · P(s_j)))
        Floored to 0 (negative PMI means anti-correlation, treated as unrelated).
        """
        n = self.entity_count
        if n == 0:
            return 0.0

        set1 = self.pred_entities.get(pred1, set())
        set2 = self.pred_entities.get(pred2, set())

        p1 = len(set1) / n
        p2 = len(set2) / n
        p12 = len(set1 & set2) / n

        if p1 == 0 or p2 == 0 or p12 == 0:
            return 0.0

        return max(0.0, math.log(p12 / (p1 * p2)))

    def val_pmi(self, val1: str, val2: str) -> float:
        """
        PMI between two values based on entity co-occurrence.
        Paper Eq. 6.
        """
        n = self.entity_count
        if n == 0:
            return 0.0

        set1 = self.val_entities.get(val1, set())
        set2 = self.val_entities.get(val2, set())

        p1 = len(set1) / n
        p2 = len(set2) / n
        p12 = len(set1 & set2) / n

        if p1 == 0 or p2 == 0 or p12 == 0:
            return 0.0

        return max(0.0, math.log(p12 / (p1 * p2)))

    # ---------- Informativeness ----------

    def feature_prob(self, pred_str: str, val_n3: str) -> float:
        """
        P(f) = |{e ∈ E : f ∈ FS(e)}| / |E|
        Paper Eq. 7-8 (unconditional).
        """
        key = (pred_str, val_n3)
        count = len(self.feature_entities.get(key, set()))
        if self.entity_count == 0:
            return 1.0
        return count / self.entity_count

    def feature_cond_prob(self, fp_pred: str, fp_val_n3: str,
                          fq_pred: str, fq_val_n3: str) -> float:
        """
        P(f_p | f_q) = |{e ∈ E : f_p, f_q ∈ FS(e)}| / |{e ∈ E : f_q ∈ FS(e)}|
        Paper Eq. 8.
        """
        key_p = (fp_pred, fp_val_n3)
        key_q = (fq_pred, fq_val_n3)
        set_p = self.feature_entities.get(key_p, set())
        set_q = self.feature_entities.get(key_q, set())

        if len(set_q) == 0:
            return 1.0
        return len(set_p & set_q) / len(set_q)


def precompute_dataset_stats(dataset_root: str, dataset_type: str) -> DatasetStats:
    """
    Scan all entities in the dataset to build global statistics.

    For each entity, extracts its features (outgoing triples from main subject)
    and records:
    - Which entities have which predicates (for predicate PMI)
    - Which entities have which values (for value PMI)
    - Which entities have which features (for informativeness)
    """
    stats = DatasetStats()
    dataset_path = os.path.join(dataset_root, f"{dataset_type}_data")

    entity_ids = [
        d for d in os.listdir(dataset_path)
        if d.isdigit() and os.path.isdir(os.path.join(dataset_path, d))
    ]

    stats.entity_count = len(entity_ids)

    for eid in entity_ids:
        desc_file = os.path.join(dataset_path, eid, f"{eid}_desc.nt")
        if not os.path.exists(desc_file):
            continue

        g = Graph()
        try:
            g.parse(desc_file, format="nt")
        except Exception:
            continue

        # Find main subject
        subject_counts = Counter(s for s, _, _ in g)
        if not subject_counts:
            continue
        subject = max(subject_counts, key=lambda s: subject_counts[s])

        # Extract features (outgoing edges)
        features = [(p, o) for s, p, o in g if s == subject]

        for p, o in features:
            pred_str = str(p)
            val_str = str(o)
            val_n3 = o.n3()

            stats.pred_entities[pred_str].add(eid)
            stats.val_entities[val_str].add(eid)
            stats.feature_entities[(pred_str, val_n3)].add(eid)

    print(f"[PRECOMPUTE] Scanned {stats.entity_count} entities | "
          f"{len(stats.pred_entities)} unique predicates | "
          f"{len(stats.val_entities)} unique values | "
          f"{len(stats.feature_entities)} unique features")

    return stats
