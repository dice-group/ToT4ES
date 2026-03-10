import numpy as np
import math
from rdflib import URIRef, Literal


class RELIN:
    def __init__(self, lam: float = 0.85, iterations: int = 10, stats=None):
        """
        RELIN: Relatedness and Informativeness-Based Centrality.

        Args:
            lam: λ parameter (Eq. 4). Controls informational jump probability.
                 λ=0.85 means 85% informativeness + 15% relatedness.
            iterations: Number of power iteration steps.
            stats: DatasetStats object from relin_precompute. If provided, uses
                   paper-faithful PMI relatedness and global informativeness.
                   If None, falls back to simple heuristics.
        """
        self.lam = lam
        self.iterations = iterations
        self.stats = stats

    # ---------- Relational move (Section 4.1) ----------

    def relatedness(self, f1, f2):
        """
        Compute relatedness between two features.

        Paper Eq. 5: M_{p,q} ∝ Rel(Prop(f_p), Prop(f_q)) · Rel(Val(f_p), Val(f_q))
        Paper Eq. 6: Rel is PMI (Pointwise Mutual Information)

        With stats: uses dataset-level PMI (multiplicative, per Eq. 5).
        Without stats: falls back to simple string-matching heuristic.
        """
        p1, o1 = f1
        p2, o2 = f2

        if self.stats is not None:
            # PMI-based relatedness (paper Eq. 5-6)
            pred_rel = self.stats.pred_pmi(str(p1), str(p2))
            val_rel = self.stats.val_pmi(str(o1), str(o2))
            # Multiplicative combination (paper Eq. 5)
            return pred_rel * val_rel
        else:
            # Fallback: simple heuristic (additive)
            score = 0.0
            if str(p1) == str(p2):
                score += 0.5
            if isinstance(o1, URIRef) and isinstance(o2, URIRef) and str(o1) == str(o2):
                score += 0.5
            elif isinstance(o1, Literal) and isinstance(o2, Literal):
                t1, t2 = set(str(o1).lower().split()), set(str(o2).lower().split())
                if len(t1 | t2) > 0:
                    score += 0.5 * len(t1 & t2) / len(t1 | t2)
            return score

    def build_M(self, features):
        """
        Build the relatedness transition matrix M (row-stochastic).

        M[i, j] = normalized relatedness from feature i to feature j.
        After transpose, M.T corresponds to the paper's column-stochastic M.
        """
        n = len(features)
        M = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                r = self.relatedness(features[i], features[j])
                M[i, j] = r
                M[j, i] = r  # PMI is symmetric

        # Normalize rows; uniform fallback for dangling nodes (zero rows)
        for i in range(n):
            row_sum = M[i].sum()
            if row_sum > 0:
                M[i] /= row_sum
            else:
                # Dangling node: distribute uniformly to all other features
                for j in range(n):
                    if j != i:
                        M[i, j] = 1.0 / (n - 1) if n > 1 else 0.0

        return M

    # ---------- Informational jump (Section 4.2) ----------

    def build_J(self, features):
        """
        Build the informativeness component.

        With stats (paper Eq. 8-9): conditional self-information matrix.
            J[i,j] = SelfInfo(f_j | f_i) = -log(P(f_j | f_i))
            Row-stochastic; J.T corresponds to the paper's column-stochastic J.

        Without stats: local predicate frequency approximation (1-D vector).
        """
        n = len(features)

        if self.stats is not None:
            # Full conditional informativeness matrix (paper Eq. 8-9)
            J = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i == j:
                        # P(f | f) = 1, so SelfInfo = -log(1) = 0
                        J[i, j] = 0.0
                        continue

                    # P(f_j | f_i) from Eq. 8
                    prob = self.stats.feature_cond_prob(
                        str(features[j][0]), features[j][1].n3(),   # target f_j
                        str(features[i][0]), features[i][1].n3()    # source f_i
                    )

                    if prob > 0:
                        J[i, j] = -math.log(prob)
                    else:
                        # Feature pair never co-occurs: maximum informativeness
                        J[i, j] = math.log(self.stats.entity_count) if self.stats.entity_count > 1 else 1.0

                # Normalize row; uniform fallback for zero rows
                row_sum = J[i].sum()
                if row_sum > 0:
                    J[i] /= row_sum
                else:
                    # All features equally informative from this source
                    for j in range(n):
                        if j != i:
                            J[i, j] = 1.0 / (n - 1) if n > 1 else 0.0

            return J
        else:
            # Fallback: local predicate frequency (1-D vector)
            pred_counts: dict[str, int] = {}
            total = len(features)
            info = []

            for p, _ in features:
                pred_counts[str(p)] = pred_counts.get(str(p), 0) + 1

            for p, _ in features:
                prob = pred_counts[str(p)] / total
                info.append(-np.log(prob))

            info = np.array(info, dtype=float)
            info /= info.sum()
            return info

    # ---------- RELIN Update (Section 3.2, Eq. 2) ----------

    def rank(self, features):
        """
        Rank features using the RELIN random surfer model.

        Paper Eq. 2: x(t+1) = ((1-λ)·M + λ·J) · x(t)
        Paper Eq. 4: Δ_{q,q} = 1-λ, Λ_{q,q} = λ

        λ controls the probability of informational jump.
        """
        M = self.build_M(features)
        J = self.build_J(features)
        n = M.shape[0]
        x = np.ones(n) / n

        if isinstance(J, np.ndarray) and J.ndim == 2:
            # Full J matrix (paper-faithful, Eq. 2)
            for _ in range(self.iterations):
                x = (1 - self.lam) * (M.T @ x) + self.lam * (J.T @ x)
                x /= x.sum()
        else:
            # J is a 1-D vector (simplified unconditional approximation)
            for _ in range(self.iterations):
                x = (1 - self.lam) * (M.T @ x) + self.lam * J
                x /= x.sum()

        ranked = sorted(zip(features, x), key=lambda x: x[1], reverse=True)
        return ranked