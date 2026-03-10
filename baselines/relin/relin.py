import numpy as np
from rdflib import URIRef, Literal


class RELIN:
    def __init__(self, lam:float = 0.85, iterations:int = 10):
        self.lam = lam
        self.iterations = iterations
    
    # ---------- Relational move ----------
    @staticmethod
    def relatedness(f1, f2):
        p1, o1 = f1
        p2, o2 = f2
        score = 0.0 

        # same predicates, weights +0.5
        if str(p1) == str(p2):
            score += 0.5

        # same objects, weights +0.5
        if isinstance(o1, URIRef) and isinstance(o2, URIRef) and str(o1) == str(o2):
            score += 0.5
        elif isinstance(o1, Literal) and isinstance(o2, Literal):
            t1, t2 = set(str(o1).lower().split()), set(str(o2).lower().split())
            if len(t1 | t2) > 0:
                score += 0.5 * len(t1 & t2) / len(t1 | t2)
        return score
    
    def build_M(self, features):
        n = len(features)
        M = np.zeros((n, n))

        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if i == j:
                    continue
                M[i, j] = self.relatedness(f1, f2)
            # Normalize row for stochastic matrix
            if M[i].sum() > 0:
                M[i] /= M[i].sum()
        
        return M
    
    # ---------- Information jump ----------
    @staticmethod
    def build_J(features):
        # Use inverse predicate freq within entity
        pred_counts: dict[str, int] = {}
        total = len(features)
        info = []
        
        # How often each predicate appears
        for p, _ in features:
            pred_counts[str(p)] = pred_counts.get(str(p), 0) + 1
        
        # if predicate appears often, less info
        for p, _ in features:
            prob = pred_counts[str(p)] / total
            info.append(-np.log(prob))

        # normalize vector
        info = np.array(info, dtype=float)
        info /= info.sum()
        return info
    
    # ---------- RELIN Update ----------
    def rank(self, features):
        M = self.build_M(features)
        J = self.build_J(features)
        n = M.shape[0]
        x = np.ones(n) / n

        # update formula
        for _ in range (self.iterations):
            x = (1 - self.lam) * (M.T @ x) + self.lam * J
            x /= x.sum()
        
        ranked = sorted(zip(features, x), key=lambda x: x[1], reverse=True)
        return ranked