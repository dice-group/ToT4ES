# RELIN Implementation Analysis

**Paper:** *RELIN: Relatedness and Informativeness-Based Centrality for Entity Summarization*
**Authors:** Gong Cheng, Thanh Tran, and Yuzhong Qu (ISWC 2011, LNCS 7031, pp. 114–129)

This document provides a point-by-point comparison between the paper and the current implementation in `relin.py`, `relin_dataLoader.py`, `relin_postProcess.py`, and `run_relin.ipynb`.

---

## 1. Core Random Surfer Model (Section 3.2 & Eq. 1–4 of the paper)

### Paper Definition

The paper defines two actions for the RELIN random surfer:

- **Relational move (M):** the surfer follows an edge to a related feature, with probability proportional to how related the two features are.
- **Informational jump (J):** the surfer jumps to a feature with probability proportional to how informative that feature is.

The update rule is (Eq. 1):

```
x_p(t+1) = Σ_{f_q ∈ FS} x_q(t) · ( P(M|f_q) · P(f_p|f_q, M) + P(J|f_q) · P(f_p|f_q, J) )
```

In matrix form (Eq. 2):

```
x(t+1) = (M · Δ + J · Λ) · x(t)
```

where:
- `M` is a |FS|×|FS| matrix with `M_{p,q} = P(f_p | f_q, M)` (column-stochastic: each column sums to 1)
- `J` is a |FS|×|FS| matrix with `J_{p,q} = P(f_p | f_q, J)`
- `Δ` is a diagonal matrix with `Δ_{q,q} = P(M|f_q)`
- `Λ` is a diagonal matrix with `Λ_{q,q} = P(J|f_q)`

The implementation section (Eq. 4) simplifies Δ and Λ:

```
Δ_{q,q} = 1 − λ,     for all q
Λ_{q,q} = λ,          for all q
```

Note the paper's convention: **λ controls the probability of an informational jump**, and **(1−λ) controls the probability of a relational move**.

Substituting these into (2):

```
x(t+1) = (1−λ) · M · x(t) + λ · J · x(t)
```

### Implementation (relin.py, lines 68–73, after the fix)

```python
x = self.lam * (M.T @ x) + (1 - self.lam) * J
```

### Analysis

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| λ meaning | λ = probability of informational jump | `self.lam` used for relational move weight | **⚠️ INVERTED** |
| M matrix structure | Column-stochastic (`M_{p,q}` → column q sums to 1) | Row-stochastic (row i sums to 1), then transposed via `M.T` | **✅ Correct after fix** |
| J matrix structure | Full |FS|×|FS| matrix (conditional informativeness) | 1-D vector (unconditional informativeness) | **⚠️ Simplified** (see §3 below) |

**Current status of the update formula:**
After the recent fix (`M.T @ x`), the matrix multiplication is structurally correct. However, the **lambda assignment is inverted** compared to the paper:

- Paper: `x = (1−λ)·M·x + λ·J·x`, where λ=0.85 means **85% jump, 15% move**
- Code: `x = λ·M.T·x + (1−λ)·J`, where λ=0.85 means **85% move, 15% jump**

**Effect:** With `lam=0.85` in the notebook, the implementation gives 85% weight to **relational move** and 15% to **informational jump**. The paper's experimental best (`λ=0.85`) gives 85% weight to **informational jump** and 15% to relational move — the exact opposite.

**Recommendation:** Either:
- (a) Swap the formula to `x = (1 - self.lam) * (M.T @ x) + self.lam * J` and keep `lam=0.85`, or
- (b) Keep the current formula but set `lam=0.15` in the notebook (since the meaning is reversed).

Option (a) is recommended for consistency with the paper's notation.

---

## 2. Relatedness Matrix M (Section 4.1)

### Paper Definition (Eq. 5)

```
M_{p,q} ∝ Rel(Prop(f_p), Prop(f_q)) · Rel(Val(f_p), Val(f_q))
```

Relatedness is computed using **Pointwise Mutual Information (PMI)** (Eq. 6) based on co-occurrence of phrases (property names, entity names, literal text) in web documents via a search engine (Google Hits).

```
PMI(s_i, s_j) = log( P(s_i, s_j) / (P(s_i) · P(s_j)) )
```

The paper proposes multiplying predicate relatedness by value relatedness.

### Implementation (relin.py, lines 12–29)

```python
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
```

### Analysis

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| Predicate relatedness | PMI (distributional, web-scale) | Exact string match (binary: 0 or 0.5) | **⚠️ Major simplification** |
| Value relatedness | PMI (distributional, web-scale) | Exact URI match or Jaccard similarity on literal tokens | **⚠️ Major simplification** |
| Combination | Multiplicative: `Rel(Prop) × Rel(Val)` | Additive: `0.5·same_pred + 0.5·val_similarity` | **⚠️ Different** |
| Score range | Real-valued (PMI can be negative) | [0, 1] | Different scale |

**Impact:**
- The paper uses **PMI via Google search hits** to capture semantic relatedness (e.g., "given name" is related to "family name" even though they are different URIs). The implementation treats two different predicates as completely unrelated (score=0), missing semantic similarity entirely.
- The paper **multiplies** predicate and value relatedness, meaning both must be non-zero. The implementation **adds** them (score = 0.5·pred_match + 0.5·val_sim), so a feature pair with only matching predicates but completely different values still gets score 0.5.

**Verdict:** This is a practical simplification. PMI via web search is expensive and may not be feasible. The implementation uses a lightweight local heuristic instead. This is a **reasonable trade-off** but should be documented. The additive vs multiplicative difference should be noted — it changes ranking behavior.

---

## 3. Informativeness Vector J (Section 4.2)

### Paper Definition (Eq. 7–9)

The paper defines a **full matrix** `J` where each entry `J_{p,q}` represents the self-information of feature `f_p` conditioned on `f_q`:

```
J_{p,q} = SelfInfo(f_p | f_q) = −log(P(f_p | f_q))
```

where (Eq. 8):

```
P(f_p | f_q) = |{e ∈ E : f_p, f_q ∈ FS(e)}| / |{e ∈ E : f_q ∈ FS(e)}|
```

This requires answering: "Among all entities that have feature `f_q`, what fraction also have `f_p`?" — a **dataset-wide, pairwise conditional** statistic.

The paper also notes (end of Section 4.2):
> "When computing SelfInfo(f_p | f_q) between all pairs of features is too costly in practice, SelfInfo(f_p) can be used as an approximation."

### Implementation (relin.py, lines 49–64)

```python
@staticmethod
def build_J(features):
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
```

### Analysis

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| J structure | |FS|×|FS| matrix (conditional) | 1-D vector | **⚠️ Simplified to unconditional** |
| Information source | Dataset-wide: how many entities have this feature | Entity-local: predicate frequency within this entity | **⚠️ Major difference** |
| What is measured | `−log(P(f_p))` across all entities in the dataset | `−log(count(predicate)/total_features_in_entity)` | **⚠️ Different semantics** |

**Impact:**
1. **1-D vector vs full matrix:** The implementation uses a 1-D vector (unconditional informativeness), which the paper mentions as an acceptable approximation. This is fine.

2. **Local vs global frequency:** This is the most significant deviation. The paper computes: "How rare is this feature (or predicate+value) across ALL entities in the dataset?" A feature like `rdf:type → Person` appears in thousands of entities, so it has low informativeness. A feature like `swrc:publication → ex:Semantic-Wikipedia` appears in very few entities, so it has high informativeness.

   The implementation computes: "How often does this predicate appear within this single entity's feature set?" For example, if an entity has 3 `rdf:type` triples out of 20 features, `prob = 3/20`. This measures **local predicate redundancy**, not **global rarity**.

   These can produce very different rankings. A predicate that appears once in an entity (high local info) might be extremely common across the dataset (low global info), or vice versa.

3. **Predicate-only vs feature-level:** The code only considers the predicate URI for informativeness, not the full (predicate, value) pair. Two features with the same predicate but different values get the same informativeness score, which is incorrect per the paper.

**Verdict:** This is the **most significant deviation** from the paper. The implementation does not access the full dataset to compute feature frequency, instead using a purely local entity-level heuristic. To faithfully implement the paper, one would need to:
- Pre-compute feature (or at minimum predicate) frequency across all entities in the dataset
- Use `−log(|{e : f ∈ FS(e)}| / |E|)` as self-information

---

## 4. Data Loading (relin_dataLoader.py)

### Paper Definition

The paper considers features as property-value pairs derived from outgoing edges of the entity in the data graph. It also mentions incoming edges but focuses on outgoing for clarity.

### Implementation

```python
features = [(p, o) for s, p, o in g if s == subject]
```

This correctly extracts outgoing triples where the entity is the subject, collecting `(predicate, object)` pairs as features. The subject is determined by the node with the most triples.

**Status: ✅ Correct** — aligns with the paper's focus on outgoing edges.

---

## 5. Notebook Configuration

### Paper's Experimental Setup (Section 6)

- **λ values tested:** 0.00, 0.15, 0.50, 0.85, 1.00
- **Best results:** λ=0.85 for k=10, λ=1.00 for k=5
- **Iterations:** 10
- **Meaning of λ:** probability of informational jump (higher λ = more emphasis on informativeness)

### Implementation

```python
relin = RELIN(lam=0.85, iterations=10)
```

Given the inverted lambda convention in the code (see §1), `lam=0.85` currently means 85% relational move weight — which is the **opposite** of the paper's best setting.

---

## 6. Summary of Issues

| # | Issue | Severity | Category |
|---|-------|----------|----------|
| 1 | λ convention inverted (code: λ=move weight; paper: λ=jump weight) | 🔴 Critical | Algorithm correctness |
| 2 | Informativeness computed locally (within entity) instead of globally (across dataset) | 🟠 High | Fidelity to paper |
| 3 | Relatedness uses string matching instead of PMI via web search | 🟡 Medium | Practical simplification |
| 4 | Relatedness combination is additive, not multiplicative | 🟡 Medium | Algorithm detail |
| 5 | J is a 1-D vector instead of a conditional matrix | 🟢 Low | Explicitly allowed by paper |
| 6 | Only predicate is used for informativeness, not full (pred,value) feature | 🟡 Medium | Algorithm detail |

---

## 7. Recommended Fix for Issue #1 (Lambda Inversion)

Change the update formula in `relin.py` line 70 from:

```python
x = self.lam * (M.T @ x) + (1 - self.lam) * J
```

to:

```python
x = (1 - self.lam) * (M.T @ x) + self.lam * J
```

This makes `lam=0.85` correctly assign 85% weight to informativeness and 15% to relatedness, matching the paper.

---

## 8. Conclusion

The implementation captures the **general structure** of RELIN (iterative ranking combining relatedness and informativeness via a random surfer model), but has **one critical bug** (inverted λ) and several **simplifications** that deviate from the paper:

- The relatedness measure is a simple heuristic rather than PMI.
- The informativeness is local rather than dataset-global.
- λ meaning is inverted.

After fixing the λ inversion, the implementation will be a **simplified but structurally correct** version of RELIN. The simplifications (local informativeness, string-match relatedness) are pragmatic trade-offs that avoid the need for web search APIs and dataset-wide pre-computation, but may affect result quality compared to the original paper's results.
