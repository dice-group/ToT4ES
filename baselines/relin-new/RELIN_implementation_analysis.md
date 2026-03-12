# RELIN Implementation Analysis (`relin-new`)

**Paper:** *RELIN: Relatedness and Informativeness-Based Centrality for Entity Summarization*
**Authors:** Gong Cheng, Thanh Tran, and Yuzhong Qu (ISWC 2011, LNCS 7031, pp. 114–129)

This document provides a point-by-point comparison between the paper and the current implementation in `relin.py` and `apply_relin_batch_all.py`.

> **Note:** This is the `relin-new/` implementation — a paper-faithful rewrite of the original `relin/` code, which had several critical deviations (inverted λ, local-only informativeness, string-match relatedness, additive combination). See `../relin/RELIN_implementation_analysis.md` for the old analysis.

---

## 1. Core Random Surfer Model (Section 3.2 & Eq. 1–4)

### Paper Definition

The random surfer has two actions:

- **Relational move (M):** follow an edge to a related feature, probability `1−λ`.
- **Informational jump (J):** jump to an informative feature, probability `λ`.

Matrix form (Eq. 2):

```
x(t+1) = (M · Δ + J · Λ) · x(t)
```

With simplified diagonal matrices (Eq. 4):

```
Δ_qq = 1 − λ,    Λ_qq = λ     →     x(t+1) = ((1−λ)·M + λ·J) · x(t)
```

### Implementation (`relin.py`, `RELIN.summarize()`)

```python
delta = np.eye(n) * (1 - self.lambda_param)
lambda_diag = np.eye(n) * self.lambda_param
transition_matrix = M @ delta + J @ lambda_diag

for _ in range(self.iterations):
    x_new = transition_matrix @ x
```

### Alignment

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| λ meaning | λ = probability of informational jump | `lambda_param` = jump probability | ✅ Correct |
| Transition matrix | `T = M·Δ + J·Λ` | `M @ delta + J @ lambda_diag` | ✅ Correct |
| Δ diagonal | `1 − λ` | `(1 - lambda_param) * I` | ✅ Correct |
| Λ diagonal | `λ` | `lambda_param * I` | ✅ Correct |
| Power iteration | `x(t+1) = T · x(t)` | `x_new = transition_matrix @ x` | ✅ Correct |
| M, J stochasticity | Column-stochastic | Column-normalized with zero-column uniform fallback | ✅ Correct |

---

## 2. Relatedness Matrix M (Section 4.1, Eq. 5–6)

### Paper Definition

**Eq. 5:**
```
M_pq ∝ sqrt( Rel(Prop(fp), Prop(fq)) · Rel(Val(fp), Val(fq)) )
```

**Eq. 6 (PMI):**
```
PMI(si, sj) = log( P(si, sj) / (P(si) · P(sj)) )
```

where `P(si) = Hits(si) / N` and `P(si, sj) = Hits(si, sj) / N`. The paper uses non-negative PMI.

Self-PMI: `PMI(si, si) = log(P(si, si) / P(si)²) = −log(P(si))`.

### Implementation (`relin.py`, `RelatednessMeasure.pmi()` and `RELIN.compute_relatedness_matrix()`)

```python
# pmi() method
if phrase1 == phrase2:
    return max(0, -math.log(p_phrase1))      # Self-PMI

pmi_value = math.log(p_both / (p_phrase1 * p_phrase2))
return max(0, pmi_value)                      # Non-negative PMI

# compute_relatedness_matrix()
M[i, j] = math.sqrt(prop_rel * val_rel)       # Eq. 5: sqrt
```

### Alignment

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| Relatedness measure | PMI (Eq. 6) | PMI with corpus-wide counts | ✅ Correct |
| Self-PMI | `−log(P(si))` | `max(0, -math.log(p_phrase1))` | ✅ Correct |
| Non-negative PMI | Yes | `max(0, pmi_value)` | ✅ Correct |
| Combination | `sqrt(Rel(Prop) × Rel(Val))` (Eq. 5) | `math.sqrt(prop_rel * val_rel)` | ✅ Correct |
| Co-occurrence storage | Symmetric | Bidirectional in `add_co_occurrence` | ✅ Correct |
| Column normalization | Column-stochastic | Column sums + zero-column uniform fallback | ✅ Correct |
| Corpus statistics | Web search hits across documents | Entity descriptions across dataset | ✅ Adapted (see §6) |

---

## 3. Informativeness Matrix J (Section 4.2, Eq. 7–9)

### Paper Definition

**Eq. 7–9:**
```
J_pq = SelfInfo(fp | fq) = −log(P(fp | fq))
P(fp | fq) = |{e ∈ E : fp, fq ∈ FS(e)}| / |{e ∈ E : fq ∈ FS(e)}|
```

Key property: when `p = q`, `P(fp | fp) = 1`, so `SelfInfo(fp | fp) = −log(1) = 0` (surfer should not jump to itself).

The paper also allows unconditional `SelfInfo(fp)` as an approximation.

### Implementation (`relin.py`, `InformativenessCalculator` and `RELIN.compute_informativeness_matrix()`)

```python
# compute_conditional_probability() — Eq. 8
for entity_id, features in self.entity_features.items():
    if feature_condition in features:
        condition_count += 1
        if feature_target in features:
            both_count += 1
return both_count / condition_count

# compute_self_information()
if prob == 0 or prob == 1:
    return 0.0                 # When p==q, prob=1 → returns 0
return -math.log(prob)

# compute_informativeness_matrix() — fills full |FS|×|FS| matrix
J[i, j] = self.informativeness_calc.compute_self_information(fi, fj)
```

### Alignment

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| J structure | Full |FS|×|FS| matrix (conditional) | Full conditional matrix | ✅ Correct |
| Conditional probability (Eq. 8) | `|{e: fp,fq ∈ FS(e)}| / |{e: fq ∈ FS(e)}|` | Counts across all entities | ✅ Correct |
| Self-information (Eq. 7) | `−log(P(fp\|fq))` | `-math.log(prob)` | ✅ Correct |
| Diagonal (p=q) | `−log(1) = 0` | Returns `0.0` when `prob == 1` | ✅ Correct |
| Scope | Dataset-wide (`E` = all entities) | Uses all entities via `prepare_informativeness` | ✅ Correct |
| Column normalization | Column-stochastic | Column sums + zero-column uniform fallback | ✅ Correct |

---

## 4. Feature Definition (Definition 2–3)

### Paper Definition

> **Def. 2:** A feature is a property-value pair `(property, value)` derived from edges in the data graph. "We actually consider both incoming and outgoing edges (i.e. where e appears as target and source node)."
>
> **Def. 3:** The feature set `FS(e)` of an entity `e` is the set of all its features.

### Implementation

```python
# Feature class (relin.py)
class Feature:
    def __init__(self, prop: str, value: str):
        self.prop = prop
        self.value = value

# Outgoing edges (apply_relin_batch_all.py, Phase 1)
for subject, predicate, obj in triples:
    entity.add_feature(prop_short, obj_short)

# Incoming edges (apply_relin_batch_all.py, Phase 1)
for s, p, o in summarizer.all_triples:
    if obj_clean == entity_clean and src_clean != entity_clean:
        entity.add_feature(p_short, s_short)
```

### Alignment

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| Feature = (property, value) | Yes | `Feature(prop, value)` | ✅ Correct |
| Outgoing edges | Included | Entity as subject → `(predicate, object)` | ✅ Correct |
| Incoming edges | Included (Def. 2) | Entity as object → `(predicate, source)` | ✅ Correct |
| Feature set | `FS(e)` as a set | `Entity.features: Set[Feature]` (no duplicates) | ✅ Correct |

---

## 5. Experimental Parameters (Section 6)

### Paper

- **λ values tested:** 0.00, 0.15, 0.50, 0.85, 1.00
- **Best results:** λ=0.85 for k=10, λ=1.00 for k=5
- **Iterations:** 10 (convergence typically reached)

### Implementation

```python
# apply_relin_batch_all.py, Phase 2
relin = RELIN(lambda_param=0.85, iterations=10)
```

### Alignment

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| λ = 0.85 (default) | Best for k=10 | `lambda_param=0.85` | ✅ Correct |
| Iterations = 10 | Paper experimental setup | `iterations=10` | ✅ Correct |
| k values | k ∈ {5, 10} | Generates for `k in [5, 10]` | ✅ Correct |

---

## 6. Corpus Statistics & Two-Pass Architecture

### Paper

The paper uses web search engine hits (Google) to estimate `P(si)` and `P(si, sj)` for PMI computation. `N` is the normalizing constant (total number of indexed documents).

For informativeness, `P(fp | fq)` is computed across all entities `E` in the dataset.

### Implementation (`apply_relin_batch_all.py`, `process_dataset()`)

The implementation uses a **corpus-wide two-pass approach**:

1. **Phase 1 — Parse all entities:** Build corpus-wide `phrase_counts` (how many entity descriptions contain each phrase) and `co_occurrence_counts` (how many entity descriptions contain both phrases). Also constructs `Entity` objects with both incoming and outgoing edges.

2. **Phase 2 — Initialize RELIN:** Single `RELIN` instance trained with global statistics. `total_docs = total_entities` (number of entity descriptions as normalizing constant).

3. **Phase 3 — Generate summaries:** Per-entity `relin.summarize(entity, k)` using the globally-trained model.

### Alignment

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| PMI scope | Corpus-wide (web search) | Corpus-wide (across all entity descriptions) | ✅ Adapted |
| Informativeness scope | Dataset-wide (all entities E) | All parsed entities | ✅ Correct |
| Single vs per-entity init | Implied global (corpus-wide stats) | Single global RELIN instance | ✅ Correct |
| Normalization constant N | Total indexed documents | `total_entities` | ✅ Adapted |

**Note:** The paper uses web search hits as a proxy for semantic relatedness across the entire web. The implementation adapts this by using entity description co-occurrence within the benchmark dataset as the corpus. This is a practical adaptation that captures dataset-specific relatedness patterns without requiring external API calls.

---

## 7. Output & Triple Mapping

### Implementation

The `map_selected_features_to_triples` method maps RELIN's selected features back to the original RDF triples in the description file. It searches **both** outgoing triples (entity as subject) and incoming triples (entity as object), consistent with the feature set construction.

Output is written as N-Triples files: `{entity_id}_top5.nt` and `{entity_id}_top10.nt`.

### Alignment

| Aspect | Status |
|--------|--------|
| Top-k selection | ✅ Correct (k=5 and k=10) |
| Feature-to-triple mapping includes outgoing edges | ✅ Correct |
| Feature-to-triple mapping includes incoming edges | ✅ Correct |

---

## 8. Summary: Paper vs Implementation Alignment

| # | Paper Element | Equation | Status | Notes |
|---|---------------|----------|--------|-------|
| 1 | Random surfer model | Eq. 1–2 | ✅ | Matrix form `T = M·Δ + J·Λ` |
| 2 | Diagonal matrices Δ, Λ | Eq. 3–4 | ✅ | `(1−λ)·I` and `λ·I` |
| 3 | Relatedness via PMI | Eq. 5–6 | ✅ | `sqrt(PMI(prop)·PMI(val))`, non-negative PMI |
| 4 | Self-PMI | Eq. 6 | ✅ | `−log(P(si))` |
| 5 | Conditional informativeness | Eq. 7–9 | ✅ | Full |FS|×|FS| matrix |
| 6 | Diagonal of J = 0 | Eq. 9 | ✅ | `P(fp\|fp)=1 → −log(1)=0` |
| 7 | Feature = (property, value) | Def. 2 | ✅ | Including incoming edges |
| 8 | Feature set FS(e) | Def. 3 | ✅ | `Set[Feature]` |
| 9 | Power iteration | Eq. 2 | ✅ | 10 iterations |
| 10 | λ = 0.85 | Sect. 6 | ✅ | Correct convention (jump probability) |
| 11 | Corpus-wide statistics | Sect. 4.1 | ✅ | Adapted from web hits → dataset co-occurrence |
| 12 | Dataset-wide informativeness | Sect. 4.2 | ✅ | All entities used for P(fp\|fq) |

---

## 9. Differences from `relin/` (Old Implementation)

The `relin-new/` implementation addresses all issues identified in the original `relin/` analysis:

| Issue in `relin/` | Severity | Fix in `relin-new/` |
|--------------------|----------|----------------------|
| λ convention inverted (`lam` = move weight, not jump weight) | 🔴 Critical | `lambda_param` correctly controls jump probability |
| Informativeness computed locally within single entity | 🔴 Critical | Computed across ALL entities in dataset (Eq. 8) |
| J is a 1-D vector (unconditional) | 🟡 Medium | Full |FS|×|FS| conditional matrix |
| Relatedness uses string matching, not PMI | 🟡 Medium | Proper PMI with corpus-wide co-occurrence |
| Relatedness combination is additive, not multiplicative with sqrt | 🟡 Medium | `sqrt(PMI(prop) · PMI(val))` per Eq. 5 |
| Only predicate used for informativeness | 🟡 Medium | Full (property, value) feature pairs |
| Per-entity RELIN initialization (empty co-occurrences) | 🔴 Critical | Single global RELIN with corpus-wide stats |
| Missing incoming edges | 🟡 Medium | Both incoming and outgoing edges included |

---

## 10. Conclusion

The `relin-new/` implementation is a **faithful reproduction** of the RELIN algorithm as described in the paper. All core equations (Eq. 1–9) and definitions (Def. 2–3) are correctly implemented. The only adaptation is using dataset-internal co-occurrence statistics instead of web search engine hits for PMI computation, which is a standard practical substitution when web API access is unavailable.
