# Tree-of-Thoughts Implementation Issues & Recommendations

## Critical Issues

### 1. **BFS Node Re-insertion Bug** (Line ~327)
**Current Code:**
```python
if children_created == 0:
    new_nodes.append(node)  # Re-adds parent
```

**Fix:**
```python
if children_created == 0:
    # Don't re-add parent - this branch is exhausted
    pass
```

### 2. **Duplicate Prevention Should Be in Prompt** (Line ~252-260)
**Current:** Filters duplicates after generation
**Fix:** Add to prompt:
```python
f"DO NOT select any of these already-chosen indices: {selected_ids}"
```

### 3. **Temperature Control Missing** (Line ~217)
**Fix:**
```python
def thought_generator(self, state: str, ...) -> List[str]:
    raw_thoughts = self.chat_completions(
        prompt=prompt,
        n=self.n_candidates,
        temperature=0.8,  # Higher for diversity
        stop=stop_string,
    )
```

### 4. **Evaluation Is Inefficient** (Line ~335)
**Current:** Evaluates all states in queue
**Recommendation:** Only evaluate newly created children, or batch more intelligently

### 5. **Missing Triple Statistics in Prompts**
**Current:** Prompts mention `triple_centrality`, `freq_property`, etc. but don't provide values
**Fix:** Either:
- Remove mentions from prompts (let LLM infer)
- Compute and inject actual statistics

### 6. **Silent Failure in Heuristic Calculator** (Line ~626)
**Fix:**
```python
if n_samples == 0:
    raise ValueError("All evaluation samples failed to parse. Check LLM outputs.")
```

### 7. **Last Step Expansion Logic** (Line ~385)
**Fix:** Don't prune to 1 before the last expansion:
```python
if step == self.n_steps:
    # On last step, expand all remaining, then pick best
    top_nodes = sorted_nodes[: self.breadth_limit]
else:
    top_nodes = sorted_nodes[: self.breadth_limit]

# After loop completes, select best from queue
```

### 8. **Add Retry Logic for JSON Parsing**
When evaluation fails, retry with a clearer prompt:
```python
if parsed is None:
    # Retry with stricter instructions
    retry_prompt = f"{original_prompt}\n\nIMPORTANT: Return ONLY valid JSON array, nothing else."
    # ... retry logic
```

### 9. **State Should Track More Context**
Consider enhancing TreeNode:
```python
@dataclass
class TreeNode:
    state: str
    thought: str
    value: float = 0.0
    depth: int = 0  # NEW
    parent: Optional["TreeNode"] = None  # NEW
    children: List["TreeNode"] = field(default_factory=list)
```

### 10. **Add Validation**
```python
def validate_state(self, state: str) -> bool:
    """Ensure state has no duplicates and valid indices"""
    ids = [int(x) for x in state.strip().splitlines() if x.strip().isdigit()]
    return len(ids) == len(set(ids)) and all(1 <= i <= self.num_triples for i in ids)
```

## Architecture Recommendations

### Use Self-Consistency for Evaluation
Instead of simple averaging, use majority voting or consistency checking across evaluation samples.

### Consider Hybrid Search
Combine BFS with Monte Carlo Tree Search (MCTS) for better exploration-exploitation tradeoff.

### Add Early Stopping
If top states converge (similar values), stop early:
```python
if max(values) - min(values) < 0.01:
    break  # States are too similar
```

### Implement Caching
Cache LLM responses for identical prompts to reduce cost.

### Add Metrics Logging
Track:
- Number of LLM calls
- Average state values per step
- Branch factor (avg children per node)
- Pruning rate

## Testing Recommendations

1. **Unit Tests:** Test each component in isolation
2. **Integration Tests:** Run on small entities (3-5 triples)
3. **Ablation Studies:** Test with/without each criterion
4. **Comparison:** Compare against greedy/random baselines

## Theoretical Alignment with ToT

**Original ToT Paper (Yao et al., 2023):**
- ✅ Uses LLM for thought generation
- ✅ Uses LLM for state evaluation
- ✅ Implements BFS with beam search
- ⚠️ Should have clearer separation of thought types
- ❌ Missing DFS alternative (paper recommends both)
- ⚠️ Vote aggregation is good, but could add value-based evaluation

**Entity Summarization Adaptation:**
- ✅ Good: Adapts "thoughts" to be triple selections
- ✅ Good: Multi-criteria evaluation fits domain
- ⚠️ Prompts could be more structured
- ❌ Missing: Compute actual triple metrics (centrality, frequency, similarity)
