# BFS vs DFS for Tree-of-Thoughts

## Overview
Both Breadth-First Search (BFS) and Depth-First Search (DFS) are now supported in our ToT implementation for entity summarization.

---

## Algorithm Comparison

### **BFS (Breadth-First Search)** - Default

**Strategy**: Explore all nodes at depth d before moving to depth d+1

**How it works**:
1. Start at root (empty state)
2. Generate thoughts for all nodes at current depth
3. Evaluate all candidate states
4. **Prune**: Keep only top-k (beam_limit=3) best states
5. Move to next depth level
6. Repeat until n_steps reached

**Characteristics**:
- ✅ **Explores multiple paths in parallel**
- ✅ **Beam pruning** prevents exponential explosion
- ✅ **Guarantees** finding best solution within beam width
- ⚠️ Higher memory usage (stores all states in current level)
- ⚠️ More LLM calls per step (evaluates all candidates)

**Best for**:
- When you want to explore diverse solution paths
- Problems where the best solution might not be in the greedy path
- When computational resources allow parallel exploration

---

### **DFS (Depth-First Search)** - New

**Strategy**: Explore one path to maximum depth before backtracking

**How it works**:
1. Start at root
2. Generate and evaluate thoughts
3. **Greedily select best thought** (highest value)
4. Recursively explore that path to full depth
5. Backtrack and try next best thought
6. Track globally best solution found

**Characteristics**:
- ✅ **Lower memory usage** (only stores current path)
- ✅ **Fewer LLM calls** initially (explores one path first)
- ✅ **Greedy optimization** finds good solutions quickly
- ⚠️ May get stuck in local optima
- ⚠️ Depends on evaluation quality at each step

**Best for**:
- Resource-constrained environments
- When greedy heuristics are reliable
- Problems with clear hierarchical structure
- Quick prototyping and testing

---

## Usage

### **BFS (Default)**
```bash
python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --search-algorithm bfs \
  --breadth-limit 3 \
  --max-summary-len 5
```

### **DFS**
```bash
python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --search-algorithm dfs \
  --max-summary-len 5
```

**Note**: `--breadth-limit` is only used for BFS

---

## Technical Details

### **BFS Implementation**
```python
def bfs(self, verbose: bool = True) -> str:
    queue = deque([self.root])
    
    for step in range(1, self.n_steps + 1):
        # Expand entire current level
        new_nodes = deque()
        for node in queue:
            thoughts = self.thought_generator(node.state)
            for thought in thoughts:
                child = create_child(node, thought)
                new_nodes.append(child)
        
        # Evaluate all candidates
        values = self.state_evaluator([n.state for n in new_nodes])
        
        # Prune to top-k (beam search)
        sorted_nodes = sorted(zip(new_nodes, values), 
                             key=lambda x: x[1], reverse=True)
        queue = deque(sorted_nodes[:self.breadth_limit])
    
    return best_node.state
```

**Complexity**:
- Time: O(n_steps × breadth_limit × n_candidates)
- Space: O(breadth_limit × n_candidates)
- LLM calls per step: ~(breadth_limit × n_candidates) evaluations

---

### **DFS Implementation**
```python
def dfs(self, verbose: bool = True) -> str:
    best_node = self.root
    best_value = float('-inf')
    
    def dfs_recursive(node, depth):
        nonlocal best_node, best_value
        
        if depth >= self.n_steps:
            # Evaluate leaf
            value = self.state_evaluator([node.state])[0]
            if value > best_value:
                best_value = value
                best_node = node
            return
        
        # Generate and evaluate thoughts
        thoughts = self.thought_generator(node.state)
        candidates = [create_child(node, t) for t in thoughts]
        values = self.state_evaluator([c.state for c in candidates])
        
        # Explore best candidates first (greedy)
        sorted_candidates = sorted(zip(candidates, values),
                                  key=lambda x: x[1], reverse=True)
        
        for child, value in sorted_candidates:
            dfs_recursive(child, depth + 1)  # Recurse
    
    dfs_recursive(self.root, 0)
    return best_node.state
```

**Complexity**:
- Time: O(n_candidates^n_steps) worst case, but greedy pruning helps
- Space: O(n_steps) - just the current path
- LLM calls: Initially fewer, but explores more branches if backtracking

---

## Performance Comparison

| Aspect | BFS | DFS |
|--------|-----|-----|
| **Memory** | High (O(breadth × depth)) | Low (O(depth)) |
| **Solution Quality** | Better (explores diverse paths) | Good (greedy optimization) |
| **Speed** | Slower initially | Faster to first solution |
| **LLM Calls** | More uniform distribution | Front-loaded, then backtracking |
| **Robustness** | High (beam search) | Medium (greedy selection) |
| **Predictability** | Deterministic within beam | Path-dependent |

---

## Recommendations

### **Use BFS when**:
- Quality is more important than speed
- You have sufficient computational resources
- The problem requires exploring diverse solutions
- Evaluation quality varies significantly

### **Use DFS when**:
- Speed/memory is critical
- Evaluation heuristics are reliable
- You want quick prototypes
- Running on resource-constrained hardware
- Testing different configurations

---

## Example Output Differences

### **BFS Output**:
```
Step 1/5: Evaluating 15 candidates → Keep top 3
Step 2/5: Evaluating 15 candidates → Keep top 3
Step 3/5: Evaluating 15 candidates → Keep top 3
Step 4/5: Evaluating 15 candidates → Keep top 3
Step 5/5: Evaluating 15 candidates → Keep top 1

Best value: 8.5
```

### **DFS Output**:
```
[DFS Step 1/5]
  Exploring thought=5 (value=0.9)
  [DFS Step 2/5]
    Exploring thought=12 (value=0.85)
    ...
    [LEAF] Value=8.2
  [DFS Step 2/5]
    Exploring thought=7 (value=0.82)
    ...
    [LEAF] Value=8.5 ★ New best!

Best value: 8.5
```

---

## Task-Decomposed Variants

Both BFS and DFS support task decomposition with multiple models:

```bash
# BFS with task decomposition
python scripts/tot_entity_summarizer_task_decomposed.py \
  --search-algorithm bfs \
  --model-relatedness meta-llama/Llama-3.2-1B-Instruct \
  --model-informativeness mistralai/Mistral-7B-Instruct-v0.2

# DFS with task decomposition
python scripts/tot_entity_summarizer_task_decomposed.py \
  --search-algorithm dfs \
  --model-diversity meta-llama/Llama-3.2-3B-Instruct
```

---

## Conclusion

- **BFS**: Standard for ToT, better quality through beam search
- **DFS**: Memory-efficient alternative with greedy exploration
- **Both**: Valid ToT implementations with different trade-offs
- **Choice**: Depends on your resource constraints and quality requirements

Try both and compare results for your specific dataset!
