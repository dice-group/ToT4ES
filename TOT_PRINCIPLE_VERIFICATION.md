# Tree of Thoughts Principle Verification

## Overview
This document verifies that our Task-Decomposed ToT for Entity Summarization follows the core principles of the Tree-of-Thoughts (ToT) framework.

## Core ToT Principles

### 1. ✅ Thought Decomposition
**Principle**: Break down complex problems into intermediate reasoning steps (thoughts).

**Our Implementation**:
- **States**: Sequences of triple indices representing partial summaries (e.g., `"2\n5\n9"`)
- **Thoughts**: Individual triple selections that extend current states
- **Deliberate Reasoning**: Each step adds exactly one triple through multi-step reasoning
- **Domain**: Entity summarization requires selecting k triples from n candidates

**Code Reference**: `TreeNode` in `tot_modules/tree_node.py`
```python
@dataclass
class TreeNode:
    state: str          # Newline-separated triple indices
    thought: str        # Last triple index added
    depth: int          # Step in reasoning process
    parent: Optional["TreeNode"]
    children: List["TreeNode"]
```

---

### 2. ✅ Thought Generation (LLM-based)
**Principle**: Use LLM to generate multiple candidate thoughts for each state.

**Our Implementation**:
- Generates **n_candidates** (default: 5) thoughts per state
- Uses **temperature=0.8** for diverse candidate generation
- **Task-decomposed variant**: Generates from 3 specialized prompts
  - Relatedness prompt: Core predicates, entity identity
  - Informativeness prompt: Unique properties, distinctive features
  - Diversity prompt: Topic coverage, complementary information

**Code Reference**: `thought_generator()` in `tot_modules/tree_search.py`
```python
def thought_generator(self, state: str) -> List[str]:
    prompt = self.get_thought_gen_prompt(self.input_seq, state)
    raw_thoughts = self.chat_completions(
        prompt=prompt,
        temperature=0.8,  # Higher temp for diverse generation
        n=self.n_candidates,  # Multiple candidates
    )
    # Extract and validate triple indices
    thought_ids = [extract_first_int(txt) for txt in raw_thoughts]
    return unique_thoughts
```

**Task-Decomposed Version**: `generate_thoughts_all_tasks()` in `tot_modules/task_decomposed_search.py`
```python
def generate_thoughts_all_tasks(self, state: str) -> Dict[str, List[str]]:
    tasks = {
        "relatedness": (self.get_relatedness_prompt, self.llm_relatedness),
        "informativeness": (self.get_informativeness_prompt, self.llm_informativeness),
        "diversity": (self.get_diversity_prompt, self.llm_diversity),
    }
    
    for task_name, (prompt_fn, task_llm) in tasks.items():
        thoughts = self.task_thought_generator(state, prompt_fn, task_name, llm=task_llm)
        all_thoughts[task_name] = thoughts
```

---

### 3. ✅ State Evaluation (LLM-based)
**Principle**: Use LLM to evaluate the quality/promise of different states.

**Our Implementation**:
- **Vote-based evaluation**: Queries LLM n_evals times (default: 3)
- Uses **temperature=0.3** for consistent, focused evaluation
- **Multi-criteria scoring**:
  - Relatedness: 40% weight
  - Informativeness: 40% weight
  - Diversity/Coverage: 20% weight
- Aggregates votes using heuristic calculator

**Code Reference**: `state_evaluator()` in `tot_modules/tree_search.py`
```python
def state_evaluator(self, states: List[str]) -> List[float]:
    prompt = self.get_state_eval_prompt(self.input_seq, states)
    state_evals = self.chat_completions(
        prompt, 
        temperature=0.3,  # Lower temp for consistent evaluation
        n=self.n_evals    # Multiple votes for robustness
    )
    vote_results = self.heuristic_calculator(states, state_evals)
    return vote_results
```

**Heuristic Calculator**: `entity_heuristic_calculator()` in `tot_modules/heuristic.py`
```python
def entity_heuristic_calculator(
    states: List[str], 
    vote_outputs: List[str]
) -> List[float]:
    # Parse LLM votes for each state
    # Extract relatedness, informativeness, coverage scores
    # Weight: 0.4 * R + 0.4 * I + 0.2 * C
    # Aggregate across multiple votes
```

---

### 4. ✅ Search Algorithm (BFS with Beam Pruning)
**Principle**: Systematically explore the thought tree using search algorithms.

**Our Implementation**:
- **Breadth-First Search (BFS)**: Explores all nodes at depth d before depth d+1
- **Beam Pruning**: Keeps only top-k (breadth_limit=3) states per layer
- **Lookahead**: Evaluates states before committing to expansion
- **n_steps**: 5 iterations (selects 5 triples total)

**Code Reference**: `bfs()` in `tot_modules/tree_search.py`
```python
def bfs(self, verbose: bool = True) -> str:
    queue = deque()
    queue.append(self.root)
    
    for step in range(1, self.n_steps + 1):
        current_layer_size = len(queue)
        new_nodes = deque()
        
        # Expand all nodes in current layer
        for i in range(current_layer_size):
            node = queue.popleft()
            thoughts = self.thought_generator(node.state)
            
            for thought in thoughts:
                child = TreeNode(
                    state=new_state,
                    thought=thought,
                    depth=node.depth + 1,
                    parent=node
                )
                new_nodes.append(child)
        
        # Evaluate and prune
        if len(new_nodes) > self.breadth_limit:
            states = [n.state for n in new_nodes]
            scores = self.state_evaluator(states)
            
            # Keep top-k nodes
            scored_nodes = sorted(
                zip(new_nodes, scores),
                key=lambda x: x[1],
                reverse=True
            )
            new_nodes = deque([n for n, s in scored_nodes[:self.breadth_limit]])
        
        queue = new_nodes
    
    # Return best final state
    return best_node.state
```

**Search Parameters**:
- `n_steps = 5`: Maximum search depth
- `breadth_limit = 3`: Beam width (top-k pruning)
- `n_candidates = 5`: Thoughts generated per node
- `n_evals = 3`: Evaluation votes per state

---

### 5. ✅ Tree Structure
**Principle**: Maintain explicit tree structure for backtracking and exploration.

**Our Implementation**:
- **Parent-child links**: Each node references its parent
- **Path tracking**: `get_path_from_root()` retrieves full reasoning chain
- **State history**: Complete expansion history preserved
- **Backtracking capable**: Can explore alternative branches

**Code Reference**: `TreeNode` in `tot_modules/tree_node.py`
```python
@dataclass
class TreeNode:
    state: str
    thought: str
    value: float = 0.0
    depth: int = 0
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)
    
    def get_path_from_root(self) -> List["TreeNode"]:
        """Get the path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
```

---

## Our Innovation: Task-Decomposed ToT

We **extend** standard ToT with multi-task thought generation:

### Enhanced Thought Generation
- **3 parallel thought generators** instead of 1:
  1. Relatedness-focused generator
  2. Informativeness-focused generator
  3. Diversity-focused generator
- Each generates `n_candidates_per_task` thoughts
- Total candidates per step: `3 × n_candidates_per_task`

### Configurable Multi-Model Architecture
- **Separate LLMs per task** (optional):
  - `llm_relatedness`: Model for relatedness task
  - `llm_informativeness`: Model for informativeness task
  - `llm_diversity`: Model for diversity task
  - `llm_evaluation`: Model for state evaluation
- **Fallback mechanism**: Uses default LLM if task-specific not provided

### Combined Multi-Criteria Evaluation
- **Single evaluation** considers all 3 criteria simultaneously
- Weighted aggregation: 40% R + 40% I + 20% C
- Vote-based robustness across multiple samples

---

## Comparison with Standard ToT

| Component | Standard ToT | Our Implementation |
|-----------|-------------|-------------------|
| **Thought Generation** | Single LLM prompt | 3 task-specific prompts |
| **LLM Configuration** | One model | Configurable per task (4 models) |
| **Evaluation** | General quality | Multi-criteria (R+I+D) |
| **Search Algorithm** | BFS/DFS/A* | BFS with beam pruning |
| **State Representation** | Problem-specific | Triple index sequences |
| **Pruning Strategy** | Top-k | Beam search (k=3) |

---

## Verification Checklist

- [x] **Deliberate Multi-Step Reasoning**: 5-step sequential triple selection
- [x] **LLM-Based Thought Generation**: Multiple candidates via LLM
- [x] **LLM-Based State Evaluation**: Vote-based quality assessment
- [x] **Tree Structure**: Explicit parent-child relationships
- [x] **Search Algorithm**: BFS with beam pruning
- [x] **Lookahead**: Evaluates states before expansion
- [x] **Temperature Control**: 0.8 for generation, 0.3 for evaluation
- [x] **Duplicate Prevention**: State validation before child creation
- [x] **Backtracking Capability**: Full path reconstruction

---

## Conclusion

✅ **Our approach fully adheres to Tree-of-Thoughts principles** while introducing:

1. **Task Decomposition**: Separate reasoning paths for different criteria
2. **Multi-Model Support**: Specialized models for specialized tasks
3. **Multi-Objective Optimization**: Simultaneous optimization of R+I+D

This creates a **Task-Decomposed Tree-of-Thoughts (TD-ToT)** framework specifically designed for entity summarization.

---

## References

**Tree-of-Thoughts Original Paper**:
- Yao et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
- Key principles: Thought decomposition, LLM-based generation/evaluation, systematic search

**Our Implementation Files**:
- `tot_modules/tree_node.py` - Tree structure
- `tot_modules/tree_search.py` - Standard ToT BFS
- `tot_modules/task_decomposed_search.py` - Task-decomposed variant
- `tot_modules/llm_wrapper.py` - LLM interface
- `tot_modules/heuristic.py` - Multi-criteria evaluation
- `tot_modules/task_prompts.py` - Task-specific prompts

**Architecture Diagram**:
- `ToT4ES_Architecture_Updated.jpg` - Visual representation of multi-model TD-ToT
