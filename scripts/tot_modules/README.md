# ToT Entity Summarizer - Modular Architecture

## Overview

This is a refactored, modular version of the Tree-of-Thought Entity Summarization system. The code has been split into focused, testable modules for easier debugging, testing, and maintenance.

## Module Structure

```
scripts/tot_modules/
├── __init__.py              # Package initialization and exports
├── tree_node.py             # TreeNode data structure
├── llm_wrapper.py           # LLM interface (LLaMA-3.2)
├── tree_search.py           # Main Tree-of-Thoughts algorithm
├── heuristic.py             # Score aggregation logic
└── utils.py                 # Utility functions (I/O, parsing)
```

## Module Descriptions

### 1. `tree_node.py`
**Purpose:** Tree node data structure  
**Key Features:**
- Stores state, thought, value, depth, parent, children
- Methods: `get_triple_ids()`, `get_path_from_root()`
- Clean separation of data structure from algorithms

### 2. `llm_wrapper.py`
**Purpose:** LLM interface abstraction  
**Key Features:**
- Wraps HuggingFace transformers pipeline
- Chat-style interface for easy use
- Configurable temperature and sampling

### 3. `tree_search.py`
**Purpose:** Core Tree-of-Thoughts algorithm  
**Key Features:**
- BFS with beam search pruning
- Thought generation with validation
- State evaluation using vote aggregation
- Comprehensive verbose output

### 4. `heuristic.py`
**Purpose:** Score aggregation  
**Key Features:**
- Multi-sample vote aggregation
- Weighted criterion combination
- Robust JSON parsing with fallbacks

### 5. `utils.py`
**Purpose:** Common utilities  
**Key Features:**
- N-Triples file loading
- State decoding
- Integer extraction

## Usage

### Basic Usage

```bash
python scripts/tot_entity_summarizer_modular.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --max-summary-len 5
```

### Advanced Usage

```bash
python scripts/tot_entity_summarizer_modular.py \
  --nt path/to/entity_desc.nt \
  --dataset dbpedia \
  --max-summary-len 10 \
  --n-candidates 7 \
  --n-evals 5 \
  --breadth-limit 5 \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --output-dir my-results
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--nt` | Required | Path to N-Triples file |
| `--dataset` | Required | Dataset name (dbpedia/lmdb/faces) |
| `--max-summary-len` | 5 | Maximum triples in summary |
| `--n-candidates` | 5 | Thought candidates per node |
| `--n-evals` | 3 | Evaluation samples for voting |
| `--breadth-limit` | 3 | Beam width for pruning |
| `--model-id` | llama-3.2-3b | HuggingFace model ID |
| `--output-dir` | tot-results-llama | Output directory |
| `--no-verbose` | False | Disable detailed output |

## Debugging Guide

### 1. Debug Individual Modules

```python
# Test tree node
from tot_modules.tree_node import TreeNode

node = TreeNode(state="1\n2\n3", thought="3", depth=1)
print(node.get_triple_ids())  # [1, 2, 3]

# Test LLM wrapper
from tot_modules.llm_wrapper import Llama32Chat

llm = Llama32Chat()
response = llm.chat([{"role": "user", "content": "Hello!"}])
print(response)

# Test utilities
from tot_modules.utils import load_entity_description_from_nt

label, triples = load_entity_description_from_nt("path/to/file.nt")
print(f"Entity: {label}, Triples: {len(triples)}")
```

### 2. Debug Search Process

Set `--no-verbose` to False (default) to see:
- Step-by-step expansion
- Generated thought candidates
- State evaluations
- Pruning decisions
- Final selection

### 3. Debug Evaluation

```python
from tot_modules.heuristic import entity_heuristic_calculator

states = ["1", "2"]
evals = ['[{"idx":0,"relatedness":0.8,"informativeness":0.7,"coverage":0.6}]']
scores = entity_heuristic_calculator(states, evals)
print(scores)
```

## Testing

### Unit Tests (TODO)

```bash
# Run all tests
python -m pytest tests/

# Run specific module tests
python -m pytest tests/test_tree_node.py
python -m pytest tests/test_utils.py
```

### Integration Tests

```bash
# Test the task-decomposed runner on a small entity
python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --max-summary-len 3 \
  --n-candidates-per-task 3 \
  --beam-width 2
```

## Benefits of Modular Design

### ✅ Easier Debugging
- Each module can be tested independently
- Clear separation of concerns
- Better error localization

### ✅ Better Maintainability
- Changes to one module don't affect others
- Clear module responsibilities
- Easier to understand code flow

### ✅ Improved Testability
- Unit tests for each module
- Mock dependencies easily
- Integration tests more focused

### ✅ Code Reusability
- Import only what you need
- Mix and match components
- Extend with new modules

### ✅ Better Documentation
- Each module documents its purpose
- Clearer API boundaries
- Easier onboarding

## Migration from Original

The original `tot_entity_summarizer.py` is preserved. The modular version provides the same functionality with better organization:

| Original | Modular |
|----------|---------|
| Single 820-line file | 7 focused modules |
| Mixed concerns | Clear separation |
| Hard to test | Easy unit testing |
| Hard to debug | Module-level debugging |

## Example: Task-Decomposed Search

```python
from tot_modules import Llama32Chat, entity_heuristic_calculator
from tot_modules.task_prompts import (
  make_relatedness_prompt,
  make_informativeness_prompt,
  make_diversity_prompt,
  make_combined_evaluation_prompt,
)
from tot_modules.task_decomposed_search import TaskDecomposedToT

all_triples = ["triple1", "triple2", "triple3"]
llm = Llama32Chat()
tot = TaskDecomposedToT(
  llm=llm,
  input_seq="\n".join(all_triples),
  get_relatedness_prompt=make_relatedness_prompt("Entity", all_triples),
  get_informativeness_prompt=make_informativeness_prompt("Entity", all_triples),
  get_diversity_prompt=make_diversity_prompt("Entity", all_triples),
  get_state_eval_prompt=make_combined_evaluation_prompt("Entity", all_triples),
  heuristic_calculator=entity_heuristic_calculator,
  num_triples=len(all_triples),
)

tot.n_steps = 3
tot.breadth_limit = 2
result = tot.bfs(verbose=True)
```

## Performance Tips

1. **Reduce LLM calls:** Lower `n_candidates` and `n_evals`
2. **Faster search:** Reduce `breadth_limit` and `max_summary_len`
3. **Better quality:** Increase all parameters (slower but better)
4. **Debug mode:** Use `--no-verbose` for production runs

## Troubleshooting

### Issue: Import errors
**Solution:** Make sure you're running from the scripts directory or adjust `sys.path`

### Issue: LLM out of memory
**Solution:** Use smaller model or reduce batch size in prompts

### Issue: All evaluations fail
**Solution:** Check LLM outputs in DEBUG section, adjust prompts if needed

### Issue: No valid children created
**Solution:** Increase `n_candidates` or check triple validation logic

## Task-Decomposed Architecture (NEW!)

A new implementation is available that decomposes the task into three separate prompts:

### Architecture Comparison

**Original (Unified):**
```
Entity → Single Prompt (all criteria) → Thoughts → Evaluation
```

**Task-Decomposed:**
```
Entity → ┌─ Relatedness Prompt    ─┐
         ├─ Informativeness Prompt ─┤ → Combined → Evaluation
         └─ Diversity Prompt       ─┘
```

### Files
- `task_prompts.py` - Three specialized prompts
- `task_decomposed_search.py` - Multi-task search algorithm
- `tot_entity_summarizer_task_decomposed.py` - Main script

### Usage
```bash
python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt data.nt \
  --dataset dbpedia \
  --max-summary-len 5 \
  --n-candidates-per-task 2  # 2 per task = 6 total
```

See `TASK_DECOMPOSED_IMPLEMENTATION.md` for details.

## Future Enhancements

- [x] Add unit tests for all modules
- [x] Implement task-decomposed architecture
- [ ] Add caching for LLM responses
- [ ] Implement DFS alternative
- [ ] Add visualization module
- [ ] Create configuration file support
- [ ] Add metrics tracking module
- [ ] Implement early stopping
- [ ] Add batch processing support

## License

Same as parent project (see main LICENSE file)
