# Task-Decomposed Architecture Implementation

## Overview

Successfully implemented the **task-decomposition architecture** shown in your diagram! This approach treats Relatedness, Informativeness, and Diversity as **separate subtasks** with specialized prompts.

## Architecture Comparison

### Original Approach
```
Entity Graph → Single Unified Prompt → Thoughts → Evaluation
                (mentions all 3 criteria)
```

### New Task-Decomposed Approach (Your Diagram)
```
                    ┌─────────────────────┐
                    │   Entity Graph      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Decomposition      │
                    │  (Relatedness,      │
                    │   Informativeness,  │
                    │   Diversity)        │
                    └──┬────────┬────────┬┘
                       │        │        │
            ┌──────────▼┐  ┌───▼────┐  ┌▼────────────┐
            │Relatedness│  │Inform. │  │Diversity    │
            │  Prompt   │  │ Prompt │  │Prompt       │
            └──────┬────┘  └───┬────┘  └┬────────────┘
                   │           │         │
                   └───────┬───┴─────────┘
                           │
                    ┌──────▼──────────┐
                    │   Combine &     │
                    │   Evaluate      │
                    └─────────────────┘
```

## Implementation Files

### 1. `task_prompts.py` - Specialized Prompts
**Three focused prompts:**
- `make_relatedness_prompt()` - Focuses ONLY on centrality/core predicates
- `make_informativeness_prompt()` - Focuses ONLY on rare/unique facts
- `make_diversity_prompt()` - Focuses ONLY on coverage/variety

**Plus combined evaluation:**
- `make_combined_evaluation_prompt()` - Evaluates all 3 criteria together

### 2. `task_decomposed_search.py` - Search Algorithm
**Key class:** `TaskDecomposedToT`

**How it works:**
1. At each step, generates thoughts from **3 separate prompts**
2. Each prompt focuses on ONE criterion
3. Combines candidates from all 3 tasks
4. Evaluates using multi-criteria scoring
5. Prunes based on combined value

### 3. `tot_entity_summarizer_task_decomposed.py` - Main Script
**Usage:**
```bash
python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --max-summary-len 5 \
  --n-candidates-per-task 2
```

## Key Differences

| Aspect | Original | Task-Decomposed |
|--------|----------|-----------------|
| **Prompts** | 1 unified prompt | 3 specialized prompts |
| **Focus** | Mentions all criteria | Each prompt has 1 focus |
| **Thoughts/step** | n_candidates total | n_candidates × 3 tasks |
| **Diversity** | Implicit | Explicit task |
| **Clarity** | Mixed objectives | Clear separation |
| **Control** | Global | Per-criterion control |

## Benefits

### ✅ **1. Clearer Task Separation**
Each prompt has a single, focused objective:
- Relatedness prompt: "Select the MOST RELATED triple"
- Informativeness prompt: "Select the MOST INFORMATIVE triple"
- Diversity prompt: "Select the triple that MAXIMIZES DIVERSITY"

### ✅ **2. Better Specialization**
Each task can use criterion-specific guidance:
- Relatedness: Core predicates, entity identity
- Informativeness: Rarity, specificity, uniqueness
- Diversity: Different aspects, semantic roles, coverage

### ✅ **3. Increased Coverage**
By generating candidates from 3 different perspectives, you're more likely to:
- Cover all aspects of quality
- Avoid bias toward one criterion
- Generate more diverse candidates

### ✅ **4. Easier Debugging**
Can test each task independently:
```python
# Test just relatedness
relatedness_prompt = make_relatedness_prompt(entity, triples)
prompt = relatedness_prompt("", "1\n2")
print(prompt)  # See what LLM sees for relatedness task
```

### ✅ **5. Configurable Balance**
Control candidates per task:
```bash
# Generate 3 from each task = 9 total candidates
--n-candidates-per-task 3

# Generate 2 from each task = 6 total candidates  
--n-candidates-per-task 2
```

## Example Workflow

### Step 1: Entity Graph Input
```
<entity> <rdf:type> <Person> .
<entity> <birthDate> "1990-01-01" .
<entity> <occupation> "Scientist" .
<entity> <award> "Nobel Prize" .
```

### Step 2: Task Decomposition

**Relatedness Task:**
- Prompt focuses on: "Which triple is most central to defining this entity?"
- Likely selects: `<rdf:type> <Person>` (core identity)

**Informativeness Task:**
- Prompt focuses on: "Which triple provides most unique information?"
- Likely selects: `<award> "Nobel Prize"` (rare/specific)

**Diversity Task:**
- Prompt focuses on: "Which triple covers a different aspect?"
- Likely selects: `<birthDate> "1990-01-01"` (temporal aspect)

### Step 3: Combined Evaluation
All candidates evaluated on all 3 criteria:
```json
[
  {"idx": 0, "relatedness": 0.9, "informativeness": 0.5, "coverage": 0.6},
  {"idx": 1, "relatedness": 0.6, "informativeness": 0.9, "coverage": 0.7},
  {"idx": 2, "relatedness": 0.5, "informativeness": 0.6, "coverage": 0.9}
]
```

### Step 4: Selection
Weighted combination (0.4R + 0.4I + 0.2C) determines best states.

## Prompt Examples

### Relatedness Prompt
```
You are a RELATEDNESS expert for entity summarization.

Your ONLY goal is to select the triple that is MOST RELATED/CENTRAL.

Focus on:
1. Core predicates that define entity identity (rdf:type, rdfs:label)
2. Properties frequently used for this entity type
3. Highly specific and central values
4. Triples that best answer "What is this entity?"
```

### Informativeness Prompt
```
You are an INFORMATIVENESS expert for entity summarization.

Your ONLY goal is to select the triple that provides MOST INFORMATIVE content.

Focus on:
1. Rare/uncommon predicates (not generic like rdf:type)
2. Specific, detailed values (not generic categories)
3. Facts providing unique, non-obvious information
4. Deep ontological specificity
5. Information NOT already covered
```

### Diversity Prompt
```
You are a DIVERSITY/COVERAGE expert for entity summarization.

Your ONLY goal is to select the triple that MAXIMIZES DIVERSITY.

Focus on:
1. Different predicate types than already selected
2. Different semantic roles (location, time, relationship, etc.)
3. Dissimilar values to existing selections
4. Covering different aspects (biography, work, relations)
5. Avoiding redundancy
```

## Usage Examples

### Basic Usage
```bash
python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --max-summary-len 5
```

### Advanced Configuration
```bash
# Generate more candidates per task
python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt path/to/entity.nt \
  --dataset dbpedia \
  --max-summary-len 10 \
  --n-candidates-per-task 3 \
  --breadth-limit 5 \
  --n-evals 5
```

### Comparison Test
```bash
# Run original version
CUDA_VISIBLE_DEVICES=0 python scripts/tot_entity_summarizer_modular.py \
  --nt data.nt --dataset dbpedia --max-summary-len 5 \
  --output-dir results-original

# Run task-decomposed version
CUDA_VISIBLE_DEVICES=0 python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt data.nt --dataset dbpedia --max-summary-len 5 \
  --output-dir results-task-decomposed

# Compare outputs
diff results-original/dbpedia/1/1_top5.nt \
     results-task-decomposed/dbpedia/1/1_top5.nt
```

## Performance Considerations

### LLM Calls Per Step
- **Original:** n_candidates calls for thought generation
- **Task-Decomposed:** n_candidates_per_task × 3 calls for thought generation

**Example:**
- Original with n_candidates=6: **6 calls/step**
- Task-decomposed with n_candidates_per_task=2: **6 calls/step** (2×3)

**Recommendation:** Use `n_candidates_per_task=2` to match original cost while gaining task decomposition benefits.

### Quality vs. Cost Trade-off
```
n_candidates_per_task=1  → 3 calls/step  (fast, lower coverage)
n_candidates_per_task=2  → 6 calls/step  (balanced)
n_candidates_per_task=3  → 9 calls/step  (slower, better coverage)
```

## Testing

### Unit Test for Task Prompts
```python
from tot_modules.task_prompts import (
    make_relatedness_prompt,
    make_informativeness_prompt,
    make_diversity_prompt,
)

triples = ["<s> <p1> <o1> .", "<s> <p2> <o2> ."]

# Test each task
rel_prompt = make_relatedness_prompt("Entity", triples)
info_prompt = make_informativeness_prompt("Entity", triples)
div_prompt = make_diversity_prompt("Entity", triples)

# Generate prompts
print(rel_prompt("", ""))  # Relatedness for root state
print(info_prompt("", "1"))  # Informativeness after selecting triple 1
print(div_prompt("", "1\n2"))  # Diversity after selecting triples 1,2
```

### Integration Test
```bash
# Small test
python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --max-summary-len 3 \
  --n-candidates-per-task 1 \
  --breadth-limit 2
```

## Expected Output Structure

```
======================================================================
Task-Decomposed Tree-of-Thought Entity Summarization
======================================================================

Architecture:
  - Separate prompts for: Relatedness, Informativeness, Diversity
  - Combined evaluation of all criteria
  - Multi-task thought generation per step
======================================================================

...

Step 1 / 5
======================================================================

--- Expanding node 1/1 ---
Current state: TreeNode(depth=0, triples=0, value=0.0000)

  Generating thoughts for RELATEDNESS...
    relatedness: ['1', '3']
  Generating thoughts for INFORMATIVENESS...
    informativeness: ['5', '7']
  Generating thoughts for DIVERSITY...
    diversity: ['2', '9']

Combined unique thoughts: ['1', '3', '5', '7', '2', '9']
Children created: 6

--- Evaluating 6 states ---
...
```

## Files Created

1. ✅ `scripts/tot_modules/task_prompts.py` - 3 specialized prompts + evaluation
2. ✅ `scripts/tot_modules/task_decomposed_search.py` - Task-decomposed search algorithm
3. ✅ `scripts/tot_entity_summarizer_task_decomposed.py` - Main entry point

## Next Steps

### Immediate
- [x] Implement task-decomposed architecture
- [ ] Run comparison tests
- [ ] Measure quality differences

### Analysis
- [ ] Compare summaries: original vs. task-decomposed
- [ ] Measure diversity of selected triples
- [ ] Analyze which tasks contribute most to final selection

### Optimization
- [ ] Tune n_candidates_per_task for best balance
- [ ] Experiment with different task weightings
- [ ] Add task-specific temperature control

## Conclusion

The task-decomposed architecture is **fully implemented and ready to use**! This mirrors your diagram's approach of separating Relatedness, Informativeness, and Diversity into distinct subtasks, then combining them for evaluation.

**Key advantages:**
- Clearer task separation
- Better specialization per criterion
- Increased candidate diversity
- Easier debugging and testing
- Configurable per-task control

**Try it out:**
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/tot_entity_summarizer_task_decomposed.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --max-summary-len 5 \
  --n-candidates-per-task 2
```

🎉 **Architecture from your diagram successfully implemented!**
