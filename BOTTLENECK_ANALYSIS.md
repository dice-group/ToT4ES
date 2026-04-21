# Bottleneck Analysis: Task-Decomposed ToT Pipeline

**Date:** 2026-03-06  
**Run analyzed:** 2026-03-03 (FACES dataset, 49 entities, 38584s total, 787s/entity avg)

---

## Configuration Used

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` (MoE, 30B params, 3B active) |
| Search | BFS (default) |
| `n_steps` (max_summary_len) | 10 |
| `breadth_limit` | 3 |
| `n_candidates_per_task` | 3 (× 3 tasks = up to 9 thoughts/node) |
| `n_evals` | 3 |
| Dataset | FACES |
| GPU | CUDA device 3 |

---

## LLM Call Count Per Entity

Every LLM call goes through a **sequential `for _ in range(n)` loop** — no batching.

| Phase | Per Step | Calls | max_new_tokens |
|---|---|---|---|
| **Thought generation** | 3 nodes × 3 tasks × 3 samples | **27** | **1024** |
| **State evaluation** | ~2-3 chunks × 3 evals | **6-9** | **1024** |
| **Per step total** | | **~33-36** | |
| **Over 10 steps** | | **~310-340** | |

---

## Root Causes (in order of impact)

### 1. `max_new_tokens=1024` for single-integer answers (CRITICAL)

The thought generation prompts say *"Output ONLY the integer index"* — the expected output is a single token like `"7"`. Yet `max_new_tokens=1024` allows the model to generate up to 1024 tokens.

**Qwen3-Coder specifically outputs `<think>...</think>` reasoning blocks** before the actual answer, potentially generating hundreds of tokens of internal chain-of-thought before outputting one integer.

If the model averages ~200 tokens per call: 330 calls × 200 tokens × ~15ms/token ≈ **990 seconds** — which matches the observed 787s/entity.

**Files:** `scripts/tot_modules/task_decomposed_search.py` (calls `chat_completions` with default `max_tokens=1024`)

### 2. Sequential generation loop — no batching (HIGH)

In `scripts/tot_modules/llm_wrapper.py`:
```python
for _ in range(n):          # n=3, called sequentially
    out = self.pipe(prompt, ...)
```

Each of the 330+ calls runs **one at a time**. The HuggingFace pipeline supports batch inputs, but it wasn't used.

### 3. No stop sequences (HIGH)

After outputting the integer answer, the model continues generating until hitting `max_new_tokens`. No stop sequences were configured to halt generation after the first integer/newline.

### 4. Qwen3 thinking mode enabled by default (HIGH)

`Qwen3CoderChat` did not pass `enable_thinking=False` to `apply_chat_template()`, so the model generates extensive `<think>...</think>` reasoning blocks before every response — even for trivial "pick one integer" tasks.

### 5. BFS expansion cost (MODERATE)

BFS expands **all** nodes in the beam (up to `breadth_limit=3`) before pruning, generating 27 thought-generation calls per step. Many of those children are immediately discarded.

---

## Fixes Applied

### Fix 1: Reduced `max_new_tokens`

**File:** `scripts/tot_modules/task_decomposed_search.py`

- Thought generation: `1024` → `32` (output is a single integer)
- State evaluation: `1024` → dynamically sized `max(64, n_states * 40)` (output is ~30 chars/state)

### Fix 2: Batched LLM inference

**File:** `scripts/tot_modules/llm_wrapper.py`

When `n > 1` with sampling enabled, all `n` prompts are now passed as a batch to the HuggingFace pipeline instead of looping sequentially:

```python
if n > 1 and do_sample:
    batch_prompts = [prompt] * n
    batch_out = self.pipe(batch_prompts, ..., batch_size=n)
```

Applied to both `Llama32Chat` and `Qwen3CoderChat`.

### Fix 3: Disabled Qwen3 thinking mode

**Files:** `scripts/tot_modules/llm_wrapper.py`, `scripts/tot_modules/task_decomposed_search.py`

- Added `enable_thinking=False` parameter to `Qwen3CoderChat.chat()`
- `chat_completions()` now introspects the LLM's `chat()` signature and passes `enable_thinking=False` when supported
- Prevents generation of internal `<think>...</think>` reasoning blocks

### Fix 4: Added stop sequences

**File:** `scripts/tot_modules/task_decomposed_search.py`

- `stop=["\n", ".", ","]` for thought generation calls — halts generation as soon as the integer is emitted

### Fix 5: Added per-step timing instrumentation

**File:** `scripts/tot_modules/task_decomposed_search.py`

Each BFS step now logs `thought_gen` and `eval` time for profiling:
```
⏱  Step 3 timing: thought_gen=4.2s, eval=2.1s, total=6.3s
```

### Fix 6: Cleaned up duplicate code

**File:** `scripts/tot_modules/llm_wrapper.py`

Removed orphaned duplicate `Llama32Chat` class definition at end of file.

---

## Expected Improvement

| Metric | Before | After (estimated) |
|---|---|---|
| Tokens/thought call | ~200-500 | ~5-15 |
| Tokens/eval call | ~200-500 | ~50-150 |
| Inference mode | Sequential (×3) | Batched |
| Qwen3 thinking | Enabled (expensive) | Disabled |
| **Estimated time/entity** | **787s** | **~50-120s** |
| **Total (49 entities)** | **~10.7 hours** | **~40-100 min** |

The dominant speedup comes from fixes 1+3 combined — reducing generated tokens from ~300-500 down to ~5-15 per thought call across the ~270 thought-generation calls per entity.

---

## Further Optimization Opportunities (not yet implemented)

1. **KV-cache reuse across n samples**: Use `num_return_sequences=n` in a single `pipeline()` call instead of batching `n` copies of the same prompt
2. **Greedy-first thought generation**: Use `temperature=0` for the first candidate per task, then sample for diversity — reduces calls while maintaining coverage
3. **Early pruning**: Evaluate after expanding each node (not after expanding all nodes in the layer) to skip generating thoughts for nodes that won't survive pruning
4. **Reduce `n_candidates_per_task` from 3 to 2**: The marginal value of the 3rd candidate is low since most produce the same integer anyway
5. **Use a smaller model for thought generation**: The task is trivial (pick an integer); a 1B-3B model suffices for thought generation, reserving the larger model for evaluation only
