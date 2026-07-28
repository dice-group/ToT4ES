# Chain-of-Thought LLM Entity Summarization

A baseline approach using **Chain-of-Thought (CoT)** prompting for entity summarization. Unlike direct prompting, this approach encourages the LLM to reason step-by-step before selecting the most important triples.

## Directory Structure

```
baseline_cot/
├── baseline_cot_llm.py           # Core CoT LLM summarizer
├── process_all_entities_cot.py   # Batch process all entities automatically
├── batch_processor_cot.py         # Process specific entity lists/ranges
└── README.md                      # This file
```

## CoT Prompting Strategy

The CoT approach guides the LLM through:

1. **Reasoning Phase**: Analyze candidate triples using Relatedness, Informativeness, and Coverage/Diversity
2. **Selection Phase**: Select the top-k triples after brief step-by-step reasoning

This structured reasoning helps the LLM produce more thoughtful and comprehensive summaries.

## Usage

### 1. Single Entity Summarization

Summarize a single entity:

```bash
python baseline_cot_llm.py \
    --entity-id 1 \
    --input-file ../datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
    --summary-size 5
```

**Arguments:**
- `--entity-id`: Entity ID (required)
- `--input-file`: Path to N-Triples file (required)
- `--entity-label`: Human-readable label (optional, auto-discovered)
- `--entity-uri`: Entity URI (optional, auto-discovered)
- `--output-dir`: Output directory (default: `baseline_cot_outputs`)
- `--summary-size`: Number of triples to select (default: 5)
- `--model`: HuggingFace model (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `--gpu`: GPU device ID (default: 0)
- `--temperature`: Sampling temperature (default: 0.1)
- `--max-new-tokens`: Maximum new tokens to generate (default: 2048)
- `--top-p`: Optional top-p nucleus sampling value
- `--no-sample`: Force greedy decoding regardless of temperature

### 2. Batch Process All Entities

Automatically process all entities in a dataset:

```bash
python process_all_entities_cot.py \
    --dataset dbpedia \
    --summary-size 5 \
    --model "Qwen/Qwen3-coder-30B-A3B-Instruct" \
    --temperature 0.8 \
    --max-new-tokens 1024 \
    --output-dir baseline_cot_outputs
```

**Arguments:**
- `--dataset`: Dataset to process - `dbpedia`, `lmdb`, or `faces` (required)
- `--dataset-root`: Root dataset directory (default: `../datasets/ESBM_benchmark_v1.2`)
- `--output-dir`: Output directory (default: `baseline_cot_outputs`)
- `--summary-size`: Summary size k (default: 5)
- `--model`: Model name (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `--gpu`: GPU device ID
- `--temperature`: Sampling temperature (default: 0.1)
- `--max-new-tokens`: Maximum new tokens to generate (default: 2048)
- `--top-p`: Optional top-p nucleus sampling value
- `--no-sample`: Force greedy decoding regardless of temperature
- `--skip-existing`: Skip entities already processed
- `--start-id`: Start from specific entity ID
- `--end-id`: End at specific entity ID

## Fairness-oriented matched setting

To reduce prompt and decoding confounds against ToT4ES without scoring:

- use the same backbone model as ToT4ES
- keep full indexed N-Triples in the prompt
- use the same temperature as ToT thought generation when comparing reasoning strategies
- keep `max-new-tokens` aligned across methods
- set `--top-p` only if you also match it in the compared setting

**Example:**
```bash
# Process dbpedia entities 1-100 with summary size 10
python process_all_entities_cot.py \
    --dataset dbpedia \
    --summary-size 10 \
    --start-id 1 \
    --end-id 100
```

### 3. Process Specific Entities

Process individual entities or a range:

```bash
# Process specific entities
python batch_processor_cot.py \
    --dataset dbpedia \
    --ids 1 5 10 15 \
    --summary-size 5

# Process a range
python batch_processor_cot.py \
    --dataset dbpedia \
    --range 1 50 \
    --summary-size 5
```

### 4. Evaluate CoT Summaries

Evaluate against ground truth with metrics:

```bash
python evaluate_cot_baseline.py \
    --dataset dbpedia \
    --summary-size 5 \
    --baseline-output-dir baseline_cot_outputs \
    --output-csv aggregated.csv \
    --detailed-csv detailed.csv
```

**Arguments:**
- `--dataset`: Dataset to evaluate
- `--all-datasets`: Evaluate all datasets
- `--summary-size`: Summary size k
- `--dataset-root`: Ground truth dataset root
- `--baseline-output-dir`: CoT output directory
- `--output-csv`: Save aggregated results
- `--detailed-csv`: Save per-entity results

**Output:**
- Aggregated metrics (precision, recall, F1) per dataset
- Detailed per-entity results with F1 per annotator

## Output Format

Summaries are saved as **N-Triples** (.nt files):

```
<http://dbpedia.org/resource/Entity_1> <http://dbpedia.org/ontology/type> <http://dbpedia.org/ontology/Person> .
<http://dbpedia.org/resource/Entity_1> <http://dbpedia.org/ontology/birthDate> "1980-01-01"^^<http://www.w3.org/2001/XMLSchema#date> .
```

**Directory Structure:**
```
baseline_cot_outputs/
├── dbpedia/
│   ├── 1/
│   │   ├── 1_top5.nt
│   │   └── 1_top10.nt
│   ├── 2/
│   │   ├── 2_top5.nt
│   │   └── 2_top10.nt
│   └── ...
├── lmdb/
│   └── ...
└── faces/
    └── ...
```

## Evaluation Metrics

The evaluation script computes:

- **Precision**: % of predicted triples in ground truth
  - P = |Sm ∩ Sh| / |Sm|
  
- **Recall**: % of ground truth triples in prediction
  - R = |Sm ∩ Sh| / |Sh|
  
- **F1-Score**: Harmonic mean
  - F = 2PR / (P + R)

For multiple annotators, F1 is **averaged** across all gold summaries.

## Comparison with Direct Baseline

| Aspect | Direct | CoT |
|--------|--------|-----|
| Prompting | Single-shot | Step-by-step reasoning |
| Reasoning | Implicit | Explicit |
| Token Usage | Lower | Higher |
| Quality | Baseline | Often better |
| Speed | Faster | Slower |

## GPU Support

Use `--gpu` to specify GPU device:

```bash
python process_all_entities_cot.py --dataset dbpedia --gpu 0
```

Or set `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python process_all_entities_cot.py --dataset dbpedia
```

## Notes

1. CoT prompting typically requires longer sequences and more tokens
2. Results may improve with higher temperature or different sampling strategies
3. The default model (LLaMA 3.2 3B) works but larger models may produce better summaries
4. Evaluation automatically discovers multiple ground truth annotations per summary size
5. Output directory automatically created if it doesn't exist

## File Summary

| File | Purpose |
|------|---------|
| `baseline_cot_llm.py` | Core CoT LLM implementation |
| `process_all_entities_cot.py` | Automatic batch processing |
| `batch_processor_cot.py` | Manual entity selection |
| `evaluate_cot_baseline.py` | Metrics evaluation |
