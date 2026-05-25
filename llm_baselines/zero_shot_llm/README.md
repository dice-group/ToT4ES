# Baseline: Zero-shot LLM Entity Summarization

This folder contains a **baseline approach** for entity triple summarization that uses **Zero-shot LLM prompting**.

## How to Usage

### Single Entity

```bash
python baseline_direct_llm.py \
  --triple-file ../datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --entity-id 1 \
  --entity-uri "http://dbpedia.org/resource/Marie_Curie" \
  --entity-label "Marie Curie" \
  --summary-size 5
```

Output saved to: `baseline_outputs/dbpedia_data/1/1_top5.nt`

### All entities

```bash
python process_all_entities.py \
  --dataset dbpedia \
  --summary-size 5 \
  --output-dir outputs \
  --temperature 0.1 \
  --max-entities 125 \
  --model "Qwen/Qwen3-coder-30B-A3B-Instruct"
  --skip-existing \
  --log-file dbpedia_results.txt
```

## How to Evaluate

This repository includes an evaluation script that computes Precision, Recall and F1-score
for baseline summaries against the gold annotations in the ESBM benchmark.

- Script: `evaluate_baseline.py`
- Default dataset root: `../datasets/ESBM_benchmark_v1.2`
- Default baseline outputs dir: `outputs`

Important: if you generated outputs using `process_all_entities.py --output-dir outputs`,
either move/rename that folder to `baseline_outputs` or pass `--baseline-output-dir outputs`
to the evaluator.

Basic usage examples:

```bash
# Evaluate DBpedia (k=5)
python evaluate_baseline.py --dataset dbpedia --summary-size 5

# Save aggregated results to CSV
python evaluate_baseline.py --dataset dbpedia --summary-size 5 --output-csv results/dbpedia_results.csv

# Evaluate all datasets and save per-entity detailed CSV
python evaluate_baseline.py --all-datasets --summary-size 5 \
  --output-csv results/all_results.csv \
  --detailed-csv results/detailed_per_entity.csv
```

Useful flags:

- `--dataset`: `dbpedia`, `lmdb`, or `faces` (default: `dbpedia`)
- `--all-datasets`: evaluate all three datasets
- `--summary-size`: summary size k (default: 5)
- `--dataset-root`: change dataset root if your ESBM files are elsewhere
- `--baseline-output-dir`: path to generated baseline outputs (default: `baseline_outputs`)
- `--output-csv`: save aggregated per-dataset metrics to a CSV file
- `--detailed-csv`: save per-entity detailed metrics to a CSV file

Expected baseline output layout (used by the evaluator):

```
