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