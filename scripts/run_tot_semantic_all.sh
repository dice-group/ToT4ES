#!/usr/bin/env bash
set -euo pipefail

# Process all DBpedia entities with Semantic-Enhanced ToT (DL)

ROOT="../datasets/ESBM_benchmark_v1.2/dbpedia_data"
OUT="outputs/tot_semantic/dbpedia"
LOGS="logs/tot_semantic/dbpedia"

# Configuration
DATASET="dbpedia"
MAX_SUMMARY_LEN=5
N_CANDIDATES_PER_TASK=2
N_EVALS=3
BREADTH_LIMIT=3
SEARCH_ALGO="bfs"  # or "dfs"
GPU_DEVICE=3

# Model configuration
MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"

mkdir -p "$OUT" "$LOGS"

# Make globs vanish instead of staying literal when they don't match
shopt -s nullglob

# Collect all *_desc.nt files exactly one level under ROOT
nt_files=("$ROOT"/*/*_desc.nt)

if ((${#nt_files[@]}==0)); then
  echo "No *_desc.nt files found under: $ROOT"
  echo "Check the path or rename your files to '<id>_desc.nt'."
  exit 1
fi

echo "========================================="
echo "Semantic ToT4ES - Batch Processing"
echo "========================================="
echo "Found ${#nt_files[@]} entities to process"
echo "Output directory: $OUT"
echo "Log directory: $LOGS"
echo "Using GPU: $GPU_DEVICE"
echo "Model: $MODEL_ID"
echo "Search Algorithm: $SEARCH_ALGO"
echo ""

# Track progress
total="${#nt_files[@]}"
current=0
failed=0

# Process each entity
for f in "${nt_files[@]}"; do
  ((current++))
  id="$(basename "$(dirname "$f")")"
  
  echo "========================================="
  echo "[$current/$total] Processing Entity: $id"
  echo "========================================="

  # Ensure per-entity output/log directories exist
  mkdir -p "$OUT/$id" "$LOGS/$id"

  # Log files
  stdout_log="$LOGS/$id/stdout.log"
  stderr_log="$LOGS/$id/stderr.log"

  # Run semantic ToT
  if CUDA_VISIBLE_DEVICES=$GPU_DEVICE python tot_entity_summarizer_semantic.py \
    --nt "$f" \
    --dataset "$DATASET" \
    --model-id "$MODEL_ID" \
    --max-summary-len "$MAX_SUMMARY_LEN" \
    --n-candidates-per-task "$N_CANDIDATES_PER_TASK" \
    --n-evals "$N_EVALS" \
    --breadth-limit "$BREADTH_LIMIT" \
    --search-algorithm "$SEARCH_ALGO" \
    --output-dir "$OUT/$id" \
    --show-semantic-analysis \
    > "$stdout_log" 2> "$stderr_log"; then
    
    echo "✓ Success: Entity $id"
  else
    ((failed++))
    echo "✗ Failed: Entity $id (see $stderr_log)"
  fi
  
  echo ""
done

echo "========================================="
echo "Batch Processing Complete"
echo "========================================="
echo "Total: $total entities"
echo "Success: $((total - failed)) entities"
echo "Failed: $failed entities"
echo ""
echo "Results: $OUT"
echo "Logs: $LOGS"
