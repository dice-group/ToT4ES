#!/usr/bin/env bash
set -euo pipefail

# Process all DBpedia entities with Semantic-Enhanced ToT (DL)

ROOT="../datasets/ESBM_benchmark_v1.2/dbpedia_data"
OUT="outputs/tot_semantic/dbpedia"
LOGS="logs/tot_semantic/dbpedia"

# Configuration (OPTIMIZED FOR SPEED)
DATASET="dbpedia"
MAX_SUMMARY_LEN=5
N_CANDIDATES_PER_TASK=3  # Reduced from 2 (2x faster)
N_EVALS=2                 # Reduced from 3 (1.5x faster)
BREADTH_LIMIT=2           # Reduced from 3 (1.5x faster)
SEARCH_ALGO="bfs"         # or "dfs"
GPU_DEVICE=3

# Model configuration
# Use 3B for better evaluation reliability (simple format works better)
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

echo "DEBUG: About to enter loop with $total files"
echo "DEBUG: First file: ${nt_files[0]}"

# Process each entity
for f in "${nt_files[@]}"; do
  echo "DEBUG: Inside loop, processing: $f"
  current=$((current + 1))  # Safer than ((current++)) with set -e
  id="$(basename "$(dirname "$f")")"
  id="$(basename "$(dirname "$f")")"
  
  echo "========================================="
  echo "[$current/$total] Processing Entity: $id"
  echo "File: $f"
  echo "========================================="

  # Ensure per-entity output/log directories exist
  mkdir -p "$OUT/$id" "$LOGS/$id"

  # Log files
  stdout_log="$LOGS/$id/stdout.log"
  stderr_log="$LOGS/$id/stderr.log"
  
  echo "Starting Python script for entity $id..."

  # Run semantic ToT
  CUDA_VISIBLE_DEVICES=$GPU_DEVICE python tot_entity_summarizer_semantic.py \
    --nt "$f" \
    --dataset "$DATASET" \
    --model-id "$MODEL_ID" \
    --max-summary-len "$MAX_SUMMARY_LEN" \
    --n-candidates-per-task "$N_CANDIDATES_PER_TASK" \
    --n-evals "$N_EVALS" \
    --breadth-limit "$BREADTH_LIMIT" \
    --search-algorithm "$SEARCH_ALGO" \
    --output-dir "$OUT/$id" \
    --no-verbose \
    > "$stdout_log" 2> "$stderr_log"
  
  # Check exit status
  exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    echo "✓ Success: Entity $id"
  else
    failed=$((failed + 1))  # Safer than ((failed++))
    echo "✗ Failed: Entity $id (exit code: $exit_code, see $stderr_log)"
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
