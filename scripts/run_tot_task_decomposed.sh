#!/usr/bin/env bash
set -euo pipefail

# Process all DBpedia entities with Task-Decomposed ToT

ROOT="../datasets/ESBM_benchmark_v1.2/dbpedia_data"
#ROOT="../datasets/FACES/faces_data"
OUT="outputs/tot_task_decomposed/dbpedia"
LOGS="logs/tot_task_decomposed/dbpedia"

# Configuration
DATASET="dbpedia"
MAX_SUMMARY_LEN=5
N_CANDIDATES_PER_TASK=3
GPU_DEVICE=3

# Model configuration (customize as needed)
MODEL_ID="Qwen/Qwen3-coder-30B"
# Uncomment to use task-specific models:
# MODEL_RELATEDNESS="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_INFORMATIVENESS="mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_DIVERSITY="meta-llama/Llama-3.2-3B-Instruct"
# MODEL_EVALUATION="meta-llama/Llama-3.2-3B-Instruct"

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

echo "Found ${#nt_files[@]} entities to process"
echo "Output directory: $OUT"
echo "Log directory: $LOGS"
echo "Using GPU: $GPU_DEVICE"
echo "Model: $MODEL_ID"
echo ""

# Process each entity
for f in "${nt_files[@]}"; do
  id="$(basename "$(dirname "$f")")"
  echo "[$id] Processing: $f"

  # Ensure per-entity output/log directories exist
  mkdir -p "$OUT/$id" "$LOGS/$id"

  # Build command
  cmd="CUDA_VISIBLE_DEVICES=$GPU_DEVICE python tot_entity_summarizer_task_decomposed.py \
    --nt \"$f\" \
    --dataset \"$DATASET\" \
    --max-summary-len $MAX_SUMMARY_LEN \
    --n-candidates-per-task $N_CANDIDATES_PER_TASK \
    --model-id \"$MODEL_ID\""

  # Add task-specific models if defined
  if [ -n "${MODEL_RELATEDNESS:-}" ]; then
    cmd="$cmd --model-relatedness \"$MODEL_RELATEDNESS\""
  fi
  if [ -n "${MODEL_INFORMATIVENESS:-}" ]; then
    cmd="$cmd --model-informativeness \"$MODEL_INFORMATIVENESS\""
  fi
  if [ -n "${MODEL_DIVERSITY:-}" ]; then
    cmd="$cmd --model-diversity \"$MODEL_DIVERSITY\""
  fi
  if [ -n "${MODEL_EVALUATION:-}" ]; then
    cmd="$cmd --model-evaluation \"$MODEL_EVALUATION\""
  fi

  # Redirect output
  cmd="$cmd > \"$LOGS/$id/stdout.log\" 2> \"$LOGS/$id/stderr.log\""

  # Execute
  eval "$cmd" && echo "[$id] ✓ Success" || echo "[$id] ✗ Failed (see $LOGS/$id/stderr.log)"
done

echo ""
echo "Processing complete!"
echo "Results saved to: $OUT"
echo "Logs saved to: $LOGS"
