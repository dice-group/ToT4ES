#!/usr/bin/env bash
set -euo pipefail

# Process WikiES benchmark test data with ToT entity summarization

# Configuration - customize these for different benchmark types
ROOT="../datasets/WikiES_benchmark/WikiCinema-s-test_data"
OUT="outputs/tot_wikies"
LOGS="logs/tot_wikies"

# Configuration
DATASET="wikicinema-s"
MAX_SUMMARY_LEN=5
N_CANDIDATES_PER_TASK=3
GPU_DEVICE=0
LIMIT_ENTITIES=0  # Set to 0 to process all, or change to 5, 10, etc. for testing

# Determinism settings
DETERMINISTIC=0   # 1 = deterministic/reproducible mode, 0 = default stochastic mode
SEED=42
THOUGHT_TEMPERATURE=0.8
EVAL_TEMPERATURE=0.3

# Model configuration (customize as needed)
MODEL_ID="Qwen/Qwen3-Coder-30B-A3B-Instruct"
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

if ((${#nt_files[@]} == 0)); then
  echo "No *_desc.nt files found under: $ROOT"
  echo "Check the path or rename your files to '{entity_id}_{entity_id}_desc.nt'."
  exit 1
fi

# Limit number of entities for testing if specified
if (( LIMIT_ENTITIES > 0 )); then
  nt_files=("${nt_files[@]:0:$LIMIT_ENTITIES}")
fi

echo "═══════════════════════════════════════════════════════════"
echo "Task-Decomposed ToT - WikiES Benchmark Entity Processing"
echo "═══════════════════════════════════════════════════════════"
echo "Total entities found: ${#nt_files[@]} to process"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Max summary length: $MAX_SUMMARY_LEN"
echo "  Candidates per task: $N_CANDIDATES_PER_TASK"
echo "  GPU device: $GPU_DEVICE"
echo "  Model: $MODEL_ID"
if (( DETERMINISTIC == 1 )); then
  echo "  Deterministic: enabled (seed=$SEED, thought_temp=0.0, eval_temp=0.0)"
else
  echo "  Deterministic: disabled (thought_temp=$THOUGHT_TEMPERATURE, eval_temp=$EVAL_TEMPERATURE)"
fi
echo ""
echo "Paths:"
echo "  Input data: $ROOT"
echo "  Output directory: $OUT"
echo "  Log directory: $LOGS"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Timing: record start time
start_time=$(date +%s)
start_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
entity_count=0
processed_count=0
skipped_count=0

# Process each entity
total_to_process=${#nt_files[@]}
current=0

for f in "${nt_files[@]}"; do
  current=$((current + 1))
  id="$(basename "$(dirname "$f")")"
  
  # Skip if output .nt file already exists (indicates processing completed)
  if [ -f "$OUT/$DATASET/$id/${id}_top${MAX_SUMMARY_LEN}.nt" ]; then
    echo "[$current/$total_to_process] [$id] ⊘ Skipped (output already exists)"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  
  proc_start=$(date +%s)
  echo ""
  echo "[$current/$total_to_process] [$id] Processing: $f"
  echo "  → Output: $OUT/$DATASET/$id/"
  echo "  → Log: $LOGS/$DATASET/$id/"

  # Ensure per-entity output/log directories exist
  mkdir -p "$OUT/$DATASET/$id" "$LOGS/$DATASET/$id"

  # Build command
  cmd="CUDA_VISIBLE_DEVICES=$GPU_DEVICE python tot_entity_summarizer_task_decomposed.py \
    --nt \"$f\" \
    --dataset \"$DATASET\" \
    --output-dir \"$OUT\" \
    --max-summary-len $MAX_SUMMARY_LEN \
    --n-candidates-per-task $N_CANDIDATES_PER_TASK \
    --thought-temperature $THOUGHT_TEMPERATURE \
    --eval-temperature $EVAL_TEMPERATURE \
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

  if (( DETERMINISTIC == 1 )); then
    cmd="$cmd --deterministic --seed $SEED"
    cmd="CUBLAS_WORKSPACE_CONFIG=:4096:8 $cmd"
  fi

  # Redirect output
  cmd="$cmd > \"$LOGS/$DATASET/$id/stdout.log\" 2> \"$LOGS/$DATASET/$id/stderr.log\""

  # Execute
  echo "  → Executing Python script..."
  if eval "$cmd"; then
    proc_end=$(date +%s)
    proc_time=$((proc_end - proc_start))
    echo "  ✓ [$id] Success (completed in ${proc_time}s)"
    processed_count=$((processed_count + 1))
  else
    echo "  ✗ [$id] Failed (see $LOGS/$DATASET/$id/stderr.log)"
    tail -10 "$LOGS/$DATASET/$id/stderr.log" | sed 's/^/    /'
  fi
  entity_count=$((entity_count + 1))
done

# Timing: record end time and calculate total and average
end_time=$(date +%s)
end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
total_time=$((end_time - start_time))

if (( processed_count > 0 )); then
  avg_time=$(awk "BEGIN {printf \"%.2f\", $total_time/$processed_count}")
else
  avg_time=0
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Processing Summary"
echo "═══════════════════════════════════════════════════════════"
echo "Start time:           $start_timestamp"
echo "End time:             $end_timestamp"
echo "Total runtime:        ${total_time}s"
echo ""
echo "Results:"
echo "  Processed:          $processed_count"
echo "  Skipped:            $skipped_count"
echo "  Total checked:      $entity_count"
echo "  Avg time per entity: ${avg_time}s"
echo ""
echo "Output locations:"
echo "  Results saved to:   $OUT"
echo "  Logs saved to:      $LOGS"
echo "═══════════════════════════════════════════════════════════"

# Save summary to overall_report.txt
REPORT_FILE="$OUT/overall_report.txt"
{
  echo "═══════════════════════════════════════════════════════════"
  echo "Task-Decomposed ToT - WikiES Benchmark Processing Report"
  echo "═══════════════════════════════════════════════════════════"
  echo "Start time:           $start_timestamp"
  echo "End time:             $end_timestamp"
  echo "Total runtime:        ${total_time}s"
  echo ""
  echo "Results:"
  echo "  Processed:          $processed_count"
  echo "  Skipped:            $skipped_count"
  echo "  Total checked:      $entity_count"
  echo "  Avg time per entity: ${avg_time}s"
  echo ""
  echo "Output locations:"
  echo "  Results saved to:   $OUT"
  echo "  Logs saved to:      $LOGS"
  echo "═══════════════════════════════════════════════════════════"
} | tee "$REPORT_FILE"

echo ""
echo "✓ All processing complete!"
