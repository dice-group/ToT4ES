#!/usr/bin/env bash
set -euo pipefail

# Process WikiES benchmark test data with ToT entity summarization

# ============================================================================
# Configuration
# ============================================================================

# Get script directory and resolve to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Dataset directories - one for each benchmark type
DATASET_DIRS=(
  "$PROJECT_ROOT/datasets/WikiES_benchmark/WikiCinema-s-test_data"
  "$PROJECT_ROOT/datasets/WikiES_benchmark/WikiLitArt-s-test_data"
  "$PROJECT_ROOT/datasets/WikiES_benchmark/WikiPro-s-test_data"
  "$PROJECT_ROOT/datasets/WikiES_benchmark/WikiProFem-s-test_data"
)

OUT="$PROJECT_ROOT/outputs/tot_wikies"
LOGS="$PROJECT_ROOT/logs/tot_wikies"

# ToT Configuration
DATASET_NAME="wikies"
MAX_SUMMARY_LEN=5
N_CANDIDATES=5
N_EVALS=3
BREADTH_LIMIT=3
GPU_DEVICE=0
LIMIT_ENTITIES=0  # Set to 0 for all, or e.g. 5 for quick test

# Model
MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"

# ============================================================================
# Script setup
# ============================================================================

mkdir -p "$OUT" "$LOGS"
shopt -s nullglob

# Parse arguments for overrides
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-len)
      MAX_SUMMARY_LEN="$2"
      shift 2
      ;;
    --n-candidates)
      N_CANDIDATES="$2"  
      shift 2
      ;;
    --n-evals)
      N_EVALS="$2"
      shift 2
      ;;
    --breadth)
      BREADTH_LIMIT="$2"
      shift 2
      ;;
    --limit)
      LIMIT_ENTITIES="$2"
      shift 2
      ;;
    --gpu)
      GPU_DEVICE="$2"
      shift 2
      ;;
    --model)
      MODEL_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ============================================================================
# Collect all entities from all benchmark types
# ============================================================================

all_nt_files=()
benchmark_types=()

for dataset_dir in "${DATASET_DIRS[@]}"; do
  if [ ! -d "$dataset_dir" ]; then
    echo "WARNING: Dataset directory not found: $dataset_dir"
    continue
  fi
  
  # Collect *_desc.nt files one level deep: entity_id/entity_id_desc.nt
  nt_files=("$dataset_dir"/*_desc.nt)
  
  if ((${#nt_files[@]} == 0)); then
    # Try nested pattern if flat pattern didn't work
    nt_files=("$dataset_dir"/*/*_desc.nt)
  fi
  
  if ((${#nt_files[@]} > 0)); then
    for f in "${nt_files[@]}"; do
      all_nt_files+=("$f")
    done
    
    type_name=$(basename "$dataset_dir" | sed 's/-test_data//')
    benchmark_types+=("$type_name")
  fi
done

# Check if any files found
if ((${#all_nt_files[@]} == 0)); then
  echo "ERROR: No *_desc.nt files found in any dataset directories:"
  for dir in "${DATASET_DIRS[@]}"; do
    echo "  - $dir"
  done
  exit 1
fi

# Apply limit if specified
if (( LIMIT_ENTITIES > 0 )); then
  all_nt_files=("${all_nt_files[@]:0:$LIMIT_ENTITIES}")
fi

# ============================================================================
# Print header
# ============================================================================

echo "═══════════════════════════════════════════════════════════════════════"
echo "ToT4ES - WikiES Benchmark Entity Summarization"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Dataset name:       $DATASET_NAME"
echo "  Max summary length: $MAX_SUMMARY_LEN triples"
echo "  Thought candidates: $N_CANDIDATES"
echo "  Evaluation votes:   $N_EVALS"
echo "  Breadth limit:      $BREADTH_LIMIT"
echo "  Model:              $MODEL_ID"
echo "  GPU device:         $GPU_DEVICE"
echo ""
echo "Benchmark types:"
for type in $(printf '%s\n' "${benchmark_types[@]}" | sort -u); do
  echo "  - $type"
done
echo ""
echo "Paths:"
echo "  Output directory:   $OUT"
echo "  Log directory:      $LOGS"
echo ""
echo "Total entities to process: ${#all_nt_files[@]}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# Main processing loop
# ============================================================================

start_time=$(date +%s)
start_timestamp=$(date '+%Y-%m-%d %H:%M:%S')

processed_count=0
skipped_count=0
failed_count=0
failed_entities=()

total_to_process=${#all_nt_files[@]}
current=0

for nt_file in "${all_nt_files[@]}"; do
  current=$((current + 1))
  
  # Extract entity ID from path: .../1234/1234_desc.nt -> 1234
  entity_id="$(basename "$(dirname "$nt_file")")"
  
  # Extract benchmark type from path
  benchmark_type=$(echo "$nt_file" | sed -E 's|.*WikiES_benchmark/([^/]*)-test_data.*|\1|')
  
  # Determine dataset label for output
  dataset_label="WikiES_${benchmark_type}"
  
  # Check if already processed
  output_file="$OUT/$dataset_label/$entity_id/${entity_id}_top${MAX_SUMMARY_LEN}.nt"
  if [ -f "$output_file" ]; then
    echo "[$current/$total_to_process] [$entity_id] ⊘ Skipped (output exists)"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  
  proc_start=$(date +%s)
  echo ""
  echo "[$current/$total_to_process] [$entity_id] Processing: $nt_file"
  echo "  → Type: $benchmark_type"
  echo "  → Output: $OUT/$dataset_label/$entity_id/"
  
  # Create per-entity directories
  mkdir -p "$OUT/$dataset_label/$entity_id" "$LOGS/$dataset_label/$entity_id"
  
  # Build and execute Python command
  cmd="cd \"$PROJECT_ROOT\" && CUDA_VISIBLE_DEVICES=$GPU_DEVICE python scripts/tot_entity_summarizer.py \
    --nt \"$nt_file\" \
    --dataset \"$dataset_label\" \
    --max-summary-len $MAX_SUMMARY_LEN \
    --n-candidates $N_CANDIDATES \
    --n-evals $N_EVALS \
    --breadth-limit $BREADTH_LIMIT \
    --model-id \"$MODEL_ID\" \
    --no-verbose"
  
  if eval "$cmd" > "$LOGS/$dataset_label/$entity_id/stdout.log" 2> "$LOGS/$dataset_label/$entity_id/stderr.log"; then
    proc_end=$(date +%s)
    proc_time=$((proc_end - proc_start))
    echo "  ✓ Success (${proc_time}s)"
    processed_count=$((processed_count + 1))
  else
    echo "  ✗ Failed (see $LOGS/$dataset_label/$entity_id/stderr.log)"
    tail -5 "$LOGS/$dataset_label/$entity_id/stderr.log" | sed 's/^/    /'
    failed_count=$((failed_count + 1))
    failed_entities+=("$entity_id")
  fi
done

# ============================================================================
# Final reporting
# ============================================================================

end_time=$(date +%s)
end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
total_time=$((end_time - start_time))

if (( processed_count > 0 )); then
  avg_time=$(awk "BEGIN {printf \"%.2f\", $total_time/$processed_count}")
  success_rate=$(awk "BEGIN {printf \"%.1f\", 100*$processed_count/($processed_count+$skipped_count+$failed_count)}")
else
  avg_time=0
  success_rate=0
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Processing Summary"
echo "═══════════════════════════════════════════════════════════════════════"
echo "Start time:           $start_timestamp"
echo "End time:             $end_timestamp"
echo "Total runtime:        ${total_time}s"
echo ""
echo "Results:"
echo "  Processed:          $processed_count"
echo "  Skipped:            $skipped_count"
echo "  Failed:             $failed_count"
echo "  Success rate:       ${success_rate}%"
echo "  Avg time per entity: ${avg_time}s"
echo ""

if (( failed_count > 0 )); then
  echo "Failed entities:"
  for id in "${failed_entities[@]}"; do
    echo "  - $id"
  done
  echo ""
fi

echo "Output locations:"
echo "  Results saved to:   $OUT"
echo "  Logs saved to:      $LOGS"
echo "═══════════════════════════════════════════════════════════════════════"

# ============================================================================
# Save report file
# ============================================================================

REPORT_FILE="$OUT/overall_report.txt"
{
  echo "═══════════════════════════════════════════════════════════════════════"
  echo "ToT4ES WikiES Benchmark - Experiment Report"
  echo "═══════════════════════════════════════════════════════════════════════"
  echo ""
  echo "Configuration:"
  echo "  Max summary length: $MAX_SUMMARY_LEN"
  echo "  Thought candidates: $N_CANDIDATES"
  echo "  Evaluation votes:   $N_EVALS"
  echo "  Breadth limit:      $BREADTH_LIMIT"
  echo "  Model:              $MODEL_ID"
  echo ""
  echo "Start time:           $start_timestamp"
  echo "End time:             $end_timestamp"
  echo "Total runtime:        ${total_time}s"
  echo ""
  echo "Results:"
  echo "  Processed:          $processed_count"
  echo "  Skipped:            $skipped_count"
  echo "  Failed:             $failed_count"
  echo "  Success rate:       ${success_rate}%"
  echo "  Avg time per entity: ${avg_time}s"
  echo ""
  if (( failed_count > 0 )); then
    echo "Failed entities:"
    for id in "${failed_entities[@]}"; do
      echo "  - $id"
    done
    echo ""
  fi
  echo "Output locations:"
  echo "  Results saved to:   $OUT"
  echo "  Logs saved to:      $LOGS"
  echo "═══════════════════════════════════════════════════════════════════════"
} | tee "$REPORT_FILE"

echo ""
if (( failed_count == 0 )); then
  echo "✓ All processing complete successfully!"
else
  echo "⚠ Processing complete with $failed_count failures. Check logs for details."
fi
