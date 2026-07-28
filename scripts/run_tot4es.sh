#!/usr/bin/env bash
set -euo pipefail

# Default paths and configuration (can be overridden via CLI flags)
ROOT="../datasets/ESBM_benchmark_v1.2/dbpedia_data"
OUT="outputs/results_1"
LOGS="logs/experiment_1"

# Default paths and configuration (can be overridden via CLI flags)
DATASET="dbpedia"
MAX_SUMMARY_LEN=5
N_CANDIDATES_PER_TASK=${N_CANDIDATES_PER_TASK:-3}  # Ablation: override to 1 for greedy
GPU_DEVICE=0
LIMIT_ENTITIES=${LIMIT_ENTITIES:-0}  # Set to 0 to process all, or change to 2, 5, etc. for testing

# Model configuration (customize as needed)
MODEL_ID="Qwen/Qwen3-coder-30B-A3B-Instruct"

# LLM temperatures (defaults used by the Python script)
THOUGHT_TEMPERATURE=0.8
EVAL_TEMPERATURE=0.3
DO_SAMPLE=${DO_SAMPLE:-auto}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -d DIR    Dataset root directory (overrides ROOT)
  -o DIR    Output directory
  -l DIR    Logs directory
  -n NAME   Dataset name (dbpedia, faces, lmdb, ...)
  -k INT    Max summary length (top-k triples)
  -m ID     Model id (HuggingFace model identifier)
  -t FLOAT  Thought temperature
  -e FLOAT  Eval temperature
  -S MODE   Sampling mode: auto|true|false
  -g INT    GPU device index or list (sets CUDA_VISIBLE_DEVICES in subcommands)
  -L INT    Limit entities (for quick tests)
  -r FILE   Runtime report JSON path
  -s MODE   No-scoring mode (currently: greedy)
  -h        Show this help
EOF
  exit 1
}

# Support long options for flags that need shell-side handling.
NEWARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      if [[ -n "${2-}" ]]; then
        GPU_DEVICE="$2"
        shift 2
        continue
      else
        echo "Option --gpu requires an argument." >&2
        usage
      fi
      ;;
    --gpu=*)
      GPU_DEVICE="${1#*=}"
      shift
      continue
      ;;
    --runtime-report)
      if [[ -n "${2-}" ]]; then
        RUNTIME_REPORT_FILE="$2"
        shift 2
        continue
      else
        echo "Option --runtime-report requires an argument." >&2
        usage
      fi
      ;;
    --runtime-report=*)
      RUNTIME_REPORT_FILE="${1#*=}"
      shift
      continue
      ;;
    --no-scoring-mode)
      if [[ -n "${2-}" ]]; then
        NO_SCORING_MODE="$2"
        shift 2
        continue
      else
        echo "Option --no-scoring-mode requires an argument." >&2
        usage
      fi
      ;;
    --no-scoring-mode=*)
      NO_SCORING_MODE="${1#*=}"
      shift
      continue
      ;;
    *)
      NEWARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${NEWARGS[@]}"

# Parse short CLI options
while getopts ":d:o:l:n:k:m:t:e:S:g:L:r:s:h" opt; do
  case $opt in
    d) ROOT="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    l) LOGS="$OPTARG" ;;
    n) DATASET="$OPTARG" ;;
    k) MAX_SUMMARY_LEN="$OPTARG" ;;
    m) MODEL_ID="$OPTARG" ;;
    t) THOUGHT_TEMPERATURE="$OPTARG" ;;
    e) EVAL_TEMPERATURE="$OPTARG" ;;
    S) DO_SAMPLE="$OPTARG" ;;
    g) GPU_DEVICE="$OPTARG" ;;
    L) LIMIT_ENTITIES="$OPTARG" ;;
    r) RUNTIME_REPORT_FILE="$OPTARG" ;;
    s) NO_SCORING_MODE="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done
shift $((OPTIND -1))

# Semantic dimension weights (for value function aggregation)
W_RELATEDNESS=${W_RELATEDNESS:-0.4}
W_INFORMATIVENESS=${W_INFORMATIVENESS:-0.4}
W_COVERAGE=${W_COVERAGE:-0.2}

# Evaluation parameters
N_EVALUATION_SAMPLES=${N_EVALUATION_SAMPLES:-3}
BEAM_WIDTH=${BEAM_WIDTH:-3}

# Candidate selection strategy
USE_RANDOM_CANDIDATES=${USE_RANDOM_CANDIDATES:-false}
NO_SCORING_MODE=${NO_SCORING_MODE:-}

# Heuristic scoring (alternative to LLM evaluation)
# When enabled, uses heuristic-based scoring instead of LLM evaluation
USE_HEURISTIC_SCORING=${USE_HEURISTIC_SCORING:-false}
HEURISTIC_METHOD=${HEURISTIC_METHOD:-fca}  # Options: fca, tfidf, random, llm (default)

# Display variant information
VARIANT_NAME="${VARIANT_NAME:-full}"  # For logging/tracking

mkdir -p "$OUT" "$LOGS"
mkdir -p "$OUT/$DATASET"

RUNTIME_REPORT_FILE="${RUNTIME_REPORT_FILE:-$OUT/$DATASET/runtime_report.json}"
RUNTIME_REPORT_DIR="$(dirname "$RUNTIME_REPORT_FILE")"
RUNTIME_ENTRIES_FILE="$RUNTIME_REPORT_DIR/.runtime_entries.jsonl"
mkdir -p "$RUNTIME_REPORT_DIR"
: > "$RUNTIME_ENTRIES_FILE"

# Make globs vanish instead of staying literal when they don't match
shopt -s nullglob

# Collect all *_desc.nt files exactly one level under ROOT
nt_files=("$ROOT"/*/*_desc.nt)

if ((${#nt_files[@]}==0)); then
  echo "No *_desc.nt files found under: $ROOT"
  echo "Check the path or rename your files to '<id>_desc.nt'."
  exit 1
fi

# Limit number of entities for testing if specified
if (( LIMIT_ENTITIES > 0 )); then
  nt_files=("${nt_files[@]:0:$LIMIT_ENTITIES}")
fi

echo "═══════════════════════════════════════════════════════════"
echo "Task-Decomposed ToT - Entity Processing"
if [ "$VARIANT_NAME" != "full" ]; then
  echo "  [ABLATION STUDY VARIANT: $VARIANT_NAME]"
fi
echo "═══════════════════════════════════════════════════════════"
echo "Total entities found: ${#nt_files[@]} to process"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Max summary length: $MAX_SUMMARY_LEN"
echo "  Candidates per task: $N_CANDIDATES_PER_TASK"
echo "  GPU device: $GPU_DEVICE"
echo "  Model: $MODEL_ID"
echo "  Sampling mode: $DO_SAMPLE"
echo ""
echo "Semantic Dimensions (Value Function):"
echo "  Relatedness weight (w_r): $W_RELATEDNESS"
echo "  Informativeness weight (w_i): $W_INFORMATIVENESS"
echo "  Coverage weight (w_c): $W_COVERAGE"
echo "  V_M(s) = ${W_RELATEDNESS}*R + ${W_INFORMATIVENESS}*I + ${W_COVERAGE}*C"
echo ""
echo "Search & Evaluation:"
echo "  Beam width: $BEAM_WIDTH"
echo "  Evaluation samples: $N_EVALUATION_SAMPLES"
echo "  Random candidates: $USE_RANDOM_CANDIDATES"
if [ -n "$NO_SCORING_MODE" ]; then
  echo "  No-scoring mode: $NO_SCORING_MODE"
fi
if [ "$USE_HEURISTIC_SCORING" = "true" ]; then
  echo "  Scoring method: HEURISTIC ($HEURISTIC_METHOD)"
else
  if [ -n "$NO_SCORING_MODE" ]; then
    echo "  Scoring method: DISABLED (no-scoring mode)"
  else
    echo "  Scoring method: LLM-based (default)"
  fi
fi
if (( LIMIT_ENTITIES > 0 )); then
  echo "  Limited to: $LIMIT_ENTITIES entities (for testing)"
fi
echo ""
echo "Paths:"
echo "  Input data: $ROOT"
echo "  Output directory: $OUT"
echo "  Log directory: $LOGS"
echo "  Runtime report: $RUNTIME_REPORT_FILE"
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
    printf '{"entity_id":"%s","status":"skipped","time":0.0,"reason":"output_exists"}\n' "$id" >> "$RUNTIME_ENTRIES_FILE"
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
    --model-id \"$MODEL_ID\" \
    --thought-temperature $THOUGHT_TEMPERATURE \
    --eval-temperature $EVAL_TEMPERATURE \
    --do-sample $DO_SAMPLE \
    --w-relatedness $W_RELATEDNESS \
    --w-informativeness $W_INFORMATIVENESS \
    --w-coverage $W_COVERAGE \
    --beam-width $BEAM_WIDTH \
    --n-evaluation-samples $N_EVALUATION_SAMPLES"

  # Add random candidates flag if specified (ablation variant)
  if [ "$USE_RANDOM_CANDIDATES" = "true" ]; then
    cmd="$cmd --use-random-candidates"
  fi

  # Add heuristic scoring if specified (ablation variant: no LLM evaluation)
  if [ "$USE_HEURISTIC_SCORING" = "true" ]; then
    cmd="$cmd --use-heuristic-scoring --heuristic-method $HEURISTIC_METHOD"
  fi

  # Add no-scoring mode if specified
  if [ -n "$NO_SCORING_MODE" ]; then
    cmd="$cmd --no-scoring-mode $NO_SCORING_MODE"
  fi

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
  cmd="$cmd > \"$LOGS/$DATASET/$id/stdout.log\" 2> \"$LOGS/$DATASET/$id/stderr.log\""

  # Execute
  echo "  → Executing Python script..."
  if eval "$cmd"; then
    proc_end=$(date +%s)
    proc_time=$((proc_end - proc_start))
    echo "  ✓ [$id] Success (completed in ${proc_time}s)"
    printf '{"entity_id":"%s","status":"success","time":%s}\n' "$id" "$proc_time" >> "$RUNTIME_ENTRIES_FILE"
    processed_count=$((processed_count + 1))
  else
    proc_end=$(date +%s)
    proc_time=$((proc_end - proc_start))
    echo "  ✗ [$id] Failed (see $LOGS/$DATASET/$id/stderr.log)"
    printf '{"entity_id":"%s","status":"failed","time":%s,"reason":"command_failed"}\n' "$id" "$proc_time" >> "$RUNTIME_ENTRIES_FILE"
    tail -10 "$LOGS/$DATASET/$id/stderr.log" | sed 's/^/    /'
  fi
  entity_count=$((entity_count + 1))
done

# Timing: record end time and calculate total and average
end_time=$(date +%s)
end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
total_time=$((end_time - start_time))

# Record ablation variant info for tracking
VARIANT_INFO=""
if [ "$W_RELATEDNESS" != "0.4" ] || [ "$W_INFORMATIVENESS" != "0.4" ] || [ "$W_COVERAGE" != "0.2" ]; then
  VARIANT_INFO="Weights: R=$W_RELATEDNESS I=$W_INFORMATIVENESS C=$W_COVERAGE"
fi
if [ "$N_CANDIDATES_PER_TASK" != "3" ]; then
  VARIANT_INFO="${VARIANT_INFO} Candidates: $N_CANDIDATES_PER_TASK"
fi
if [ "$USE_RANDOM_CANDIDATES" = "true" ]; then
  VARIANT_INFO="${VARIANT_INFO} RandomCandidates: ON"
fi
if [ "$BEAM_WIDTH" != "3" ]; then
  VARIANT_INFO="${VARIANT_INFO} BeamWidth: $BEAM_WIDTH"
fi

if (( processed_count > 0 )); then
  avg_time=$(awk "BEGIN {printf \"%.2f\", $total_time/$processed_count}")
else
  avg_time=0
fi

python - "$RUNTIME_ENTRIES_FILE" "$RUNTIME_REPORT_FILE" "$DATASET" "$MODEL_ID" "$MAX_SUMMARY_LEN" "$total_time" <<'PY'
import json
import sys
from pathlib import Path

entries_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
dataset = sys.argv[3]
model = sys.argv[4]
summary_size = int(sys.argv[5])
total_time = float(sys.argv[6])

entities = []
if entries_path.exists():
    with entries_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entities.append(json.loads(line))

measured_times = [entry["time"] for entry in entities if entry.get("status") != "skipped"]
avg_time = sum(measured_times) / len(measured_times) if measured_times else 0.0
min_time = min(measured_times) if measured_times else 0.0
max_time = max(measured_times) if measured_times else 0.0

report = {
    "dataset": dataset,
    "model": model,
    "summary_size": summary_size,
    "total_runtime_seconds": total_time,
    "average_runtime_seconds": avg_time,
    "min_runtime_seconds": min_time,
    "max_runtime_seconds": max_time,
    "entities": entities,
}

report_path.parent.mkdir(parents=True, exist_ok=True)
with report_path.open("w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
PY
rm -f "$RUNTIME_ENTRIES_FILE"

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
echo "  Runtime report:     $RUNTIME_REPORT_FILE"
echo "═══════════════════════════════════════════════════════════"

# Save summary to overall_report.txt
REPORT_FILE="$OUT/overall_report.txt"
{
  echo "═══════════════════════════════════════════════════════════"
  echo "Task-Decomposed ToT Processing Report"
  if [ "$VARIANT_NAME" != "full" ]; then
    echo "  [ABLATION STUDY VARIANT: $VARIANT_NAME]"
  fi
  echo "═══════════════════════════════════════════════════════════"
  echo "Start time:           $start_timestamp"
  echo "End time:             $end_timestamp"
  echo "Total runtime:        ${total_time}s"
  echo ""
  echo "Configuration:"
  echo "  Dataset: $DATASET"
  echo "  Model: $MODEL_ID"
  echo "  Weights: R=$W_RELATEDNESS I=$W_INFORMATIVENESS C=$W_COVERAGE"
  echo "  Beam Width: $BEAM_WIDTH"
  echo "  Evaluation Samples: $N_EVALUATION_SAMPLES"
  echo "  Candidates Per Task: $N_CANDIDATES_PER_TASK"
  if [ "$USE_RANDOM_CANDIDATES" = "true" ]; then
    echo "  Selection Method: RANDOM (Ablation - For RQ2)"
  fi
  if [ "$USE_HEURISTIC_SCORING" = "true" ]; then
    echo "  Scoring Method: HEURISTIC - $HEURISTIC_METHOD (Ablation - For RQ2)"
  fi
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
  echo "  Runtime report:     $RUNTIME_REPORT_FILE"
  echo "═══════════════════════════════════════════════════════════"
} | tee "$REPORT_FILE"

echo ""
echo "✓ All processing complete!"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Ablation Study (RQ2) - Running Variants from This Script"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "To run ablation study variants, use environment variables:"
echo ""
echo "  # Baseline (full model)"
echo "  ./run_tot_task_decomposed.sh"
echo ""
echo "  # Ablation 1: No thought policy (random candidates)"
echo "  USE_RANDOM_CANDIDATES=true VARIANT_NAME=no_thought ./run_tot_task_decomposed.sh"
echo ""
echo "  # Ablation 2: No branching (greedy, beam_width=1)"
echo "  BEAM_WIDTH=1 VARIANT_NAME=no_branch ./run_tot_task_decomposed.sh"
echo ""
echo "  # Ablation 3: Only relatedness (ignore I and C)"
echo "  W_RELATEDNESS=1.0 W_INFORMATIVENESS=0.0 W_COVERAGE=0.0 VARIANT_NAME=only_relatedness ./run_tot_task_decomposed.sh"
echo ""
echo "  # Ablation 4: Only informativeness"
echo "  W_RELATEDNESS=0.0 W_INFORMATIVENESS=1.0 W_COVERAGE=0.0 VARIANT_NAME=only_informativeness ./run_tot_task_decomposed.sh"
echo ""
echo "  # Ablation 5: Only coverage"
echo "  W_RELATEDNESS=0.0 W_INFORMATIVENESS=0.0 W_COVERAGE=1.0 VARIANT_NAME=only_coverage ./run_tot_task_decomposed.sh"
echo ""
echo "  # Ablation 6: Uniform weights"
echo "  W_RELATEDNESS=0.333 W_INFORMATIVENESS=0.333 W_COVERAGE=0.334 VARIANT_NAME=uniform_weights ./run_tot_task_decomposed.sh"
echo ""
echo "  # Ablation 7: FCA-based heuristic scoring (no LLM evaluation)"
echo "  USE_HEURISTIC_SCORING=true HEURISTIC_METHOD=fca VARIANT_NAME=fca_heuristic ./run_tot_task_decomposed.sh"
echo ""
echo "  # Ablation 8: TF-IDF heuristic scoring (traditional baseline)"
echo "  USE_HEURISTIC_SCORING=true HEURISTIC_METHOD=tfidf VARIANT_NAME=tfidf_heuristic ./run_tot_task_decomposed.sh"
echo ""
echo "  # Ablation 9: Random scoring baseline"
echo "  USE_HEURISTIC_SCORING=true HEURISTIC_METHOD=random VARIANT_NAME=random_scoring ./run_tot_task_decomposed.sh"
echo ""
echo "  LIMIT_ENTITIES=5 ./run_tot_task_decomposed.sh"
echo ""
echo "  # Combine: Ablation variant + limited entities"
echo "  USE_RANDOM_CANDIDATES=true LIMIT_ENTITIES=10 VARIANT_NAME=no_thought ./run_tot_task_decomposed.sh"
echo ""
echo "For complete ablation study automation, use ablation_runner.py:"
echo "  cd .. && python scripts/ablation_runner.py --dataset wikicinema-s"
echo ""
echo "═══════════════════════════════════════════════════════════"
