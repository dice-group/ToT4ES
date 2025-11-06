#!/usr/bin/env bash
set -euo pipefail

#ROOT="../datasets/ESBM_benchmark_v1.2/lmdb_data"
ROOT="../datasets/FACES/faces_data"
OUT="out/informativeness/faces"
LOGS="logs/informativeness/faces"

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

for f in "${nt_files[@]}"; do
  id="$(basename "$(dirname "$f")")"
  echo "processing $id -> $f"

  # Ensure per-entity output/log directories exist
  mkdir -p "$OUT/$id" "$LOGS/$id"

  python informativeness.py \
    --k 5 \
    --diversity none \
    --batch-size 12 \
    --max-new-tokens 1024 \
    --llm-model-id meta-llama/Llama-3.2-3B-Instruct \
    --llm-device auto \
    --nt "$f" \
    --emit-nt \
    --out-dir "$OUT/$id" \
    --log-dir "$LOGS/$id" \
    --pretty-console \
  || echo "failed: $id"
done
