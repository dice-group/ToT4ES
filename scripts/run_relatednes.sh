#!/usr/bin/env bash
set -euo pipefail

ROOT=../datasets/FACES/faces_data
OUT=out/relatedness/faces
LOGS=logs/faces

mkdir -p "$OUT" "$LOGS"

for d in "$ROOT"/*/; do
  id="$(basename "$d")"
  f="$d/${id}_desc.nt"
  if [ -f "$f" ]; then
    echo "processing $id -> $f"
    python relatedness.py \
      --rel-mode linear \
      --alpha 0.5 \
      --k 10 \
      --mode relatedness \
      --batch-size 12 \
      --max-new-tokens 1024 \
      --llm-model-id meta-llama/Llama-3.2-3B-Instruct \
      --llm-device auto \
      --nt "$f" \
      --fallback-local-freq \
      --emit-nt \
      --out-dir "$OUT/$id" \
      --log-dir "$LOGS/$id" \
      --diversity none \
      --pretty-console \
    || echo "failed: $id"
  else
    echo "skip (no ${id}_desc.nt): $d"
  fi
done
