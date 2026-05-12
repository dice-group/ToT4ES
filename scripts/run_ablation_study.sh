#!/bin/bash
# Quick-start script for ablation study
# Run all variants and generate analysis automatically

set -euo pipefail

# Configuration
DATASET="${1:-wikicinema-s}"
LIMIT_ENTITIES="${2:-0}"  # 0 = all, or set to 10 for quick test
OUTPUT_DIR="outputs/ablation_study"

echo "═══════════════════════════════════════════════════════════════════════"
echo "ToT4ES Ablation Study (RQ2) - Quick Start"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Dataset:        $DATASET"
echo "  Entity limit:   $([ "$LIMIT_ENTITIES" -eq 0 ] && echo 'All' || echo $LIMIT_ENTITIES)"
echo "  Output dir:     $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: List available variants
echo "Available variants:"
python scripts/ablation_runner.py --list-variants

# Step 2: Run all variants
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Step 1: Running ablation variants..."
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

cd scripts

if [ "$LIMIT_ENTITIES" -eq 0 ]; then
    python ablation_runner.py \
        --dataset "$DATASET" \
        --output-dir "../$OUTPUT_DIR" \
        --skip-failed
else
    python ablation_runner.py \
        --dataset "$DATASET" \
        --output-dir "../$OUTPUT_DIR" \
        --limit-entities "$LIMIT_ENTITIES" \
        --skip-failed
fi

cd ..

# Step 3: Analyze results
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Step 3: Analyzing results..."
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

cd scripts

python ablation_evaluator.py \
    --ablation-dir "../$OUTPUT_DIR" \
    --baseline full \
    --output-csv "../$OUTPUT_DIR/comparison.csv" \
    --output-report "../$OUTPUT_DIR/analysis_report.txt"

cd ..

# Step 4: Print summary
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✓ Ablation study complete!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Results location: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  • ablation_execution.json   - Execution log and metadata"
echo "  • comparison.csv            - Metrics comparison across variants"
echo "  • analysis_report.txt       - Detailed analysis and insights"
echo ""
echo "Next steps:"
echo "  1. Review: cat $OUTPUT_DIR/analysis_report.txt"
echo "  2. Compare: cat $OUTPUT_DIR/comparison.csv"
echo "  3. Investigate: Check logs in $OUTPUT_DIR/<variant>/run.log"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
