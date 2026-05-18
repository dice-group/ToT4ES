#!/bin/bash
# Demo script to run baseline on sample entities

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Baseline Direct LLM Summarizer Demo ===${NC}\n"

# Change to baseline directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directory
mkdir -p results

# Array of sample entities to summarize
declare -a ENTITIES=(
    "1:http://dbpedia.org/resource/Marie_Curie:Marie Curie:5"
    "2:http://dbpedia.org/resource/Albert_Einstein:Albert Einstein:5"
    "100:http://dbpedia.org/resource/Barack_Obama:Barack Obama:5"
)

echo -e "${BLUE}Running baseline summarization on sample entities...${NC}\n"

for entity_info in "${ENTITIES[@]}"; do
    IFS=':' read -r entity_id entity_uri entity_label summary_size <<< "$entity_info"
    
    triple_file="../datasets/ESBM_benchmark_v1.2/dbpedia_data/$entity_id/${entity_id}_desc.nt"
    output_file="results/baseline_${entity_id}_summary.nt"
    
    if [ ! -f "$triple_file" ]; then
        echo -e "${BLUE}Entity $entity_id: Triple file not found at $triple_file${NC}"
        continue
    fi
    
    echo -e "${BLUE}Processing: $entity_label (ID: $entity_id)${NC}"
    echo "  Triple file: $triple_file"
    echo "  Output: $output_file"
    echo "  Target size: $summary_size triples"
    
    python baseline_direct_llm.py \
        --triple-file "$triple_file" \
        --entity-uri "$entity_uri" \
        --entity-label "$entity_label" \
        --summary-size "$summary_size" \
        --temperature 0.1 \
        --output-file "$output_file"
    
    echo -e "${GREEN}✓ Summary saved to $output_file${NC}\n"
done

echo -e "${GREEN}=== Demo Complete ===${NC}"
echo -e "Results saved in: $SCRIPT_DIR/results/\n"
