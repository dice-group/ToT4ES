#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantically-Enhanced ToT Entity Summarization

Uses Description Logic principles for better informativeness, diversity, and relatedness.
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from tot_modules import (
    Llama32Chat,
    decode_state_to_triples,
    load_entity_description_from_nt,
    entity_heuristic_calculator,
    SemanticAnalyzer,
)
from tot_modules.semantic_prompts import (
    make_semantic_relatedness_prompt,
    make_semantic_informativeness_prompt,
    make_semantic_diversity_prompt,
)
from tot_modules.task_prompts import make_combined_evaluation_prompt
from tot_modules.task_decomposed_search import TaskDecomposedToT


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Semantically-Enhanced ToT Entity Summarization (DL-inspired)"
    )
    parser.add_argument("--nt", required=True, help="Path to N-Triples file")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max-summary-len", type=int, default=5)
    parser.add_argument("--n-candidates-per-task", type=int, default=2)
    parser.add_argument("--n-evals", type=int, default=3)
    parser.add_argument("--breadth-limit", type=int, default=3)
    parser.add_argument("--search-algorithm", choices=["bfs", "dfs"], default="bfs")
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--output-dir", default="tot-results-semantic")
    parser.add_argument("--show-semantic-analysis", action="store_true",
                        help="Show semantic analysis of triples before search")
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_arguments()
    
    print("="*70)
    print("SEMANTICALLY-ENHANCED ToT for Entity Summarization")
    print("Using Description Logic Principles")
    print("="*70)
    
    # Load entity
    print(f"\nLoading: {args.nt}")
    entity_label, all_triples = load_entity_description_from_nt(args.nt)
    print(f"  Entity: {entity_label}")
    print(f"  Triples: {len(all_triples)}")
    
    entity_id = os.path.basename(os.path.dirname(args.nt))
    
    # Semantic Analysis
    print("\n" + "="*70)
    print("SEMANTIC ANALYSIS (DL-Inspired)")
    print("="*70)
    
    analyzer = SemanticAnalyzer(all_triples)
    stats = analyzer.get_summary_statistics()
    
    print(f"\nEntity Statistics:")
    print(f"  Total triples: {stats['total_triples']}")
    print(f"  Unique predicates: {stats['unique_predicates']}")
    print(f"  Defining triples (rdf:type, label): {stats['defining_count']}")
    print(f"  Functional properties (birthDate, etc.): {stats['functional_count']}")
    print(f"  Literals: {stats['literal_count']}")
    print(f"  Entity links: {stats['entity_link_count']}")
    
    if args.show_semantic_analysis:
        print("\nSemantic Annotations:")
        informativeness = analyzer.get_informativeness_scores()
        relatedness = analyzer.get_relatedness_scores()
        
        for idx in range(1, min(10, len(all_triples) + 1)):
            annotation = analyzer.get_enriched_triple_info(idx)
            info = informativeness.get(idx, 0.5)
            rel = relatedness.get(idx, 0.5)
            print(f"  {idx}. Info={info:.2f}, Rel={rel:.2f} | {annotation}")
            print(f"     {all_triples[idx-1][:80]}...")
    
    # Setup LLM
    print(f"\nInitializing LLM: {args.model_id}")
    llm = Llama32Chat(
        model_id=args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Create SEMANTIC prompts
    print("\n" + "="*70)
    print("Creating Semantically-Enhanced Prompts")
    print("="*70)
    print("  Relatedness: DL-based (defining predicates, common patterns)")
    print("  Informativeness: Specificity-based (rare predicates, unique values)")
    print("  Diversity: Type-based (predicate variety, value types)")
    
    input_seq = "\n".join(all_triples)
    
    get_relatedness_prompt = make_semantic_relatedness_prompt(entity_label, all_triples)
    get_informativeness_prompt = make_semantic_informativeness_prompt(entity_label, all_triples)
    get_diversity_prompt = make_semantic_diversity_prompt(entity_label, all_triples)
    get_eval_prompt = make_combined_evaluation_prompt(entity_label, all_triples)
    
    # Create search engine
    tot = TaskDecomposedToT(
        llm=llm,
        input_seq=input_seq,
        get_relatedness_prompt=get_relatedness_prompt,
        get_informativeness_prompt=get_informativeness_prompt,
        get_diversity_prompt=get_diversity_prompt,
        get_state_eval_prompt=get_eval_prompt,
        heuristic_calculator=entity_heuristic_calculator,
        num_triples=len(all_triples),
    )
    
    tot.n_steps = args.max_summary_len
    tot.n_candidates_per_task = args.n_candidates_per_task
    tot.n_evals = args.n_evals
    tot.breadth_limit = args.breadth_limit
    
    print(f"\nSearch Configuration: {tot}")
    
    # Run search
    print("\n" + "="*70)
    print(f"Running Semantic ToT Search ({args.search_algorithm.upper()})")
    print("="*70)
    
    if args.search_algorithm == "bfs":
        best_state = tot.bfs(verbose=not args.no_verbose)
    else:
        best_state = tot.dfs(verbose=not args.no_verbose)
    
    best_triples = decode_state_to_triples(best_state, all_triples)
    
    # Analyze selected triples
    print("\n" + "="*70)
    print("SELECTED SUMMARY WITH SEMANTIC ANALYSIS")
    print("="*70)
    
    selected_ids = [int(x) for x in best_state.strip().splitlines() if x.strip().isdigit()]
    informativeness = analyzer.get_informativeness_scores()
    relatedness = analyzer.get_relatedness_scores()
    
    for i, (triple_id, triple) in enumerate(zip(selected_ids, best_triples), 1):
        info = informativeness.get(triple_id, 0.5)
        rel = relatedness.get(triple_id, 0.5)
        annotation = analyzer.get_enriched_triple_info(triple_id)
        
        print(f"\n{i}. Triple #{triple_id}")
        print(f"   Informativeness: {info:.2f} | Relatedness: {rel:.2f}")
        print(f"   Semantic: {annotation}")
        print(f"   {triple}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{entity_id}_summary.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"Entity: {entity_label}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Method: Semantic ToT ({args.search_algorithm.upper()})\n\n")
        f.write("Selected Triples:\n")
        for i, (triple_id, triple) in enumerate(zip(selected_ids, best_triples), 1):
            info = informativeness.get(triple_id, 0.5)
            rel = relatedness.get(triple_id, 0.5)
            f.write(f"{i}. Info={info:.2f}, Rel={rel:.2f} | {triple}\n")
    
    print(f"\nResults saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
