#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-of-Thought Entity Summarization - Modular Version

This is a refactored version of tot_entity_summarizer.py with improved modularity.

Usage example:
    python tot_entity_summarizer_modular.py \
      --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
      --dataset dbpedia \
      --max-summary-len 5 \
      --n-candidates 5 \
      --n-evals 3 \
      --breadth-limit 3
"""

import argparse
import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from tot_modules import (
    TreeNode,
    Llama32Chat,
    TreeOfThoughts,
    make_entity_thought_gen_prompt,
    make_entity_state_eval_prompt,
    entity_heuristic_calculator,
    decode_state_to_triples,
    load_entity_description_from_nt,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tree-of-Thought Entity Summarization (Modular Version) with LLaMA-3.2-3B-Instruct"
    )
    parser.add_argument(
        "--nt",
        required=True,
        help="Path to entity description N-Triples file, e.g., dbpedia/1/1_desc.nt",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g., dbpedia, lmdb, faces",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model ID (default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--max-summary-len",
        type=int,
        default=5,
        help="Maximum number of triples in the summary (number of ToT steps)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=5,
        help="Number of thought candidates (triple indices) per node",
    )
    parser.add_argument(
        "--n-evals",
        type=int,
        default=3,
        help="Number of evaluation votes per state (for robustness)",
    )
    parser.add_argument(
        "--breadth-limit",
        type=int,
        default=3,
        help="Beam width (number of states kept at each level)",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose search output",
    )
    parser.add_argument(
        "--output-dir",
        default="tot-results-llama",
        help="Output directory for results (default: tot-results-llama)",
    )

    return parser.parse_args()


def setup_llm(model_id: str):
    """
    Initialize the LLM.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Llama32Chat instance
    """
    print(f"Initializing LLM: {model_id}")
    llm = Llama32Chat(
        model_id=model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"LLM initialized: {llm}")
    return llm


def create_search_engine(
    llm: Llama32Chat,
    entity_label: str,
    all_triples: list,
    max_summary_len: int,
    n_candidates: int,
    n_evals: int,
    breadth_limit: int,
):
    """
    Create and configure Tree-of-Thoughts search engine.
    
    Args:
        llm: LLM wrapper instance
        entity_label: Human-readable entity name
        all_triples: List of all triples for this entity
        max_summary_len: Maximum triples in summary
        n_candidates: Thought candidates per node
        n_evals: Number of evaluation samples
        breadth_limit: Beam width
        
    Returns:
        Configured TreeOfThoughts instance
    """
    # Global input sequence
    input_seq = "\n".join(all_triples)

    # Create prompt factories
    get_thought_gen_prompt = make_entity_thought_gen_prompt(
        entity_label=entity_label,
        all_triples=all_triples,
        max_summary_len=max_summary_len,
    )

    get_state_eval_prompt = make_entity_state_eval_prompt(
        entity_label=entity_label,
        all_triples=all_triples,
    )

    # Create search engine
    tot = TreeOfThoughts(
        llm=llm,
        input_seq=input_seq,
        get_thought_gen_prompt=get_thought_gen_prompt,
        get_state_eval_prompt=get_state_eval_prompt,
        heuristic_calculator=entity_heuristic_calculator,
        num_triples=len(all_triples),
    )

    # Configure parameters
    tot.n_steps = max_summary_len
    tot.n_candidates = n_candidates
    tot.n_evals = n_evals
    tot.breadth_limit = breadth_limit

    print(f"Search engine configured: {tot}")
    return tot


def save_summary(
    best_triples: list,
    args,
    entity_id: str,
):
    """
    Save the selected summary to file.
    
    Args:
        best_triples: List of selected triple strings
        args: Command-line arguments
        entity_id: Entity identifier
    """
    # Build output path
    in_dir = os.path.dirname(args.nt)
    root, ext = os.path.splitext(in_dir)
    root_split = root.split("/")
    
    if not ext:
        ext = ".nt"
    
    out_base = f"{args.dataset}/{entity_id}/{entity_id}_top{args.max_summary_len}{ext}"
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, out_base)
    parent_dir = os.path.dirname(out_path)
    os.makedirs(parent_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for triple in best_triples:
            f.write(triple.rstrip() + "\n")

    print(f"\n✓ Saved summary ({len(best_triples)} triples) to: {out_path}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("="*70)
    print("Tree-of-Thought Entity Summarization (Modular Version)")
    print("="*70)
    
    # Load entity description
    print(f"\nLoading entity description from: {args.nt}")
    entity_label, all_triples = load_entity_description_from_nt(args.nt)
    print(f"  Entity: {entity_label}")
    print(f"  Triples loaded: {len(all_triples)}")
    
    # Extract entity ID from path
    entity_id = os.path.basename(os.path.dirname(args.nt))
    
    # Setup LLM
    llm = setup_llm(args.model_id)
    
    # Create search engine
    tot = create_search_engine(
        llm=llm,
        entity_label=entity_label,
        all_triples=all_triples,
        max_summary_len=args.max_summary_len,
        n_candidates=args.n_candidates,
        n_evals=args.n_evals,
        breadth_limit=args.breadth_limit,
    )
    
    # Run search
    print("\n" + "="*70)
    print("Starting Tree-of-Thought search...")
    print("="*70)
    
    best_state = tot.bfs(verbose=not args.no_verbose)
    best_triples = decode_state_to_triples(best_state, all_triples)
    
    # Display results
    print("\n" + "="*70)
    print("FINAL SELECTED SUMMARY")
    print("="*70)
    for i, t in enumerate(best_triples, 1):
        print(f"{i}. {t}")
    
    # Save results
    save_summary(best_triples, args, entity_id)
    
    print("\n" + "="*70)
    print("✓ Process completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
