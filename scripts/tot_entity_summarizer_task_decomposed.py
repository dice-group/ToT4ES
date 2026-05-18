#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-of-Thought Entity Summarization with Task Decomposition

This implements the architecture where Relatedness, Informativeness, and Diversity
are handled by separate task-specific prompts.

Usage:
    python tot_entity_summarizer_task_decomposed.py \
      --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
      --dataset dbpedia \
      --max-summary-len 5
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from tot_modules import (
    Llama32Chat,
    decode_state_to_triples,
    load_entity_description_from_nt,
    entity_heuristic_calculator,
)
from tot_modules.task_prompts import (
    make_relatedness_prompt,
    make_informativeness_prompt,
    make_diversity_prompt,
    make_combined_evaluation_prompt,
)
from tot_modules.task_decomposed_search import TaskDecomposedToT


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Task-Decomposed Tree-of-Thought Entity Summarization"
    )
    parser.add_argument(
        "--nt",
        required=True,
        help="Path to entity description N-Triples file",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (dbpedia, lmdb, faces)",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Default HuggingFace model ID (used if task-specific models not provided)",
    )
    parser.add_argument(
        "--model-relatedness",
        default=None,
        help="Model for relatedness task (optional, defaults to --model-id)",
    )
    parser.add_argument(
        "--model-informativeness",
        default=None,
        help="Model for informativeness task (optional, defaults to --model-id)",
    )
    parser.add_argument(
        "--model-diversity",
        default=None,
        help="Model for diversity task (optional, defaults to --model-id)",
    )
    parser.add_argument(
        "--model-evaluation",
        default=None,
        help="Model for evaluation task (optional, defaults to --model-id)",
    )
    parser.add_argument(
        "--max-summary-len",
        type=int,
        default=5,
        help="Maximum triples in summary",
    )
    parser.add_argument(
        "--n-candidates-per-task",
        type=int,
        default=2,
        help="Number of candidates generated per task (relatedness/informativeness/diversity)",
    )
    parser.add_argument(
        "--n-evals",
        type=int,
        default=3,
        help="Number of evaluation votes per state",
    )
    parser.add_argument(
        "--thought-temperature",
        type=float,
        default=0.8,
        help="Temperature for thought generation",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.3,
        help="Temperature for state evaluation",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (sets temperatures to 0 and fixed seeds)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --deterministic is enabled",
    )
    parser.add_argument(
        "--breadth-limit",
        type=int,
        default=3,
        help="Beam width",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="Beam width (alias for --breadth-limit, used by ablation study)",
    )
    parser.add_argument(
        "--n-evaluation-samples",
        type=int,
        default=None,
        help="Number of evaluation samples (alias for --n-evals, used by ablation study)",
    )
    parser.add_argument(
        "--w-relatedness",
        type=float,
        default=0.4,
        help="Weight for relatedness dimension in value function [0,1]",
    )
    parser.add_argument(
        "--w-informativeness",
        type=float,
        default=0.4,
        help="Weight for informativeness dimension in value function [0,1]",
    )
    parser.add_argument(
        "--w-coverage",
        type=float,
        default=0.2,
        help="Weight for coverage dimension in value function [0,1]",
    )
    parser.add_argument(
        "--use-random-candidates",
        action="store_true",
        help="Use random candidates instead of LLM-guided thought policy (ablation variant)",
    )
    parser.add_argument(
        "--use-heuristic-scoring",
        action="store_true",
        help="Use heuristic-based scoring instead of LLM evaluation (ablation variant)",
    )
    parser.add_argument(
        "--heuristic-method",
        type=str,
        default="fca",
        choices=["fca", "tfidf", "random"],
        help="Heuristic method for scoring when --use-heuristic-scoring is enabled",
    )
    parser.add_argument(
        "--limit-entities",
        type=int,
        default=0,
        help="Limit to first N entities (0=all, for testing)",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output",
    )
    parser.add_argument(
        "--output-dir",
        default="tot-results-task-decomposed",
        help="Output directory",
    )

    args = parser.parse_args()
    
    # Handle aliases: --beam-width overrides --breadth-limit if provided
    if args.beam_width is not None:
        args.breadth_limit = args.beam_width
    
    # Handle aliases: --n-evaluation-samples overrides --n-evals if provided
    if args.n_evaluation_samples is not None:
        args.n_evals = args.n_evaluation_samples
    
    return args


def _categorize_semantic_roles(all_triples):
    """
    Categorize predicates by semantic roles for diversity evaluation.
    
    Args:
        all_triples: List of RDF triple strings
        
    Returns:
        Dict mapping predicates to semantic role categories
    """
    # Semantic role patterns based on common predicate names
    role_patterns = {
        'location': ['place', 'location', 'birthplace', 'birthPlace', 'deathplace', 'deathPlace', 'hometown'],
        'time': ['date', 'born', 'died', 'founded', 'year', 'birthdate', 'birthDate', 'deathdate', 'deathDate'],
        'relationship': ['knows', 'friend', 'spouse', 'parent', 'child', 'sibling', 'related', 'ownedBy', 'owns'],
        'attribute': ['name', 'label', 'title', 'description', 'comment', 'type', 'class', 'category'],
        'work': ['work', 'wrote', 'created', 'author', 'composed', 'directed', 'produced', 'album', 'book', 'film'],
        'organization': ['member', 'organization', 'company', 'institution', 'team', 'group', 'affiliation'],
    }
    
    semantic_roles = {}
    for triple in all_triples:
        parts = triple.split(maxsplit=2)
        if len(parts) >= 2:
            predicate = parts[1]
            if predicate not in semantic_roles:
                # Extract the predicate name (last part after ':' or '/')
                pred_name = predicate.split('/')[-1].split(':')[-1].lower()
                
                # Match against patterns
                assigned_role = 'other'
                for role, patterns in role_patterns.items():
                    if any(pattern in pred_name for pattern in patterns):
                        assigned_role = role
                        break
                
                semantic_roles[predicate] = assigned_role
    
    return semantic_roles


def _compute_predicate_frequencies(all_triples):
    """
    Compute predicate frequency distribution from triples.
    Used to identify rare/uncommon predicates for informativeness evaluation.
    
    Args:
        all_triples: List of RDF triple strings
        
    Returns:
        Dict mapping predicates to occurrence count
    """
    predicates = {}
    for triple in all_triples:
        # Extract predicate (second element in RDF triple)
        parts = triple.split(maxsplit=2)
        if len(parts) >= 2:
            predicate = parts[1]
            predicates[predicate] = predicates.get(predicate, 0) + 1
    return predicates


def _set_deterministic_mode(seed: int) -> None:
    """Configure deterministic behavior across Python, NumPy, and PyTorch."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main execution."""
    args = parse_arguments()

    if args.deterministic:
        _set_deterministic_mode(args.seed)
        args.thought_temperature = 0.0
        args.eval_temperature = 0.0

    print("="*70)
    print("Task-Decomposed Tree-of-Thought Entity Summarization")
    print("="*70)
    print("\nArchitecture:")
    print("  - Separate prompts for: Relatedness, Informativeness, Diversity")
    print("  - Combined evaluation of all criteria")
    print("  - Multi-task thought generation per step")
    if args.deterministic:
        print("  - Deterministic mode: enabled")
        print(f"    seed={args.seed}, thought_temperature=0.0, eval_temperature=0.0")
    else:
        print("  - Deterministic mode: disabled")
        print(
            f"    thought_temperature={args.thought_temperature}, "
            f"eval_temperature={args.eval_temperature}"
        )
    print("="*70)
    
    # Display ablation study parameters
    print("\nAblation Study Parameters (RQ2):")
    print(f"  Semantic Weights (Value Function):")
    print(f"    w_relatedness:      {args.w_relatedness}")
    print(f"    w_informativeness:  {args.w_informativeness}")
    print(f"    w_coverage:         {args.w_coverage}")
    print(f"  Search Configuration:")
    print(f"    beam_width:         {args.breadth_limit}")
    print(f"    eval_samples:       {args.n_evals}")
    print(f"    use_random_candidates: {args.use_random_candidates}")
    if args.use_heuristic_scoring:
        print(f"  Heuristic Scoring (No LLM):")
        print(f"    method:             {args.heuristic_method}")
    print("="*70)
    
    # Load entity
    print(f"\nLoading: {args.nt}")
    entity_label, all_triples = load_entity_description_from_nt(args.nt)
    print(f"  Entity: {entity_label}")
    print(f"  Triples: {len(all_triples)}")
    
    # Compute predicate frequencies for informativeness evaluation
    predicate_frequencies = _compute_predicate_frequencies(all_triples)
    print(f"  Unique predicates: {len(predicate_frequencies)}")
    
    # Categorize predicates by semantic roles for diversity evaluation
    semantic_roles = _categorize_semantic_roles(all_triples)
    unique_roles = set(semantic_roles.values())
    print(f"  Semantic roles identified: {', '.join(sorted(unique_roles))}")
    
    entity_id = os.path.basename(os.path.dirname(args.nt))
    
    # Setup LLMs
    print(f"\nInitializing LLMs...")
    print(f"  Default model: {args.model_id}")
    llm = Llama32Chat(
        model_id=args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Initialize task-specific LLMs if provided
    llm_relatedness = None
    llm_informativeness = None
    llm_diversity = None
    llm_evaluation = None
    
    if args.model_relatedness:
        print(f"  Relatedness model: {args.model_relatedness}")
        llm_relatedness = Llama32Chat(
            model_id=args.model_relatedness,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    if args.model_informativeness:
        print(f"  Informativeness model: {args.model_informativeness}")
        llm_informativeness = Llama32Chat(
            model_id=args.model_informativeness,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    if args.model_diversity:
        print(f"  Diversity model: {args.model_diversity}")
        llm_diversity = Llama32Chat(
            model_id=args.model_diversity,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    if args.model_evaluation:
        print(f"  Evaluation model: {args.model_evaluation}")
        llm_evaluation = Llama32Chat(
            model_id=args.model_evaluation,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    
    # Create task-specific prompt generators
    print("\nCreating task-specific prompts...")
    input_seq = "\n".join(all_triples)
    
    # Use dataset name for loading predicate mappings (passed from CLI / bash)
    dataset_for_mapping = args.dataset
    
    get_relatedness_prompt = make_relatedness_prompt(
        entity_label,
        all_triples,
        predicate_frequencies=predicate_frequencies,
        dataset_name=dataset_for_mapping,
    )
    get_informativeness_prompt = make_informativeness_prompt(
        entity_label,
        all_triples,
        predicate_frequencies=predicate_frequencies,
        dataset_name=dataset_for_mapping,
    )
    get_diversity_prompt = make_diversity_prompt(
        entity_label,
        all_triples,
        semantic_roles=semantic_roles,
        dataset_name=dataset_for_mapping,
    )
    get_eval_prompt = make_combined_evaluation_prompt(entity_label, all_triples)
    
    print("  ✓ Relatedness prompt created")
    print("  ✓ Informativeness prompt created")
    print("  ✓ Diversity/Coverage prompt created")
    print("  ✓ Evaluation prompt created")
    
    # Create heuristic scorer if enabled (for ablation study without LLM evaluation)
    heuristic_scorer = None
    if args.use_heuristic_scoring:
        print("\n  Initializing heuristic scorer (ablation variant)...")
        from tot_modules.heuristic_scorers import create_heuristic_scorer
        heuristic_scorer = create_heuristic_scorer(args.heuristic_method, all_triples)
        print(f"  ✓ {args.heuristic_method.upper()} heuristic scorer created")
    
    # Create search engine
    print("\nCreating Task-Decomposed ToT search engine...")
    tot = TaskDecomposedToT(
        llm=llm,
        input_seq=input_seq,
        get_relatedness_prompt=get_relatedness_prompt,
        get_informativeness_prompt=get_informativeness_prompt,
        get_diversity_prompt=get_diversity_prompt,
        get_state_eval_prompt=get_eval_prompt,
        heuristic_calculator=entity_heuristic_calculator,
        num_triples=len(all_triples),
        llm_relatedness=llm_relatedness,
        llm_informativeness=llm_informativeness,
        llm_diversity=llm_diversity,
        llm_evaluation=llm_evaluation,
        heuristic_scorer=heuristic_scorer,
        w_relatedness=args.w_relatedness,
        w_informativeness=args.w_informativeness,
        w_coverage=args.w_coverage,
    )
    
    # Configure
    tot.n_steps = args.max_summary_len
    tot.n_candidates_per_task = args.n_candidates_per_task
    tot.n_evals = args.n_evals
    tot.breadth_limit = args.breadth_limit
    tot.thought_temperature = args.thought_temperature
    tot.eval_temperature = args.eval_temperature
    
    print(f"  Configuration: {tot}")
    print(f"  Total candidates per step: ~{args.n_candidates_per_task * 3} (from 3 tasks)")
    
    # Run search
    print("\n" + "="*70)
    print("Starting Task-Decomposed Search...")
    print("="*70)
    
    best_state = tot.bfs(verbose=not args.no_verbose)
    best_triples = decode_state_to_triples(best_state, all_triples)
    
    # Display results
    print("\n" + "="*70)
    print("FINAL SELECTED SUMMARY")
    print("="*70)
    for i, t in enumerate(best_triples, 1):
        print(f"{i}. {t}")
    
    # Save
    out_base = f"{args.dataset}/{entity_id}/{entity_id}_top{args.max_summary_len}.nt"
    out_path = os.path.join(args.output_dir, out_base)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for triple in best_triples:
            f.write(triple.rstrip() + "\n")
    
    print(f"\n✓ Saved ({len(best_triples)} triples) to: {out_path}")
    
    print("\n" + "="*70)
    print("✓ Task-Decomposed Process Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
