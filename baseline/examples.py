#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using the Baseline Summarizer Programmatically

This file demonstrates how to use the BaselineLLMSummarizer in your own code.
"""

import logging
from baseline_direct_llm import BaselineLLMSummarizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_simple_summarization():
    """Example 1: Basic summarization of a single entity."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Summarization")
    print("=" * 80)
    
    # Initialize summarizer
    summarizer = BaselineLLMSummarizer()
    
    # Load triples
    triple_file = "../datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt"
    raw_triples = summarizer.load_triples(triple_file)
    logger.info(f"Loaded {len(raw_triples)} raw triples")
    
    # Summarize
    summary = summarizer.summarize(
        entity_uri="http://dbpedia.org/resource/Marie_Curie",
        entity_label="Marie Curie",
        raw_triples=raw_triples,
        summary_size=5,
        temperature=0.1,
    )
    
    logger.info(f"Generated summary with {len(summary)} triples")
    
    # Print results
    print("\nSummary:")
    for triple in summary:
        print(f"  {triple}")
    
    return summary


def example_2_comparison_different_sizes():
    """Example 2: Compare summaries of different sizes."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Different Summary Sizes")
    print("=" * 80)
    
    summarizer = BaselineLLMSummarizer()
    triple_file = "../datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt"
    raw_triples = summarizer.load_triples(triple_file)
    
    sizes = [3, 5, 10]
    summaries = {}
    
    for size in sizes:
        logger.info(f"\nGenerating {size}-triple summary...")
        summary = summarizer.summarize(
            entity_uri="http://dbpedia.org/resource/Marie_Curie",
            entity_label="Marie Curie",
            raw_triples=raw_triples,
            summary_size=size,
            temperature=0.1,
        )
        summaries[size] = summary
        
        print(f"\n{size}-Triple Summary:")
        for triple in summary:
            print(f"  {triple}")
    
    return summaries


def example_3_using_preprocessed_triples():
    """Example 3: Manual preprocessing and summarization."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Manual Preprocessing")
    print("=" * 80)
    
    summarizer = BaselineLLMSummarizer()
    triple_file = "../datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt"
    
    # Load raw triples
    raw_triples = summarizer.load_triples(triple_file)
    logger.info(f"Raw triples: {len(raw_triples)}")
    
    # Preprocess (filter & deduplicate)
    processed = summarizer.preprocess_triples(raw_triples)
    logger.info(f"Processed triples: {len(processed)}")
    
    # Show top predicates
    from collections import Counter
    predicates = [p for _, p, _ in processed]
    top_predicates = Counter(predicates).most_common(10)
    
    print("\nTop 10 Most Common Predicates:")
    for pred, count in top_predicates:
        pred_short = pred.split('/')[-1].rstrip('>')
        print(f"  {pred_short}: {count}")
    
    # Show sample triples
    print("\nSample Processed Triples:")
    for s, p, o in processed[:5]:
        s_short = s.split('/')[-1].rstrip('>')
        p_short = p.split('/')[-1].rstrip('>')
        o_short = o.split('/')[-1].rstrip('>') if o.startswith('<') else o[:50]
        print(f"  {s_short} -> {p_short} -> {o_short}")


def example_4_batch_processing():
    """Example 4: Process multiple entities."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 80)
    
    summarizer = BaselineLLMSummarizer()
    
    # Entity list: (id, uri, label)
    entities = [
        (1, "http://dbpedia.org/resource/Marie_Curie", "Marie Curie"),
        (2, "http://dbpedia.org/resource/Albert_Einstein", "Albert Einstein"),
    ]
    
    summaries = {}
    
    for entity_id, uri, label in entities:
        logger.info(f"\nProcessing {label}...")
        
        # Load triples
        triple_file = f"../datasets/ESBM_benchmark_v1.2/dbpedia_data/{entity_id}/{entity_id}_desc.nt"
        
        try:
            raw_triples = summarizer.load_triples(triple_file)
            
            # Summarize
            summary = summarizer.summarize(
                entity_uri=uri,
                entity_label=label,
                raw_triples=raw_triples,
                summary_size=5,
                temperature=0.1,
            )
            
            summaries[label] = summary
            logger.info(f"✓ {label}: {len(summary)} triples")
            
        except Exception as e:
            logger.error(f"✗ {label}: {e}")
    
    # Print summary
    print("\nBatch Processing Results:")
    for label, summary in summaries.items():
        print(f"\n{label} ({len(summary)} triples):")
        for triple in summary[:3]:  # Show first 3
            print(f"  {triple}")
        if len(summary) > 3:
            print(f"  ... and {len(summary) - 3} more")


def example_5_predicate_analysis():
    """Example 5: Analyze predicate coverage."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Predicate Analysis")
    print("=" * 80)
    
    from collections import defaultdict
    
    summarizer = BaselineLLMSummarizer()
    triple_file = "../datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt"
    
    # Load and preprocess
    raw_triples = summarizer.load_triples(triple_file)
    processed = summarizer.preprocess_triples(raw_triples)
    
    # Analyze predicates
    predicate_counts = defaultdict(int)
    for _, p, _ in processed:
        p_short = p.split('/')[-1].rstrip('>')
        predicate_counts[p_short] += 1
    
    # Sort by frequency
    sorted_preds = sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTotal unique predicates: {len(predicate_counts)}")
    print("\nTop 15 predicates by frequency:")
    print(f"{'Predicate':<40} {'Count':>10}")
    print("-" * 51)
    for pred, count in sorted_preds[:15]:
        print(f"{pred:<40} {count:>10}")
    
    # Metadata predicates
    logger.info("\nMetadata filtering analysis:")
    print(f"Original triples: {len(raw_triples)}")
    print(f"After filtering metadata: {len(processed)}")
    print(f"Reduction: {len(raw_triples) - len(processed)} triples")


def example_6_export_results():
    """Example 6: Export summaries to file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Export Results")
    print("=" * 80)
    
    import os
    from pathlib import Path
    
    summarizer = BaselineLLMSummarizer()
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    triple_file = "../datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt"
    raw_triples = summarizer.load_triples(triple_file)
    
    # Summarize
    summary = summarizer.summarize(
        entity_uri="http://dbpedia.org/resource/Marie_Curie",
        entity_label="Marie Curie",
        raw_triples=raw_triples,
        summary_size=5,
        temperature=0.1,
    )
    
    # Export as N-Triples
    output_file = output_dir / "marie_curie_summary.nt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Baseline Summary for Marie Curie\n")
        f.write("# Generated by BaselineLLMSummarizer\n")
        f.write("#\n")
        for triple in summary:
            f.write(triple + "\n")
    
    logger.info(f"Summary exported to {output_file}")
    
    # Export as JSON (for convenience)
    import json
    
    json_file = output_dir / "marie_curie_summary.json"
    summary_data = {
        "entity": "Marie Curie",
        "uri": "http://dbpedia.org/resource/Marie_Curie",
        "summary_size": 5,
        "triples": summary,
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"Summary exported to {json_file}")
    
    print(f"\nExported files:")
    print(f"  - {output_file}")
    print(f"  - {json_file}")


if __name__ == "__main__":
    # Run examples
    # Uncomment the examples you want to run:
    
    example_1_simple_summarization()
    # example_2_comparison_different_sizes()
    # example_3_using_preprocessed_triples()
    # example_4_batch_processing()
    # example_5_predicate_analysis()
    # example_6_export_results()
