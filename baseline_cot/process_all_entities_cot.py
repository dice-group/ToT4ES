#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processor for CoT Baseline Summaries

Automatically discovers all entities and generates summaries for the entire dataset
using the CoT (Chain-of-Thought) LLM approach.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_entity_uri(entity_id: int, dataset_root: str, dataset_name: str) -> Tuple[str, str]:
    """
    Extract entity URI from input _desc.nt file.
    
    Args:
        entity_id: Entity ID
        dataset_root: Root dataset directory
        dataset_name: Dataset name (dbpedia, lmdb, faces)
        
    Returns:
        Tuple of (uri, label) extracted from first triple
    """
    if dataset_name == "dbpedia":
        desc_file = Path(dataset_root) / "dbpedia_data" / str(entity_id) / f"{entity_id}_desc.nt"
    elif dataset_name == "lmdb":
        desc_file = Path(dataset_root) / "lmdb_data" / str(entity_id) / f"{entity_id}_desc.nt"
    elif dataset_name == "faces":
        desc_file = Path(dataset_root) / "FACES" / "faces_data" / str(entity_id) / f"{entity_id}_desc.nt"
    else:
        return None, None
    
    if not desc_file.exists():
        logger.warning(f"Description file not found: {desc_file}")
        return None, None
    
    try:
        with open(desc_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            
            if not first_line:
                return None, None
            
            # Parse first triple to extract subject URI
            parts = first_line.split(None, 2)
            if len(parts) >= 3:
                uri = parts[0]
                # Extract label from URI
                label = uri.split('/')[-1].rstrip('>')
                return uri, label
    
    except Exception as e:
        logger.warning(f"Error extracting entity URI: {e}")
    
    return None, None


def discover_entities(dataset_root: str, dataset_name: str) -> List[int]:
    """
    Discover all entity IDs in a dataset.
    
    Args:
        dataset_root: Root dataset directory
        dataset_name: Dataset name
        
    Returns:
        Sorted list of entity IDs
    """
    if dataset_name == "dbpedia":
        entity_dir = Path(dataset_root) / "dbpedia_data"
    elif dataset_name == "lmdb":
        entity_dir = Path(dataset_root) / "lmdb_data"
    elif dataset_name == "faces":
        entity_dir = Path(dataset_root) / "FACES" / "faces_data"
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        return []
    
    if not entity_dir.exists():
        logger.error(f"Dataset directory not found: {entity_dir}")
        return []
    
    entity_ids = []
    for item in entity_dir.iterdir():
        if item.is_dir():
            try:
                entity_id = int(item.name)
                entity_ids.append(entity_id)
            except ValueError:
                continue
    
    return sorted(entity_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Batch Process All Entities with CoT Baseline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dbpedia",
        choices=["dbpedia", "lmdb", "faces"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="../datasets/ESBM_benchmark_v1.2",
        help="Root directory of datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_cot_outputs",
        help="Output directory for summaries"
    )
    parser.add_argument(
        "--summary-size",
        type=int,
        default=5,
        help="Summary size (number of triples to select)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model to use"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip entities that already have output files"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="Start from specific entity ID"
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=None,
        help="End at specific entity ID"
    )
    
    args = parser.parse_args()
    
    # Import CoT summarizer here to avoid loading model if just printing help
    from baseline_cot_llm import CoTLLMSummarizer
    
    # Discover entities
    logger.info(f"Discovering entities in {args.dataset}...")
    entity_ids = discover_entities(args.dataset_root, args.dataset)
    
    if args.start_id:
        entity_ids = [eid for eid in entity_ids if eid >= args.start_id]
    if args.end_id:
        entity_ids = [eid for eid in entity_ids if eid <= args.end_id]
    
    logger.info(f"Found {len(entity_ids)} entities to process")
    
    # Initialize summarizer
    logger.info("Initializing CoT Summarizer...")
    summarizer = CoTLLMSummarizer(
        model_name=args.model,
        device=args.gpu,
    )
    
    # Process entities
    successful = 0
    failed = 0
    skipped = 0
    
    print("\n" + "=" * 80)
    print(f"Processing {len(entity_ids)} entities from {args.dataset.upper()}")
    print(f"Summary size: {args.summary_size}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")
    
    for entity_id in tqdm(entity_ids, desc=f"Processing {args.dataset}"):
        # Get entity data
        if args.dataset == "dbpedia":
            input_dir = Path(args.dataset_root) / "dbpedia_data" / str(entity_id)
            output_subdir = "dbpedia"
        elif args.dataset == "lmdb":
            input_dir = Path(args.dataset_root) / "lmdb_data" / str(entity_id)
            output_subdir = "lmdb"
        elif args.dataset == "faces":
            input_dir = Path(args.dataset_root) / "FACES" / "faces_data" / str(entity_id)
            output_subdir = "faces"
        
        input_file = input_dir / f"{entity_id}_desc.nt"
        output_file = Path(args.output_dir) / output_subdir / str(entity_id) / f"{entity_id}_top{args.summary_size}.nt"
        
        # Check if input exists
        if not input_file.exists():
            logger.warning(f"Entity {entity_id}: input file not found")
            failed += 1
            continue
        
        # Check if already processed
        if args.skip_existing and output_file.exists():
            skipped += 1
            continue
        
        # Extract entity URI
        uri, label = extract_entity_uri(entity_id, args.dataset_root, args.dataset)
        if not uri:
            logger.warning(f"Entity {entity_id}: could not extract URI")
            failed += 1
            continue
        
        # Generate summary
        try:
            summary = summarizer.summarize(
                entity_id=str(entity_id),
                entity_uri=uri,
                entity_label=label,
                triple_file=str(input_file),
                summary_size=args.summary_size,
            )
            
            if summary:
                # Save output
                os.makedirs(output_file.parent, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    for triple in summary:
                        f.write(triple + '\n')
                
                successful += 1
            else:
                failed += 1
        
        except Exception as e:
            logger.error(f"Entity {entity_id}: {e}")
            failed += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(entity_ids)}")
    print(f"Success rate: {100*successful/len(entity_ids):.1f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
