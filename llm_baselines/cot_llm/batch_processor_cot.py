#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processor for CoT Baseline - Entity List Processing

Process a list of specific entities with CoT summarization.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple
import argparse

from baseline_cot_llm import CoTLLMSummarizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_entity(
    entity_id: int,
    dataset_name: str,
    dataset_root: str,
    output_dir: str,
    summary_size: int,
    summarizer: CoTLLMSummarizer,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    no_sample: bool,
) -> bool:
    """
    Process a single entity.
    
    Args:
        entity_id: Entity ID
        dataset_name: Dataset name
        dataset_root: Root dataset directory
        output_dir: Output directory
        summary_size: Summary size
        summarizer: CoT Summarizer instance
        
    Returns:
        True if successful, False otherwise
    """
    # Get paths
    if dataset_name == "dbpedia":
        input_dir = Path(dataset_root) / "dbpedia_data" / str(entity_id)
        output_subdir = "dbpedia"
    elif dataset_name == "lmdb":
        input_dir = Path(dataset_root) / "lmdb_data" / str(entity_id)
        output_subdir = "lmdb"
    elif dataset_name == "faces":
        input_dir = Path(dataset_root) / "FACES" / "faces_data" / str(entity_id)
        output_subdir = "faces"
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False
    
    input_file = input_dir / f"{entity_id}_desc.nt"
    output_file = Path(output_dir) / output_subdir / str(entity_id) / f"{entity_id}_top{summary_size}.nt"
    
    # Check input
    if not input_file.exists():
        logger.warning(f"Entity {entity_id}: input file not found")
        return False
    
    # Extract URI
    try:
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                return False
            
            parts = first_line.split(None, 2)
            if len(parts) < 3:
                return False
            
            uri = parts[0]
            label = uri.split('/')[-1].rstrip('>')
    except Exception as e:
        logger.warning(f"Entity {entity_id}: could not extract URI: {e}")
        return False
    
    # Generate summary
    try:
        summary = summarizer.summarize(
            entity_id=str(entity_id),
            entity_uri=uri,
            entity_label=label,
            triple_file=str(input_file),
            summary_size=summary_size,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            do_sample=None if not no_sample else False,
        )
        
        if summary:
            os.makedirs(output_file.parent, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for triple in summary:
                    f.write(triple + '\n')
            
            logger.info(f"Entity {entity_id}: successfully processed")
            return True
        else:
            logger.warning(f"Entity {entity_id}: no summary generated")
            return False
    
    except Exception as e:
        logger.error(f"Entity {entity_id}: {e}")
        return False


def process_entities_list(
    entity_ids: List[int],
    dataset_name: str,
    dataset_root: str,
    output_dir: str,
    summary_size: int,
    summarizer: CoTLLMSummarizer,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    no_sample: bool,
) -> int:
    """Process multiple entities and return count of successful."""
    successful = 0
    for entity_id in entity_ids:
        if process_entity(
            entity_id,
            dataset_name,
            dataset_root,
            output_dir,
            summary_size,
            summarizer,
            temperature,
            max_new_tokens,
            top_p,
            no_sample,
        ):
            successful += 1
    
    return successful


def process_id_range(
    start_id: int,
    end_id: int,
    dataset_name: str,
    dataset_root: str,
    output_dir: str,
    summary_size: int,
    summarizer: CoTLLMSummarizer,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    no_sample: bool,
) -> int:
    """Process a range of entity IDs."""
    entity_ids = list(range(start_id, end_id + 1))
    return process_entities_list(
        entity_ids,
        dataset_name,
        dataset_root,
        output_dir,
        summary_size,
        summarizer,
        temperature,
        max_new_tokens,
        top_p,
        no_sample,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch CoT Processor - Process Entity List"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["dbpedia", "lmdb", "faces"],
        help="Dataset name"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="../datasets/ESBM_benchmark_v1.2",
        help="Root dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_cot_outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--summary-size",
        type=int,
        default=5,
        help="Summary size"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name"
    )
    parser.add_argument(
        "--model-local-dir",
        type=str,
        default=None,
        help="Optional local directory containing model files"
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download model to --model-local-dir if not present"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top-p nucleus sampling value when sampling is enabled"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Force greedy decoding regardless of temperature"
    )
    parser.add_argument(
        "--ids",
        nargs='+',
        type=int,
        help="Specific entity IDs to process"
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        help="Entity ID range (start end)"
    )
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = CoTLLMSummarizer(
        model_name=args.model,
        device=args.gpu,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        model_local_dir=args.model_local_dir,
        download_model=args.download_model,
    )
    
    # Process entities
    if args.ids:
        successful = process_entities_list(
            args.ids, args.dataset, args.dataset_root,
            args.output_dir, args.summary_size, summarizer,
            args.temperature, args.max_new_tokens, args.top_p, args.no_sample
        )
        print(f"Processed {successful}/{len(args.ids)} entities successfully")
    
    elif args.range:
        successful = process_id_range(
            args.range[0], args.range[1], args.dataset,
            args.dataset_root, args.output_dir, args.summary_size, summarizer,
            args.temperature, args.max_new_tokens, args.top_p, args.no_sample
        )
        print(f"Processed {successful}/{args.range[1] - args.range[0] + 1} entities successfully")
    
    else:
        print("Please specify either --ids or --range")


if __name__ == "__main__":
    main()
