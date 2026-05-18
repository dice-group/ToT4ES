#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processor: Process multiple entities and save to structured .nt files

This script processes multiple entities from your dataset and saves
summaries in the same directory structure as the input data.

Output structure:
baseline_outputs/
└── dbpedia_data/
    ├── 1/
    │   └── 1_top5.nt
    ├── 2/
    │   └── 2_top5.nt
    └── ...
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple
import argparse
from baseline_direct_llm import BaselineLLMSummarizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple entities in batch."""
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        dataset_root: str = "../datasets/ESBM_benchmark_v1.2",
    ):
        """
        Initialize batch processor.
        
        Args:
            model_id: LLM model identifier
            dataset_root: Root path to datasets
        """
        self.summarizer = BaselineLLMSummarizer(model_id=model_id)
        self.dataset_root = Path(dataset_root)
    
    def process_entity(
        self,
        entity_id: int,
        entity_uri: str,
        entity_label: str,
        summary_size: int = 5,
        dataset_name: str = "dbpedia",
        output_dir: str = "baseline_outputs",
        temperature: float = 0.1,
    ) -> Tuple[bool, str]:
        """
        Process a single entity and save to .nt file.
        
        Args:
            entity_id: Entity ID number
            entity_uri: Entity URI
            entity_label: Entity label
            summary_size: Target summary size
            dataset_name: Dataset name (dbpedia, lmdb, faces)
            output_dir: Output directory
            temperature: LLM temperature
            
        Returns:
            Tuple of (success, output_path or error_message)
        """
        try:
            # Build input path
            if dataset_name == "dbpedia":
                input_dir = self.dataset_root / "dbpedia_data" / str(entity_id)
                triple_file = input_dir / f"{entity_id}_desc.nt"
            elif dataset_name == "lmdb":
                input_dir = self.dataset_root / "lmdb_data" / str(entity_id)
                triple_file = input_dir / f"{entity_id}_desc.nt"
            elif dataset_name == "faces":
                input_dir = self.dataset_root / "FACES" / "faces_data" / str(entity_id)
                triple_file = input_dir / f"{entity_id}_desc.nt"
            else:
                return False, f"Unknown dataset: {dataset_name}"
            
            # Check if file exists
            if not triple_file.exists():
                return False, f"Triple file not found: {triple_file}"
            
            # Load and process
            raw_triples = self.summarizer.load_triples(str(triple_file))
            summary = self.summarizer.summarize(
                entity_uri=entity_uri,
                entity_label=entity_label,
                raw_triples=raw_triples,
                summary_size=summary_size,
                temperature=temperature,
            )
            
            # Save
            output_path = self.summarizer.save_summary_with_metadata(
                summary=summary,
                entity_id=entity_id,
                entity_uri=entity_uri,
                entity_label=entity_label,
                summary_size=summary_size,
                output_dir=output_dir,
                dataset_name=dataset_name,
            )
            
            return True, output_path
        
        except Exception as e:
            logger.error(f"Error processing entity {entity_id}: {e}")
            return False, str(e)
    
    def process_entities_list(
        self,
        entities: List[Tuple[int, str, str]],
        summary_size: int = 5,
        dataset_name: str = "dbpedia",
        output_dir: str = "baseline_outputs",
        temperature: float = 0.1,
    ) -> dict:
        """
        Process multiple entities.
        
        Args:
            entities: List of (entity_id, entity_uri, entity_label) tuples
            summary_size: Target summary size
            dataset_name: Dataset name
            output_dir: Output directory
            temperature: LLM temperature
            
        Returns:
            Dict with results
        """
        results = {
            "total": len(entities),
            "success": 0,
            "failed": 0,
            "outputs": [],
            "errors": [],
        }
        
        for i, (entity_id, entity_uri, entity_label) in enumerate(entities, 1):
            logger.info(f"[{i}/{len(entities)}] Processing {entity_label} (ID: {entity_id})")
            
            success, result = self.process_entity(
                entity_id=entity_id,
                entity_uri=entity_uri,
                entity_label=entity_label,
                summary_size=summary_size,
                dataset_name=dataset_name,
                output_dir=output_dir,
                temperature=temperature,
            )
            
            if success:
                results["success"] += 1
                results["outputs"].append({
                    "entity_id": entity_id,
                    "entity_label": entity_label,
                    "output_path": result,
                })
                logger.info(f"  ✓ Saved to {result}")
            else:
                results["failed"] += 1
                results["errors"].append({
                    "entity_id": entity_id,
                    "entity_label": entity_label,
                    "error": result,
                })
                logger.warning(f"  ✗ Error: {result}")
        
        return results
    
    def process_id_range(
        self,
        start_id: int,
        end_id: int,
        entity_uri_template: str = "http://dbpedia.org/resource/{label}",
        entity_label_mapping: dict = None,
        summary_size: int = 5,
        dataset_name: str = "dbpedia",
        output_dir: str = "baseline_outputs",
        temperature: float = 0.1,
    ) -> dict:
        """
        Process a range of entity IDs.
        
        Args:
            start_id: Starting entity ID
            end_id: Ending entity ID (inclusive)
            entity_uri_template: Template for entity URI
            entity_label_mapping: Dict mapping ID to labels (optional)
            summary_size: Target summary size
            dataset_name: Dataset name
            output_dir: Output directory
            temperature: LLM temperature
            
        Returns:
            Results dict
        """
        entities = []
        
        for entity_id in range(start_id, end_id + 1):
            # Get entity label
            if entity_label_mapping and entity_id in entity_label_mapping:
                label = entity_label_mapping[entity_id]
            else:
                label = f"Entity_{entity_id}"
            
            # Build URI
            uri = entity_uri_template.format(label=label, id=entity_id)
            
            entities.append((entity_id, uri, label))
        
        return self.process_entities_list(
            entities=entities,
            summary_size=summary_size,
            dataset_name=dataset_name,
            output_dir=output_dir,
            temperature=temperature,
        )


def main():
    """CLI for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process entities and save to .nt files"
    )
    
    # Input options
    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "--entity-ids",
        type=str,
        help="Comma-separated entity IDs (e.g., '1,2,100,11')"
    )
    input_group.add_argument(
        "--id-range",
        type=str,
        help="Range of entity IDs (e.g., '1-10' for 1 to 10)"
    )
    input_group.add_argument(
        "--entity-file",
        type=str,
        help="File with entities (one per line: ID:URI:Label)"
    )
    
    # Output options
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="baseline_outputs",
        help="Base output directory (default: baseline_outputs)"
    )
    output_group.add_argument(
        "--dataset",
        type=str,
        default="dbpedia",
        choices=["dbpedia", "lmdb", "faces"],
        help="Dataset name (default: dbpedia)"
    )
    
    # Processing options
    proc_group = parser.add_argument_group("processing")
    proc_group.add_argument(
        "--summary-size",
        type=int,
        default=5,
        help="Target summary size (default: 5)"
    )
    proc_group.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )
    proc_group.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="LLM model identifier"
    )
    proc_group.add_argument(
        "--dataset-root",
        type=str,
        default="../datasets/ESBM_benchmark_v1.2",
        help="Dataset root path"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_count = sum([
        args.entity_ids is not None,
        args.id_range is not None,
        args.entity_file is not None,
    ])
    
    if input_count != 1:
        parser.error("Exactly one of --entity-ids, --id-range, or --entity-file must be specified")
    
    # Initialize processor
    processor = BatchProcessor(
        model_id=args.model,
        dataset_root=args.dataset_root,
    )
    
    # Process based on input type
    if args.entity_ids:
        # Parse comma-separated IDs
        entity_ids = [int(x.strip()) for x in args.entity_ids.split(',')]
        # Build entity list (placeholder labels)
        entities = [
            (eid, f"http://dbpedia.org/resource/Entity_{eid}", f"Entity_{eid}")
            for eid in entity_ids
        ]
        results = processor.process_entities_list(
            entities=entities,
            summary_size=args.summary_size,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            temperature=args.temperature,
        )
    
    elif args.id_range:
        # Parse range
        start_id, end_id = args.id_range.split('-')
        start_id, end_id = int(start_id), int(end_id)
        results = processor.process_id_range(
            start_id=start_id,
            end_id=end_id,
            summary_size=args.summary_size,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            temperature=args.temperature,
        )
    
    else:  # entity_file
        # Load entities from file
        entities = []
        with open(args.entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(':')
                if len(parts) >= 3:
                    entity_id = int(parts[0])
                    uri = parts[1]
                    label = parts[2]
                    entities.append((entity_id, uri, label))
        
        results = processor.process_entities_list(
            entities=entities,
            summary_size=args.summary_size,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            temperature=args.temperature,
        )
    
    # Print results
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING RESULTS")
    print(f"{'=' * 80}")
    print(f"Total entities: {results['total']}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    
    if results['success'] > 0:
        print(f"\nOutputs:")
        for item in results['outputs']:
            print(f"  {item['entity_label']:30} → {item['output_path']}")
    
    if results['errors']:
        print(f"\nErrors:")
        for item in results['errors']:
            print(f"  {item['entity_label']:30} → {item['error']}")
    
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
