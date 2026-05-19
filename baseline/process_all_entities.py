#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process All Entities: Automatically discover and process all entities in dataset

This script automatically finds all entities in your dataset and processes them
all at once, saving summaries to the structured .nt format.

Usage:
    python process_all_entities.py --dataset dbpedia
    python process_all_entities.py --dataset dbpedia --summary-size 5
    python process_all_entities.py --dataset lmdb --output-dir results
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm

import sys
from pathlib import Path

# Add baseline directory to path
baseline_dir = Path(__file__).parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))

from batch_processor import BatchProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AllEntitiesProcessor:
    """Automatically discover and process all entities in a dataset."""
    
    def __init__(
        self,
        dataset_root: str = "../datasets/ESBM_benchmark_v1.2",
        dataset_name: str = "dbpedia",
    ):
        """
        Initialize processor for all entities.
        
        Args:
            dataset_root: Root path to datasets
            dataset_name: Dataset name (dbpedia, lmdb, faces)
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.batch_processor = BatchProcessor(dataset_root=dataset_root)
    
    def discover_entities(self) -> List[int]:
        """
        Automatically discover all entity IDs in dataset.
        
        Returns:
            Sorted list of entity IDs
        """
        if self.dataset_name == "dbpedia":
            entity_dir = self.dataset_root / "dbpedia_data"
        elif self.dataset_name == "lmdb":
            entity_dir = self.dataset_root / "lmdb_data"
        elif self.dataset_name == "faces":
            entity_dir = self.dataset_root / "FACES" / "faces_data"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        if not entity_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {entity_dir}")
        
        entity_ids = []
        
        # Find all numeric directories
        for item in entity_dir.iterdir():
            if item.is_dir():
                try:
                    entity_id = int(item.name)
                    entity_ids.append(entity_id)
                except ValueError:
                    # Skip non-numeric directories
                    continue
        
        entity_ids.sort()
        logger.info(f"Discovered {len(entity_ids)} entities in {self.dataset_name}")
        
        return entity_ids
    
    def extract_entity_uri(self, entity_id: int) -> Tuple[str, str]:
        """
        Extract actual entity URI from the input triple file.
        
        Args:
            entity_id: Entity ID number
            
        Returns:
            Tuple of (uri, label) extracted from the data
        """
        if self.dataset_name == "dbpedia":
            entity_dir = self.dataset_root / "dbpedia_data" / str(entity_id)
            triple_file = entity_dir / f"{entity_id}_desc.nt"
        elif self.dataset_name == "lmdb":
            entity_dir = self.dataset_root / "lmdb_data" / str(entity_id)
            triple_file = entity_dir / f"{entity_id}_desc.nt"
        elif self.dataset_name == "faces":
            entity_dir = self.dataset_root / "FACES" / "faces_data" / str(entity_id)
            triple_file = entity_dir / f"{entity_id}_desc.nt"
        else:
            return f"http://example.com/entity/{entity_id}", f"Entity_{entity_id}"
        
        # Try to extract URI from first triple in file
        if triple_file.exists():
            try:
                with open(triple_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse the first triple to get subject (entity URI)
                            parts = line.split(' ', 1)
                            if parts:
                                uri = parts[0].strip('<>').strip()
                                # Extract label from URI (last part after /)
                                label = uri.split('/')[-1].replace('_', ' ')
                                return uri, label
            except Exception as e:
                logger.warning(f"Could not extract URI from {triple_file}: {e}")
        
        # Fallback to default
        return f"http://dbpedia.org/resource/Entity_{entity_id}", f"Entity_{entity_id}"
    
    def build_entity_list(
        self,
        entity_ids: List[int],
        use_labels: bool = False,
        label_mapping: Dict[int, str] = None,
    ) -> List[Tuple[int, str, str]]:
        """
        Build list of (entity_id, uri, label) tuples.
        
        Args:
            entity_ids: List of entity IDs
            use_labels: Whether to fetch real labels (slower)
            label_mapping: Dict mapping ID to label (optional)
            
        Returns:
            List of (entity_id, uri, label) tuples
        """
        entities = []
        
        for entity_id in entity_ids:
            # Extract actual entity URI from input data
            uri, label = self.extract_entity_uri(entity_id)
            
            # Override with label mapping if provided
            if label_mapping and entity_id in label_mapping:
                label = label_mapping[entity_id]
            
            entities.append((entity_id, uri, label))
        
        return entities
    
    def process_all(
        self,
        summary_size: int = 5,
        output_dir: str = "baseline_outputs",
        temperature: float = 0.1,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_entities: int = None,
        skip_existing: bool = True,
    ) -> Dict:
        """
        Process all discovered entities.
        
        Args:
            summary_size: Target summary size
            output_dir: Output directory
            temperature: LLM temperature
            model_id: LLM model identifier
            max_entities: Max entities to process (None = all)
            skip_existing: Skip if output file already exists
            
        Returns:
            Results dict
        """
        # Discover entities
        entity_ids = self.discover_entities()
        
        if max_entities:
            entity_ids = entity_ids[:max_entities]
            logger.info(f"Processing first {max_entities} entities (limited)")
        
        # Build entity list
        entities = self.build_entity_list(entity_ids)
        
        logger.info(f"Processing {len(entities)} entities...")
        
        results = {
            "total": len(entities),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "outputs": [],
            "errors": [],
        }
        
        # Process with progress bar
        for i, (entity_id, uri, label) in enumerate(tqdm(entities, desc="Processing"), 1):
            # Check if output already exists
            if skip_existing:
                output_path = (
                    Path(output_dir)
                    / f"{self.dataset_name}_data"
                    / str(entity_id)
                    / f"{entity_id}_top{summary_size}.nt"
                )
                
                if output_path.exists():
                    results["skipped"] += 1
                    results["outputs"].append({
                        "entity_id": entity_id,
                        "entity_label": label,
                        "output_path": str(output_path),
                        "status": "skipped",
                    })
                    continue
            
            # Process entity
            try:
                success, result = self.batch_processor.process_entity(
                    entity_id=entity_id,
                    entity_uri=uri,
                    entity_label=label,
                    summary_size=summary_size,
                    dataset_name=self.dataset_name,
                    output_dir=output_dir,
                    temperature=temperature,
                )
                
                if success:
                    results["success"] += 1
                    results["outputs"].append({
                        "entity_id": entity_id,
                        "entity_label": label,
                        "output_path": result,
                        "status": "success",
                    })
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "entity_id": entity_id,
                        "entity_label": label,
                        "error": result,
                    })
            
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "entity_id": entity_id,
                    "entity_label": label,
                    "error": str(e),
                })
        
        return results
    
    def save_results_log(self, results: Dict, log_file: str = "processing_results.txt"):
        """Save results to log file."""
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ALL ENTITIES PROCESSING RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total entities: {results['total']}\n")
            f.write(f"Successful: {results['success']}\n")
            f.write(f"Failed: {results['failed']}\n")
            f.write(f"Skipped: {results['skipped']}\n\n")
            
            if results['success'] > 0:
                f.write("SUCCESSFUL PROCESSES\n")
                f.write("-" * 40 + "\n")
                for item in results['outputs']:
                    if item['status'] == 'success':
                        f.write(f"ID {item['entity_id']:5} | {item['entity_label']:30} | {item['output_path']}\n")
                f.write("\n")
            
            if results['skipped'] > 0:
                f.write("SKIPPED (ALREADY EXIST)\n")
                f.write("-" * 40 + "\n")
                for item in results['outputs']:
                    if item['status'] == 'skipped':
                        f.write(f"ID {item['entity_id']:5} | {item['entity_label']:30}\n")
                f.write("\n")
            
            if results['errors']:
                f.write("ERRORS\n")
                f.write("-" * 40 + "\n")
                for item in results['errors']:
                    f.write(f"ID {item['entity_id']:5} | {item['entity_label']:30}\n")
                    f.write(f"  Error: {item['error']}\n\n")
        
        logger.info(f"Results log saved to {log_file}")


def main():
    """CLI for processing all entities."""
    parser = argparse.ArgumentParser(
        description="Process ALL entities in a dataset"
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        default="dbpedia",
        choices=["dbpedia", "lmdb", "faces"],
        help="Dataset name (default: dbpedia)"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root path (auto-discovered if not provided)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_outputs",
        help="Base output directory (default: baseline_outputs)"
    )
    
    # Processing options
    parser.add_argument(
        "--summary-size",
        type=int,
        default=5,
        help="Target summary size (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="LLM model identifier"
    )
    
    # Processing control
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="Max entities to process (default: all)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip if output already exists (default: True)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_false",
        dest="skip_existing",
        help="Process all, even if output exists"
    )
    
    # Output options
    parser.add_argument(
        "--log-file",
        type=str,
        default="processing_results.txt",
        help="Log file for results (default: processing_results.txt)"
    )
    
    # GPU options
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (default: 0). Set to -1 to use CPU only"
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES string (e.g., '0,1' for GPUs 0 and 1)"
    )
    
    args = parser.parse_args()
    
    # Auto-discover dataset root if not provided
    if args.dataset_root is None:
        # Try to find datasets directory
        current = Path.cwd()
        found = False
        
        # Search up to 5 levels up the directory tree
        for _ in range(5):
            candidate = current / "datasets" / "ESBM_benchmark_v1.2"
            if candidate.exists():
                args.dataset_root = str(candidate)
                logger.info(f"Auto-discovered dataset root: {args.dataset_root}")
                found = True
                break
            current = current.parent
        
        if not found:
            # Try common relative paths
            for rel_path in ["../datasets/ESBM_benchmark_v1.2", "datasets/ESBM_benchmark_v1.2", "../../datasets/ESBM_benchmark_v1.2"]:
                candidate = Path(rel_path).resolve()
                if candidate.exists():
                    args.dataset_root = str(candidate)
                    logger.info(f"Auto-discovered dataset root: {args.dataset_root}")
                    found = True
                    break
        
        if not found:
            logger.error(
                "Could not auto-discover dataset root. Please specify --dataset-root\n"
                "Example: python process_all_entities.py --dataset faces --dataset-root /path/to/datasets/ESBM_benchmark_v1.2"
            )
            sys.exit(1)
    
    # Resolve to absolute path
    args.dataset_root = str(Path(args.dataset_root).resolve())
    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    elif args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Initialize processor
    processor = AllEntitiesProcessor(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset,
    )
    
    print(f"\n{'=' * 80}")
    print(f"Processing ALL entities from {args.dataset.upper()}")
    print(f"{'=' * 80}\n")
    
    # Process all
    results = processor.process_all(
        summary_size=args.summary_size,
        output_dir=args.output_dir,
        temperature=args.temperature,
        model_id=args.model,
        max_entities=args.max_entities,
        skip_existing=args.skip_existing,
    )
    
    # Save results log
    processor.save_results_log(results, args.log_file)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total entities: {results['total']}")
    print(f"Successful: {results['success']} ✓")
    print(f"Failed: {results['failed']} ✗")
    print(f"Skipped: {results['skipped']} ⊘")
    
    if results['success'] > 0 and results['success'] <= 10:
        print(f"\nProcessed entities:")
        for item in results['outputs'][:10]:
            if item['status'] == 'success':
                print(f"  {item['entity_label']} → {Path(item['output_path']).name}")
        if len(results['outputs']) > 10:
            print(f"  ... and {len(results['outputs']) - 10} more")
    
    print(f"\nResults saved to: {args.log_file}")
    print(f"Outputs in: {args.output_dir}/\n")


if __name__ == "__main__":
    main()
