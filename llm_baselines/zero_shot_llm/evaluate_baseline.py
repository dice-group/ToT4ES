#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Evaluation: Calculate Precision, Recall, F1-Score for Baseline Summaries

This script evaluates baseline summaries (in baseline_outputs/) against ground truth
triple files (in datasets/ESBM_benchmark_v1.2/) by computing:
- Precision: % of predicted triples that are in ground truth
- Recall: % of ground truth triples that are in predicted summary
- F1-Score: Harmonic mean of precision and recall

Usage:
    python evaluate_baseline.py --dataset dbpedia --summary-size 5
    python evaluate_baseline.py --dataset dbpedia --summary-size 5 --output results.csv
    python evaluate_baseline.py --all-datasets --output all_results.csv
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
import csv
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NTriplesParser:
    """Parser for N-Triples format."""
    
    @staticmethod
    def parse_triple(line: str) -> Tuple[str, str, str]:
        """
        Parse N-Triples line into (subject, predicate, object).
        
        Args:
            line: N-Triples line
            
        Returns:
            Tuple (subject, predicate, object) or (None, None, None)
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None, None
        
        # Remove trailing period and whitespace
        if line.endswith(' .'):
            line = line[:-2].strip()
        elif line.endswith('.'):
            line = line[:-1].strip()
        
        # Parse: split by whitespace but preserve URIs and literals
        parts = []
        current = ""
        in_uri = False
        in_literal = False
        escape = False
        
        for char in line:
            if char == '<' and not in_literal:
                in_uri = True
                current += char
            elif char == '>' and in_uri:
                in_uri = False
                current += char
            elif char == '"' and not escape:
                in_literal = not in_literal
                current += char
            elif char == '\\' and in_literal:
                escape = True
                current += char
                continue
            elif char == ' ' and not in_uri and not in_literal:
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += char
            escape = False
        
        if current:
            parts.append(current)
        
        if len(parts) >= 3:
            return parts[0], parts[1], ' '.join(parts[2:])
        
        return None, None, None
    
    @staticmethod
    def normalize_triple(subject: str, predicate: str, obj: str) -> Tuple[str, str, str]:
        """
        Normalize a triple for comparison:
        - Strip angle brackets from URIs
        - Normalize whitespace
        
        Args:
            subject, predicate, obj: Triple components
            
        Returns:
            Normalized triple tuple
        """
        # Remove angle brackets from URIs
        subject = subject.strip('<>').strip()
        predicate = predicate.strip('<>').strip()
        
        # For objects, only strip if it's a URI (starts with <)
        if obj.startswith('<'):
            obj = obj.strip('<>').strip()
        else:
            obj = obj.strip()
        
        return (subject, predicate, obj)
    
    @staticmethod
    def load_triples_from_file(filepath: str) -> Set[Tuple[str, str, str]]:
        """
        Load and parse all triples from an N-Triples file.
        
        Args:
            filepath: Path to .nt file
            
        Returns:
            Set of normalized (subject, predicate, object) tuples
        """
        triples = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    s, p, o = NTriplesParser.parse_triple(line)
                    if s is not None and p is not None and o is not None:
                        # Normalize for comparison
                        norm = NTriplesParser.normalize_triple(s, p, o)
                        triples.add(norm)
        except Exception as e:
            logger.warning(f"Error loading triples from {filepath}: {e}")
        
        return triples


class BaselineEvaluator:
    """Evaluate baseline summaries against ground truth."""
    
    def __init__(
        self,
        dataset_root: str = "../datasets/ESBM_benchmark_v1.2",
        baseline_output_dir: str = "baseline_outputs",
    ):
        """
        Initialize evaluator.
        
        Args:
            dataset_root: Root directory of datasets
            baseline_output_dir: Directory containing baseline outputs
        """
        self.dataset_root = Path(dataset_root)
        self.baseline_output_dir = Path(baseline_output_dir)
    
    def get_entity_ids(self, dataset_name: str) -> List[int]:
        """
        Get all entity IDs for a dataset.
        
        Args:
            dataset_name: Dataset name (dbpedia, lmdb, faces)
            
        Returns:
            Sorted list of entity IDs
        """
        if dataset_name == "dbpedia":
            entity_dir = self.dataset_root / "dbpedia_data"
        elif dataset_name == "lmdb":
            entity_dir = self.dataset_root / "lmdb_data"
        elif dataset_name == "faces":
            entity_dir = self.dataset_root / "FACES" / "faces_data"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if not entity_dir.exists():
            logger.warning(f"Dataset directory not found: {entity_dir}")
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
    
    def get_ground_truth_all_annotators(
        self, 
        dataset_name: str, 
        entity_id: int, 
        summary_size: int,
        max_files: int = 6,
    ) -> List[Set[Tuple[str, str, str]]]:
        """
        Load all ground truth annotations for an entity at a specific summary size.
        
        The dataset contains multiple annotators/gold summaries:
        {entity_id}_gold_top{summary_size}_{0}.nt
        {entity_id}_gold_top{summary_size}_{1}.nt
        ... up to file_n
        
        Args:
            dataset_name: Dataset name
            entity_id: Entity ID
            summary_size: Summary size (k)
            max_files: Maximum number of gold files to check
            
        Returns:
            List of Sets, each containing normalized triples from one annotator
        """
        if dataset_name == "dbpedia":
            entity_dir = self.dataset_root / "dbpedia_data" / str(entity_id)
        elif dataset_name == "lmdb":
            entity_dir = self.dataset_root / "lmdb_data" / str(entity_id)
        elif dataset_name == "faces":
            entity_dir = self.dataset_root / "FACES" / "faces_data" / str(entity_id)
        else:
            return []
        
        if not entity_dir.exists():
            logger.warning(f"Entity directory not found: {entity_dir}")
            return []
        
        gold_summaries = []
        
        # Try to load gold files with indices: 0, 1, 2, 3, 4, 5
        for file_idx in range(max_files):
            gold_file = entity_dir / f"{entity_id}_gold_top{summary_size}_{file_idx}.nt"
            
            if gold_file.exists():
                gold_triples = NTriplesParser.load_triples_from_file(str(gold_file))
                if gold_triples:
                    gold_summaries.append(gold_triples)
                    logger.debug(f"Loaded {len(gold_triples)} triples from {gold_file}")
            else:
                # Stop if we encounter a gap (assume indices are contiguous)
                if gold_summaries:
                    break
        
        if not gold_summaries:
            logger.warning(f"No gold summaries found for entity {entity_id} at k={summary_size}")
        
        return gold_summaries
    
    def get_baseline_summary(self, dataset_name: str, entity_id: int, summary_size: int) -> Set[Tuple[str, str, str]]:
        """
        Load baseline summary for an entity.
        
        Args:
            dataset_name: Dataset name
            entity_id: Entity ID
            summary_size: Summary size (k)
            
        Returns:
            Set of normalized triples
        """
        if dataset_name == "dbpedia":
            subdir = "dbpedia"
        elif dataset_name == "lmdb":
            subdir = "lmdb"
        elif dataset_name == "faces":
            subdir = "faces"
        else:
            return set()
        
        summary_file = self.baseline_output_dir / subdir / str(entity_id) / f"{entity_id}_top{summary_size}.nt"
        
        if not summary_file.exists():
            logger.warning(f"Baseline summary not found: {summary_file}")
            return set()
        
        return NTriplesParser.load_triples_from_file(str(summary_file))
    
    def calculate_metrics(
        self,
        predicted: Set[Tuple[str, str, str]],
        ground_truths: List[Set[Tuple[str, str, str]]],
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score against multiple ground truths.
        
        Averages F1-scores across all annotators (gold summaries), similar to
        multi-reference evaluation. This matches the reference evaluation approach.
        
        Args:
            predicted: Set of predicted triples
            ground_truths: List of ground truth sets (one per annotator)
            
        Returns:
            Dict with precision, recall, f1 (averaged), and per-annotator f1 scores
        """
        if not predicted or not ground_truths:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "f1_per_annotator": [],
                "num_annotators": len(ground_truths),
            }
        
        f1_scores = []
        
        # Calculate F1 against each ground truth annotator and average
        for gt_idx, ground_truth in enumerate(ground_truths):
            if not ground_truth:
                continue
            
            # Calculate TP, FP, FN for this annotator
            tp = len(predicted & ground_truth)  # Intersection
            fp = len(predicted - ground_truth)  # Predicted but not in ground truth
            fn = len(ground_truth - predicted)  # Ground truth but not predicted
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
        
        # Average F1 across annotators
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        # For overall precision/recall, calculate against union of all gold summaries
        all_gold_triples = set()
        for gt in ground_truths:
            all_gold_triples.update(gt)
        
        tp_overall = len(predicted & all_gold_triples)
        fp_overall = len(predicted - all_gold_triples)
        fn_overall = len(all_gold_triples - predicted)
        
        precision_overall = tp_overall / (tp_overall + fp_overall) if (tp_overall + fp_overall) > 0 else 0.0
        recall_overall = tp_overall / (tp_overall + fn_overall) if (tp_overall + fn_overall) > 0 else 0.0
        
        return {
            "precision": precision_overall,
            "recall": recall_overall,
            "f1": avg_f1,
            "f1_per_annotator": f1_scores,
            "num_annotators": len(ground_truths),
            "tp": tp_overall,
            "fp": fp_overall,
            "fn": fn_overall,
        }
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        summary_size: int = 5,
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate baseline for all entities in a dataset against multiple ground truths.
        
        Args:
            dataset_name: Dataset name
            summary_size: Summary size (k)
            
        Returns:
            Tuple of (aggregated metrics, per-entity results)
        """
        entity_ids = self.get_entity_ids(dataset_name)
        
        if not entity_ids:
            logger.warning(f"No entities found for {dataset_name}")
            return {}, []
        
        logger.info(f"Evaluating {len(entity_ids)} entities from {dataset_name} (k={summary_size})")
        
        results = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        evaluated = 0
        missing_summaries = 0
        
        for entity_id in entity_ids:
            # Load all ground truths (multiple annotators) for this summary size
            ground_truths = self.get_ground_truth_all_annotators(
                dataset_name, entity_id, summary_size
            )
            predicted = self.get_baseline_summary(dataset_name, entity_id, summary_size)
            
            # Skip only when the entity has no usable reference data.
            # A missing prediction should be counted as an empty summary so the
            # aggregate metrics still reflect all dataset entities.
            if not ground_truths:
                logger.debug(f"Skipping entity {entity_id}: missing ground truth")
                continue

            if not predicted:
                missing_summaries += 1
            
            # Calculate metrics against all annotators
            metrics = self.calculate_metrics(predicted, ground_truths)
            
            # Accumulate scores
            f1_scores.append(metrics["f1"])
            precision_scores.append(metrics["precision"])
            recall_scores.append(metrics["recall"])
            evaluated += 1
            
            results.append({
                "entity_id": entity_id,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "num_annotators": metrics["num_annotators"],
                "f1_per_annotator": ";".join(f"{f:.4f}" for f in metrics["f1_per_annotator"]),
                "pred_size": len(predicted),
                "has_prediction": bool(predicted),
            })
        
        # Calculate aggregated metrics - average across entities
        if evaluated > 0:
            agg_metrics = {
                "dataset": dataset_name,
                "summary_size": summary_size,
                "entities_evaluated": evaluated,
                "missing_summaries": missing_summaries,
                "precision": sum(precision_scores) / len(precision_scores),
                "recall": sum(recall_scores) / len(recall_scores),
                "f1": sum(f1_scores) / len(f1_scores),
            }
        else:
            agg_metrics = {}
        
        return agg_metrics, results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Baseline Summaries: Precision, Recall, F1-Score"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dbpedia",
        choices=["dbpedia", "lmdb", "faces"],
        help="Dataset to evaluate (default: dbpedia)"
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Evaluate all datasets (dbpedia, lmdb, faces)"
    )
    parser.add_argument(
        "--summary-size",
        type=int,
        default=5,
        help="Summary size k (default: 5)"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="../datasets/ESBM_benchmark_v1.2",
        help="Root directory of datasets"
    )
    parser.add_argument(
        "--baseline-output-dir",
        type=str,
        default="baseline_outputs",
        help="Directory containing baseline outputs"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="CSV file to save results"
    )
    parser.add_argument(
        "--detailed-csv",
        type=str,
        default=None,
        help="CSV file to save per-entity detailed results"
    )
    
    args = parser.parse_args()
    
    # Determine datasets to evaluate
    datasets_to_eval = ["dbpedia", "lmdb", "faces"] if args.all_datasets else [args.dataset]
    
    # Initialize evaluator
    evaluator = BaselineEvaluator(
        dataset_root=args.dataset_root,
        baseline_output_dir=args.baseline_output_dir,
    )
    
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION: Precision, Recall, F1-Score")
    print("=" * 80 + "\n")
    
    all_results = []
    all_detailed = []
    
    for dataset_name in datasets_to_eval:
        print(f"Evaluating {dataset_name.upper()} (k={args.summary_size})...")
        
        agg_metrics, detailed_results = evaluator.evaluate_dataset(
            dataset_name=dataset_name,
            summary_size=args.summary_size,
        )
        
        if agg_metrics:
            all_results.append(agg_metrics)
            all_detailed.extend(detailed_results)
            
            # Print results
            print(f"\n{dataset_name.upper()} Results:")
            print(f"  Entities Evaluated: {agg_metrics['entities_evaluated']}")
            if 'missing_summaries' in agg_metrics:
                print(f"  Missing Summaries: {agg_metrics['missing_summaries']}")
            print(f"  Precision: {agg_metrics['precision']:.4f}")
            print(f"  Recall:    {agg_metrics['recall']:.4f}")
            print(f"  F1-Score:  {agg_metrics['f1']:.4f}")
        else:
            print(f"  No results for {dataset_name}")
    
    # Calculate overall metrics
    if all_results:
        overall_precision = sum(r["precision"] for r in all_results) / len(all_results)
        overall_recall = sum(r["recall"] for r in all_results) / len(all_results)
        overall_f1 = sum(r["f1"] for r in all_results) / len(all_results)
        
        print("\n" + "-" * 80)
        print("OVERALL RESULTS (All Datasets Combined):")
        print(f"  Total Entities: {sum(r['entities_evaluated'] for r in all_results)}")
        print(f"  Precision: {overall_precision:.4f}")
        print(f"  Recall:    {overall_recall:.4f}")
        print(f"  F1-Score:  {overall_f1:.4f}")
        print("=" * 80 + "\n")
    
    # Save CSV results
    if args.output_csv and all_results:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["dataset", "summary_size", "entities_evaluated", "missing_summaries", "precision", "recall", "f1"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Saved aggregated results to: {args.output_csv}")
    
    # Save detailed CSV results
    if args.detailed_csv and all_detailed:
        os.makedirs(os.path.dirname(os.path.abspath(args.detailed_csv)) or ".", exist_ok=True)
        with open(args.detailed_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["entity_id", "precision", "recall", "f1", "num_annotators", "f1_per_annotator", "pred_size", "has_prediction"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detailed)
        print(f"Saved detailed results to: {args.detailed_csv}")


if __name__ == "__main__":
    main()
