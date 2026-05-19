#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoT Baseline Evaluation: Calculate Precision, Recall, F1-Score

Evaluates CoT baseline summaries against ground truth triple files.

Usage:
    python evaluate_cot_baseline.py --dataset dbpedia --summary-size 5
    python evaluate_cot_baseline.py --baseline-output-dir /path/to/results --output-csv results.csv
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
import csv

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
        """Parse N-Triples line into (subject, predicate, object)."""
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None, None
        
        if line.endswith(' .'):
            line = line[:-2].strip()
        elif line.endswith('.'):
            line = line[:-1].strip()
        
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
        """Normalize a triple for comparison."""
        subject = subject.strip('<>').strip()
        predicate = predicate.strip('<>').strip()
        
        if obj.startswith('<'):
            obj = obj.strip('<>').strip()
        else:
            obj = obj.strip()
        
        return (subject, predicate, obj)
    
    @staticmethod
    def load_triples_from_file(filepath: str) -> Set[Tuple[str, str, str]]:
        """Load and parse all triples from an N-Triples file."""
        triples = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    s, p, o = NTriplesParser.parse_triple(line)
                    if s is not None and p is not None and o is not None:
                        norm = NTriplesParser.normalize_triple(s, p, o)
                        triples.add(norm)
        except Exception as e:
            logger.warning(f"Error loading triples from {filepath}: {e}")
        
        return triples


class CoTEvaluator:
    """Evaluate CoT baseline summaries against ground truth."""
    
    def __init__(
        self,
        dataset_root: str = "../datasets/ESBM_benchmark_v1.2",
        baseline_output_dir: str = "baseline_cot_outputs",
    ):
        """Initialize evaluator."""
        self.dataset_root = Path(dataset_root)
        self.baseline_output_dir = Path(baseline_output_dir)
    
    def get_entity_ids(self, dataset_name: str) -> List[int]:
        """Get all entity IDs for a dataset."""
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
        """Load all ground truth annotations for an entity."""
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
        
        for file_idx in range(max_files):
            gold_file = entity_dir / f"{entity_id}_gold_top{summary_size}_{file_idx}.nt"
            
            if gold_file.exists():
                gold_triples = NTriplesParser.load_triples_from_file(str(gold_file))
                if gold_triples:
                    gold_summaries.append(gold_triples)
                    logger.debug(f"Loaded {len(gold_triples)} triples from {gold_file}")
            else:
                if gold_summaries:
                    break
        
        if not gold_summaries:
            logger.warning(f"No gold summaries found for entity {entity_id} at k={summary_size}")
        
        return gold_summaries
    
    def get_cot_summary(self, dataset_name: str, entity_id: int, summary_size: int) -> Set[Tuple[str, str, str]]:
        """Load CoT summary for an entity."""
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
            logger.warning(f"CoT summary not found: {summary_file}")
            return set()
        
        return NTriplesParser.load_triples_from_file(str(summary_file))
    
    def calculate_metrics(
        self,
        predicted: Set[Tuple[str, str, str]],
        ground_truths: List[Set[Tuple[str, str, str]]],
    ) -> Dict[str, float]:
        """Calculate metrics against multiple ground truths."""
        if not predicted or not ground_truths:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "f1_per_annotator": [],
                "num_annotators": len(ground_truths),
            }
        
        f1_scores = []
        
        for gt_idx, ground_truth in enumerate(ground_truths):
            if not ground_truth:
                continue
            
            tp = len(predicted & ground_truth)
            fp = len(predicted - ground_truth)
            fn = len(ground_truth - predicted)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        # Overall precision/recall
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
        """Evaluate CoT baseline for all entities in a dataset."""
        entity_ids = self.get_entity_ids(dataset_name)
        
        if not entity_ids:
            logger.warning(f"No entities found for {dataset_name}")
            return {}, []
        
        logger.info(f"Evaluating {len(entity_ids)} entities from {dataset_name} (k={summary_size})")
        
        results = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        successful = 0
        
        for entity_id in entity_ids:
            ground_truths = self.get_ground_truth_all_annotators(
                dataset_name, entity_id, summary_size
            )
            predicted = self.get_cot_summary(dataset_name, entity_id, summary_size)
            
            if not ground_truths or not predicted:
                logger.debug(f"Skipping entity {entity_id}: missing data")
                continue
            
            metrics = self.calculate_metrics(predicted, ground_truths)
            
            f1_scores.append(metrics["f1"])
            precision_scores.append(metrics["precision"])
            recall_scores.append(metrics["recall"])
            successful += 1
            
            results.append({
                "entity_id": entity_id,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "num_annotators": metrics["num_annotators"],
                "f1_per_annotator": ";".join(f"{f:.4f}" for f in metrics["f1_per_annotator"]),
                "pred_size": len(predicted),
            })
        
        if successful > 0:
            agg_metrics = {
                "dataset": dataset_name,
                "summary_size": summary_size,
                "entities_evaluated": successful,
                "precision": sum(precision_scores) / len(precision_scores),
                "recall": sum(recall_scores) / len(recall_scores),
                "f1": sum(f1_scores) / len(f1_scores),
            }
        else:
            agg_metrics = {}
        
        return agg_metrics, results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoT Baseline Summaries"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dbpedia",
        choices=["dbpedia", "lmdb", "faces"],
        help="Dataset to evaluate"
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Evaluate all datasets"
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
        default="baseline_cot_outputs",
        help="Directory containing CoT outputs"
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
    
    datasets_to_eval = ["dbpedia", "lmdb", "faces"] if args.all_datasets else [args.dataset]
    
    evaluator = CoTEvaluator(
        dataset_root=args.dataset_root,
        baseline_output_dir=args.baseline_output_dir,
    )
    
    print("\n" + "=" * 80)
    print("COT BASELINE EVALUATION: Precision, Recall, F1-Score")
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
            
            print(f"\n{dataset_name.upper()} Results:")
            print(f"  Entities Evaluated: {agg_metrics['entities_evaluated']}")
            print(f"  Precision: {agg_metrics['precision']:.4f}")
            print(f"  Recall:    {agg_metrics['recall']:.4f}")
            print(f"  F1-Score:  {agg_metrics['f1']:.4f}")
        else:
            print(f"  No results for {dataset_name}")
    
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
    
    if args.output_csv and all_results:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["dataset", "summary_size", "entities_evaluated", "precision", "recall", "f1"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Saved aggregated results to: {args.output_csv}")
    
    if args.detailed_csv and all_detailed:
        os.makedirs(os.path.dirname(os.path.abspath(args.detailed_csv)) or ".", exist_ok=True)
        with open(args.detailed_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["entity_id", "precision", "recall", "f1", "num_annotators", "f1_per_annotator", "pred_size"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detailed)
        print(f"Saved detailed results to: {args.detailed_csv}")


if __name__ == "__main__":
    main()
