#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation utilities for comparing Baseline vs ToT4ES summaries.

This module provides metrics to compare the quality of summaries produced by:
1. Baseline: Direct LLM prompt
2. ToT4ES: Tree-of-thought approach
"""

import os
import logging
from typing import List, Tuple, Set, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


class TripleNormalizer:
    """Normalize triples for comparison."""
    
    @staticmethod
    def parse_triple(line: str) -> Tuple[str, str, str]:
        """Parse an N-Triple line."""
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None, None
        
        # Remove trailing period
        if line.endswith(' .'):
            line = line[:-2].strip()
        elif line.endswith('.'):
            line = line[:-1].strip()
        
        # Simple split (works for most N-Triples)
        parts = line.split(None, 2)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        
        return None, None, None
    
    @staticmethod
    def normalize_predicate(pred: str) -> str:
        """Normalize predicate for comparison (extract local name)."""
        # Remove angle brackets
        pred = pred.strip('<>')
        # Get the last part (local name)
        return pred.split('/')[-1].split('#')[-1]
    
    @staticmethod
    def normalize_object(obj: str) -> str:
        """Normalize object for comparison."""
        # Remove quotes and angle brackets
        obj = obj.strip('<>"')
        # Get the last part if it's a URI
        if obj.startswith('http'):
            return obj.split('/')[-1]
        return obj


class SummaryComparer:
    """Compare two entity summaries."""
    
    def __init__(self):
        self.normalizer = TripleNormalizer()
    
    def load_summary(self, summary_file: str) -> Set[Tuple[str, str, str]]:
        """
        Load a summary file and return set of (subject, predicate, object) tuples.
        
        Args:
            summary_file: Path to N-Triples file
            
        Returns:
            Set of normalized (s, p, o) tuples
        """
        triples = set()
        
        if not os.path.exists(summary_file):
            logger.warning(f"Summary file not found: {summary_file}")
            return triples
        
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                for line in f:
                    s, p, o = self.normalizer.parse_triple(line)
                    if s is not None:
                        # Normalize for comparison
                        p_norm = self.normalizer.normalize_predicate(p)
                        o_norm = self.normalizer.normalize_object(o)
                        triples.add((s, p_norm, o_norm))
        except Exception as e:
            logger.error(f"Error loading summary from {summary_file}: {e}")
        
        return triples
    
    def predicate_coverage(
        self,
        baseline_summary: Set[Tuple[str, str, str]],
        tot4es_summary: Set[Tuple[str, str, str]],
    ) -> Dict[str, int]:
        """
        Calculate predicate coverage in each summary.
        
        Args:
            baseline_summary: Baseline summary triples
            tot4es_summary: ToT4ES summary triples
            
        Returns:
            Dict with coverage statistics
        """
        baseline_preds = defaultdict(int)
        tot4es_preds = defaultdict(int)
        
        for _, p, _ in baseline_summary:
            baseline_preds[p] += 1
        
        for _, p, _ in tot4es_summary:
            tot4es_preds[p] += 1
        
        return {
            "baseline_predicate_count": len(baseline_preds),
            "tot4es_predicate_count": len(tot4es_preds),
            "baseline_predicates": dict(baseline_preds),
            "tot4es_predicates": dict(tot4es_preds),
            "unique_to_baseline": list(set(baseline_preds.keys()) - set(tot4es_preds.keys())),
            "unique_to_tot4es": list(set(tot4es_preds.keys()) - set(baseline_preds.keys())),
            "shared_predicates": list(set(baseline_preds.keys()) & set(tot4es_preds.keys())),
        }
    
    def overlap_metrics(
        self,
        baseline_summary: Set[Tuple[str, str, str]],
        tot4es_summary: Set[Tuple[str, str, str]],
    ) -> Dict[str, float]:
        """
        Calculate overlap and similarity metrics.
        
        Args:
            baseline_summary: Baseline summary triples
            tot4es_summary: ToT4ES summary triples
            
        Returns:
            Dict with metrics
        """
        baseline_size = len(baseline_summary)
        tot4es_size = len(tot4es_summary)
        
        # Exact overlap
        exact_overlap = baseline_summary & tot4es_summary
        
        # Predicate-only overlap (same predicates regardless of values)
        baseline_preds = {(s, p) for s, p, o in baseline_summary}
        tot4es_preds = {(s, p) for s, p, o in tot4es_summary}
        pred_overlap = baseline_preds & tot4es_preds
        
        # Calculate metrics
        exact_overlap_count = len(exact_overlap)
        pred_overlap_count = len(pred_overlap)
        
        metrics = {
            "baseline_size": baseline_size,
            "tot4es_size": tot4es_size,
            "exact_overlap_count": exact_overlap_count,
            "predicate_overlap_count": pred_overlap_count,
            "jaccard_exact": (
                exact_overlap_count / len(baseline_summary | tot4es_summary)
                if (baseline_summary | tot4es_summary) else 0.0
            ),
            "jaccard_predicate": (
                pred_overlap_count / (
                    len(baseline_preds) + len(tot4es_preds) - pred_overlap_count
                )
                if (len(baseline_preds) + len(tot4es_preds) - pred_overlap_count) > 0
                else 0.0
            ),
            "baseline_exact_recall": (
                exact_overlap_count / tot4es_size
                if tot4es_size > 0 else 0.0
            ),
            "baseline_exact_precision": (
                exact_overlap_count / baseline_size
                if baseline_size > 0 else 0.0
            ),
        }
        
        return metrics
    
    def diversity_metrics(
        self,
        summary: Set[Tuple[str, str, str]],
    ) -> Dict[str, any]:
        """
        Calculate diversity metrics for a summary.
        
        Args:
            summary: Summary triples
            
        Returns:
            Dict with diversity metrics
        """
        if not summary:
            return {"predicate_diversity": 0.0, "unique_predicates": 0}
        
        predicates = [p for _, p, _ in summary]
        unique_predicates = set(predicates)
        
        # Diversity = number of unique predicates / total triples
        diversity = len(unique_predicates) / len(summary) if summary else 0.0
        
        return {
            "total_triples": len(summary),
            "unique_predicates": len(unique_predicates),
            "predicate_diversity": diversity,
            "predicates": sorted(unique_predicates),
        }
    
    def compare(
        self,
        baseline_file: str,
        tot4es_file: str,
        output_file: str = None,
    ) -> Dict[str, any]:
        """
        Comprehensive comparison of two summaries.
        
        Args:
            baseline_file: Path to baseline summary
            tot4es_file: Path to ToT4ES summary
            output_file: Optional file to save comparison report
            
        Returns:
            Dict with all comparison metrics
        """
        # Load summaries
        baseline = self.load_summary(baseline_file)
        tot4es = self.load_summary(tot4es_file)
        
        logger.info(f"Baseline: {len(baseline)} triples")
        logger.info(f"ToT4ES: {len(tot4es)} triples")
        
        # Calculate metrics
        comparison = {
            "files": {
                "baseline": baseline_file,
                "tot4es": tot4es_file,
            },
            "overlap": self.overlap_metrics(baseline, tot4es),
            "predicate_coverage": self.predicate_coverage(baseline, tot4es),
            "baseline_diversity": self.diversity_metrics(baseline),
            "tot4es_diversity": self.diversity_metrics(tot4es),
        }
        
        # Optional: save report
        if output_file:
            self._save_report(comparison, output_file)
        
        return comparison
    
    def _save_report(self, comparison: Dict, output_file: str):
        """Save comparison report to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BASELINE vs ToT4ES COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary sizes
            f.write("SUMMARY SIZES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Baseline: {comparison['overlap']['baseline_size']} triples\n")
            f.write(f"ToT4ES: {comparison['overlap']['tot4es_size']} triples\n\n")
            
            # Overlap metrics
            f.write("OVERLAP METRICS\n")
            f.write("-" * 40 + "\n")
            overlap = comparison['overlap']
            f.write(f"Exact Overlap: {overlap['exact_overlap_count']} triples\n")
            f.write(f"Predicate Overlap: {overlap['predicate_overlap_count']} triples\n")
            f.write(f"Jaccard (Exact): {overlap['jaccard_exact']:.3f}\n")
            f.write(f"Jaccard (Predicate): {overlap['jaccard_predicate']:.3f}\n")
            f.write(f"Baseline Precision: {overlap['baseline_exact_precision']:.3f}\n")
            f.write(f"Baseline Recall: {overlap['baseline_exact_recall']:.3f}\n\n")
            
            # Diversity
            f.write("DIVERSITY METRICS\n")
            f.write("-" * 40 + "\n")
            bl_div = comparison['baseline_diversity']
            tot_div = comparison['tot4es_diversity']
            f.write(f"Baseline Predicate Diversity: {bl_div['predicate_diversity']:.3f} ")
            f.write(f"({bl_div['unique_predicates']} unique)\n")
            f.write(f"ToT4ES Predicate Diversity: {tot_div['predicate_diversity']:.3f} ")
            f.write(f"({tot_div['unique_predicates']} unique)\n\n")
            
            # Predicate coverage
            f.write("PREDICATE COVERAGE\n")
            f.write("-" * 40 + "\n")
            pred_cov = comparison['predicate_coverage']
            f.write(f"Baseline unique predicates: {pred_cov['baseline_predicate_count']}\n")
            f.write(f"ToT4ES unique predicates: {pred_cov['tot4es_predicate_count']}\n")
            f.write(f"Shared predicates: {len(pred_cov['shared_predicates'])}\n")
            f.write(f"Unique to Baseline: {pred_cov['unique_to_baseline']}\n")
            f.write(f"Unique to ToT4ES: {pred_cov['unique_to_tot4es']}\n")
        
        logger.info(f"Report saved to {output_file}")


def main():
    """CLI for comparing summaries."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare Baseline vs ToT4ES entity summaries"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline summary file"
    )
    parser.add_argument(
        "--tot4es",
        type=str,
        required=True,
        help="Path to ToT4ES summary file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for comparison report"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    comparer = SummaryComparer()
    comparison = comparer.compare(
        args.baseline,
        args.tot4es,
        args.output
    )
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80 + "\n")
    print(f"Baseline size: {comparison['overlap']['baseline_size']}")
    print(f"ToT4ES size: {comparison['overlap']['tot4es_size']}")
    print(f"Exact overlap: {comparison['overlap']['exact_overlap_count']}")
    print(f"Jaccard similarity: {comparison['overlap']['jaccard_exact']:.3f}")
    print(f"Baseline diversity: {comparison['baseline_diversity']['predicate_diversity']:.3f}")
    print(f"ToT4ES diversity: {comparison['tot4es_diversity']['predicate_diversity']:.3f}")


if __name__ == "__main__":
    main()
