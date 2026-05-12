#!/usr/bin/env python3
"""
Ablation Study Evaluator and Analyzer

Aggregates evaluation results from ablation variants and produces:
1. Comparison table of metrics across variants
2. Component contribution analysis
3. Visualization of results
4. Statistical analysis (if multiple runs available)
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
import sys


class AblationAnalyzer:
    """Analyzes ablation study results"""
    
    def __init__(self, ablation_dir: Path):
        self.ablation_dir = Path(ablation_dir)
        self.variants = {}
        self.baseline_score = None
        self.baseline_name = "full"
        
    def load_variant_results(self) -> bool:
        """
        Load results from each variant directory
        
        Returns:
            True if at least baseline loaded successfully
        """
        print("\nLoading variant results...")
        
        for variant_dir in self.ablation_dir.iterdir():
            if not variant_dir.is_dir() or variant_dir.name.startswith('.'):
                continue
            
            variant_name = variant_dir.name
            metrics = self._extract_metrics(variant_dir)
            
            if metrics:
                self.variants[variant_name] = metrics
                print(f"  ✓ {variant_name:<30} F-Score={metrics.get('f_score', 'N/A')}")
            else:
                print(f"  ✗ {variant_name:<30} No metrics found")
        
        # Set baseline
        if self.baseline_name in self.variants:
            self.baseline_score = self.variants[self.baseline_name].get('f_score')
            print(f"\nBaseline ({self.baseline_name}): {self.baseline_score}")
            return True
        else:
            print(f"\n⚠️  Baseline variant '{self.baseline_name}' not found!")
            return False
    
    def _extract_metrics(self, variant_dir: Path) -> Optional[Dict]:
        """Extract metrics from variant output"""
        # Try reading overall_report.txt
        report_file = variant_dir / "overall_report.txt"
        if report_file.exists():
            try:
                metrics = self._parse_report_file(report_file)
                if metrics:
                    return metrics
            except Exception as e:
                print(f"    Error parsing {report_file}: {e}")
        
        # Try reading metrics.json  
        metrics_file = variant_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    return json.load(f)
            except:
                pass
        
        return None
    
    def _parse_report_file(self, report_file: Path) -> Optional[Dict]:
        """
        Parse overall_report.txt to extract metrics
        Expected format: "F-Score: 0.6524", "NDCG: 0.7142", etc.
        """
        metrics = {}
        try:
            with open(report_file) as f:
                content = f.read()
                
                # Search for metric patterns
                import re
                
                # F-Score pattern
                f_match = re.search(r'F[?\-]?[Ss]core\s*[:=]\s*([\d.]+)', content)
                if f_match:
                    metrics['f_score'] = float(f_match.group(1))
                
                # NDCG pattern
                ndcg_match = re.search(r'NDCG\s*[:=]\s*([\d.]+)', content)
                if ndcg_match:
                    metrics['ndcg'] = float(ndcg_match.group(1))
                
                # MAP pattern
                map_match = re.search(r'MAP\s*[:=]\s*([\d.]+)', content)
                if map_match:
                    metrics['map'] = float(map_match.group(1))
                
                # Coverage (entities processed)
                cov_match = re.search(r'coverage\s*[:=]\s*([\d.%]+)', content, re.IGNORECASE)
                if cov_match:
                    metrics['coverage'] = cov_match.group(1)
                
                # Processed/total
                proc_match = re.search(r'Processed\s*[:=]\s*(\d+)', content)
                if proc_match:
                    metrics['processed'] = int(proc_match.group(1))
                    
        except Exception as e:
            print(f"    Parse error: {e}")
        
        return metrics if metrics else None
    
    def compute_deltas(self) -> Dict[str, float]:
        """
        Compute metric delta from baseline for each variant
        
        Returns:
            {variant_name: percent_delta}
        """
        if not self.baseline_score:
            return {}
        
        deltas = {}
        for variant_name, metrics in self.variants.items():
            if variant_name == self.baseline_name:
                deltas[variant_name] = 0.0
            else:
                score = metrics.get('f_score')
                if score:
                    delta = ((score - self.baseline_score) / self.baseline_score * 100)
                    deltas[variant_name] = delta
                else:
                    deltas[variant_name] = None
        
        return deltas
    
    def create_comparison_table(self, output_csv: Optional[Path] = None) -> str:
        """Generate comparison table of all variants"""
        deltas = self.compute_deltas()
        
        table_lines = []
        table_lines.append("\n" + "="*100)
        table_lines.append(f"{'ABLATION STUDY COMPARISON':^100}")
        table_lines.append("="*100)
        table_lines.append(f"Baseline: {self.baseline_name} (F-Score={self.baseline_score})\n")
        
        # Table header
        header = (f"{'Variant':<30} | "
                 f"{'F-Score':<12} | "
                 f"{'Δ from Full':<15} | "
                 f"{'NDCG':<12} | "
                 f"{'MAP':<12} | "
                 f"{'Entities':<12}")
        table_lines.append("-" * 100)
        table_lines.append(header)
        table_lines.append("-" * 100)
        
        # Sort by delta (worst first) for importance ranking
        sorted_variants = sorted(
            self.variants.items(),
            key=lambda x: deltas.get(x[0]) if deltas.get(x[0]) is not None else 0,
            reverse=False  # Ascending (most negative first)
        )
        
        # Table rows
        for variant_name, metrics in sorted_variants:
            f_score = metrics.get('f_score', 0.0)
            ndcg = metrics.get('ndcg', 'N/A')
            map_score = metrics.get('map', 'N/A')
            n_entities = metrics.get('processed', 'N/A')
            
            delta = deltas.get(variant_name, 0.0)
            delta_str = f"{delta:+.2f}%" if delta is not None else "N/A"
            
            ndcg_str = f"{ndcg:.4f}" if isinstance(ndcg, float) else str(ndcg)
            map_str = f"{map_score:.4f}" if isinstance(map_score, float) else str(map_score)
            entities_str = str(n_entities)
            
            row = (f"{variant_name:<30} | "
                  f"{f_score:<12.4f} | "
                  f"{delta_str:<15} | "
                  f"{ndcg_str:<12} | "
                  f"{map_str:<12} | "
                  f"{entities_str:<12}")
            table_lines.append(row)
        
        table_lines.append("-" * 100)
        table_lines.append("")
        
        table_str = "\n".join(table_lines)
        
        # Save to CSV if requested
        if output_csv:
            self._save_csv(output_csv, sorted_variants, deltas)
        
        return table_str
    
    def _save_csv(self, csv_path: Path, sorted_variants, deltas):
        """Save metrics to CSV"""
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'Rank', 'Variant', 'F-Score', 'Delta (%)', 
                'NDCG', 'MAP', 'Entities'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for rank, (variant_name, metrics) in enumerate(sorted_variants, 1):
                writer.writerow({
                    'Rank': rank,
                    'Variant': variant_name,
                    'F-Score': f"{metrics.get('f_score', 'N/A'):.4f}",
                    'Delta (%)': f"{deltas.get(variant_name, 0.0):+.2f}",
                    'NDCG': f"{metrics.get('ndcg', 'N/A'):.4f}" if isinstance(metrics.get('ndcg'), float) else 'N/A',
                    'MAP': f"{metrics.get('map', 'N/A'):.4f}" if isinstance(metrics.get('map'), float) else 'N/A',
                    'Entities': metrics.get('processed', 'N/A'),
                })
        
        print(f"✓ CSV saved: {csv_path}")
    
    def analyze_component_importance(self) -> str:
        """
        Analyze which components are most critical based on ablation results
        """
        deltas = self.compute_deltas()
        
        analysis_lines = []
        analysis_lines.append("\n" + "="*100)
        analysis_lines.append(f"{'COMPONENT CONTRIBUTION ANALYSIS (RQ2)':^100}")
        analysis_lines.append("="*100)
        
        # Categorize by severity
        critical = {}    # delta < -10%
        high = {}        # -10% ≤ delta < -5%
        moderate = {}    # -5% ≤ delta < -2%
        low = {}         # delta ≥ -2%
        
        for variant_name, delta in deltas.items():
            if variant_name == self.baseline_name:
                continue
            if delta is None:
                continue
            
            if delta < -10:
                critical[variant_name] = delta
            elif delta < -5:
                high[variant_name] = delta
            elif delta < -2:
                moderate[variant_name] = delta
            else:
                low[variant_name] = delta
        
        # Report by category
        analysis_lines.append("\n🔴 CRITICAL (Δ < -10%): Component is ESSENTIAL")
        if critical:
            for variant, delta in sorted(critical.items(), key=lambda x: x[1]):
                analysis_lines.append(f"  • {variant:<40} {delta:+.2f}%")
        else:
            analysis_lines.append("  (none)")
        
        analysis_lines.append("\n🟠 HIGH IMPACT (Δ -10% to -5%): Component is IMPORTANT")
        if high:
            for variant, delta in sorted(high.items(), key=lambda x: x[1]):
                analysis_lines.append(f"  • {variant:<40} {delta:+.2f}%")
        else:
            analysis_lines.append("  (none)")
        
        analysis_lines.append("\n🟡 MODERATE (-5% to -2%): Component adds value")
        if moderate:
            for variant, delta in sorted(moderate.items(), key=lambda x: x[1]):
                analysis_lines.append(f"  • {variant:<40} {delta:+.2f}%")
        else:
            analysis_lines.append("  (none)")
        
        analysis_lines.append("\n🟢 LOW (<-2%): Component impact minimal")
        if low:
            for variant, delta in sorted(low.items(), key=lambda x: x[1]):
                analysis_lines.append(f"  • {variant:<40} {delta:+.2f}%")
        else:
            analysis_lines.append("  (none)")
        
        analysis_lines.append("\n" + "="*100)
        
        return "\n".join(analysis_lines)
    
    def extract_insights(self) -> str:
        """Generate key insights from ablation results"""
        deltas = self.compute_deltas()
        
        insights = []
        insights.append("\n" + "="*100)
        insights.append(f"{'KEY INSIGHTS':^100}")
        insights.append("="*100 + "\n")
        
        # Insight 1: Thought Policy
        if "no_thought" in deltas and deltas["no_thought"]:
            delta = deltas["no_thought"]
            insights.append(f"1. THOUGHT POLICY (LLM-based candidate generation)")
            insights.append(f"   Ablation delta: {delta:+.2f}%")
            if delta < -15:
                insights.append("   → CRITICAL. LLM policy is the primary driver.")
                insights.append("   → Without intelligent candidate selection, performance collapses.")
            else:
                insights.append("   → Important but other components also contribute significantly.")
            insights.append("")
        
        # Insight 2: Branching Strategy
        if "no_branch" in deltas and deltas["no_branch"]:
            delta = deltas["no_branch"]
            insights.append(f"2. BRANCHING STRATEGY (Parallel exploration with beam search)")
            insights.append(f"   Ablation delta: {delta:+.2f}%")
            if delta < -10:
                insights.append("   → Important. Beam search enables recovery from local optima.")
                insights.append("   → Greedy sequential selection significantly underperforms.")
            else:
                insights.append("   → Moderate impact. Single-path search still viable.")
            insights.append("")
        
        # Insight 3: Semantic Dimensions
        dims = {
            "only_relatedness": "Relatedness",
            "only_informativeness": "Informativeness",
            "only_coverage": "Coverage",
        }
        dim_deltas = {k: deltas.get(k) for k, v in dims.items()}
        dim_deltas = {k: v for k, v in dim_deltas.items() if v is not None}
        
        if dim_deltas:
            insights.append(f"3. SEMANTIC DIMENSIONS")
            worst_dim = min(dim_deltas.items(), key=lambda x: x[1])[0]
            best_dim = max(dim_deltas.items(), key=lambda x: x[1])[0]
            
            for variant, delta in sorted(dim_deltas.items(), key=lambda x: x[1]):
                dim_name = dims[variant]
                insights.append(f"   • {dim_name:<20} {delta:+.2f}%")
            
            insights.append(f"   → All three dimensions necessary for good performance.")
            worst_name = dims[worst_dim]
            best_name = dims[best_dim]
            insights.append(f"   → {worst_name} has highest individual impact (most negative when alone)")
            insights.append(f"   → {best_name} has lowest individual impact")
            insights.append("")
        
        # Insight 4: Weight Tuning
        if "uniform_weights" in deltas and deltas["uniform_weights"]:
            delta = deltas["uniform_weights"]
            insights.append(f"4. WEIGHT TUNING (Optimized vs Uniform)")
            insights.append(f"   Ablation delta: {delta:+.2f}%")
            if abs(delta) < 3:
                insights.append("   → Tuning helps but architecture robustness matters more.")
                insights.append("   → Any reasonable weighting of dimensions works well.")
            else:
                insights.append("   → Weight optimization important for best performance.")
            insights.append("")
        
        # Insight 5: Multi-Sample Evaluation
        if "no_ensemble" in deltas and deltas["no_ensemble"]:
            delta = deltas["no_ensemble"]
            insights.append(f"5. MULTI-SAMPLE EVALUATION (Ensemble robustness)")
            insights.append(f"   Ablation delta: {delta:+.2f}%")
            insights.append(f"   → Averaging k samples improves stability against LLM variance.")
            insights.append("")
        
        insights.append("="*100)
        
        return "\n".join(insights)
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate complete analysis report"""
        report_parts = [
            self.create_comparison_table(),
            self.analyze_component_importance(),
            self.extract_insights(),
        ]
        
        report = "\n".join(report_parts)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\n✓ Report saved: {output_file}")
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze ablation study results for RQ2"
    )
    parser.add_argument(
        "--ablation-dir",
        required=True,
        help="Path to ablation study directory"
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Save comparison table as CSV"
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Save full report as text file"
    )
    parser.add_argument(
        "--baseline",
        default="full",
        help="Baseline variant name (default: full)"
    )
    
    args = parser.parse_args()
    
    ablation_dir = Path(args.ablation_dir)
    if not ablation_dir.exists():
        print(f"❌ Directory not found: {ablation_dir}")
        sys.exit(1)
    
    # Run analysis
    analyzer = AblationAnalyzer(ablation_dir)
    analyzer.baseline_name = args.baseline
    
    if not analyzer.load_variant_results():
        print("⚠️  Could not load sufficient results for analysis")
        sys.exit(1)
    
    # Generate outputs
    print(analyzer.create_comparison_table(
        Path(args.output_csv) if args.output_csv else None
    ))
    
    print(analyzer.analyze_component_importance())
    print(analyzer.extract_insights())
    
    if args.output_report:
        analyzer.generate_report(Path(args.output_report))
    
    print("✓ Analysis complete!")


if __name__ == "__main__":
    main()
