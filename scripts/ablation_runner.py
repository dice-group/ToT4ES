#!/usr/bin/env python3
"""
Ablation Study Runner for ToT4ES RQ2

Systematically evaluates ToT4ES variants to measure component contributions:
- Thought Policy (LLM-based candidate selection)
- Branching Strategy (beam search vs greedy)
- Semantic Dimensions (Relatedness, Informativeness, Coverage)
- Weight tuning and multi-sample evaluation

Usage:
    python ablation_runner.py --variants full no_thought only_relatedness --dataset wikicinema-s
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class AblationConfig:
    """Configuration for one ablation variant"""
    name: str
    display_name: str
    description: str
    config_overrides: Dict  # CLI argument overrides
    expected_delta: str     # Expected performance change


# Define all ablation variants
ABLATION_VARIANTS = {
    "full": AblationConfig(
        name="full",
        display_name="ToT4ES-Full",
        description="Baseline: Full ToT4ES with all components and optimized weights",
        config_overrides={
            "n_candidates_per_task": "3",
            "beam_width": "3",
            "n_evaluation_samples": "3",
            "w_relatedness": "0.4",
            "w_informativeness": "0.4",
            "w_coverage": "0.2",
        },
        expected_delta="+0.0% (Baseline)"
    ),
    
    "no_thought": AblationConfig(
        name="no_thought",
        display_name="ToT4ES-NoThought",
        description="Ablation: Random candidate selection instead of LLM policy",
        config_overrides={
            "beam_width": "3",
            "n_evaluation_samples": "3",
            "use_random_candidates": "true",
            "w_relatedness": "0.4",
            "w_informativeness": "0.4",
            "w_coverage": "0.2",
        },
        expected_delta="-15% to -25% (Validates policy importance)"
    ),
    
    "no_branch": AblationConfig(
        name="no_branch",
        display_name="ToT4ES-NoBranch",
        description="Ablation: Linear greedy search (beam_width=1) instead of branching",
        config_overrides={
            "beam_width": "1",
            "n_evaluation_samples": "3",
            "w_relatedness": "0.4",
            "w_informativeness": "0.4",
            "w_coverage": "0.2",
        },
        expected_delta="-10% to -15% (Validates branching importance)"
    ),
    
    "only_relatedness": AblationConfig(
        name="only_relatedness",
        display_name="ToT4ES-OnlyRel",
        description="Ablation: Value function uses only Relatedness (V=R, I=0, C=0)",
        config_overrides={
            "beam_width": "3",
            "n_evaluation_samples": "3",
            "w_relatedness": "1.0",
            "w_informativeness": "0.0",
            "w_coverage": "0.0",
        },
        expected_delta="-8% to -12% (Tests single dimension)"
    ),
    
    "only_informativeness": AblationConfig(
        name="only_informativeness",
        display_name="ToT4ES-OnlyInfo",
        description="Ablation: Value function uses only Informativeness (V=I, R=0, C=0)",
        config_overrides={
            "beam_width": "3",
            "n_evaluation_samples": "3",
            "w_relatedness": "0.0",
            "w_informativeness": "1.0",
            "w_coverage": "0.0",
        },
        expected_delta="-10% to -15% (Tests single dimension)"
    ),
    
    "only_coverage": AblationConfig(
        name="only_coverage",
        display_name="ToT4ES-OnlyCov",
        description="Ablation: Value function uses only Coverage (V=C, R=0, I=0)",
        config_overrides={
            "beam_width": "3",
            "n_evaluation_samples": "3",
            "w_relatedness": "0.0",
            "w_informativeness": "0.0",
            "w_coverage": "1.0",
        },
        expected_delta="-15% to -20% (Tests single dimension)"
    ),
    
    "rel_and_info": AblationConfig(
        name="rel_and_info",
        display_name="ToT4ES-RelInfo",
        description="Ablation: Only Relatedness + Informativeness (V=R+I, C=0)",
        config_overrides={
            "beam_width": "3",
            "n_evaluation_samples": "3",
            "w_relatedness": "0.5",
            "w_informativeness": "0.5",
            "w_coverage": "0.0",
        },
        expected_delta="-5% to -10% (Tests dimension interaction)"
    ),
    
    "uniform_weights": AblationConfig(
        name="uniform_weights",
        display_name="ToT4ES-Uniform",
        description="Ablation: Equal weights for all dimensions (w_r = w_i = w_c = 1/3)",
        config_overrides={
            "beam_width": "3",
            "n_evaluation_samples": "3",
            "w_relatedness": "0.333",
            "w_informativeness": "0.333",
            "w_coverage": "0.334",
        },
        expected_delta="-2% to -5% (Tests weight tuning)"
    ),
    
    "no_ensemble": AblationConfig(
        name="no_ensemble",
        display_name="ToT4ES-NoEnsemble",
        description="Ablation: Single evaluation sample (k=1) instead of k=3",
        config_overrides={
            "beam_width": "3",
            "n_evaluation_samples": "1",
            "w_relatedness": "0.4",
            "w_informativeness": "0.4",
            "w_coverage": "0.2",
        },
        expected_delta="-3% to -8% (Tests multi-sample robustness)"
    ),
}


class AblationRunner:
    """Orchestrates ablation study execution"""
    
    def __init__(self, output_base: Path, summarizer_script: str = "tot_entity_summarizer_semantic.py"):
        self.output_base = Path(output_base)
        self.summarizer_script = summarizer_script
        self.results = {}
        self.start_time = datetime.now()
        
    def run_variant(self, variant_name: str, dataset: str, limit_entities: int = 0) -> bool:
        """
        Run a single ablation variant
        
        Args:
            variant_name: Key in ABLATION_VARIANTS
            dataset: Dataset name (e.g., 'wikicinema-s')
            limit_entities: Max entities to process (0=all)
            
        Returns:
            True if successful
        """
        if variant_name not in ABLATION_VARIANTS:
            print(f"❌ Unknown variant: {variant_name}")
            return False
        
        config = ABLATION_VARIANTS[variant_name]
        variant_output = self.output_base / variant_name
        variant_output.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"  Variant: {config.display_name}")
        print(f"  Description: {config.description}")
        print(f"  Expected Delta: {config.expected_delta}")
        print(f"{'='*80}")
        
        # Build command
        cmd = [
            "python", self.summarizer_script,
            "--dataset", dataset,
            "--output-dir", str(variant_output),
            "--max-summary-len", "5",
        ]
        
        # Add config overrides
        for key, value in config.config_overrides.items():
            cmd.append(f"--{key}")
            cmd.append(value)
        
        # Add entity limit if specified
        if limit_entities > 0:
            cmd.append("--limit-entities")
            cmd.append(str(limit_entities))
        
        # Redirect stdout/stderr
        log_file = variant_output / "run.log"
        
        print(f"  Executing: {' '.join(cmd)}")
        print(f"  Logs: {log_file}")
        
        try:
            with open(log_file, 'w') as logf:
                result = subprocess.run(
                    cmd,
                    cwd=".",  # Adjust if needed
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=3600  # 1 hour timeout per variant
                )
            
            success = result.returncode == 0
            
            if success:
                print(f"  ✓ SUCCESS")
                self.results[variant_name] = {
                    "status": "completed",
                    "output_dir": str(variant_output),
                    "config": config.config_overrides,
                }
            else:
                print(f"  ✗ FAILED (exit code {result.returncode})")
                print(f"  Check log: {log_file}")
                self.results[variant_name] = {
                    "status": "failed",
                    "output_dir": str(variant_output),
                }
            
            return success
            
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT (exceeded 1 hour)")
            self.results[variant_name] = {
                "status": "timeout",
                "output_dir": str(variant_output),
            }
            return False
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            self.results[variant_name] = {
                "status": "error",
                "output_dir": str(variant_output),
                "error": str(e),
            }
            return False
    
    def save_results(self):
        """Save execution results"""
        results_file = self.output_base / "ablation_execution.json"
        
        execution_log = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "varieties_executed": len(self.results),
            "results": self.results,
            "variants_definitions": {
                name: {
                    "display_name": cfg.display_name,
                    "description": cfg.description,
                    "expected_delta": cfg.expected_delta,
                }
                for name, cfg in ABLATION_VARIANTS.items()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(execution_log, f, indent=2)
        
        print(f"\n✓ Execution log: {results_file}")
        return results_file
    
    def print_summary(self):
        """Print execution summary"""
        total = len(self.results)
        completed = sum(1 for r in self.results.values() if r["status"] == "completed")
        failed = sum(1 for r in self.results.values() if r["status"] == "failed")
        
        print(f"\n{'='*80}")
        print(f"{'ABLATION STUDY EXECUTION SUMMARY':^80}")
        print(f"{'='*80}")
        print(f"Total variants run:      {total}")
        print(f"Completed successfully:  {completed}")
        print(f"Failed:                  {failed}")
        print(f"Success rate:            {completed/total*100:.1f}%" if total > 0 else "N/A")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for ToT4ES (RQ2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Run all variants on wikicinema-s dataset
  python ablation_runner.py --dataset wikicinema-s

  # Run specific variants with entity limit (for quick testing)
  python ablation_runner.py --variants full no_thought only_relatedness \\
                           --dataset wikicinema-s --limit-entities 10

  # List available variants
  python ablation_runner.py --list-variants
        """
    )
    
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ABLATION_VARIANTS.keys()),
        help="Which variants to run (default: all)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., wikicinema-s, faces, esbm)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ablation_study",
        help="Base output directory for ablation results"
    )
    parser.add_argument(
        "--limit-entities",
        type=int,
        default=0,
        help="Limit entities per variant (0=all, useful for testing)"
    )
    parser.add_argument(
        "--summarizer",
        default="tot_entity_summarizer_semantic.py",
        help="Summarizer script to use"
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="Print available variants and exit"
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Continue if a variant fails"
    )
    
    args = parser.parse_args()
    
    # List variants and exit if requested
    if args.list_variants:
        print(f"\n{'Available Ablation Variants':^80}\n")
        print(f"{'Name':<25} {'Display Name':<25} {'Expected Delta':<20}")
        print("-" * 80)
        for name in sorted(ABLATION_VARIANTS.keys()):
            cfg = ABLATION_VARIANTS[name]
            print(f"{name:<25} {cfg.display_name:<25} {cfg.expected_delta:<20}")
        print()
        return
    
    # Validate variants
    invalid = [v for v in args.variants if v not in ABLATION_VARIANTS]
    if invalid:
        print(f"❌ Unknown variants: {invalid}")
        sys.exit(1)
    
    # Run ablation study
    print(f"\n{'='*80}")
    print(f"{'ToT4ES Ablation Study (RQ2)':^80}")
    print(f"{'='*80}")
    print(f"Dataset:              {args.dataset}")
    print(f"Variants to run:      {len(args.variants)}")
    print(f"Output directory:     {args.output_dir}")
    print(f"Entity limit:         {'All' if args.limit_entities == 0 else args.limit_entities}")
    print(f"{'='*80}\n")
    
    runner = AblationRunner(
        output_base=Path(args.output_dir),
        summarizer_script=args.summarizer
    )
    
    # Run each variant
    failed_variants = []
    for variant_name in args.variants:
        success = runner.run_variant(
            variant_name,
            args.dataset,
            args.limit_entities
        )
        if not success:
            failed_variants.append(variant_name)
            if not args.skip_failed:
                print(f"\n❌ Stopping on first failure. Use --skip-failed to continue.")
                break
    
    # Save results
    runner.save_results()
    runner.print_summary()
    
    if failed_variants:
        print(f"⚠️  Failed variants: {failed_variants}")
        print(f"Check logs in outputs/ablation_study/<variant>/run.log")
    
    sys.exit(0 if not failed_variants else 1)


if __name__ == "__main__":
    main()
