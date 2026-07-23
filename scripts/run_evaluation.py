#!/usr/bin/env python3
# run_evaluation.py
import argparse
import csv
import os
import sys
from datetime import datetime

def ensure_rdflib_nt_parser():
    """
    Make rdflib >=6/<7 compatible by aliasing W3CNTriplesParser -> NTriplesParser
    when the latter is missing. Safe no-op on older versions.
    """
    try:
        import importlib
        nt = importlib.import_module('rdflib.plugins.parsers.ntriples')
        if not hasattr(nt, 'NTriplesParser') and hasattr(nt, 'W3CNTriplesParser'):
            nt.NTriplesParser = nt.W3CNTriplesParser  # alias
    except Exception as e:
        # Don't block execution; ESBenchmark may not need the symbol if it uses Graph.parse
        print(f"[warn] rdflib ntriples shim not applied: {e}", file=sys.stderr)

# Apply the shim BEFORE importing the project modules that import NTriplesParser
ensure_rdflib_nt_parser()

# import your evaluation function
try:
    from evaluation import evaluation
except Exception as e:
    print(f"Failed to import evaluation.py: {e}", file=sys.stderr)
    sys.exit(1)

# ESBenchmark is expected by evaluation(); import from your project
try:
    from classes.dataset import ESBenchmark
except Exception as e:
    print("Could not import ESBenchmark from classes.dataset.\n"
          "Make sure your project has classes/dataset.py with ESBenchmark.\n"
          f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

def coerce_ds_list(raw):
    # comma-separated list, strip whitespace
    return [x.strip().lower() for x in raw.split(",") if x.strip()]

def coerce_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

class Tee:
    """Write to stdout and an optional file simultaneously."""
    def __init__(self, fileobj=None):
        self.fileobj = fileobj
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        if self.fileobj:
            self.fileobj.write(data)
    def flush(self):
        self.stdout.flush()
        if self.fileobj:
            self.fileobj.flush()

def main():
    parser = argparse.ArgumentParser(
        description="Run ESBM-style evaluation for ranked and top-k triples."
    )
    parser.add_argument(
        "--datasets",
        type=coerce_ds_list,
        default="dbpedia,faces,lmdb",
        help="Comma-separated dataset names (e.g., dbpedia,faces,lmdb)."
    )
    parser.add_argument(
        "--topk",
        type=coerce_int_list,
        default="5,10",
        help="Comma-separated list of k values (e.g., 5,10,15)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name used to locate out/<model>/... directories."
    )
    parser.add_argument(
        "--file-n",
        type=int,
        default=6,
        help="file_n passed to ESBenchmark (default: 6)."
    )
    parser.add_argument(
        "--use-gold-split",
        action="store_true",
        help="Pass True for the 'gold split' flag of ESBenchmark (4th ctor arg)."
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Optional path to save aggregated results as CSV."
    )
    args = parser.parse_args()

    print("Evaluation in progress ...")
    rows = []
    for ds_name in args.datasets:
        for k in args.topk:
            # ESBenchmark(ds_name, file_n, topk, split_flag)
            dataset = ESBenchmark(ds_name, args.file_n, k, args.use_gold_split)
            metrics = evaluation(dataset, k, args.model_name)
            metrics["model"] = args.model_name
            metrics["timestamp"] = datetime.now().isoformat(timespec="seconds")
            rows.append(metrics)

    print("Evaluation is done ...")

    # Write a CSV index of what was evaluated, including completeness stats.
    if args.results_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)
        with open(args.results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved run index to {args.results_csv}")

if __name__ == "__main__":
    main()
