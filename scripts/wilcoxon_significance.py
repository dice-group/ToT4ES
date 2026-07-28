#!/usr/bin/env python3
"""Wilcoxon signed-rank significance tests for final-comparison outputs.

This script compares ToT4ES against other methods using paired per-entity
F-measure scores computed with the same triple-encoding pipeline used by the
repository evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


DATASET_GOLD_PATHS = {
    "dbpedia": os.path.join("datasets", "ESBM_benchmark_v1.2", "dbpedia_data"),
    "lmdb": os.path.join("datasets", "ESBM_benchmark_v1.2", "lmdb_data"),
    "faces": os.path.join("datasets", "FACES", "faces_data"),
}


TRIPLE_RE = re.compile(r'^(?P<s><[^>]+>)\s+(?P<p><[^>]+>)\s+(?P<o>.+?)(?:\s*\.)?$')
WS_RE = re.compile(r"\s+")


def collapse_ws(text: str) -> str:
    return WS_RE.sub(" ", text.strip())


def split_triple(line: str) -> Tuple[str | None, str | None, str | None]:
    match = TRIPLE_RE.match(line.strip())
    if not match:
        return None, None, None
    return match.group("s"), match.group("p"), match.group("o")


def literal_variants(obj_raw: str) -> List[str]:
    value = obj_raw.strip()
    if not value.startswith('"'):
        return [value]

    core_end = None
    escaped = False
    for idx, char in enumerate(value[1:], start=1):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == '"' and not escaped:
            core_end = idx
            break
        escaped = False

    if core_end is None:
        return [value]

    core = value[: core_end + 1]
    rest = value[core_end + 1 :].strip()
    out = [value, core]

    if rest.startswith("@"):
        m = re.match(r"^@[a-zA-Z]+(?:-[a-zA-Z0-9]+)?(.*)$", rest)
        rest_wo_lang = m.group(1).strip() if m else rest
        out.append(collapse_ws(f"{core} {rest_wo_lang}".strip()))

    if "^^<" in rest:
        before, _, _ = rest.partition("^^<")
        out.append(collapse_ws(f"{core} {before.strip()}".strip()))

    dedup: List[str] = []
    seen = set()
    for item in out:
        item = item.strip()
        if item and item not in seen:
            dedup.append(item)
            seen.add(item)
    return dedup


def triple_variants(line: str) -> List[str]:
    subj, pred, obj = split_triple(line)
    if not subj:
        base = line.strip()
        if base.endswith("."):
            base = base[:-1].strip()
        return [collapse_ws(base)]

    variants = [collapse_ws(f"{subj} {pred} {obj_variant}") for obj_variant in literal_variants(obj)]
    base = line.strip()
    if base.endswith("."):
        base = base[:-1]
    variants.append(collapse_ws(base))

    dedup: List[str] = []
    seen = set()
    for item in variants:
        if item not in seen:
            dedup.append(item)
            seen.add(item)
    return dedup


def load_gold_and_dict(gold_db_path: str, entity_id: int, top_k: int, file_n: int) -> Tuple[List[List[int]], Dict[str, int]]:
    entity_dir = os.path.join(gold_db_path, str(entity_id))
    desc_path = os.path.join(entity_dir, f"{entity_id}_desc.nt")

    triples_dict: Dict[str, int] = {}
    with open(desc_path, "r", encoding="utf-8") as reader:
        for raw in reader:
            triple = raw.strip()
            if not triple:
                continue
            variants = triple_variants(triple)
            canonical = variants[0]
            if canonical not in triples_dict:
                triples_dict[canonical] = len(triples_dict)
            cur_id = triples_dict[canonical]
            for variant in variants[1:]:
                triples_dict.setdefault(variant, cur_id)

    if os.path.basename(gold_db_path) == "faces_data":
        available = [
            n for n in os.listdir(entity_dir)
            if n.startswith(f"{entity_id}_gold_top{top_k}_") and n.endswith(".nt")
        ]
        file_n = len(available)

    gold_list: List[List[int]] = []
    for i in range(file_n):
        gold_path = os.path.join(entity_dir, f"{entity_id}_gold_top{top_k}_{i}.nt")
        if not os.path.isfile(gold_path):
            continue
        encoded_gold: List[int] = []
        with open(gold_path, "r", encoding="utf-8") as reader:
            for raw in reader:
                triple = raw.strip()
                if not triple:
                    continue
                variants = triple_variants(triple)
                enc = None
                for variant in variants:
                    if variant in triples_dict:
                        enc = triples_dict[variant]
                        break
                if enc is None:
                    triples_dict[variants[0]] = len(triples_dict)
                    enc = triples_dict[variants[0]]
                    for variant in variants[1:]:
                        triples_dict.setdefault(variant, enc)
                encoded_gold.append(enc)
        gold_list.append(encoded_gold)

    return gold_list, triples_dict


def load_encoded_prediction(method_dataset_dir: str, entity_id: int, top_k: int, triples_dict: Dict[str, int]) -> List[int]:
    pred_path = os.path.join(method_dataset_dir, str(entity_id), f"{entity_id}_top{top_k}.nt")
    if not os.path.isfile(pred_path):
        return []

    encoded: List[int] = []
    with open(pred_path, "r", encoding="utf-8") as reader:
        for raw in reader:
            triple = raw.strip()
            if not triple:
                continue
            for variant in triple_variants(triple):
                if variant in triples_dict:
                    encoded.append(triples_dict[variant])
                    break
    return encoded


@dataclass
class WilcoxonResult:
    n_pairs: int
    n_nonzero: int
    w_plus: float
    w_minus: float
    statistic_min: float
    z_value: float
    p_two_sided: float
    p_greater: float
    p_less: float
    median_diff: float
    mean_diff: float


def list_entity_ids(method_dataset_dir: str, top_k: int) -> List[int]:
    if not os.path.isdir(method_dataset_dir):
        return []

    entity_ids: List[int] = []
    for name in os.listdir(method_dataset_dir):
        if not name.isdigit():
            continue
        entity_id = int(name)
        summary_path = os.path.join(method_dataset_dir, name, f"{name}_top{top_k}.nt")
        if os.path.isfile(summary_path):
            entity_ids.append(entity_id)

    entity_ids.sort()
    return entity_ids


def f_measure(predicted: Sequence[int], gold_list: Sequence[Sequence[int]]) -> float:
    f_list: List[float] = []
    pred_size = len(predicted)
    for gold in gold_list:
        gold_size = len(gold)
        if pred_size == 0 or gold_size == 0:
            f_list.append(0.0)
            continue
        corr = len([tid for tid in predicted if tid in gold])
        if corr == 0:
            f_list.append(0.0)
            continue
        precision = corr / pred_size
        recall = corr / gold_size
        f_list.append(2 * ((precision * recall) / (precision + recall)))
    return float(sum(f_list) / len(f_list)) if f_list else 0.0


def score_entity(gold_db_path: str, method_dataset_dir: str, entity_id: int, top_k: int, file_n: int) -> float:
    gold_list, triples_dict = load_gold_and_dict(gold_db_path, entity_id, top_k, file_n)
    encoded_topk = load_encoded_prediction(method_dataset_dir, entity_id, top_k, triples_dict)
    return f_measure(encoded_topk, gold_list)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> WilcoxonResult:
    if len(x) != len(y):
        raise ValueError("Paired samples must have equal length.")

    diffs = [a - b for a, b in zip(x, y)]
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)

    if n == 0:
        return WilcoxonResult(
            n_pairs=len(diffs),
            n_nonzero=0,
            w_plus=0.0,
            w_minus=0.0,
            statistic_min=0.0,
            z_value=0.0,
            p_two_sided=1.0,
            p_greater=1.0,
            p_less=1.0,
            median_diff=0.0,
            mean_diff=0.0,
        )

    abs_with_sign = [(abs(d), 1 if d > 0 else -1) for d in nonzero]
    abs_with_sign.sort(key=lambda item: item[0])

    ranks_with_sign: List[Tuple[float, int, float]] = []
    tie_counts: List[int] = []
    i = 0
    rank = 1
    while i < n:
        j = i
        while j < n and abs_with_sign[j][0] == abs_with_sign[i][0]:
            j += 1
        group_size = j - i
        avg_rank = (rank + (rank + group_size - 1)) / 2.0
        tie_counts.append(group_size)
        for k in range(i, j):
            ranks_with_sign.append((avg_rank, abs_with_sign[k][1], abs_with_sign[k][0]))
        rank += group_size
        i = j

    w_plus = sum(r for r, sign, _ in ranks_with_sign if sign > 0)
    total_rank_sum = n * (n + 1) / 2.0
    w_minus = total_rank_sum - w_plus
    w_min = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0
    tie_correction = sum((t**3 - t) for t in tie_counts) / 48.0
    var_w = (n * (n + 1) * (2 * n + 1)) / 24.0 - tie_correction

    if var_w <= 0:
        z = 0.0
        p_two_sided = 1.0
        p_greater = 1.0
        p_less = 1.0
    else:
        if w_plus > mean_w:
            cc = 0.5
        elif w_plus < mean_w:
            cc = -0.5
        else:
            cc = 0.0
        z = (w_plus - mean_w - cc) / math.sqrt(var_w)
        cdf = _normal_cdf(z)
        p_two_sided = min(1.0, 2.0 * min(cdf, 1.0 - cdf))
        p_greater = 1.0 - cdf
        p_less = cdf

    return WilcoxonResult(
        n_pairs=len(diffs),
        n_nonzero=n,
        w_plus=w_plus,
        w_minus=w_minus,
        statistic_min=w_min,
        z_value=z,
        p_two_sided=p_two_sided,
        p_greater=p_greater,
        p_less=p_less,
        median_diff=statistics.median(diffs),
        mean_diff=sum(diffs) / len(diffs),
    )


def compare_methods(
    repo_root: str,
    comparison_dir: str,
    datasets: Sequence[str],
    topk_values: Sequence[int],
    reference_method: str,
    methods: Sequence[str],
    file_n: int,
    alpha: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for dataset in datasets:
        gold_rel_path = DATASET_GOLD_PATHS[dataset]
        gold_db_path = os.path.join(repo_root, gold_rel_path)

        for top_k in topk_values:
            ref_dir = os.path.join(comparison_dir, reference_method, dataset)
            ref_ids = set(list_entity_ids(ref_dir, top_k))

            for method in methods:
                if method == reference_method:
                    continue

                other_dir = os.path.join(comparison_dir, method, dataset)
                other_ids = set(list_entity_ids(other_dir, top_k))
                paired_ids = sorted(ref_ids | other_ids)

                if not paired_ids:
                    continue

                ref_scores: List[float] = []
                other_scores: List[float] = []
                for entity_id in paired_ids:
                    ref_scores.append(score_entity(gold_db_path, ref_dir, entity_id, top_k, file_n))
                    other_scores.append(score_entity(gold_db_path, other_dir, entity_id, top_k, file_n))

                result = wilcoxon_signed_rank(ref_scores, other_scores)
                rows.append(
                    {
                        "dataset": dataset,
                        "top_k": top_k,
                        "reference_method": reference_method,
                        "other_method": method,
                        "n_pairs": result.n_pairs,
                        "n_nonzero": result.n_nonzero,
                        "mean_f_reference": sum(ref_scores) / len(ref_scores),
                        "mean_f_other": sum(other_scores) / len(other_scores),
                        "mean_diff": result.mean_diff,
                        "median_diff": result.median_diff,
                        "w_plus": result.w_plus,
                        "w_minus": result.w_minus,
                        "w_min": result.statistic_min,
                        "z_value": result.z_value,
                        "p_two_sided": result.p_two_sided,
                        "p_greater_ref_better": result.p_greater,
                        "p_less_ref_worse": result.p_less,
                        "alpha": alpha,
                        "significant_two_sided": "Significant" if result.p_two_sided < alpha else "Not Significant",
                        "significant_ref_better_one_sided": "Significant" if result.p_greater < alpha else "Not Significant",
                    }
                )

    return rows


def print_table(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        print("No comparable pairs found.")
        return

    header = (
        "dataset", "k", "reference", "other", "n", "n!=0",
        "mean_ref", "mean_other", "mean_diff", "p_two_sided", "p_ref>other",
        "sig(two)", "sig(ref>other)",
    )
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["dataset"]),
                    str(row["top_k"]),
                    str(row["reference_method"]),
                    str(row["other_method"]),
                    str(row["n_pairs"]),
                    str(row["n_nonzero"]),
                    f"{float(row['mean_f_reference']):.6f}",
                    f"{float(row['mean_f_other']):.6f}",
                    f"{float(row['mean_diff']):.6f}",
                    f"{float(row['p_two_sided']):.6g}",
                    f"{float(row['p_greater_ref_better']):.6g}",
                    str(row["significant_two_sided"]),
                    str(row["significant_ref_better_one_sided"]),
                ]
            )
        )


def write_csv(rows: Sequence[Dict[str, object]], output_csv: str) -> None:
    if not rows:
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as writer:
        fieldnames = list(rows[0].keys())
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wilcoxon signed-rank tests for ToT4ES vs other methods in final-comparison"
    )
    parser.add_argument(
        "--comparison-dir",
        default="final-comparison",
        help="Path to directory containing method folders (default: final-comparison)",
    )
    parser.add_argument(
        "--reference-method",
        default="ToT4ES",
        help="Reference method folder name (default: ToT4ES)",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["zero_shot", "cot", "IRES"],
        help="Other method folder names to compare with reference",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["dbpedia", "faces", "lmdb"],
        choices=["dbpedia", "faces", "lmdb"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--topk",
        nargs="*",
        type=int,
        default=[5, 10],
        help="Top-k values to evaluate (default: 5 10)",
    )
    parser.add_argument(
        "--file-n",
        type=int,
        default=6,
        help="Number of gold files per entity for non-faces datasets (default: 6)",
    )
    parser.add_argument(
        "--output-csv",
        default="final-comparison/wilcoxon_tot4es_vs_others.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level alpha (default: 0.05)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    comparison_dir = (
        args.comparison_dir
        if os.path.isabs(args.comparison_dir)
        else os.path.join(repo_root, args.comparison_dir)
    )

    rows = compare_methods(
        repo_root=repo_root,
        comparison_dir=comparison_dir,
        datasets=args.datasets,
        topk_values=args.topk,
        reference_method=args.reference_method,
        methods=args.methods,
        file_n=args.file_n,
        alpha=args.alpha,
    )
    print_table(rows)
    write_csv(rows, args.output_csv)
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()