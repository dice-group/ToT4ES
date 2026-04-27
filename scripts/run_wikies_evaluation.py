#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


SUPPORTED_DOMAINS: Dict[str, str] = {
    "wikicinema-s": "WikiCinema-s-test_data",
    "wikilitart-s": "WikiLitArt-s-test_data",
    "wikipro-s": "WikiPro-s-test_data",
    "wikiprofem-s": "WikiProFem-s-test_data",
}

DOMAIN_ALIASES: Dict[str, str] = {
    "wikicinema": "wikicinema-s",
    "wikilitart": "wikilitart-s",
    "wikipro": "wikipro-s",
    "wikiprofem": "wikiprofem-s",
    "wikies_wikicinema-s": "wikicinema-s",
    "wikies_wikilitart-s": "wikilitart-s",
    "wikies_wikipro-s": "wikipro-s",
    "wikies_wikiprofem-s": "wikiprofem-s",
}

WS_RE = re.compile(r"\s+")


def normalize_triple(line: str) -> str:
    triple = line.strip()
    if triple.endswith("."):
        triple = triple[:-1].strip()
    return WS_RE.sub(" ", triple)


def read_nt_file(path: Path) -> List[str]:
    triples: List[str] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            triples.append(normalize_triple(line))
    return triples


def ndcg_score(predicted: Sequence[str], gold: Sequence[str]) -> float:
    if not predicted or not gold:
        return 0.0

    triple_grade: Dict[str, int] = {}
    for triple in gold:
        triple_grade[triple] = triple_grade.get(triple, 0) + 1
    grade_list = sorted(triple_grade.values(), reverse=True)

    dcg = 0.0
    for idx, triple in enumerate(predicted, start=1):
        rel = float(triple_grade.get(triple, 0))
        dcg += rel / np.log2(idx + 1)

    ideal_len = min(len(grade_list), len(predicted))
    if ideal_len == 0:
        return 0.0

    idcg = sum(grade_list[idx - 1] / np.log2(idx + 1) for idx in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(predicted: Sequence[str], gold: Sequence[str]) -> float:
    if not predicted or not gold:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for idx, triple in enumerate(predicted, start=1):
        if triple in gold:
            hit_count += 1
            precision_sum += hit_count / idx

    return precision_sum / len(gold) if hit_count > 0 else 0.0


def f1_at_k(predicted: Sequence[str], gold: Sequence[str]) -> float:
    if not predicted or not gold:
        return 0.0

    corr = len([triple for triple in predicted if triple in gold])
    if corr == 0:
        return 0.0

    precision = corr / len(predicted)
    recall = corr / len(gold)
    return 2 * precision * recall / (precision + recall)


def parse_topk(raw: str) -> List[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("--topk must contain at least one integer")
    return values


def normalize_domain_name(name: str) -> str:
    key = name.strip().lower()
    if key in SUPPORTED_DOMAINS:
        return key
    if key in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[key]
    raise ValueError(
        f"Unsupported domain '{name}'. Supported: {', '.join(SUPPORTED_DOMAINS.keys())}"
    )


def parse_domains(raw: str) -> List[str]:
    requested = [token.strip() for token in raw.split(",") if token.strip()]
    if not requested:
        raise ValueError("--domains must contain at least one domain")
    normalized = [normalize_domain_name(name) for name in requested]
    unique = []
    seen = set()
    for domain in normalized:
        if domain not in seen:
            seen.add(domain)
            unique.append(domain)
    return unique


def parse_domain_path_overrides(raw: Optional[str]) -> Dict[str, Path]:
    if not raw:
        return {}

    result: Dict[str, Path] = {}
    items = [token.strip() for token in raw.split(",") if token.strip()]
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid path override '{item}'. Expected format: domain=/absolute/or/relative/path"
            )
        domain_raw, path_raw = item.split("=", 1)
        domain = normalize_domain_name(domain_raw.strip())
        directory = Path(path_raw.strip()).expanduser().resolve()
        result[domain] = directory

    return result


def discover_prediction_dirs(pred_root: Path) -> Dict[str, List[Path]]:
    discovered: Dict[str, List[Path]] = {domain: [] for domain in SUPPORTED_DOMAINS}
    if not pred_root.exists():
        return discovered

    for child in pred_root.iterdir():
        if not child.is_dir():
            continue
        alias = DOMAIN_ALIASES.get(child.name.lower())
        if alias:
            discovered[alias].append(child)
            continue

        key = child.name.lower()
        if key in SUPPORTED_DOMAINS:
            discovered[key].append(child)

    return discovered


def resolve_test_dirs(
    wikies_root: Path,
    domains: Sequence[str],
    overrides: Dict[str, Path],
) -> Dict[str, Path]:
    test_dirs: Dict[str, Path] = {}
    for domain in domains:
        if domain in overrides:
            test_dir = overrides[domain]
        else:
            test_dir = wikies_root / SUPPORTED_DOMAINS[domain]

        if not test_dir.is_dir():
            raise FileNotFoundError(f"Test folder not found for {domain}: {test_dir}")
        test_dirs[domain] = test_dir

    return test_dirs


def resolve_prediction_dirs(
    pred_root: Path,
    domains: Sequence[str],
    overrides: Dict[str, Path],
) -> Dict[str, List[Path]]:
    discovered = discover_prediction_dirs(pred_root)
    resolved: Dict[str, List[Path]] = {}

    for domain in domains:
        if domain in overrides:
            domain_dir = overrides[domain]
            if not domain_dir.is_dir():
                raise FileNotFoundError(
                    f"Prediction folder override not found for {domain}: {domain_dir}"
                )
            resolved[domain] = [domain_dir]
        else:
            resolved[domain] = discovered.get(domain, [])

    return resolved


def print_resolved_inputs(
    test_dirs: Dict[str, Path],
    pred_dirs_map: Dict[str, List[Path]],
) -> None:
    print("Resolved input directories:")
    for domain in sorted(test_dirs.keys()):
        pred_dirs = pred_dirs_map.get(domain, [])
        pred_label = ", ".join(str(path) for path in pred_dirs) if pred_dirs else "<none found>"
        print(f"  - {domain}")
        print(f"    test: {test_dirs[domain]}")
        print(f"    pred: {pred_label}")


def resolve_pred_file(
    pred_dirs: Sequence[Path],
    entity_id: str,
    k: int,
) -> Optional[Path]:
    for base in pred_dirs:
        candidate = base / entity_id / f"{entity_id}_top{k}.nt"
        if candidate.is_file():
            return candidate
    return None


def evaluate_domain(
    domain_key: str,
    test_dir: Path,
    pred_dirs: Sequence[Path],
    topk_values: Sequence[int],
    missing_as_zero: bool,
) -> List[dict]:
    entity_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    rows: List[dict] = []

    for k in topk_values:
        metric_f1: List[float] = []
        metric_ndcg: List[float] = []
        metric_map: List[float] = []

        evaluated = 0
        missing_pred = 0
        missing_gold = 0

        for entity_dir in entity_dirs:
            entity_id = entity_dir.name
            gold_path = entity_dir / f"{entity_id}_gold_top{k}.nt"

            if not gold_path.is_file():
                missing_gold += 1
                if missing_as_zero:
                    metric_f1.append(0.0)
                    metric_ndcg.append(0.0)
                    metric_map.append(0.0)
                continue

            pred_path = resolve_pred_file(pred_dirs, entity_id, k)
            if pred_path is None:
                missing_pred += 1
                if missing_as_zero:
                    metric_f1.append(0.0)
                    metric_ndcg.append(0.0)
                    metric_map.append(0.0)
                continue

            gold = read_nt_file(gold_path)
            pred = read_nt_file(pred_path)
            if len(pred) > k:
                pred = pred[:k]

            f1 = f1_at_k(pred, gold)
            ndcg = ndcg_score(pred, gold)
            map_score = average_precision(pred, gold)

            metric_f1.append(f1)
            metric_ndcg.append(ndcg)
            metric_map.append(map_score)
            evaluated += 1

        total_entities = len(entity_dirs)
        denominator = total_entities if missing_as_zero else max(evaluated, 1)
        coverage = (evaluated / total_entities) if total_entities > 0 else 0.0

        rows.append(
            {
                "domain": domain_key,
                "k": k,
                "entities_total": total_entities,
                "entities_evaluated": evaluated,
                "missing_gold": missing_gold,
                "missing_prediction": missing_pred,
                "coverage": coverage,
                "f1": float(np.sum(metric_f1) / denominator) if metric_f1 else 0.0,
                "ndcg": float(np.sum(metric_ndcg) / denominator) if metric_ndcg else 0.0,
                "map": float(np.sum(metric_map) / denominator) if metric_map else 0.0,
            }
        )

    return rows


def print_results(rows: Iterable[dict]) -> None:
    print("=" * 78)
    print("WikiES Benchmark (Test Data) Evaluation")
    print("=" * 78)
    print(
        f"{'Domain':<15} {'K':<4} {'Eval/Total':<12} {'MissPred':<9} {'Cov':<8} "
        f"{'F1':<10} {'NDCG':<10} {'MAP':<10}"
    )
    print("-" * 78)
    for row in rows:
        print(
            f"{row['domain']:<15} {row['k']:<4} "
            f"{row['entities_evaluated']}/{row['entities_total']:<12} "
            f"{row['missing_prediction']:<9} "
            f"{row['coverage']:<8.2%} "
            f"{row['f1']:<10.4f} {row['ndcg']:<10.4f} {row['map']:<10.4f}"
        )
    print("=" * 78)


def write_csv(rows: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "k",
        "entities_total",
        "entities_evaluated",
        "missing_gold",
        "missing_prediction",
        "coverage",
        "f1",
        "ndcg",
        "map",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as writer_handle:
        writer = csv.DictWriter(writer_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Evaluate ToT outputs on WikiES benchmark test domains."
    )
    parser.add_argument(
        "--wikies-root",
        type=Path,
        default=repo_root / "datasets" / "WikiES_benchmark",
        help="Root path to WikiES_benchmark.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=repo_root / "outputs" / "tot_wikies",
        help="Root path containing model predictions.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=",".join(SUPPORTED_DOMAINS.keys()),
        help=(
            "Comma-separated WikiES domains. Supported: "
            + ", ".join(SUPPORTED_DOMAINS.keys())
        ),
    )
    parser.add_argument(
        "--test-dirs",
        type=str,
        default=None,
        help=(
            "Optional per-domain test directory overrides. "
            "Format: wikicinema-s=/path1,wikipro-s=/path2"
        ),
    )
    parser.add_argument(
        "--pred-dirs",
        type=str,
        default=None,
        help=(
            "Optional per-domain prediction directory overrides. "
            "Format: wikicinema-s=/path1,wikipro-s=/path2"
        ),
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="5,10",
        help="Comma-separated K values, e.g. 5,10.",
    )
    parser.add_argument(
        "--exclude-missing",
        action="store_true",
        help="Exclude missing gold/prediction entities from averaging.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV file path for evaluation table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = parse_domains(args.domains)
    topk_values = parse_topk(args.topk)

    test_overrides = parse_domain_path_overrides(args.test_dirs)
    pred_overrides = parse_domain_path_overrides(args.pred_dirs)
    test_dirs = resolve_test_dirs(args.wikies_root, domains, test_overrides)
    pred_dirs_map = resolve_prediction_dirs(args.pred_root, domains, pred_overrides)

    print_resolved_inputs(test_dirs, pred_dirs_map)
    missing_as_zero = not args.exclude_missing

    all_rows: List[dict] = []

    for domain in domains:
        test_dir = test_dirs[domain]
        pred_dirs = pred_dirs_map.get(domain, [])
        rows = evaluate_domain(
            domain_key=domain,
            test_dir=test_dir,
            pred_dirs=pred_dirs,
            topk_values=topk_values,
            missing_as_zero=missing_as_zero,
        )
        all_rows.extend(rows)

    print_results(all_rows)

    if args.csv_out is not None:
        write_csv(all_rows, args.csv_out)
        print(f"Saved CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
