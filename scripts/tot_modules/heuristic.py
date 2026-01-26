#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heuristic Calculator - Aggregates multi-sample evaluations into scores
"""

import json
from typing import List


def entity_heuristic_calculator(
    states: List[str],
    state_evals: List[str],
    w_relatedness: float = 0.4,
    w_informativeness: float = 0.4,
    w_coverage: float = 0.2,
) -> List[float]:
    """
    Aggregate multiple LLM evaluation samples using vote-based averaging.
    
    Args:
        states: List of state strings being evaluated
        state_evals: List of raw LLM evaluation outputs (JSON strings)
        w_relatedness: Weight for relatedness criterion
        w_informativeness: Weight for informativeness criterion
        w_coverage: Weight for coverage/diversity criterion
        
    Returns:
        List of aggregated scores (one per state)
        
    Notes:
        - Uses robust JSON extraction (finds [...] in response)
        - Averages across multiple evaluation samples
        - Returns neutral scores (0.5) if all parsing fails
    """
    n_states = len(states)
    agg = [{"relatedness": 0.0, "informativeness": 0.0, "coverage": 0.0}
           for _ in range(n_states)]
    n_samples = 0

    for raw in state_evals:
        raw = raw.strip()
        if not raw:
            continue

        # Salvage: find JSON between [ ... ]
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            continue

        json_str = raw[start:end+1]

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, list):
            continue
        if len(parsed) != n_states:
            continue

        # Accumulate scores
        for i, entry in enumerate(parsed):
            try:
                agg[i]["relatedness"]     += float(entry["relatedness"])
                agg[i]["informativeness"] += float(entry["informativeness"])
                agg[i]["coverage"]        += float(entry["coverage"])
            except (KeyError, ValueError, TypeError):
                pass

        n_samples += 1

    if n_samples == 0:
        # All evaluation samples failed to parse
        print("\n[ERROR] All evaluation samples failed JSON parsing!")
        print("Raw evaluation outputs were shown above in DEBUG section.")
        print("Falling back to uniform scores (0.5) for all states.")
        return [0.5] * n_states  # Return neutral scores instead of zeros

    # Average and compute weighted sum
    factor = 1.0 / n_samples
    final_values = []
    for i in range(n_states):
        r   = agg[i]["relatedness"]     * factor
        inf = agg[i]["informativeness"] * factor
        cov = agg[i]["coverage"]        * factor
        score = w_relatedness * r + w_informativeness * inf + w_coverage * cov
        final_values.append(score)

    return final_values
