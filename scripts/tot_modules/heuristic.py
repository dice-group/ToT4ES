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
    
    Supports two formats:
    1. JSON: [{"idx": 0, "relatedness": 0.8, ...}, ...]
    2. Simple: SUMMARY_0: R=0.8 I=0.7 C=0.9
    
    Args:
        states: List of state strings being evaluated
        state_evals: List of raw LLM evaluation outputs
        w_relatedness: Weight for relatedness criterion
        w_informativeness: Weight for informativeness criterion
        w_coverage: Weight for coverage/diversity criterion
        
    Returns:
        List of aggregated scores (one per state)
    """
    import re
    
    n_states = len(states)
    agg = [{"relatedness": 0.0, "informativeness": 0.0, "coverage": 0.0}
           for _ in range(n_states)]
    n_samples = 0

    for raw in state_evals:
        raw = raw.strip()
        if not raw:
            continue

        # Try JSON format first - more robust extraction
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_str = raw[start:end+1]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and len(parsed) == n_states:
                    valid_sample = True
                    temp_agg = {"r": [], "i": [], "c": []}
                    
                    for i, entry in enumerate(parsed):
                        try:
                            r_val = float(entry.get("relatedness", entry.get("r", 0)))
                            i_val = float(entry.get("informativeness", entry.get("i", 0)))
                            c_val = float(entry.get("coverage", entry.get("c", 0)))
                            
                            # Validate scores are in [0, 1]
                            if not (0.0 <= r_val <= 1.0 and 0.0 <= i_val <= 1.0 and 0.0 <= c_val <= 1.0):
                                valid_sample = False
                                break
                            
                            temp_agg["r"].append(r_val)
                            temp_agg["i"].append(i_val)
                            temp_agg["c"].append(c_val)
                        except (KeyError, ValueError, TypeError):
                            valid_sample = False
                            break
                    
                    # Only accept if all entries were valid
                    if valid_sample and all(len(v) == n_states for v in temp_agg.values()):
                        for i in range(n_states):
                            agg[i]["relatedness"] += temp_agg["r"][i]
                            agg[i]["informativeness"] += temp_agg["i"][i]
                            agg[i]["coverage"] += temp_agg["c"][i]
                        n_samples += 1
                        continue
            except (json.JSONDecodeError, ValueError):
                pass

        # Try simple format: SUMMARY_X: R=0.X I=0.X C=0.X
        # More lenient pattern to handle variations
        pattern = r'SUMMARY[_\s]*(\d+)\s*:?\s*R\s*=\s*([0-9.]+)\s*I\s*=\s*([0-9.]+)\s*C\s*=\s*([0-9.]+)'
        matches = re.findall(pattern, raw, re.IGNORECASE)
        
        if matches:
            # Create mapping from index to scores
            score_map = {}
            for idx_str, r_str, i_str, c_str in matches:
                try:
                    idx = int(idx_str)
                    if 0 <= idx < n_states:
                        score_map[idx] = {
                            'r': float(r_str),
                            'i': float(i_str),
                            'c': float(c_str)
                        }
                except (ValueError, IndexError):
                    pass
            
            # Check if we got scores for all states
            if len(score_map) == n_states:
                for idx in range(n_states):
                    agg[idx]["relatedness"]     += score_map[idx]['r']
                    agg[idx]["informativeness"] += score_map[idx]['i']
                    agg[idx]["coverage"]        += score_map[idx]['c']
                n_samples += 1
                continue

    if n_samples == 0:
        # All evaluation samples failed to parse
        print("\n[ERROR] All evaluation samples failed parsing!")
        print("Raw evaluation outputs were shown above in DEBUG section.")
        print("Falling back to uniform scores (0.5) for all states.")
        return [0.5] * n_states

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
