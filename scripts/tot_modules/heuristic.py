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

        # Try JSON format first - with robust extraction
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_str = raw[start:end+1]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Be lenient: accept partial evaluations
                    # If we get at least one valid entry, use what we have
                    scored_indices = set()
                    valid_sample = True
                    
                    for entry in parsed:
                        try:
                            idx = entry.get("idx", None)
                            if idx is None:
                                valid_sample = False
                                continue
                            
                            idx = int(idx)
                            if not (0 <= idx < n_states):
                                continue
                            
                            r_val = float(entry.get("relatedness", entry.get("r", 0.5)))
                            i_val = float(entry.get("informativeness", entry.get("i", 0.5)))
                            c_val = float(entry.get("coverage", entry.get("c", 0.5)))
                            
                            # Clamp to [0, 1] if slightly out of bounds (floating point tolerance)
                            r_val = max(0.0, min(1.0, r_val))
                            i_val = max(0.0, min(1.0, i_val))
                            c_val = max(0.0, min(1.0, c_val))
                            
                            agg[idx]["relatedness"] += r_val
                            agg[idx]["informativeness"] += i_val
                            agg[idx]["coverage"] += c_val
                            scored_indices.add(idx)
                        except (ValueError, TypeError):
                            continue
                    
                    # Accept if we got at least 1 valid score
                    if scored_indices:
                        n_samples += 1
                        continue
            except (json.JSONDecodeError, ValueError):
                pass

        # Try simple format: SUMMARY_X: R=0.X I=0.X C=0.X
        pattern = r'SUMMARY[_\s]*(\d+)\s*:?\s*R\s*=\s*([0-9.]+)\s*I\s*=\s*([0-9.]+)\s*C\s*=\s*([0-9.]+)'
        matches = re.findall(pattern, raw, re.IGNORECASE)
        
        if matches:
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
            
            # Accept if we got at least 1 score
            if score_map:
                for idx in score_map:
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

    # Average only the states that were actually scored
    # For states that weren't scored in any sample, reset to 0.5
    final_values = []
    for i in range(n_states):
        if agg[i]["relatedness"] > 0 or agg[i]["informativeness"] > 0 or agg[i]["coverage"] > 0:
            # This state was scored at least once
            factor = 1.0 / n_samples
            r   = agg[i]["relatedness"]     * factor
            inf = agg[i]["informativeness"] * factor
            cov = agg[i]["coverage"]        * factor
        else:
            # This state was never scored, use neutral
            r = inf = cov = 0.5
        
        score = w_relatedness * r + w_informativeness * inf + w_coverage * cov
        final_values.append(score)

    return final_values
