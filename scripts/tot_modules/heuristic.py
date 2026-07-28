#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heuristic Calculator - Aggregates multi-sample evaluations into scores
"""

import json
from typing import List


def _extract_json_objects(raw: str) -> List[dict]:
    """Best-effort extraction of JSON objects from model output.

    Handles valid JSON arrays, truncated arrays, and outputs that contain
    one object per line without a closing bracket.
    """
    objects: List[dict] = []

    # Strip common markdown/code-fence wrappers first.
    cleaned = raw.strip()
    cleaned = cleaned.replace("```json", "```").replace("```JSON", "```")
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Prefer a full JSON array when possible.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, list):
                return [entry for entry in parsed if isinstance(entry, dict)]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract every {...} block and try to decode it individually.
    depth = 0
    block_start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                block_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and block_start is not None:
                    block = cleaned[block_start : i + 1]
                    try:
                        obj = json.loads(block)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    block_start = None

    return objects


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
        parsed_entries = _extract_json_objects(raw)
        if parsed_entries:
            scored_indices = set()

            for entry in parsed_entries:
                try:
                    idx = entry.get("idx", None)
                    if idx is None:
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
                except (ValueError, TypeError, AttributeError):
                    continue

            # Accept if we got at least 1 valid score
            if scored_indices:
                n_samples += 1
                continue

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
        
        # Ensure score stays in valid range
        score = max(0.0, min(1.0, score))
        final_values.append(score)

    return final_values
