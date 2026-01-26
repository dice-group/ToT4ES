#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt Factory - Generate prompts for thought generation and state evaluation
"""

from typing import Callable, List


def make_entity_thought_gen_prompt(
    entity_label: str,
    all_triples: List[str],
    max_summary_len: int,
) -> Callable[[str, str], str]:
    """
    Create a prompt generator for thought generation (triple selection).
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        max_summary_len: Maximum number of triples in summary
        
    Returns:
        Function that takes (input_seq, state) and returns a prompt string
    """

    def _inner(input_seq: str, state: str) -> str:
        """
        Generate prompt for selecting the next triple.
        
        Args:
            input_seq: Global input sequence (currently unused, kept for interface)
            state: Current state - newline-separated triple indices
            
        Returns:
            Prompt string for LLM
        """
        # Parse current state into a set of selected triple ids
        selected_ids: List[int] = []
        if state.strip():
            selected_ids = [
                int(x)
                for x in state.strip().splitlines()
                if x.strip().isdigit()
            ]
        selected_set = set(selected_ids)

        # Build candidate list (only unselected triples)
        candidate_lines = []
        for idx, triple in enumerate(all_triples, start=1):
            if idx not in selected_set:
                candidate_lines.append(f"{idx}. {triple}")

        if selected_ids:
            selected_triples_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            already_selected_note = f"\nDO NOT select any of these already-chosen indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_triples_text = "None yet."
            already_selected_note = ""

        candidates_text = (
            "\n".join(candidate_lines) if candidate_lines else "<no remaining candidates>"
        )

        return f"""
You are an expert to construct a concise RDF triple summary for the entity:

  {entity_label}

You are given:
- The full candidate triples describing this entity.
- The subset of triples already selected for the current summary.
- The remaining candidate triples that are still available.

The goal is to gradually grow a good summary with at most {max_summary_len} triples, 
optimizing three criteria:
1) Relatedness: emphasize triple_centrality and how core the predicates/values are.
2) Informativeness: emphasize low freq_property/freq_value (rarer = more informative) and high type_depth.
3) Diversity/Coverage: prefer summaries where triples cover different predicates/entity roles and whose values are not too similar (you can also look at the numeric similarity scores provided).

Current selected summary (by index):
{selected_triples_text}

Remaining candidate triples (index: triple):
{candidates_text}

Task:
- Propose exactly ONE additional triple index from the remaining candidates that should
  be added next to improve the summary with respect to all three criteria.
- Prefer triples that:
  * Are strongly related to the entity.
  * Add new, important information not already covered by the selected triples.
  * Increase diversity of types/roles/topics.{already_selected_note}

Output format (IMPORTANT):
Return ONLY the integer index of the chosen triple, with no explanation, no text, no JSON.
For example, a valid answer is simply:
7
        """.strip()

    return _inner


def make_entity_state_eval_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, List[str]], str]:
    """
    Create a prompt generator for state evaluation.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        
    Returns:
        Function that takes (input_seq, states) and returns a prompt string
    """

    def _inner(input_seq: str, states: List[str]) -> str:
        """
        Generate prompt for evaluating multiple candidate summaries.
        
        Args:
            input_seq: Global input sequence (currently unused, kept for interface)
            states: List of state strings to evaluate
            
        Returns:
            Prompt string for LLM
        """
        formatted_states = []
        n_triples = len(all_triples)

        for idx, state in enumerate(states):
            triple_ids: List[int] = []
            if state.strip():
                triple_ids = [
                    int(x)
                    for x in state.strip().splitlines()
                    if x.strip().isdigit()
                ]

            # keep only valid triple ids
            triple_ids = [tid for tid in triple_ids if 1 <= tid <= n_triples]

            if triple_ids:
                triples_txt = "\n".join(
                    f"- {tid}. {all_triples[tid - 1]}" for tid in triple_ids
                )
            else:
                triples_txt = "(empty summary or invalid triple ids)"

            formatted_states.append(f"SUMMARY {idx}:\n{triples_txt}")

        states_block = "\n\n".join(formatted_states)

        return f"""
You are evaluating alternative RDF triple summaries for the entity:

  {entity_label}

Each candidate summary is a set of triples about this entity. You must rate each summary on:
- relatedness (0.0–1.0): how central and relevant the selected triples are to the entity. Emphasize triple_centrality and how core the predicates and values are to describing this entity.
- informativeness (0.0–1.0): how much important, non-trivial information the summary contains. Emphasize low freq_property and freq_value (rarer facts are more informative) and higher type_depth (more specific / deeper ontological types).
- coverage (0.0–1.0): how diverse and comprehensive the summary is across different aspects (types, roles, locations, time, etc.) while avoiding redundancy. Prefer summaries whose triples use different predicates/entity roles and whose values are not too similar (you can also use the provided numeric similarity scores for this).

There are {len(states)} candidate summaries. For each SUMMARY i (i from 0 to {len(states)-1}),
you must return numeric scores in JSON format, as a JSON array of objects, one per summary,
in the same order, with this schema:

[
  {{
    "idx": 0,
    "relatedness": 0.0,
    "informativeness": 0.0,
    "coverage": 0.0
  }},
  ...
]

Do NOT add any extra keys.
Do NOT add comments or explanations.
Return ONLY the JSON array.

Candidate summaries:

{states_block}
        """.strip()

    return _inner
