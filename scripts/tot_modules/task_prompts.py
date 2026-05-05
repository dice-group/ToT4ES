#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-Specific Prompt Factories
Implements decomposed prompts for Relatedness, Informativeness, and Diversity
with human-readable triple presentation
"""

import json
from pathlib import Path
from typing import Callable, List, Dict, Optional


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-Specific Prompt Factories
Implements decomposed prompts for Relatedness, Informativeness, and Diversity
with human-readable triple presentation
"""

import json
from pathlib import Path
from typing import Callable, List, Dict, Optional


def _load_predicate_mapping(dataset_name: str) -> Dict[str, str]:
    """Load predicate ID -> label mapping for a dataset."""
    
    # Try to find the mapping file
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    mapping_path = repo_root / "wikies_property_mappings" / f"{dataset_name}.json"
    
    if not mapping_path.exists():
        return {}
    
    try:
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        predicates = data.get("predicates", {})
        return {p_id: p_info.get("label", p_id) for p_id, p_info in predicates.items()}
    except Exception:
        return {}


def _format_triple_readable(
    triple_str: str, 
    predicate_map: Dict[str, str] = None
) -> str:
    """
    Format an RDF triple string into a more human-readable format.
    
    Input:  <http://www.wikidata.org/entity/Q2709> <http://www.wikidata.org/prop/direct/P91> <http://www.wikidata.org/entity/Q43200>
    Output: Avatar (Q2709) --[production_company]--> 20th Century Fox (Q43200)
    """
    if not triple_str.strip():
        return ""
    
    predicate_map = predicate_map or {}
    parts = triple_str.split()
    
    if len(parts) < 3:
        return triple_str
    
    subject = parts[0]
    predicate = parts[1]
    obj = parts[2]
    
    # Extract entity IDs (Q-numbers) and predicate codes (P-numbers)
    subject_id = subject.split("/")[-1].rstrip(">")
    predicate_id = predicate.split("/")[-1].rstrip(">")
    obj_id = obj.split("/")[-1].rstrip(">")
    
    # Get human-readable predicate label
    pred_label = predicate_map.get(predicate_id, predicate_id)
    
    # Format as: Subject (ID) --[predicate]--> Object (ID)
    return f"{subject_id} --[{pred_label}]--> {obj_id}"


def make_relatedness_prompt(
    entity_label: str,
    all_triples: List[str],
    predicate_frequencies: dict = None,
    dataset_name: str = None,
) -> Callable[[str, str], str]:
    """
    Create prompt for RELATEDNESS-focused triple selection with explicit criteria.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        predicate_frequencies: Dict mapping predicates to their occurrence count
                              (used to identify core/central predicates)
        dataset_name: WikiES dataset name (e.g. "wikiprofem-s") for loading predicate mappings
        
    Returns:
        Function that generates relatedness-focused prompts
    """
    
    # Load predicate mapping if dataset name provided
    predicate_map = {}
    if dataset_name:
        predicate_map = _load_predicate_mapping(dataset_name)
    
    def _extract_predicate(triple: str) -> str:
        """Extract predicate ID from RDF triple string."""
        parts = triple.split()
        if len(parts) >= 2:
            pred = parts[1]
            return pred.split("/")[-1].rstrip(">")
        return ""
    
    def _get_core_predicates(top_n: int = 8) -> str:
        """Get most common/core predicates from corpus statistics with labels."""
        if not predicate_frequencies:
            return "Not available in this run."
        
        sorted_preds = sorted(
            predicate_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        core = [(p, predicate_map.get(p, p)) for p, _ in sorted_preds[:top_n]]
        return ", ".join([f"{p} ({label})" for p, label in core]) if core else "Analysis shows no clear core predicates"
    
    def _inner(input_seq: str, state: str) -> str:
        selected_ids: List[int] = []
        if state.strip():
            selected_ids = [
                int(x) for x in state.strip().splitlines() if x.strip().isdigit()
            ]
        selected_set = set(selected_ids)

        candidate_lines = []
        for idx, triple in enumerate(all_triples, start=1):
            if idx not in selected_set:
                readable = _format_triple_readable(triple, predicate_map)
                candidate_lines.append(f"{idx}. {readable}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {_format_triple_readable(all_triples[i - 1], predicate_map)}" 
                for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"
        core_preds_text = _get_core_predicates()

        return f"""
You are evaluating triples for RELATEDNESS to the entity.

Entity: {entity_label}

RELATEDNESS DEFINITION:
A triple is RELATED/CENTRAL if it:
1. CENTRALITY: Uses core/frequent predicates that define entity types (identity, classification, basic properties)
2. SPECIFICITY: Provides distinctive values NOT generic descriptions
3. ESSENTIALITY: Best answers "What fundamentally IS this entity?"

DOMAIN CONTEXT - Core/frequent predicates (entity-defining):
{core_preds_text}

SELECTION CRITERION:
Choose the candidate combining: Centrality (frequent predicate?) + Specificity (distinctive value?) + Essentiality (defines the entity?)

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

For each candidate, briefly evaluate: predicate frequency + value specificity + whether it defines the entity.
Select ONE triple index that is MOST RELATED/CENTRAL to the entity.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_informativeness_prompt(
    entity_label: str,
    all_triples: List[str],
    predicate_frequencies: dict = None,
    selected_triples: List[int] = None,
    dataset_name: str = None,
) -> Callable[[str, str], str]:
    """
    Create prompt for INFORMATIVENESS-focused triple selection with explicit criteria.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        predicate_frequencies: Dict mapping predicates to their occurrence count in corpus
                              (used to identify rare predicates)
        selected_triples: List of indices already selected (for topic coverage tracking)
        dataset_name: WikiES dataset name (e.g. "wikiprofem-s") for loading predicate mappings
        
    Returns:
        Function that generates informativeness-focused prompts
    """
    
    # Load predicate mapping if dataset name provided
    predicate_map = {}
    if dataset_name:
        predicate_map = _load_predicate_mapping(dataset_name)
    
    def _extract_predicate(triple: str) -> str:
        """Extract predicate ID from RDF triple string."""
        parts = triple.split()
        if len(parts) >= 2:
            pred = parts[1]
            return pred.split("/")[-1].rstrip(">")
        return ""
    
    def _get_selected_predicates(selected_ids: List[int]) -> set:
        """Get set of predicates already selected with labels."""
        predicates = set()
        for idx in selected_ids:
            if 1 <= idx <= len(all_triples):
                pred_id = _extract_predicate(all_triples[idx - 1])
                pred_label = predicate_map.get(pred_id, pred_id)
                predicates.add(f"{pred_id} ({pred_label})")
        return predicates
    
    def _get_rare_predicates(top_n: int = 8) -> str:
        """Get rare predicates from corpus statistics with labels."""
        if not predicate_frequencies:
            return "Not available in this run."
        
        sorted_preds = sorted(
            predicate_frequencies.items(),
            key=lambda x: x[1]
        )
        rare = [(p, predicate_map.get(p, p)) for p, _ in sorted_preds[:top_n]]
        return ", ".join([f"{p} ({label})" for p, label in rare]) if rare else "Analysis shows no clear rare predicates"
    
    def _inner(input_seq: str, state: str) -> str:
        selected_ids: List[int] = []
        if state.strip():
            selected_ids = [
                int(x) for x in state.strip().splitlines() if x.strip().isdigit()
            ]
        selected_set = set(selected_ids)

        candidate_lines = []
        for idx, triple in enumerate(all_triples, start=1):
            if idx not in selected_set:
                readable = _format_triple_readable(triple, predicate_map)
                candidate_lines.append(f"{idx}. {readable}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {_format_triple_readable(all_triples[i - 1], predicate_map)}" 
                for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"
        
        covered_predicates = _get_selected_predicates(selected_ids)
        covered_predicates_text = ", ".join(covered_predicates) if covered_predicates else "None yet"
        rare_preds_text = _get_rare_predicates()

        return f"""
You are evaluating triples for INFORMATIVENESS.

Entity: {entity_label}

INFORMATIVENESS DEFINITION:
A triple is informative if it combines:
1. RARITY: Uses uncommon predicates (not generic like rdf:type, rdfs:label, or rdf:comment)
2. NOVELTY: Introduces predicates NOT already in your selection
3. SPECIFICITY: Provides concrete, detailed values (not generic categories/descriptions)

DOMAIN CONTEXT - Rare predicates in this dataset:
{rare_preds_text}

CURRENT STATE - Already selected predicates:
{covered_predicates_text}

SELECTION CRITERION:
Choose the candidate with highest information gain:
- PredicateRarity (is it uncommon?) + TopicNovelty (covers new predicate?) + Specificity (concrete value?)

Already selected:
{selected_text}

Remaining candidates:
Select ONE triple index that is MOST INFORMATIVE (best combines rarity + novelty + specificity).{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_diversity_prompt(
    entity_label: str,
    all_triples: List[str],
    semantic_roles: dict = None,
    dataset_name: str = None,
) -> Callable[[str, str], str]:
    """
    Create prompt for DIVERSITY/COVERAGE-focused triple selection with explicit coverage analysis.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        semantic_roles: Dict mapping predicates to semantic role categories
                        (e.g., {"dbpedia:birthPlace": "location", "dbo:birthDate": "time"})
        dataset_name: WikiES dataset name (e.g. "wikiprofem-s") for loading predicate mappings
        
    Returns:
        Function that generates diversity-focused prompts
    """
    
    # Load predicate mapping if dataset name provided
    predicate_map = {}
    if dataset_name:
        predicate_map = _load_predicate_mapping(dataset_name)
    
    def _extract_predicate(triple: str) -> str:
        """Extract predicate ID from RDF triple string."""
        parts = triple.split()
        if len(parts) >= 2:
            pred = parts[1]
            return pred.split("/")[-1].rstrip(">")
        return ""
    
    def _get_semantic_roles_coverage(selected_ids: List[int]) -> str:
        """Analyze coverage of semantic roles in selection."""
        if not semantic_roles:
            return "Not available in this run."
        
        covered_roles = {}
        for idx in selected_ids:
            if 1 <= idx <= len(all_triples):
                pred = _extract_predicate(all_triples[idx - 1])
                role = semantic_roles.get(pred, "other")
                covered_roles[role] = covered_roles.get(role, 0) + 1
        
        if not covered_roles:
            return "None yet (no roles covered)"
        
        return ", ".join([f"{role}({count})" for role, count in sorted(covered_roles.items())])
    
    def _get_available_roles(selected_ids: List[int]) -> str:
        """Get semantic roles NOT yet covered in candidates."""
        if not semantic_roles:
            return "Not available in this run."
        
        covered_roles = set()
        for idx in selected_ids:
            if 1 <= idx <= len(all_triples):
                pred = _extract_predicate(all_triples[idx - 1])
                role = semantic_roles.get(pred, "other")
                covered_roles.add(role)
        
        all_roles = set(semantic_roles.values())
        available_roles = all_roles - covered_roles
        
        return ", ".join(sorted(available_roles)) if available_roles else "All roles already covered"
    
    def _inner(input_seq: str, state: str) -> str:
        selected_ids: List[int] = []
        if state.strip():
            selected_ids = [
                int(x) for x in state.strip().splitlines() if x.strip().isdigit()
            ]
        selected_set = set(selected_ids)

        candidate_lines = []
        for idx, triple in enumerate(all_triples, start=1):
            if idx not in selected_set:
                readable = _format_triple_readable(triple, predicate_map)
                candidate_lines.append(f"{idx}. {readable}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {_format_triple_readable(all_triples[i - 1], predicate_map)}" 
                for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"
        coverage_text = _get_semantic_roles_coverage(selected_ids)
        available_roles_text = _get_available_roles(selected_ids)

        return f"""
You are evaluating triples for DIVERSITY and coverage.

Entity: {entity_label}

DIVERSITY DEFINITION:
A triple MAXIMIZES DIVERSITY if it:
1. ROLE VARIETY: Covers semantic roles NOT yet represented (location, time, relationship, attribute, etc.)
2. PREDICATE NOVELTY: Uses predicates different from already selected (no redundant predicates)
3. PERSPECTIVE BREADTH: Views entity from different aspects (not just repeating the same type of info)

CURRENT STATE - Semantic roles covered:
{coverage_text}

AVAILABLE OPPORTUNITIES - Roles NOT yet covered:
{available_roles_text}

SELECTION CRITERION:
Choose the candidate with highest diversity impact:
- RoleNovelty (covers missing semantic role?) + PredicateNovelty (different predicate?) + PerspectiveBreadth (new aspect?)

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

For each candidate, briefly evaluate: semantic role novelty + predicate distinctness + perspective breadth.
Select ONE triple index that MAXIMIZES DIVERSITY and coverage.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_combined_evaluation_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, List[str]], str]:
    """
    Create evaluation prompt that assesses all three criteria.
    This remains the same as before.
    """
    
    def _inner(input_seq: str, states: List[str]) -> str:
        formatted_states = []
        n_triples = len(all_triples)

        for idx, state in enumerate(states):
            triple_ids: List[int] = []
            if state.strip():
                triple_ids = [
                    int(x) for x in state.strip().splitlines() if x.strip().isdigit()
                ]

            triple_ids = [tid for tid in triple_ids if 1 <= tid <= n_triples]

            if triple_ids:
                triples_txt = "\n".join(
                    f"- {tid}. {all_triples[tid - 1]}" for tid in triple_ids
                )
            else:
                triples_txt = "(empty summary)"

            formatted_states.append(f"SUMMARY {idx}:\n{triples_txt}")

        states_block = "\n\n".join(formatted_states)

        return f"""
You are evaluating RDF triple summaries for: {entity_label}

Rate each summary on three criteria (0.0–1.0):

1. RELATEDNESS: How central/core are the triples to the entity?
2. INFORMATIVENESS: How much unique, valuable information is provided?
3. COVERAGE/DIVERSITY: How diverse are the aspects covered?

There are {len(states)} summaries. Return JSON array:

[
  {{
    "idx": 0,
    "relatedness": 0.0,
    "informativeness": 0.0,
    "coverage": 0.0
  }},
  ...
]

Return ONLY the JSON array, no explanations.

Summaries:

{states_block}
        """.strip()

    return _inner
