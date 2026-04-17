#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-Specific Prompt Factories
Implements decomposed prompts for Relatedness, Informativeness, and Diversity
"""

from typing import Callable, List


def make_relatedness_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, str], str]:
    """
    Create prompt for RELATEDNESS-focused triple selection.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        
    Returns:
        Function that generates relatedness-focused prompts
    """
    
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
                candidate_lines.append(f"{idx}. {triple}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"

        return f"""
You are a RELATEDNESS expert for entity summarization.

Entity: {entity_label}

Your ONLY goal is to select the triple that is MOST RELATED/CENTRAL to this entity.

RELATEDNESS SCORING RULES:
- Score 1.0 (Select): Core identity properties unique to this entity
  Example: Person → occupation, nationality, role
  Example: Place → coordinates, population, country
- Score 0.5 (Maybe): Generic properties shared by entity type
  Example: rdf:type, rdfs:label (too generic)
- Score 0.1 (Avoid): Metadata, irrelevant properties
  Example: Internal IDs, timestamps, formalities

Focus on:
1. Central predicates that define WHO/WHAT this entity is
2. Properties frequently used for this entity TYPE
3. Values highly specific to THIS entity (not generic)
4. Triples that best answer "What is this entity?"

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

Task: Select ONE triple index that is MOST RELATED/CENTRAL to the entity.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_informativeness_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, str], str]:
    """
    Create prompt for INFORMATIVENESS-focused triple selection.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        
    Returns:
        Function that generates informativeness-focused prompts
    """
    
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
                candidate_lines.append(f"{idx}. {triple}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"

        return f"""
You are an INFORMATIVENESS expert for entity summarization.

Entity: {entity_label}

Your ONLY goal is to select the triple that provides the MOST INFORMATIVE content.
Informativeness = How RARE, SPECIFIC, and UNIQUE is this information?

INFORMATIVENESS SCORING RULES:
- Score 1.0 (Select): Rare predicates + Specific unique values
  Example: "nobelPrize: Physics 1923" (only 0.01% have this)
  Example: "disease: MalignantMelanoma" (specific medical term)
  Why: Very surprising, distinguishes this entity

- Score 0.5 (Acceptable): Common predicates + Specific values
  Example: "birthDate: 1965-03-15" (everyone has birth date, but this specific date is unique)
  Example: "employer: Apple Inc" (company is known, but relationship is specific)
  Why: Moderate information gain

- Score 0.1 (Avoid): Generic predicates + Generic values
  Example: "rdf:type: Person" (95% have this, generic category)
  Example: "name: <same as label>" (redundant with rdfs:label)
  Why: No surprises, wastes summary space

Focus on:
1. RARE predicates (not rdf:type, rdfs:label, birthDate)
2. SPECIFIC, detailed values (not broad categories like "Person", "Organization")
3. Facts providing UNIQUE, non-obvious information
4. Information NOT already covered by selected triples
5. Properties that DISTINGUISH this entity from others

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

Task: Select ONE triple index that is MOST INFORMATIVE.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_diversity_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, str], str]:
    """
    Create prompt for DIVERSITY/COVERAGE-focused triple selection.
    
    Args:
        entity_label: Human-readable entity name
        all_triples: Complete list of triples for this entity
        
    Returns:
        Function that generates diversity-focused prompts
    """
    
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
                candidate_lines.append(f"{idx}. {triple}")

        if selected_ids:
            selected_text = "\n".join(
                f"{i}. {all_triples[i - 1]}" for i in selected_ids if 1 <= i <= len(all_triples)
            )
            exclusion_note = f"\nDO NOT select indices: {', '.join(map(str, selected_ids))}"
        else:
            selected_text = "None yet."
            exclusion_note = ""

        candidates_text = "\n".join(candidate_lines) if candidate_lines else "<no candidates>"

        return f"""
You are a DIVERSITY/COVERAGE expert for entity summarization.

Entity: {entity_label}

Your ONLY goal is to select the triple that MAXIMIZES DIVERSITY and coverage.
Diversity = Covering DIFFERENT ASPECTS and AVOIDING REDUNDANCY

DIVERSITY SCORING RULES:
- Score 1.0 (Select): Different predicate type + Different semantic aspect
  Example: If selected has [birthDate, birthPlace], pick [occupation]
  Example: If selected has [employer], pick [award] (different role)
  Why: Maximizes information variety

- Score 0.5 (Acceptable): Same predicate family but different value
  Example: If selected has [birthPlace: Berlin], pick [workPlace: Paris]
  Example: If selected has [award: Nobel], pick [award: Emmy]
  Why: Moderate diversity benefit

- Score 0.1 (Avoid): Redundant with already selected
  Example: If selected has [birthDate], avoid [deathDate] (both temporal)
  Example: If selected has [birthPlace], avoid [hometown] (same semantic role)
  Why: Wastes summary space on similar information

Semantic Roles to Consider:
- TEMPORAL: birthDate, deathDate, founded, era
- SPATIAL: birthPlace, workPlace, location, country
- ACHIEVEMENT: award, nobelPrize, degree, publication
- IDENTITY: rdf:type, occupation, role, nationality
- RELATIONSHIPS: spouse, child, employer, colleague
- CHARACTERISTICS: height, language, disease, alias

Focus on:
1. Different PREDICATE TYPES than already selected (e.g., avoid date when date selected)
2. Different SEMANTIC ROLES (location, time, achievement, etc.)
3. Values DISSIMILAR to already selected (don't cluster similar facts)
4. Covering DIFFERENT ASPECTS of the entity (bio, work, relations, achievements)
5. AVOIDING REDUNDANCY (don't repeat similar predicates)

Already selected:
{selected_text}

Remaining candidates:
{candidates_text}

Task: Select ONE triple index that MAXIMIZES DIVERSITY.{exclusion_note}

Output ONLY the integer index:
""".strip()

    return _inner


def make_combined_evaluation_prompt(
    entity_label: str,
    all_triples: List[str],
) -> Callable[[str, List[str]], str]:
    """
    Create evaluation prompt that assesses all three criteria.
    Uses simpler format for better compatibility with smaller models.
    """
    
    def _inner(input_seq: str, states: List[str]) -> str:
        formatted_states = []
        n_triples = len(all_triples)

1. RELATEDNESS (R): How central/essential are the triples to entity identity?
   - 1.0: All triples define core identity (occupation, nationality, type)
   - 0.5: Mix of identity and contextual properties
   - 0.0: No identity properties, only generic metadata

2. INFORMATIVENESS (I): How unique/rare/valuable is the information?
   - 1.0: Rare facts (unique achievements, rare properties < 5% frequency)
   - 0.5: Mixed (common predicates + specific values)
   - 0.0: Generic facts (rdf:type, generic labels, common properties > 90%)

3. COVERAGE (C): How diverse are the aspects covered?
   - 1.0: All different predicate types (temporal, spatial, achievement, identity, etc.)
   - 0.5: Mostly different (1-2 predicate type overlaps)
   - 0.0: Highly redundant (multiple temporal, multiple spatial, clustered facts)
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
You are evaluating RDF triple summaries for entity: {entity_label}

Rate each summary on 3 criteria (scale 0.0 to 1.0):
- RELATEDNESS (R): How central are triples to entity identity?
- INFORMATIVENESS (I): How unique/valuable is the information?
- COVERAGE (C): How diverse are the aspects covered?

IMPORTANT: Output EXACTLY one line per summary in this format:
SUMMARY_0: R=0.8 I=0.7 C=0.9

Below are {len(states)} summaries to rate:

{states_block}

Now output your ratings (one line per summary):
        """.strip()

    return _inner
