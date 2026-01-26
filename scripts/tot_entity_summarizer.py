#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-of-Thought Entity Summarization with LLaMA-3.2-3B-Instruct (Hugging Face)
for a single entity description in N-Triples (.nt) format.

Usage example:

python tot_entity_summarizer.py \
  --nt dbpedia/1/1_desc.nt \
  --max-summary-len 5 \
  --n-candidates 5 \
  --n-evals 3 \
  --breadth-limit 3
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, List, Dict, Optional

import torch
from transformers import pipeline


# =========================
# Basic tree structures
# =========================

@dataclass
class TreeNode:
    state: str            # newline-separated triple indices (as strings), e.g. "2\n5\n9"
    thought: str          # the last triple index added, e.g. "5"
    value: float = 0.0    # scalar score of this state
    depth: int = 0        # depth in the search tree
    parent: Optional["TreeNode"] = None  # parent node reference
    children: List["TreeNode"] = field(default_factory=list)


# =========================
# LLaMA 3.2 3B wrapper
# =========================

class Llama32Chat:
    """
    Wrapper around Hugging Face transformers pipeline with
    meta-llama/Llama-3.2-3B-Instruct used in chat style.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        device_map: str = "auto",
        torch_dtype = torch.bfloat16,
    ):
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.tokenizer = self.pipe.tokenizer

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        n: int = 1,
    ) -> List[str]:
        """
        Simple ChatGPT-like interface using HF pipeline.

        messages: list of { "role": "user"/"assistant"/"system", "content": "..." }

        Returns: list of n generated assistant strings.
        """
        # Convert chat messages into a single prompt using the model's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs: List[str] = []
        do_sample = temperature > 0.0

        for _ in range(n):
            out = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,  # only return the completion
            )
            # text-generation pipeline returns a list of dicts
            # [{'generated_text': '...'}]
            text = out[0]["generated_text"]
            outputs.append(text.strip())
        return outputs


# =========================
# Utility helpers
# =========================

def extract_first_int(text: str) -> Optional[int]:
    """
    Extracts the first integer occurrence in the text.
    Returns None if no integer is found.
    """
    m = re.search(r"\d+", text)
    if m:
        return int(m.group(0))
    return None


def decode_state_to_triples(state: str, all_triples: List[str]) -> List[str]:
    """
    Convert a state "2\n5\n9" into the corresponding list of triples.
    """
    if not state.strip():
        return []
    ids = [int(x) for x in state.strip().splitlines() if x.strip().isdigit()]
    return [all_triples[i - 1] for i in ids if 1 <= i <= len(all_triples)]


def load_entity_description_from_nt(nt_path: str) -> (str, List[str]):
    """
    Load an entity description from an .nt file.

    Returns:
      entity_label: a human-readable label if found (rdfs:label / foaf:name), else subject URI.
      all_triples:  list of raw triple strings (one per line, stripped).
    """
    triples: List[str] = []
    entity_label: Optional[str] = None
    subject_uri: Optional[str] = None

    # Simple literal pattern: "label"@en . or "label" .
    literal_pattern = re.compile(r'"(.*?)"(?:@[a-zA-Z\-]+|\^\^<[^>]+>)?\s*\.\s*$')

    with open(nt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            triples.append(line)

            # Extract subject URI if not set
            if subject_uri is None:
                parts = line.split()
                if parts:
                    subject_uri = parts[0].strip("<>")

            # Try to get a label from rdfs:label or foaf:name
            if entity_label is None:
                if ("rdf-schema#label" in line) or ("/label>" in line) or ("foaf/0.1/name" in line):
                    m = literal_pattern.search(line)
                    if m:
                        entity_label = m.group(1)

    if entity_label is None:
        # Fallback: use subject URI as label
        if subject_uri is not None:
            entity_label = subject_uri
        else:
            entity_label = os.path.basename(nt_path)

    if not triples:
        raise ValueError(f"No triples found in {nt_path}")

    return entity_label, triples


# =========================
# Tree-of-Thoughts core
# =========================

class TreeOfThoughts:
    def __init__(
        self,
        llm: Llama32Chat,
        input_seq: str,
        get_thought_gen_prompt: Callable[[str, str], str],
        get_state_eval_prompt: Callable[[str, List[str]], str],
        heuristic_calculator: Callable[[List[str], List[str]], List[float]],
        num_triples: int,
    ):
        """
        llm: Llama32Chat instance
        input_seq: a global description string (e.g., concatenation of all triples)
        get_thought_gen_prompt(input_seq, state) -> prompt string
        get_state_eval_prompt(input_seq, states) -> prompt string
        heuristic_calculator(states, state_evals) -> List[float]
        num_triples: total number of triples (for index range checking)
        """
        self.llm = llm
        self.input_seq = input_seq
        self.root = TreeNode(state="", thought="")
        self.n_steps = 5                 # number of triple additions
        self.thought_gen_strategy = "sample"
        self.n_candidates = 5            # number of candidate thoughts per node
        self.state_eval_strategy = "vote"
        self.n_evals = 3                 # number of evaluation samples
        self.heuristic_calculator = heuristic_calculator
        self.breadth_limit = 3           # beam width
        self.get_thought_gen_prompt = get_thought_gen_prompt
        self.get_state_eval_prompt = get_state_eval_prompt
        self.stop_strings: Optional[List[str]] = None  # optional list of stop substrings
        self.num_triples = num_triples   # new: used to filter invalid indices

    def validate_state(self, state: str) -> bool:
        """
        Validate that a state has no duplicate indices and all are within valid range.
        """
        if not state.strip():
            return True  # Empty state is valid (root)
        
        try:
            ids = [int(x) for x in state.strip().splitlines() if x.strip().isdigit()]
            # Check for duplicates
            if len(ids) != len(set(ids)):
                return False
            # Check valid range
            if not all(1 <= i <= self.num_triples for i in ids):
                return False
            return True
        except (ValueError, TypeError):
            return False

    def chat_completions(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        n: int = 1,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generic wrapper to query LLaMA-3.2-3B-Instruct in a ChatGPT-like way.
        """
        messages = [{"role": "user", "content": prompt}]
        raw_outputs = self.llm.chat(
            messages=messages,
            temperature=temperature,
            max_new_tokens=max_tokens,
            n=n,
        )
        outputs: List[str] = []
        for text in raw_outputs:
            if stop:
                for s in stop:
                    if s:
                        idx = text.find(s)
                        if idx != -1:
                            text = text[:idx]
            outputs.append(text.strip())
        return outputs

    def thought_generator(self, state: str, stop_string: Optional[List[str]] = None) -> List[str]:
        """
        Generate candidate thoughts for a given state.

        In this entity summarization setting, each thought is a triple index (as string).
        """
        prompt = self.get_thought_gen_prompt(self.input_seq, state)
        raw_thoughts = self.chat_completions(
            prompt=prompt,
            temperature=0.8,  # Higher temperature for diverse thought generation
            n=self.n_candidates,
            stop=stop_string,
        )

        thought_ids: List[str] = []
        for txt in raw_thoughts:
            tid = extract_first_int(txt)
            # keep only valid indices within [1, num_triples]
            if tid is not None and 1 <= tid <= self.num_triples:
                thought_ids.append(str(tid))

        # Remove duplicates while preserving order
        unique_thoughts = list(dict.fromkeys(thought_ids))
        return unique_thoughts

    def state_evaluator(self, states: List[str]) -> List[float]:
        if self.state_eval_strategy == 'vote':
            prompt = self.get_state_eval_prompt(self.input_seq, states)
            state_evals = self.chat_completions(prompt, temperature=0.3, n=self.n_evals)  # Lower temp for consistent evaluation

            print("\n[DEBUG] Raw LLM evaluation outputs:")
            for i, s in enumerate(state_evals):
                print(f"--- sample {i} ---")
                print(s)
                print("---------------")

            vote_results = self.heuristic_calculator(states, state_evals)
            return vote_results

    def bfs(self, verbose: bool = True) -> str:
        """
        Breadth-first Tree-of-Thought search with pruning.

        Returns the best state's string (newline-separated triple indices).
        """
        queue = deque()
        queue.append(self.root)

        for step in range(1, self.n_steps + 1):
            if verbose:
                print(f"Step {step} / {self.n_steps}")
                print("---------------")

            current_layer_size = len(queue)
            if current_layer_size == 0:
                if verbose:
                    print("Queue is empty; stopping early.")
                break

            # Expand all nodes in the current layer
            new_nodes = deque()
            for i in range(current_layer_size):
                node = queue.popleft()

                if verbose:
                    print(f"Expanding node {i + 1} in this layer.")
                    if node.state:
                        print("Current state (triple ids):")
                        print(node.state)
                    else:
                        print("<root state: empty>")

                thoughts = self.thought_generator(
                    state=node.state,
                    stop_string=self.stop_strings,
                )

                # Parse current state's triple IDs to avoid duplicates
                existing_ids = set()
                if node.state.strip():
                    existing_ids = {
                        int(x)
                        for x in node.state.strip().splitlines()
                        if x.strip().isdigit()
                    }

                if verbose:
                    print(f"Generated {len(thoughts)} thought candidates: {thoughts}")

                children_created = 0
                for t_str in thoughts:
                    try:
                        t_id = int(t_str)
                    except ValueError:
                        continue

                    if t_id in existing_ids:
                        # Skip duplicates
                        continue

                    if node.state == "":
                        new_state = t_str
                    else:
                        new_state = node.state + "\n" + t_str

                    # Validate the new state before creating child
                    if not self.validate_state(new_state):
                        if verbose:
                            print(f"WARNING: Invalid state generated, skipping: {new_state}")
                        continue

                    child = TreeNode(
                        state=new_state, 
                        thought=t_str,
                        depth=node.depth + 1,
                        parent=node
                    )
                    node.children.append(child)
                    new_nodes.append(child)
                    children_created += 1

                if verbose:
                    print(f"Children created from this node: {children_created}")
                    if children_created == 0:
                        print("WARNING: No valid children created for this node - branch exhausted")
                    print("---")

                # Note: If no children created, this branch is exhausted (don't re-add parent)

            queue = new_nodes

            if len(queue) == 0:
                if verbose:
                    print("No nodes left after expansion; breaking.")
                break

            # Evaluate all states in the queue
            if verbose:
                print("Evaluating states in the queue...")

            states = [node.state for node in queue]
            values = self.state_evaluator(states=states)
            for node, val in zip(queue, values):
                node.value = val

            if verbose:
                for idx, node in enumerate(queue):
                    print(f"State {idx}: value = {node.value:.4f}, depth = {node.depth}, triples = {len(node.state.splitlines()) if node.state else 0}")
                print("Pruning...")

            # Sort by value descending and prune
            sorted_nodes = sorted(queue, key=lambda n: n.value, reverse=True)
            if step == self.n_steps:
                # Last step: keep only the best state
                top_nodes = sorted_nodes[:1]
            else:
                top_nodes = sorted_nodes[: self.breadth_limit]

            keep_states = set(node.state for node in top_nodes)
            new_queue = deque()
            for node in queue:
                if node.state in keep_states:
                    new_queue.append(node)

            queue = new_queue

            if verbose:
                print(f"Queue size after pruning: {len(queue)}")
                print("~~~")

        if not queue:
            if verbose:
                print("Search finished with empty queue; returning root state.")
            return self.root.state

        # Best node is the one with the highest value
        best_node = max(queue, key=lambda n: n.value)
        if verbose:
            print("\n" + "="*50)
            print("Search finished.")
            print("Best state summary:")
            print(f"  - Value: {best_node.value:.4f}")
            print(f"  - Depth: {best_node.depth}")
            print(f"  - Triple count: {len(best_node.state.splitlines()) if best_node.state else 0}")
            print("  - Selected triple IDs:")
            print(f"    {best_node.state}")
            print("="*50)

        return best_node.state


# =========================
# Prompt factories
# =========================

def make_entity_thought_gen_prompt(
    entity_label: str,
    all_triples: List[str],
    max_summary_len: int,
) -> Callable[[str, str], str]:
    """
    Returns get_thought_gen_prompt(input_seq, state) -> prompt string.

    state: newline-separated triple indices already selected, e.g. "2\n5".
    """

    def _inner(input_seq: str, state: str) -> str:
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
    Returns get_state_eval_prompt(input_seq, states) -> prompt string.

    states: list of newline-separated triple-id strings.
    """

    def _inner(input_seq: str, states: List[str]) -> str:
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


# =========================
# Heuristic aggregator
# =========================

def entity_heuristic_calculator(
    states: List[str],
    state_evals: List[str],
    w_relatedness: float = 0.4,
    w_informativeness: float = 0.4,
    w_coverage: float = 0.2,
) -> List[float]:
    n_states = len(states)
    agg = [{"relatedness": 0.0, "informativeness": 0.0, "coverage": 0.0}
           for _ in range(n_states)]
    n_samples = 0

    for raw in state_evals:
        raw = raw.strip()
        if not raw:
            continue

        # (optional salvage) find JSON between [ ... ]
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

    factor = 1.0 / n_samples
    final_values = []
    for i in range(n_states):
        r   = agg[i]["relatedness"]     * factor
        inf = agg[i]["informativeness"] * factor
        cov = agg[i]["coverage"]        * factor
        score = w_relatedness * r + w_informativeness * inf + w_coverage * cov
        final_values.append(score)

    return final_values


# =========================
# CLI + main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Tree-of-Thought Entity Summarization with LLaMA-3.2-3B-Instruct for a single .nt description"
    )
    parser.add_argument(
        "--nt",
        required=True,
        help="Path to entity description N-Triples file, e.g., dbpedia/1/1_desc.nt",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g., dbpedia, lmdb, faces",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model ID (default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--max-summary-len",
        type=int,
        default=5,
        help="Maximum number of triples in the summary (number of ToT steps)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=5,
        help="Number of thought candidates (triple indices) per node",
    )
    parser.add_argument(
        "--n-evals",
        type=int,
        default=3,
        help="Number of evaluation votes per state (for robustness)",
    )
    parser.add_argument(
        "--breadth-limit",
        type=int,
        default=3,
        help="Beam width (number of states kept at each level)",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose search output",
    )

    args = parser.parse_args()

    entity_label, all_triples = load_entity_description_from_nt(args.nt)
    print(f"Loaded {len(all_triples)} triples from {args.nt}")
    print(f"Entity label: {entity_label}")

    # Global input sequence for prompts (can be all triples)
    input_seq = "\n".join(all_triples)

    # Build LLaMA client
    llm = Llama32Chat(
        model_id=args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Prompt factories
    get_thought_gen_prompt = make_entity_thought_gen_prompt(
        entity_label=entity_label,
        all_triples=all_triples,
        max_summary_len=args.max_summary_len,
    )

    get_state_eval_prompt = make_entity_state_eval_prompt(
        entity_label=entity_label,
        all_triples=all_triples,
    )

    # Tree-of-Thoughts
    tot = TreeOfThoughts(
        llm=llm,
        input_seq=input_seq,
        get_thought_gen_prompt=get_thought_gen_prompt,
        get_state_eval_prompt=get_state_eval_prompt,
        heuristic_calculator=entity_heuristic_calculator,
        num_triples=len(all_triples),
    )

    tot.n_steps = args.max_summary_len
    tot.n_candidates = args.n_candidates
    tot.n_evals = args.n_evals
    tot.breadth_limit = args.breadth_limit

    best_state = tot.bfs(verbose=not args.no_verbose)
    best_triples = decode_state_to_triples(best_state, all_triples)

    print("\n=== Final selected summary triples ===")
    for t in best_triples:
        print("-", t)

    # =============== NEW: save summary to *_topK.nt ===============

    in_dir = os.path.dirname(args.nt)
    in_base = os.path.basename(args.nt)

    # Build output filename: 1_top5.nt / 1_top10.nt / etc.
    root, ext = os.path.splitext(in_dir)
    root_split = root.split("/")
    #print(root_split)
    entity_id = root_split[-1]
    if not ext:
        ext = ".nt"
    out_base = f"{args.dataset}/{entity_id}/{entity_id}_top{args.max_summary_len}{ext}"

    # Output directory under "tot-results", mirroring the input subdirectory
    # Example: "tot-results/dbpedia/1"
    out_dir = "tot-results-llama"

    # Create directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)

    # Final output path, e.g. "tot-results/dbpedia/1/1_top5.nt"
    out_path = os.path.join(out_dir, out_base)
    parent_dir = os.path.dirname(out_path)
    os.makedirs(parent_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for triple in best_triples:
            f.write(triple.rstrip() + "\n")

    print(f"\nSaved summary ({len(best_triples)} triples) to: {out_path}")


if __name__ == "__main__":
    main()