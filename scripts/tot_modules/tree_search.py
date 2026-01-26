#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-of-Thoughts Search Algorithm
"""

from collections import deque
from typing import Callable, List, Dict, Optional

from .tree_node import TreeNode
from .llm_wrapper import Llama32Chat
from .utils import extract_first_int


class TreeOfThoughts:
    """
    Tree-of-Thoughts search engine for entity summarization.
    
    Implements breadth-first search with beam pruning and LLM-based
    thought generation and state evaluation.
    """

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
        Initialize Tree-of-Thoughts search.
        
        Args:
            llm: LLM wrapper instance
            input_seq: Global description string (concatenation of all triples)
            get_thought_gen_prompt: Function to generate thought generation prompts
            get_state_eval_prompt: Function to generate state evaluation prompts
            heuristic_calculator: Function to aggregate evaluation scores
            num_triples: Total number of triples (for validation)
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
        self.stop_strings: Optional[List[str]] = None
        self.num_triples = num_triples

    def validate_state(self, state: str) -> bool:
        """
        Validate that a state has no duplicate indices and all are within valid range.
        
        Args:
            state: State string to validate
            
        Returns:
            True if state is valid, False otherwise
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
        Query LLM with chat interface.
        
        Args:
            prompt: Prompt string
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n: Number of completions
            stop: Stop strings for early termination
            
        Returns:
            List of generated text strings
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
        Generate candidate thoughts (triple indices) for a given state.
        
        Args:
            state: Current state string
            stop_string: Optional stop strings
            
        Returns:
            List of unique thought strings (triple indices)
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
        """
        Evaluate a list of states using vote-based aggregation.
        
        Args:
            states: List of state strings to evaluate
            
        Returns:
            List of evaluation scores (one per state)
        """
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
        Perform breadth-first Tree-of-Thought search with beam pruning.
        
        Args:
            verbose: Whether to print detailed progress
            
        Returns:
            Best state string (newline-separated triple indices)
        """
        queue = deque()
        queue.append(self.root)

        for step in range(1, self.n_steps + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Step {step} / {self.n_steps}")
                print('='*60)

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
                    print(f"\n--- Expanding node {i + 1}/{current_layer_size} ---")
                    if node.state:
                        print(f"Current state: {node}")
                        print(f"Triple IDs: {node.state}")
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
                    print(f"Children created: {children_created}")
                    if children_created == 0:
                        print("WARNING: No valid children created - branch exhausted")

                # Note: If no children created, this branch is exhausted (don't re-add parent)

            queue = new_nodes

            if len(queue) == 0:
                if verbose:
                    print("\nNo nodes left after expansion; breaking.")
                break

            # Evaluate all states in the queue
            if verbose:
                print(f"\n--- Evaluating {len(queue)} states ---")

            states = [node.state for node in queue]
            values = self.state_evaluator(states=states)
            for node, val in zip(queue, values):
                node.value = val

            if verbose:
                print("\nState evaluations:")
                for idx, node in enumerate(queue):
                    print(f"  State {idx}: value={node.value:.4f}, depth={node.depth}, triples={len(node.get_triple_ids())}")

            # Sort by value descending and prune
            if verbose:
                print(f"\nPruning to top {self.breadth_limit if step < self.n_steps else 1}...")
            
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

        if not queue:
            if verbose:
                print("\nSearch finished with empty queue; returning root state.")
            return self.root.state

        # Best node is the one with the highest value
        best_node = max(queue, key=lambda n: n.value)
        if verbose:
            print("\n" + "="*60)
            print("SEARCH COMPLETE")
            print("="*60)
            print("Best state summary:")
            print(f"  Value: {best_node.value:.4f}")
            print(f"  Depth: {best_node.depth}")
            print(f"  Triple count: {len(best_node.get_triple_ids())}")
            print(f"  Selected triple IDs: {best_node.state}")
            print("="*60)

        return best_node.state
    
    def __repr__(self) -> str:
        return (f"TreeOfThoughts(n_steps={self.n_steps}, "
                f"n_candidates={self.n_candidates}, "
                f"breadth_limit={self.breadth_limit})")
