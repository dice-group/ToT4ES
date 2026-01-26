#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-Decomposed Tree Search
Implements the architecture with separate task-specific thought generators
"""

from collections import deque
from typing import Callable, List, Dict, Optional

from .tree_node import TreeNode
from .llm_wrapper import Llama32Chat
from .utils import extract_first_int


class TaskDecomposedToT:
    """
    Task-Decomposed Tree-of-Thoughts for entity summarization.
    
    Uses three separate thought generators, one for each criterion:
    - Relatedness generator
    - Informativeness generator  
    - Diversity/Coverage generator
    
    This implements the architecture shown in the diagram.
    """

    def __init__(
        self,
        llm: Llama32Chat,
        input_seq: str,
        get_relatedness_prompt: Callable[[str, str], str],
        get_informativeness_prompt: Callable[[str, str], str],
        get_diversity_prompt: Callable[[str, str], str],
        get_state_eval_prompt: Callable[[str, List[str]], str],
        heuristic_calculator: Callable[[List[str], List[str]], List[float]],
        num_triples: int,
    ):
        """
        Initialize Task-Decomposed ToT search.
        
        Args:
            llm: LLM wrapper instance
            input_seq: Global description string
            get_relatedness_prompt: Prompt function for relatedness task
            get_informativeness_prompt: Prompt function for informativeness task
            get_diversity_prompt: Prompt function for diversity task
            get_state_eval_prompt: Prompt function for state evaluation
            heuristic_calculator: Function to aggregate evaluation scores
            num_triples: Total number of triples
        """
        self.llm = llm
        self.input_seq = input_seq
        self.root = TreeNode(state="", thought="")
        self.n_steps = 5
        self.n_candidates_per_task = 2  # Generate 2 candidates per task
        self.n_evals = 3
        self.heuristic_calculator = heuristic_calculator
        self.breadth_limit = 3
        
        # Task-specific prompt generators
        self.get_relatedness_prompt = get_relatedness_prompt
        self.get_informativeness_prompt = get_informativeness_prompt
        self.get_diversity_prompt = get_diversity_prompt
        self.get_state_eval_prompt = get_state_eval_prompt
        
        self.num_triples = num_triples

    def validate_state(self, state: str) -> bool:
        """Validate state has no duplicates and valid indices."""
        if not state.strip():
            return True
        
        try:
            ids = [int(x) for x in state.strip().splitlines() if x.strip().isdigit()]
            if len(ids) != len(set(ids)):
                return False
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
        """Query LLM with chat interface."""
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

    def task_thought_generator(
        self, 
        state: str, 
        task_prompt_fn: Callable[[str, str], str],
        task_name: str,
    ) -> List[str]:
        """
        Generate thoughts using a specific task prompt.
        
        Args:
            state: Current state string
            task_prompt_fn: Task-specific prompt generator
            task_name: Name of task (for logging)
            
        Returns:
            List of unique thought strings (triple indices)
        """
        prompt = task_prompt_fn(self.input_seq, state)
        raw_thoughts = self.chat_completions(
            prompt=prompt,
            temperature=0.8,
            n=self.n_candidates_per_task,
        )

        thought_ids: List[str] = []
        for txt in raw_thoughts:
            tid = extract_first_int(txt)
            if tid is not None and 1 <= tid <= self.num_triples:
                thought_ids.append(str(tid))

        unique_thoughts = list(dict.fromkeys(thought_ids))
        return unique_thoughts

    def generate_thoughts_all_tasks(self, state: str, verbose: bool = False) -> Dict[str, List[str]]:
        """
        Generate thoughts from all three task-specific generators.
        
        Args:
            state: Current state
            verbose: Whether to print debug info
            
        Returns:
            Dictionary mapping task name to thought list
        """
        tasks = {
            "relatedness": self.get_relatedness_prompt,
            "informativeness": self.get_informativeness_prompt,
            "diversity": self.get_diversity_prompt,
        }
        
        all_thoughts = {}
        
        for task_name, prompt_fn in tasks.items():
            if verbose:
                print(f"  Generating thoughts for {task_name.upper()}...")
            
            thoughts = self.task_thought_generator(state, prompt_fn, task_name)
            all_thoughts[task_name] = thoughts
            
            if verbose:
                print(f"    {task_name}: {thoughts}")
        
        return all_thoughts

    def state_evaluator(self, states: List[str]) -> List[float]:
        """Evaluate states using vote-based aggregation."""
        prompt = self.get_state_eval_prompt(self.input_seq, states)
        state_evals = self.chat_completions(prompt, temperature=0.3, n=self.n_evals)

        print("\n[DEBUG] Raw LLM evaluation outputs:")
        for i, s in enumerate(state_evals):
            print(f"--- sample {i} ---")
            print(s)
            print("---------------")

        vote_results = self.heuristic_calculator(states, state_evals)
        return vote_results

    def bfs(self, verbose: bool = True) -> str:
        """
        Perform task-decomposed BFS search.
        
        At each step, generates candidates from three task-specific prompts,
        then evaluates and prunes based on combined criteria.
        """
        queue = deque()
        queue.append(self.root)

        for step in range(1, self.n_steps + 1):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Step {step} / {self.n_steps}")
                print('='*70)

            current_layer_size = len(queue)
            if current_layer_size == 0:
                if verbose:
                    print("Queue is empty; stopping early.")
                break

            # Expand all nodes in current layer
            new_nodes = deque()
            for i in range(current_layer_size):
                node = queue.popleft()

                if verbose:
                    print(f"\n--- Expanding node {i + 1}/{current_layer_size} ---")
                    print(f"Current state: {node}")
                    if node.state:
                        print(f"Triple IDs: {node.state}")

                # Generate thoughts from all three tasks
                all_task_thoughts = self.generate_thoughts_all_tasks(node.state, verbose)

                # Parse current state to avoid duplicates
                existing_ids = set()
                if node.state.strip():
                    existing_ids = {
                        int(x) for x in node.state.strip().splitlines()
                        if x.strip().isdigit()
                    }

                # Combine thoughts from all tasks
                all_thoughts = []
                for task_name, thoughts in all_task_thoughts.items():
                    all_thoughts.extend(thoughts)
                
                # Remove duplicates while preserving order
                all_thoughts = list(dict.fromkeys(all_thoughts))
                
                if verbose:
                    print(f"\nCombined unique thoughts: {all_thoughts}")

                # Create children
                children_created = 0
                for t_str in all_thoughts:
                    try:
                        t_id = int(t_str)
                    except ValueError:
                        continue

                    if t_id in existing_ids:
                        continue

                    new_state = t_str if node.state == "" else node.state + "\n" + t_str

                    if not self.validate_state(new_state):
                        if verbose:
                            print(f"WARNING: Invalid state, skipping: {new_state}")
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
                        print("WARNING: No valid children - branch exhausted")

            queue = new_nodes

            if len(queue) == 0:
                if verbose:
                    print("\nNo nodes left after expansion; breaking.")
                break

            # Evaluate all states
            if verbose:
                print(f"\n--- Evaluating {len(queue)} states ---")

            states = [node.state for node in queue]
            values = self.state_evaluator(states)
            for node, val in zip(queue, values):
                node.value = val

            if verbose:
                print("\nState evaluations:")
                for idx, node in enumerate(queue):
                    print(f"  State {idx}: value={node.value:.4f}, depth={node.depth}, triples={len(node.get_triple_ids())}")

            # Prune
            if verbose:
                print(f"\nPruning to top {self.breadth_limit if step < self.n_steps else 1}...")

            sorted_nodes = sorted(queue, key=lambda n: n.value, reverse=True)
            if step == self.n_steps:
                top_nodes = sorted_nodes[:1]
            else:
                top_nodes = sorted_nodes[:self.breadth_limit]

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

        # Best node
        best_node = max(queue, key=lambda n: n.value)
        if verbose:
            print("\n" + "="*70)
            print("SEARCH COMPLETE (Task-Decomposed)")
            print("="*70)
            print("Best state summary:")
            print(f"  Value: {best_node.value:.4f}")
            print(f"  Depth: {best_node.depth}")
            print(f"  Triple count: {len(best_node.get_triple_ids())}")
            print(f"  Selected triple IDs: {best_node.state}")
            print("="*70)

        return best_node.state

    def __repr__(self) -> str:
        return (f"TaskDecomposedToT(n_steps={self.n_steps}, "
                f"n_candidates_per_task={self.n_candidates_per_task}, "
                f"breadth_limit={self.breadth_limit})")
