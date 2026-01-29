"""
Tree-of-Thought Entity Summarization - Modular Components
"""

from .tree_node import TreeNode
from .llm_wrapper import Llama32Chat
from .tree_search import TreeOfThoughts
from .prompt_factory import (
    make_entity_thought_gen_prompt,
    make_entity_state_eval_prompt
)
from .heuristic import entity_heuristic_calculator
from .utils import (
    extract_first_int,
    decode_state_to_triples,
    load_entity_description_from_nt
)

# Task-decomposed components
try:
    from .task_prompts import (
        make_relatedness_prompt,
        make_informativeness_prompt,
        make_diversity_prompt,
        make_combined_evaluation_prompt,
    )
    from .task_decomposed_search import TaskDecomposedToT
    
    _task_decomposed_available = True
except ImportError:
    _task_decomposed_available = False

# Semantic enhancement components
try:
    from .semantic_analyzer import SemanticAnalyzer
    from .semantic_prompts import (
        make_semantic_relatedness_prompt,
        make_semantic_informativeness_prompt,
        make_semantic_diversity_prompt,
    )
    
    _semantic_available = True
except ImportError:
    _semantic_available = False

__all__ = [
    'TreeNode',
    'Llama32Chat',
    'TreeOfThoughts',
    'make_entity_thought_gen_prompt',
    'make_entity_state_eval_prompt',
    'entity_heuristic_calculator',
    'extract_first_int',
    'decode_state_to_triples',
    'load_entity_description_from_nt',
]

if _task_decomposed_available:
    __all__.extend([
        'make_relatedness_prompt',
        'make_informativeness_prompt',
        'make_diversity_prompt',
        'make_combined_evaluation_prompt',
        'TaskDecomposedToT',
    ])

if _semantic_available:
    __all__.extend([
        'SemanticAnalyzer',
        'make_semantic_relatedness_prompt',
        'make_semantic_informativeness_prompt',
        'make_semantic_diversity_prompt',
    ])
