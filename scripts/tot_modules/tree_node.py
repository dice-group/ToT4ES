#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree Node Structure
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TreeNode:
    """
    Represents a node in the Tree-of-Thought search.
    
    Attributes:
        state: Newline-separated triple indices (as strings), e.g., "2\n5\n9"
        thought: The last triple index added, e.g., "5"
        value: Scalar score of this state from evaluation
        depth: Depth in the search tree (0 for root)
        parent: Reference to parent node (None for root)
        children: List of child nodes
    """
    state: str
    thought: str
    value: float = 0.0
    depth: int = 0
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)
    
    def get_triple_ids(self) -> List[int]:
        """Extract triple IDs from state as integers."""
        if not self.state.strip():
            return []
        return [
            int(x) 
            for x in self.state.strip().splitlines() 
            if x.strip().isdigit()
        ]
    
    def get_path_from_root(self) -> List["TreeNode"]:
        """Get the path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def __repr__(self) -> str:
        triple_count = len(self.get_triple_ids())
        return f"TreeNode(depth={self.depth}, triples={triple_count}, value={self.value:.4f})"
