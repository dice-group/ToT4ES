#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit Tests for ToT Modules

Run with: python -m pytest tests/test_tot_modules.py -v
Or simply: python tests/test_tot_modules.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import unittest
from tot_modules.tree_node import TreeNode
from tot_modules.utils import extract_first_int, decode_state_to_triples
from tot_modules.heuristic import entity_heuristic_calculator


class TestTreeNode(unittest.TestCase):
    """Test TreeNode functionality."""
    
    def test_create_node(self):
        """Test basic node creation."""
        node = TreeNode(state="1\n2\n3", thought="3")
        self.assertEqual(node.state, "1\n2\n3")
        self.assertEqual(node.thought, "3")
        self.assertEqual(node.value, 0.0)
        self.assertEqual(node.depth, 0)
    
    def test_get_triple_ids(self):
        """Test extracting triple IDs from state."""
        node = TreeNode(state="1\n2\n3", thought="3")
        ids = node.get_triple_ids()
        self.assertEqual(ids, [1, 2, 3])
    
    def test_get_triple_ids_empty(self):
        """Test extracting IDs from empty state."""
        node = TreeNode(state="", thought="")
        ids = node.get_triple_ids()
        self.assertEqual(ids, [])
    
    def test_get_path_from_root(self):
        """Test getting path from root to node."""
        root = TreeNode(state="", thought="")
        child1 = TreeNode(state="1", thought="1", depth=1, parent=root)
        child2 = TreeNode(state="1\n2", thought="2", depth=2, parent=child1)
        
        path = child2.get_path_from_root()
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], root)
        self.assertEqual(path[1], child1)
        self.assertEqual(path[2], child2)
    
    def test_node_repr(self):
        """Test node string representation."""
        node = TreeNode(state="1\n2", thought="2", value=0.85, depth=2)
        repr_str = repr(node)
        self.assertIn("depth=2", repr_str)
        self.assertIn("triples=2", repr_str)
        self.assertIn("0.85", repr_str)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_extract_first_int(self):
        """Test extracting first integer from text."""
        self.assertEqual(extract_first_int("The answer is 42"), 42)
        self.assertEqual(extract_first_int("123 and 456"), 123)
        self.assertEqual(extract_first_int("No numbers"), None)
        self.assertEqual(extract_first_int(""), None)
    
    def test_decode_state_to_triples(self):
        """Test decoding state to triples."""
        triples = ["triple1", "triple2", "triple3", "triple4"]
        state = "1\n3"
        result = decode_state_to_triples(state, triples)
        self.assertEqual(result, ["triple1", "triple3"])
    
    def test_decode_empty_state(self):
        """Test decoding empty state."""
        triples = ["triple1", "triple2"]
        state = ""
        result = decode_state_to_triples(state, triples)
        self.assertEqual(result, [])
    
    def test_decode_state_out_of_range(self):
        """Test decoding state with invalid indices."""
        triples = ["triple1", "triple2"]
        state = "1\n5"  # 5 is out of range
        result = decode_state_to_triples(state, triples)
        self.assertEqual(result, ["triple1"])  # Only valid index


class TestHeuristic(unittest.TestCase):
    """Test heuristic calculator."""
    
    def test_basic_aggregation(self):
        """Test basic score aggregation."""
        states = ["1", "2"]
        evals = [
            '[{"idx":0,"relatedness":0.8,"informativeness":0.6,"coverage":0.4}, '
            '{"idx":1,"relatedness":0.5,"informativeness":0.7,"coverage":0.8}]'
        ]
        scores = entity_heuristic_calculator(states, evals)
        
        # Check we got 2 scores
        self.assertEqual(len(scores), 2)
        
        # Check scores are in valid range
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_multi_sample_aggregation(self):
        """Test aggregation across multiple evaluation samples."""
        states = ["1"]
        evals = [
            '[{"idx":0,"relatedness":1.0,"informativeness":1.0,"coverage":1.0}]',
            '[{"idx":0,"relatedness":0.0,"informativeness":0.0,"coverage":0.0}]',
        ]
        scores = entity_heuristic_calculator(states, evals)
        
        # Average should be 0.5
        self.assertAlmostEqual(scores[0], 0.5, places=2)
    
    def test_fallback_on_parse_failure(self):
        """Test fallback when all parsing fails."""
        states = ["1", "2"]
        evals = ["invalid json", "also invalid"]
        scores = entity_heuristic_calculator(states, evals)
        
        # Should return neutral scores (0.5)
        self.assertEqual(scores, [0.5, 0.5])
    
    def test_weighted_combination(self):
        """Test weighted combination of criteria."""
        states = ["1"]
        # relatedness=1.0, informativeness=0.0, coverage=0.0
        # Expected: 0.4*1.0 + 0.4*0.0 + 0.2*0.0 = 0.4
        evals = [
            '[{"idx":0,"relatedness":1.0,"informativeness":0.0,"coverage":0.0}]'
        ]
        scores = entity_heuristic_calculator(states, evals)
        self.assertAlmostEqual(scores[0], 0.4, places=2)


class TestIntegration(unittest.TestCase):
    """Integration tests for combined functionality."""
    
    def test_full_workflow_simulation(self):
        """Test a simplified version of the full workflow."""
        # Create root node
        root = TreeNode(state="", thought="", depth=0)
        
        # Create children
        child1 = TreeNode(state="1", thought="1", depth=1, parent=root)
        child2 = TreeNode(state="2", thought="2", depth=1, parent=root)
        root.children = [child1, child2]
        
        # Simulate evaluation
        states = [child1.state, child2.state]
        evals = [
            '[{"idx":0,"relatedness":0.9,"informativeness":0.8,"coverage":0.7}, '
            '{"idx":1,"relatedness":0.5,"informativeness":0.6,"coverage":0.8}]'
        ]
        scores = entity_heuristic_calculator(states, evals)
        
        # Assign scores
        child1.value = scores[0]
        child2.value = scores[1]
        
        # Select best
        best = max([child1, child2], key=lambda n: n.value)
        
        # child1 should have higher score
        self.assertEqual(best, child1)
        self.assertGreater(child1.value, child2.value)


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    print("="*70)
    print("Running ToT Modules Unit Tests")
    print("="*70)
    run_tests()
