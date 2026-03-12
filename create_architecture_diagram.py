#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate architecture diagram for ToT4ES (Tree-of-Thought Entity Summarization)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Color scheme
color_input = '#E8F4F8'
color_llm = '#FFE5CC'
color_tot = '#D4E6F1'
color_eval = '#E8DAEF'
color_output = '#D5F4E6'
color_arrow = '#34495E'

# Title
ax.text(5, 13.5, 'ToT4ES Architecture: Tree-of-Thought Entity Summarization', 
        fontsize=20, fontweight='bold', ha='center', va='top')
ax.text(5, 13.0, 'Using LLaMA-3.2-3B-Instruct for RDF Triple Selection', 
        fontsize=14, ha='center', va='top', style='italic', color='#555')

# ===== INPUT LAYER =====
input_box = FancyBboxPatch((0.5, 11), 2, 1.2, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='#2C3E50', facecolor=color_input, linewidth=2)
ax.add_patch(input_box)
ax.text(1.5, 11.8, 'Input', fontsize=12, fontweight='bold', ha='center')
ax.text(1.5, 11.5, 'N-Triples File', fontsize=10, ha='center')
ax.text(1.5, 11.2, '(.nt format)', fontsize=9, ha='center', style='italic')

# Entity Description Box
entity_box = FancyBboxPatch((3.5, 11), 3, 1.2, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='#2C3E50', facecolor=color_input, linewidth=2)
ax.add_patch(entity_box)
ax.text(5, 11.8, 'Entity Description', fontsize=11, fontweight='bold', ha='center')
ax.text(5, 11.5, '• Entity Label Extraction', fontsize=9, ha='center')
ax.text(5, 11.25, '• Triple Parsing & Indexing', fontsize=9, ha='center')

# ===== LLM LAYER =====
llm_box = FancyBboxPatch((0.5, 9), 6, 1.4, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='#D35400', facecolor=color_llm, linewidth=3)
ax.add_patch(llm_box)
ax.text(3.5, 10.1, 'LLM: Llama32Chat Wrapper', fontsize=12, fontweight='bold', ha='center')
ax.text(3.5, 9.75, 'Model: meta-llama/Llama-3.2-3B-Instruct', fontsize=10, ha='center')
ax.text(3.5, 9.45, '• HuggingFace Transformers Pipeline', fontsize=9, ha='center')
ax.text(3.5, 9.2, '• Chat-based Interface (User/Assistant)', fontsize=9, ha='center')

# ===== TREE-OF-THOUGHTS CORE =====
tot_main_box = FancyBboxPatch((0.5, 5.5), 6, 3, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#1F618D', facecolor=color_tot, linewidth=3)
ax.add_patch(tot_main_box)
ax.text(3.5, 8.2, 'Tree-of-Thoughts Engine (BFS)', fontsize=13, fontweight='bold', ha='center')

# ToT Components
# 1. Thought Generator
thought_box = FancyBboxPatch((0.8, 6.8), 2.5, 1.2, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='#1F618D', facecolor='#EBF5FB', linewidth=1.5)
ax.add_patch(thought_box)
ax.text(2.05, 7.7, 'Thought Generator', fontsize=10, fontweight='bold', ha='center')
ax.text(2.05, 7.45, '• Sample n candidates', fontsize=8, ha='center')
ax.text(2.05, 7.2, '• Select next triple index', fontsize=8, ha='center')
ax.text(2.05, 6.95, '• Avoid duplicates', fontsize=8, ha='center')

# 2. State Evaluator
eval_box = FancyBboxPatch((3.7, 6.8), 2.5, 1.2, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='#7D3C98', facecolor='#F4ECF7', linewidth=1.5)
ax.add_patch(eval_box)
ax.text(4.95, 7.7, 'State Evaluator', fontsize=10, fontweight='bold', ha='center')
ax.text(4.95, 7.45, '• Vote strategy (n evals)', fontsize=8, ha='center')
ax.text(4.95, 7.2, '• Multi-criteria scoring', fontsize=8, ha='center')
ax.text(4.95, 6.95, '• JSON-based output', fontsize=8, ha='center')

# 3. Search Algorithm
search_box = FancyBboxPatch((0.8, 5.7), 5.4, 0.9, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='#1F618D', facecolor='#D6EAF8', linewidth=1.5)
ax.add_patch(search_box)
ax.text(3.5, 6.35, 'BFS with Beam Search Pruning', fontsize=10, fontweight='bold', ha='center')
ax.text(3.5, 6.0, 'Parameters: max_summary_len, n_candidates, breadth_limit', fontsize=8, ha='center')
ax.text(3.5, 5.8, 'Iteratively expand → evaluate → prune → repeat', fontsize=8, ha='center', style='italic')

# ===== EVALUATION CRITERIA =====
criteria_box = FancyBboxPatch((7.2, 7), 2.4, 3.5, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='#7D3C98', facecolor=color_eval, linewidth=2)
ax.add_patch(criteria_box)
ax.text(8.4, 10.2, 'Evaluation Criteria', fontsize=11, fontweight='bold', ha='center')

# Criterion 1
ax.text(8.4, 9.7, '1. Relatedness', fontsize=10, fontweight='bold', ha='center', color='#C0392B')
ax.text(7.4, 9.4, '• Triple centrality', fontsize=8, ha='left')
ax.text(7.4, 9.15, '• Core predicates', fontsize=8, ha='left')
ax.text(7.4, 8.9, '• Entity relevance', fontsize=8, ha='left')

# Criterion 2
ax.text(8.4, 8.5, '2. Informativeness', fontsize=10, fontweight='bold', ha='center', color='#1F618D')
ax.text(7.4, 8.2, '• Low freq_property', fontsize=8, ha='left')
ax.text(7.4, 7.95, '• Low freq_value', fontsize=8, ha='left')
ax.text(7.4, 7.7, '• High type_depth', fontsize=8, ha='left')

# Criterion 3
ax.text(8.4, 7.3, '3. Coverage/Diversity', fontsize=10, fontweight='bold', ha='center', color='#27AE60')
ax.text(7.4, 7.0, '• Diverse predicates', fontsize=8, ha='left')
ax.text(7.4, 6.75, '• Varied entity roles', fontsize=8, ha='left')
ax.text(7.4, 6.5, '• Low value similarity', fontsize=8, ha='left')

# Heuristic Calculator
heuristic_box = FancyBboxPatch((7.2, 5.5), 2.4, 0.8, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='#7D3C98', facecolor='#FADBD8', linewidth=1.5)
ax.add_patch(heuristic_box)
ax.text(8.4, 6.05, 'Heuristic Calculator', fontsize=9, fontweight='bold', ha='center')
ax.text(8.4, 5.75, 'Weighted sum: 0.4R + 0.4I + 0.2C', fontsize=8, ha='center')

# ===== PROMPT ENGINEERING =====
prompt_box = FancyBboxPatch((0.5, 3.5), 6, 1.6, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='#E67E22', facecolor='#FEF5E7', linewidth=2)
ax.add_patch(prompt_box)
ax.text(3.5, 4.9, 'Prompt Engineering', fontsize=12, fontweight='bold', ha='center')

# Thought Gen Prompt
ax.text(2, 4.5, 'Thought Generation Prompt:', fontsize=9, fontweight='bold', ha='center')
ax.text(2, 4.25, '• Current selected triples', fontsize=8, ha='center')
ax.text(2, 4.0, '• Remaining candidates', fontsize=8, ha='center')
ax.text(2, 3.75, '• Output: single triple index', fontsize=8, ha='center')

# State Eval Prompt
ax.text(5, 4.5, 'State Evaluation Prompt:', fontsize=9, fontweight='bold', ha='center')
ax.text(5, 4.25, '• Multiple candidate summaries', fontsize=8, ha='center')
ax.text(5, 4.0, '• Score all 3 criteria (0-1)', fontsize=8, ha='center')
ax.text(5, 3.75, '• Output: JSON array', fontsize=8, ha='center')

# ===== OUTPUT LAYER =====
output_box = FancyBboxPatch((0.5, 1.5), 6, 1.6, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='#27AE60', facecolor=color_output, linewidth=3)
ax.add_patch(output_box)
ax.text(3.5, 2.9, 'Output Generation', fontsize=12, fontweight='bold', ha='center')
ax.text(3.5, 2.55, 'Best Summary Selection (highest value node)', fontsize=10, ha='center')
ax.text(3.5, 2.25, '• Decode state to triples', fontsize=9, ha='center')
ax.text(3.5, 1.95, '• Save to: tot-results-llama/{dataset}/{entity_id}/{entity_id}_top{k}.nt', fontsize=8, ha='center')
ax.text(3.5, 1.65, '• Format: N-Triples (.nt)', fontsize=9, ha='center')

# ===== PARAMETERS BOX =====
params_box = FancyBboxPatch((7.2, 1.5), 2.4, 3.6, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='#2C3E50', facecolor='#F8F9F9', linewidth=2)
ax.add_patch(params_box)
ax.text(8.4, 4.9, 'Key Parameters', fontsize=11, fontweight='bold', ha='center')

param_text = [
    ('--max-summary-len', '5', 'Max triples'),
    ('--n-candidates', '5', 'Thought samples'),
    ('--n-evals', '3', 'Vote samples'),
    ('--breadth-limit', '3', 'Beam width'),
]

y_pos = 4.5
for param, default, desc in param_text:
    ax.text(7.4, y_pos, f'{param}', fontsize=8, ha='left', fontweight='bold', family='monospace')
    ax.text(7.4, y_pos - 0.2, f'  default: {default} ({desc})', fontsize=7, ha='left', style='italic')
    y_pos -= 0.5

ax.text(8.4, 2.5, 'Datasets Supported:', fontsize=9, fontweight='bold', ha='center')
ax.text(8.4, 2.2, '• DBpedia', fontsize=8, ha='center')
ax.text(8.4, 1.95, '• LMDB', fontsize=8, ha='center')
ax.text(8.4, 1.7, '• FACES', fontsize=8, ha='center')

# ===== ARROWS - Data Flow =====
def draw_arrow(x1, y1, x2, y2, color=color_arrow, style='solid', width=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=width, color=color, linestyle=style,
                           connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)

# Input to Entity Description
draw_arrow(2.5, 11.6, 3.5, 11.6)

# Entity to LLM
draw_arrow(3.5, 11, 3.5, 10.4, width=2.5)

# LLM to ToT
draw_arrow(3.5, 9, 3.5, 8.5, width=2.5)

# ToT to Prompt Engineering (bidirectional)
draw_arrow(2, 5.5, 2, 5.1, style='dashed')
draw_arrow(5, 5.1, 5, 5.5, style='dashed')

# Criteria to State Evaluator (connection)
draw_arrow(7.2, 7.4, 6.2, 7.4, style='dotted', width=1.5)

# Heuristic to ToT
draw_arrow(7.2, 5.9, 6.5, 6.2, style='dotted', width=1.5)

# ToT to Output
draw_arrow(3.5, 5.5, 3.5, 3.1, width=2.5)

# ===== WORKFLOW ANNOTATIONS =====
ax.text(0.3, 10.2, '①', fontsize=14, fontweight='bold', 
        bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))
ax.text(0.3, 8.7, '②', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))
ax.text(0.3, 7, '③', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))
ax.text(0.3, 4.2, '④', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))
ax.text(0.3, 2.3, '⑤', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))

# ===== LEGEND =====
ax.text(0.5, 0.9, 'Workflow:', fontsize=10, fontweight='bold')
ax.text(0.5, 0.6, '① Load entity N-Triples → ② Initialize LLM → ③ Tree-of-Thought search (BFS)', fontsize=8)
ax.text(0.5, 0.35, '④ Generate thoughts & evaluate states → ⑤ Select best summary & save output', fontsize=8)

# Footer
ax.text(5, 0.1, '© ToT4ES: Tree-of-Thought for Entity Summarization | LLaMA-3.2-3B-Instruct', 
        fontsize=8, ha='center', style='italic', color='#7F8C8D')

# Adjust layout and save
plt.tight_layout()
plt.savefig('/home/asepff/Documents/Github/dice/ToT4ES/ToT4ES_Architecture.jpg', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved to: ToT4ES_Architecture.jpg")
plt.close()
