"""
Coreset Selection for Instruction Tuning - Processing Pipeline

This module contains the processing scripts for the diversity-guided
dataset pruning experiment:

1. 01_prepare_data.py - Data preparation and embedding generation
2. 02_select_subsets.py - Diversity-based subset selection
3. 03_finetune.py - Fine-tuning with LoRA (Unsloth-optimized)
4. 04_evaluate.py - AlpacaEval 2.0 and local benchmark evaluation
5. 05_generate_report.py - Report and visualization generation
"""

__version__ = "0.1.0"
