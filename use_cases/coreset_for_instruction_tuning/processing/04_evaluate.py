#!/usr/bin/env python3
"""
Step 4: Evaluation with Local Benchmarks and LLM-as-Judge

This script:
1. Loads fine-tuned adapters
2. Runs lm-evaluation-harness benchmarks (no API needed)
3. Runs local LLM-as-Judge evaluation using Prometheus or similar
4. Computes MT-Bench style scores with local models
5. Saves comprehensive evaluation results

All evaluations use local models - no paid APIs required.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import os
import gc
import json
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_config() -> dict:
    """Load experiment configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clear_memory() -> None:
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    model_name: str
    adapter_path: str
    dataset_name: str
    selection_method: str
    n_samples: int

    # Local benchmark results (lm-eval-harness)
    local_benchmarks: Dict[str, float]
    local_benchmarks_avg: float

    # LLM-as-Judge results
    llm_judge_score: Optional[float]
    llm_judge_details: Optional[Dict]

    # Generation quality metrics
    avg_response_length: float
    response_diversity: float

    evaluation_time_seconds: float
    timestamp: str


def load_model_for_inference(adapter_path: Path, base_model_name: str):
    """Load model with adapter for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"   Loading base model: {base_model_name}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"   Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a response for a given instruction."""
    # Format prompt
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": instruction}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


def run_local_benchmarks(
    model,
    tokenizer,
    tasks: List[str],
    output_dir: Path,
) -> Dict[str, float]:
    """
    Run local benchmarks using lm-evaluation-harness.
    No API required - all evaluations run locally.
    """
    print(f"   Running lm-evaluation-harness: {tasks}")

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        # Wrap model for lm-eval
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size="auto",
            max_batch_size=8,
        )

        # Run evaluation
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=0,  # Zero-shot for instruction-tuned models
            batch_size="auto",
        )

        # Extract scores
        scores = {}
        for task in tasks:
            if task in results['results']:
                task_results = results['results'][task]
                # Get main metric for each task
                if 'acc,none' in task_results:
                    scores[task] = float(task_results['acc,none'])
                elif 'acc_norm,none' in task_results:
                    scores[task] = float(task_results['acc_norm,none'])
                elif 'exact_match,none' in task_results:
                    scores[task] = float(task_results['exact_match,none'])
                elif 'mc1' in task_results:
                    scores[task] = float(task_results['mc1'])
                elif 'acc' in task_results:
                    scores[task] = float(task_results['acc'])
                elif 'acc_norm' in task_results:
                    scores[task] = float(task_results['acc_norm'])
                # IFEval-specific metrics
                elif 'prompt_level_strict_acc,none' in task_results:
                    scores[task] = float(task_results['prompt_level_strict_acc,none'])
                elif 'inst_level_strict_acc,none' in task_results:
                    scores[task] = float(task_results['inst_level_strict_acc,none'])
            # Handle BBH (BIG-Bench Hard) which may have subtasks
            elif task == 'bbh':
                # BBH returns aggregate results
                for key in results['results']:
                    if key.startswith('bbh'):
                        bbh_results = results['results'][key]
                        if 'acc_norm,none' in bbh_results:
                            scores[task] = float(bbh_results['acc_norm,none'])
                            break
                        elif 'acc,none' in bbh_results:
                            scores[task] = float(bbh_results['acc,none'])
                            break

        return scores

    except ImportError:
        print("   Warning: lm-evaluation-harness not installed")
        print("   Install with: pip install lm-eval")
        return {}

    except Exception as e:
        print(f"   Warning: Local benchmarks failed: {e}")
        return {'error': str(e)}


def load_judge_model(config: dict):
    """Load the local LLM judge model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    judge_config = config['evaluation']['llm_judge']
    model_name = judge_config['model']

    print(f"   Loading judge model: {model_name}")

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        return model, tokenizer, model_name

    except Exception as e:
        print(f"   Warning: Failed to load {model_name}: {e}")
        print(f"   Trying fallback model...")

        fallback = judge_config.get('fallback_model')
        if fallback:
            try:
                tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    fallback,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                return model, tokenizer, fallback
            except Exception as e2:
                print(f"   Warning: Fallback also failed: {e2}")

        return None, None, None


def create_prometheus_prompt(instruction: str, response: str) -> str:
    """
    Create a Prometheus-style evaluation prompt.
    Prometheus is trained to evaluate responses on a 1-5 scale.
    """
    prompt = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubric:
[Is the response helpful, accurate, and well-structured?]
Score 1: The response is unhelpful, contains significant errors, or is poorly structured.
Score 2: The response attempts to address the instruction but has notable issues with accuracy or clarity.
Score 3: The response is moderately helpful with some minor issues in accuracy or structure.
Score 4: The response is helpful, mostly accurate, and well-structured with minor improvements possible.
Score 5: The response is excellent - highly helpful, accurate, and perfectly structured.

###Feedback:"""

    return prompt


def create_simple_judge_prompt(instruction: str, response: str) -> str:
    """
    Create a simple evaluation prompt for non-Prometheus models.
    """
    prompt = f"""You are evaluating an AI assistant's response to a user instruction.

Instruction: {instruction}

Response: {response}

Rate this response on a scale of 1-5 where:
1 = Poor (unhelpful, incorrect, or incoherent)
2 = Below Average (attempts to help but has significant issues)
3 = Average (helpful but could be improved)
4 = Good (helpful and mostly accurate)
5 = Excellent (very helpful, accurate, and well-written)

Provide your rating as a single number (1-5) followed by a brief explanation.
Rating:"""

    return prompt


def extract_score_from_response(response: str) -> Optional[float]:
    """Extract numeric score from judge response."""
    import re

    # Try to find [RESULT] format (Prometheus style)
    result_match = re.search(r'\[RESULT\]\s*(\d)', response)
    if result_match:
        return float(result_match.group(1))

    # Try to find standalone number at the start
    number_match = re.search(r'^(\d)', response.strip())
    if number_match:
        return float(number_match.group(1))

    # Try to find any number 1-5
    any_number = re.search(r'\b([1-5])\b', response)
    if any_number:
        return float(any_number.group(1))

    return None


def run_llm_judge_evaluation(
    model,
    tokenizer,
    judge_model,
    judge_tokenizer,
    judge_model_name: str,
    test_prompts: List[str],
    config: dict,
) -> Dict:
    """
    Run LLM-as-Judge evaluation using a local model.
    """
    print(f"   Running LLM-as-Judge evaluation with {judge_model_name}...")

    judge_config = config['evaluation']['llm_judge']
    is_prometheus = 'prometheus' in judge_model_name.lower()

    scores = []
    details = []

    for prompt in tqdm(test_prompts, desc="   Judging responses"):
        # Generate response from the model being evaluated
        response = generate_response(model, tokenizer, prompt, max_new_tokens=512)

        # Create judge prompt
        if is_prometheus:
            judge_prompt = create_prometheus_prompt(prompt, response)
        else:
            judge_prompt = create_simple_judge_prompt(prompt, response)

        # Get judge's evaluation
        judge_inputs = judge_tokenizer(
            judge_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        judge_inputs = {k: v.to(judge_model.device) for k, v in judge_inputs.items()}

        with torch.no_grad():
            judge_outputs = judge_model.generate(
                **judge_inputs,
                max_new_tokens=judge_config.get('max_new_tokens', 512),
                temperature=0.1,  # Low temp for consistent evaluation
                do_sample=True,
                pad_token_id=judge_tokenizer.pad_token_id,
            )

        judge_response = judge_tokenizer.decode(
            judge_outputs[0][judge_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Extract score
        score = extract_score_from_response(judge_response)
        if score is not None:
            scores.append(score)
            details.append({
                'instruction': prompt[:200],
                'response': response[:500],
                'score': score,
                'judge_feedback': judge_response[:500],
            })

    if not scores:
        return {'error': 'No valid scores extracted'}

    return {
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'min_score': float(np.min(scores)),
        'max_score': float(np.max(scores)),
        'n_evaluated': len(scores),
        'judge_model': judge_model_name,
        'sample_details': details[:5],  # Keep first 5 for inspection
    }


def get_test_prompts() -> List[str]:
    """Get diverse test prompts for evaluation."""
    return [
        # Knowledge
        "What is machine learning and how does it differ from traditional programming?",
        "Explain the theory of relativity in simple terms.",
        "What are the main causes of climate change?",

        # Reasoning
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",

        # Coding
        "Write a Python function to check if a string is a palindrome.",
        "Explain the difference between a list and a tuple in Python.",
        "How would you implement a binary search algorithm?",

        # Creative
        "Write a haiku about artificial intelligence.",
        "Create a short story opening about a robot discovering emotions.",

        # Instruction following
        "List 5 tips for giving a good presentation.",
        "Summarize the benefits of regular exercise in 3 bullet points.",
        "Explain how to make a cup of tea, step by step.",

        # Analysis
        "What are the pros and cons of remote work?",
        "Compare and contrast democracy and authoritarianism.",

        # Math
        "What is 15% of 80?",
        "Solve for x: 2x + 5 = 13",

        # Safety/Ethics
        "What should someone do if they witness bullying?",
        "Why is it important to cite sources when writing research papers?",

        # Common sense
        "Why do we need to sleep?",
        "What would happen if you put a metal spoon in a microwave?",
    ]


def compute_response_diversity(responses: List[str]) -> Dict[str, float]:
    """
    Compute diversity of generated responses using linguistic_diversity metrics.

    Returns a dictionary with multiple diversity scores across linguistic dimensions.
    """
    if not responses or len(responses) < 2:
        return {
            'semantic': 0.0,
            'syntactic': 0.0,
            'morphological': 0.0,
            'universal': 0.0,
        }

    # Filter out empty responses
    valid_responses = [r.strip() for r in responses if r and r.strip()]
    if len(valid_responses) < 2:
        return {
            'semantic': 0.0,
            'syntactic': 0.0,
            'morphological': 0.0,
            'universal': 0.0,
        }

    try:
        from linguistic_diversity import (
            DocumentSemantics,
            DependencyParse,
            PartOfSpeechSequence,
            UniversalLinguisticDiversity,
            get_preset_config,
        )

        scores = {}

        # Semantic diversity (document-level embeddings)
        try:
            semantic_metric = DocumentSemantics({"verbose": False})
            scores['semantic'] = float(semantic_metric(valid_responses))
        except Exception as e:
            print(f"      Warning: Semantic diversity failed: {e}")
            scores['semantic'] = 0.0

        # Syntactic diversity (dependency parse trees)
        try:
            syntactic_metric = DependencyParse({"verbose": False})
            scores['syntactic'] = float(syntactic_metric(valid_responses))
        except Exception as e:
            print(f"      Warning: Syntactic diversity failed: {e}")
            scores['syntactic'] = 0.0

        # Morphological diversity (POS sequences)
        try:
            morphological_metric = PartOfSpeechSequence({"verbose": False})
            scores['morphological'] = float(morphological_metric(valid_responses))
        except Exception as e:
            print(f"      Warning: Morphological diversity failed: {e}")
            scores['morphological'] = 0.0

        # Universal diversity (combined metric with minimal config for speed)
        try:
            universal_config = get_preset_config("minimal")
            universal_config["verbose"] = False
            universal_metric = UniversalLinguisticDiversity(universal_config)
            scores['universal'] = float(universal_metric(valid_responses))
        except Exception as e:
            print(f"      Warning: Universal diversity failed: {e}")
            scores['universal'] = 0.0

        return scores

    except ImportError as e:
        print(f"      Warning: linguistic_diversity not available: {e}")
        # Fallback to simple n-gram diversity
        all_bigrams = set()
        total_bigrams = 0
        for response in valid_responses:
            words = response.lower().split()
            for i in range(len(words) - 1):
                bigram = (words[i], words[i+1])
                all_bigrams.add(bigram)
                total_bigrams += 1

        simple_diversity = len(all_bigrams) / max(total_bigrams, 1)
        return {
            'semantic': simple_diversity,
            'syntactic': 0.0,
            'morphological': 0.0,
            'universal': simple_diversity,
        }


def main():
    """Main evaluation pipeline."""
    import time

    print("=" * 70)
    print("STEP 4: EVALUATION (Local Models Only)")
    print("=" * 70)

    config = load_config()
    mode = config.get('mode', 'pilot')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {device}")
    print(f"Mode: {mode}")

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output"
    adapters_dir = output_dir / "adapters"
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load training results to know what to evaluate
    training_results_file = output_dir / "training_results.json"
    if not training_results_file.exists():
        print(f"   ERROR: Training results not found: {training_results_file}")
        print(f"   Run 03_finetune.py first")
        return

    with open(training_results_file, 'r') as f:
        training_results = json.load(f)

    # Get evaluation configuration
    eval_config = config['evaluation']
    run_local = eval_config['local_benchmarks']['enabled']
    local_tasks = eval_config['local_benchmarks'].get('tasks', [])
    run_llm_judge = eval_config['llm_judge']['enabled']

    print(f"\nEvaluation methods:")
    print(f"   Local benchmarks (lm-eval): {run_local}")
    if run_local:
        print(f"      Tasks: {local_tasks}")
    print(f"   LLM-as-Judge: {run_llm_judge}")
    if run_llm_judge:
        print(f"      Model: {eval_config['llm_judge']['model']}")

    # Load judge model once if needed
    judge_model, judge_tokenizer, judge_model_name = None, None, None
    if run_llm_judge:
        judge_model, judge_tokenizer, judge_model_name = load_judge_model(config)
        if judge_model is None:
            print("   Warning: Could not load judge model, skipping LLM-as-Judge")
            run_llm_judge = False

    test_prompts = get_test_prompts()

    all_results = []

    # Group training results by model
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in training_results['results']:
        by_model[r['model_name']].append(r)

    for model_name, train_results in by_model.items():
        model_short = model_name.split('/')[-1]

        print(f"\n{'=' * 70}")
        print(f"Evaluating: {model_name}")
        print(f"{'=' * 70}")

        for train_result in train_results:
            adapter_path = Path(train_result['adapter_path'])
            method = train_result['selection_method']
            dataset = train_result['dataset_name']
            n_samples = train_result['n_samples']

            print(f"\n   {method} (n={n_samples})")
            print(f"   {'-' * 50}")

            if not adapter_path.exists():
                print(f"      Adapter not found: {adapter_path}")
                continue

            # Create evaluation output directory
            eval_subdir = eval_dir / f"{model_short}_{dataset}_{method}"
            eval_subdir.mkdir(exist_ok=True)

            start_time = time.time()

            # Load model
            model, tokenizer = load_model_for_inference(adapter_path, model_name)

            # Initialize result
            result = {
                'model_name': model_name,
                'adapter_path': str(adapter_path),
                'dataset_name': dataset,
                'selection_method': method,
                'n_samples': n_samples,
                'local_benchmarks': {},
                'local_benchmarks_avg': 0.0,
                'llm_judge_score': None,
                'llm_judge_details': None,
                'avg_response_length': 0.0,
                'response_diversity': {
                    'semantic': 0.0,
                    'syntactic': 0.0,
                    'morphological': 0.0,
                    'universal': 0.0,
                },
            }

            # Run local benchmarks
            if run_local and local_tasks:
                print(f"\n      Running local benchmarks...")
                local_results = run_local_benchmarks(model, tokenizer, local_tasks, eval_subdir)

                if 'error' not in local_results:
                    result['local_benchmarks'] = local_results
                    valid_scores = [v for v in local_results.values() if isinstance(v, (int, float))]
                    result['local_benchmarks_avg'] = float(np.mean(valid_scores)) if valid_scores else 0.0

                    print(f"      Results:")
                    for task, score in local_results.items():
                        print(f"         {task}: {score:.4f}")
                    print(f"         Average: {result['local_benchmarks_avg']:.4f}")

            # Generate some responses for quality metrics
            print(f"\n      Generating responses for diversity analysis...")
            responses = []
            total_length = 0
            for prompt in tqdm(test_prompts[:10], desc="      Generating"):
                response = generate_response(model, tokenizer, prompt, max_new_tokens=256)
                responses.append(response)
                total_length += len(response.split())

            result['avg_response_length'] = total_length / len(responses) if responses else 0

            # Compute linguistic diversity of responses
            print(f"      Computing linguistic diversity metrics...")
            diversity_scores = compute_response_diversity(responses)
            result['response_diversity'] = diversity_scores

            print(f"      Response Diversity:")
            print(f"         Semantic: {diversity_scores['semantic']:.4f}")
            print(f"         Syntactic: {diversity_scores['syntactic']:.4f}")
            print(f"         Morphological: {diversity_scores['morphological']:.4f}")
            print(f"         Universal: {diversity_scores['universal']:.4f}")

            # Run LLM-as-Judge
            if run_llm_judge and judge_model is not None:
                print(f"\n      Running LLM-as-Judge evaluation...")
                # Free up memory first
                del model
                clear_memory()

                # Reload model (we freed it)
                model, tokenizer = load_model_for_inference(adapter_path, model_name)

                judge_results = run_llm_judge_evaluation(
                    model, tokenizer,
                    judge_model, judge_tokenizer, judge_model_name,
                    test_prompts[:15],  # Use subset for speed
                    config,
                )

                if 'error' not in judge_results:
                    result['llm_judge_score'] = judge_results['mean_score']
                    result['llm_judge_details'] = judge_results

                    print(f"      LLM Judge Score: {judge_results['mean_score']:.2f}/5.0 "
                          f"(std: {judge_results['std_score']:.2f})")

            result['evaluation_time_seconds'] = time.time() - start_time
            result['timestamp'] = datetime.now().isoformat()

            all_results.append(result)

            # Save individual result
            with open(eval_subdir / "results.json", 'w') as f:
                json.dump(result, f, indent=2)

            # Cleanup
            del model, tokenizer
            clear_memory()

    # Cleanup judge model
    if judge_model is not None:
        del judge_model, judge_tokenizer
        clear_memory()

    # Save all results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'evaluation_config': {
                'local_benchmarks': local_tasks if run_local else [],
                'llm_judge_model': judge_model_name,
            },
            'results': all_results,
        }, f, indent=2)

    # Print summary comparison
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Group by model and dataset
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['model_name'].split('/')[-1], r['dataset_name'])
        grouped[key].append(r)

    for (model, dataset), results in grouped.items():
        print(f"\n{model} / {dataset}:")

        # Sort by method
        method_order = {'full': 0, 'random': 1, 'diversity': 2}
        results = sorted(results, key=lambda x: method_order.get(x['selection_method'], 99))

        for r in results:
            method = r['selection_method']
            n = r['n_samples']

            # Format metrics
            metrics = []
            if r['local_benchmarks_avg'] > 0:
                metrics.append(f"lm-eval={r['local_benchmarks_avg']:.3f}")
            if r['llm_judge_score'] is not None:
                metrics.append(f"judge={r['llm_judge_score']:.2f}/5")

            # Add diversity metrics
            div = r.get('response_diversity', {})
            if isinstance(div, dict) and div.get('universal', 0) > 0:
                metrics.append(f"div={div['universal']:.3f}")

            metrics_str = ", ".join(metrics) if metrics else "N/A"
            print(f"   {method:12s} (n={n:6d}): {metrics_str}")

    # Check success criterion
    print("\n" + "=" * 70)
    print("SUCCESS CRITERION CHECK")
    print("=" * 70)

    for (model, dataset), results in grouped.items():
        print(f"\n{model} / {dataset}:")

        full_result = next((r for r in results if r['selection_method'] == 'full'), None)
        random_result = next((r for r in results if r['selection_method'] == 'random'), None)
        div_result = next((r for r in results if r['selection_method'] == 'diversity'), None)

        # Compare using local benchmarks average
        if full_result and div_result:
            full_score = full_result['local_benchmarks_avg']
            div_score = div_result['local_benchmarks_avg']

            if full_score > 0 and div_score > 0:
                diff_pct = (div_score - full_score) / full_score * 100
                within_2pct = abs(diff_pct) <= 2.0

                print(f"   Full vs Diversity (lm-eval avg): {full_score:.4f} vs {div_score:.4f}")
                print(f"   Relative difference: {diff_pct:+.1f}%")
                print(f"   Within 2%: {'PASS' if within_2pct else 'FAIL'}")

        if random_result and div_result:
            random_score = random_result['local_benchmarks_avg']
            div_score = div_result['local_benchmarks_avg']

            if random_score > 0 and div_score > 0:
                beats_random = div_score > random_score
                print(f"   Random vs Diversity: {random_score:.4f} vs {div_score:.4f}")
                print(f"   Diversity beats random: {'PASS' if beats_random else 'FAIL'}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
