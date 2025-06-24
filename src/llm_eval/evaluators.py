from abc import ABC, abstractmethod
from typing import Any, Dict, List
import re
import string
from typing import List, Dict
import time
import tqdm.asyncio as tqdm_as
from loguru import logger
from llm_eval.llm_judge import LLMJudge
from llm_eval.utils.configuration import MainConfig, ExperimentConfig

from pathlib import Path
from llm_eval.utils.file_io import load_jsonl, save_jsonl
from collections import defaultdict

def compute_per_example_agreement(scores_by_judge: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute agreement scores per metric across judges. Args: scores_by_judge: Dict mapping judge names to metric scores. Returns: Dict of agreement scores per metric."""
    from itertools import combinations
    from statistics import mean
    agreement_per_metric = {}
    per_metric_scores = defaultdict(list)

    for judge_scores in scores_by_judge.values():
        for metric, score in judge_scores.items():
            per_metric_scores[metric].append(score)

    for metric, scores in per_metric_scores.items():
        if len(scores) <= 1:
            agreement_per_metric[metric] = 1.0
        else:
            pairwise_diffs = [abs(a - b) for a, b in combinations(scores, 2)]
            avg_diff = mean(pairwise_diffs)
            agreement = round(1 - (avg_diff / 5.0), 2)
            agreement_per_metric[metric] = max(0.0, min(1.0, agreement))

    return agreement_per_metric

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing certain phrases, and stripping punctuation. Args: text: Input string. Returns: Normalized string."""
    text = text.lower()
    text = re.sub(r'\b(the answer is|answer:)\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())
class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    def __init__(self, config: MainConfig, experiment_config: ExperimentConfig, use_cache):
        """Initialize the evaluator with configs and cache flag. Args: config: MainConfig object. experiment_config: ExperimentConfig object. use_cache: Whether to use cache."""
        self.config=config
        self.experiment_config=experiment_config
        self.use_cache = use_cache
        
    @abstractmethod
    def evaluate_all(self, examples: List[Dict]) -> List[List[Dict]]:
        """Evaluate all examples and return results. Args: examples: List of example dictionaries. Returns: List of lists of result dictionaries."""
        pass
    
    @abstractmethod
    def evaluate_and_save(self, experiment_file: Path) -> None:
        """Evaluate and save results to the experiment file. Args: experiment_file: Path to the experiment file."""
        pass

class ExactMatchEvaluator(BaseEvaluator):
    """Evaluator for exact match between model output and accepted answers."""
    def evaluate_all(self, examples: List[Dict]) -> List[List[Dict]]:
        """Evaluate all examples for exact match. Args: examples: List of example dictionaries. Returns: List of lists of result dictionaries."""
        results = []
        for example in examples:
            targets_norm = [normalize_text(t) for t in example['original_doc']['accepted_answers']]
            model_output = example['inference_result'].get('model_answer')
            
            if not model_output:
                results.append([{'metric': 'exact_match', 'result': "invalid_format"}])
                continue

            pred = normalize_text(model_output)

            if any(pred == target for target in targets_norm):
                results.append([{'metric': 'exact_match', 'result': {'scores': {'exact_match_score': 1.0}}}])
            else:
                results.append([{'metric': 'exact_match', 'result': {'scores': {'exact_match_score': 0.0}}}])

        return results
    
    def evaluate_and_save(self, experiment_file: Path) -> None:
        """Evaluate and save exact match results to the experiment file. Args: experiment_file: Path to the experiment file."""
        examples = load_jsonl(str(experiment_file))
        results = []
        for example in examples:
            targets_norm = [normalize_text(t) for t in example['original_doc']['accepted_answers']]
            model_output = example['inference_result'].get('model_answer')
            
            if not model_output:
                results.append([{'metric': 'exact_match', 'result': "invalid_format"}])
                continue

            pred = normalize_text(model_output)

            if any(pred == target for target in targets_norm):
                results.append([{'metric': 'exact_match', 'result': {'scores': {'exact_match_score': 1.0}}}])
            else:
                results.append([{'metric': 'exact_match', 'result': {'scores': {'exact_match_score': 0.0}}}])

        for rec, res in zip(examples, results):
                rec["process_results"] = res
        save_jsonl(experiment_file, examples)
    
class OptionMatchEvaluator(BaseEvaluator):
    """Evaluator for matching model output to the correct option letter."""
    def evaluate_all(self, examples: List[Dict]) -> List[List[Dict]]:
        """Evaluate all examples for option match. Args: examples: List of example dictionaries. Returns: List of lists of result dictionaries."""
        results = []
        for example in examples:
            model_output = example['inference_result'].get('model_answer')
            target = example['original_doc']['correct_letter'].strip()
            
            result = (model_output or '').strip()

            if not result:
                results.append([{'metric': 'option_match', "result": "invalid_format"}])
                continue

            pred = result[0]
            option_letters = [chr(ord("A") + i) for i in range(7)]
            if (len(result) > 1 and result[1].isalpha()) or pred not in option_letters:
                results.append([{'metric': 'option_match', "result": "invalid_format"}])
                continue

            results.append([{
                'metric': 'option_match',
                'result': {'scores': {'option_match_score': 1.0 if pred == target else 0.0}}
            }])
        return results
    
    def evaluate_and_save(self, experiment_file: Path) -> None:
        """Evaluate and save option match results to the experiment file. Args: experiment_file: Path to the experiment file."""
        results = []
        examples = load_jsonl(str(experiment_file))
        for example in examples:
            model_output = example['inference_result'].get('model_answer')
            target = example['original_doc']['correct_letter'].strip()
            
            result = (model_output or '').strip()

            if not result:
                results.append([{'metric': 'option_match', "result": "invalid_format"}])
                continue

            pred = result[0]
            option_letters = [chr(ord("A") + i) for i in range(7)]
            if (len(result) > 1 and result[1].isalpha()) or pred not in option_letters:
                results.append([{'metric': 'option_match', "result": "invalid_format"}])
                continue

            results.append([{
                'metric': 'option_match',
                'result': {'scores': {'option_match_score': 1.0 if pred == target else 0.0}}
            }])
        for rec, res in zip(examples, results):
                rec["process_results"] = res
        save_jsonl(experiment_file, examples)

class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluator that uses an LLM judge to evaluate model outputs."""
    def evaluate_all(self, examples: List[Dict]) -> List[List[Dict]]:
        """Evaluate all examples using the LLM judge. Args: examples: List of example dictionaries. Returns: List of lists of result dictionaries."""
        return []
    
    def evaluate_and_save(self, experiment_file: Path) -> None:
        """Evaluate and save LLM judge results to the experiment file. Args: experiment_file: Path to the experiment file."""
        prompts_dict = self.config.evaluators.prompts_dict
        all_provider_results = []
        examples = load_jsonl(str(experiment_file))
        for evaluator_config in self.config.evaluators.providers:
            # Safety: skip if already present and cache is enabled
            first_result = examples[0] if examples else {}
            already_present_models = [
                res.get('model') for res in first_result.get('process_results', [])
            ]
            if self.use_cache and evaluator_config.model in already_present_models:
                logger.info(
                    f"The file {self.experiment_config.dataset.name} already evaluated with model {evaluator_config.model}."
                )
                continue

            judge = LLMJudge(
                evaluator_config,
                self.experiment_config.process_results.evaluation_type,
                prompts_dict
            )
            
            eval_results = judge.evaluate_until(examples)
            time.sleep(2)

            
            # Attach model name to each result for clarity (optional)
            provider_results = [
                {'metric': 'llm_judge', 'model': evaluator_config.model, 'result': r} for r in eval_results
            ]
            all_provider_results.append(provider_results)
            
            results = [list(item_tuple) for item_tuple in zip(*all_provider_results)]
            
            for rec, res in zip(examples, results):
                rec["process_results"] = res
            
            # ⬇️ Add agreement per example here
            for rec in examples:
                scores_by_model = {}
                for pr in rec.get("process_results", []):
                    if 'result' in pr and 'scores' in pr['result']:
                        model = pr.get("model", "default")
                        scores_by_model[model] = pr["result"]["scores"]

                if len(scores_by_model) > 1:
                    agreement = compute_per_example_agreement(scores_by_model)
                    rec["judges_agreement"] = agreement
                    
            save_jsonl(experiment_file, examples)


EVALUATOR_REGISTRY = {
    "exact_match": ExactMatchEvaluator,
    "option_match": OptionMatchEvaluator,
    "llm_judge": LLMJudgeEvaluator,
}

def get_evaluator(eval_type, config, experiment_config, use_cache):
    """Get the evaluator class instance for the given type. Args: eval_type: Type of evaluator. config: MainConfig object. experiment_config: ExperimentConfig object. use_cache: Whether to use cache. Returns: Evaluator class instance."""
    evaluator_cls = EVALUATOR_REGISTRY[eval_type]
    return evaluator_cls(config, experiment_config, use_cache)