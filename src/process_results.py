import os
import re
import string
import asyncio
from typing import Any, Dict, List, Union
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import tqdm.asyncio as tqdm_as

from utils import load_jsonl, save_jsonl, load_yaml
from GEvaluator import GEvaluator

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(the answer is|answer:)\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())

def word_overlap_score(pred: str, target: str) -> float:
    pred_words = set(pred.split())
    target_words = set(target.split())
    return len(pred_words & target_words) / len(target_words) if target_words else 0.0

def process_option_match(target: str, result: str) -> List[Dict]:
    
    target = target.strip()
    result = result.strip()
    pred = result[0]
    option_letters = [chr(ord("A") + i) for i in range(7)]
    #target = target.strip()
    if not result or (len(result) > 1 and result[1].isalpha()) or pred not in option_letters:
        return [{'metric': 'option_match', "result": "invalid_format"}]
    
    
    return [{'metric': 'option_match', 'result': {'scores': {'option_match_score' : 1.0 if pred == target else 0.0}}}]
    

def process_exact_match(targets_list: list, result: str) -> List[Dict]:
    result = result.strip()
    if not result or "Model Error" in result:
        return [{'metric': 'exact_match', 'result': "invalid_format"}]

    pred = normalize_text(result)
    targets_norm = [normalize_text(t) for t in targets_list]
    for target in targets_norm:
        if pred==target:
            return [{'metric': 'exact_match', 'result': {'scores': {'exact_match_score' :1.0}}}]
    
    return [{'metric': 'exact_match', 'result': {'scores': {'exact_match_score' :0.0}}}]
    # pred_parts = [p.strip() for p in pred.split('or')]

    # exact_match = best_overlap = 0.0
    # for part in pred_parts:
    #     for target in targets_norm:
    #         if part == target:
    #             return {"exact_match": 1.0}
    #         best_overlap = max(best_overlap, word_overlap_score(part, target))

def process_option_match_file(file_path: Path) -> None:
    results = load_jsonl(file_path)
    for result in results:
        result.update({'process_results': process_option_match(result['inference_result']['correct_letter'], result['inference_result'].get('model_answer'))})
    save_jsonl(file_path, results)

def process_exact_match_file(file_path: Path) -> None:
    results = load_jsonl(file_path)
    for result in results:
        result.update({'process_results': process_exact_match(result['original_doc']['accepted_answers'], result['inference_result'].get('model_answer'))})
    save_jsonl(file_path, results)

def process_g_eval_file(evaluator: GEvaluator, file_path: Path) -> None:
    results = load_jsonl(file_path)
    coroutines = [evaluator.evaluate(result['original_doc'], result['inference_result'].get('model_answer')) for result in results]
    
    evaluated_results = asyncio.run(tqdm_as.tqdm_asyncio.gather(*coroutines, desc=f"G-eval with model {evaluator.model_name}: {file_path.name}"))

    for result, eval_result in zip(results, evaluated_results):
        if eval_result:
            result.setdefault('process_results', []).append({
            'metric': 'g_eval',
            'model': evaluator.model_name,
            'result': eval_result
        })
        else:
            logger.error(f"G-eval failed for {file_path.name} with model {evaluator.model_name}")

    save_jsonl(file_path, results)

def load_experiments_config(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    config_dict = {}
    for file in base_dir.glob("*.yaml"):
        config = load_yaml(file)
        name = config.get("dataset", {}).get("name")
        if name:
            config_dict[name] = config
        else:
            logger.warning(f"No dataset name in {file.name}, skipping.")
    return config_dict

def process_results(config_path: Path, use_cache: bool = True) -> None:
    general_config = load_yaml(config_path)
    experiments_dir = Path(general_config['experiments']['base_dir'])
    results_dir = Path(general_config['models']['results_directory'])

    if not results_dir.exists():
        logger.error(f"Results directory {results_dir} not found.")
        return

    experiments_config = load_experiments_config(experiments_dir)

    for model_subdir in results_dir.iterdir():
        if not model_subdir.is_dir():
            continue

        model_name = model_subdir.name
        for experiment_file in model_subdir.glob("*.jsonl"):
            experiment_name = experiment_file.stem
            experiment_config = experiments_config.get(experiment_name)

            if not experiment_config:
                logger.warning(f"No config found for experiment: {experiment_name}")
                continue

            metric = experiment_config['process_results'].get('metric')
            logger.info(f"Processing {model_name}/{experiment_name} with metric: {metric}")

            if metric == 'option_match':
                process_option_match_file(experiment_file)
            elif metric == 'exact_match':
                process_exact_match_file(experiment_file)
            elif metric == 'g_eval':
                for evaluator in general_config['evaluators'].get('providers', []):
                    results = load_jsonl(experiment_file)
                    already_present_models = [result.get('model') for result in results[0].get('process_results',[])] # only check the first element
                    if use_cache and evaluator['model'] in already_present_models:
                        logger.info(f"{experiment_file} already g-evaluated with the model {evaluator['model']}.")
                        continue
                    evaluator = GEvaluator(evaluator, experiment_config['process_results'].get('evaluation_type'))
                    process_g_eval_file(evaluator, experiment_file)
