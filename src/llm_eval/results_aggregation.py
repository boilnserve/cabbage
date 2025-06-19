from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
from tqdm import tqdm
from loguru import logger
from collections import defaultdict

from llm_eval.utils.file_io import load_jsonl, save_json
from llm_eval.utils.configuration import MainConfig

# Type alias
ScoreDict = Dict[str, List[Dict[str, float]]]

def invalid_format_aggregation(results: List[Any]) -> Union[float, None]:
    """Compute the proportion of 'invalid_format' entries."""
    try:
        total = len(results)
        if total == 0:
            return 0.0
        invalid_count = sum(1 for r in results if r == 'invalid_format')
        return round(invalid_count / total, 2)
    except Exception as e:
        logger.error(f"Error during invalid format aggregation: {e}")
        return None

# def aggregate_results(model_dir: Path) -> Dict[str, Any]:
#     """Aggregate results from JSONL files in the specified directory."""
#     aggregated_results = {}
#     for file_path in tqdm(model_dir.glob("*.jsonl"), desc=f"Processing {model_dir.name}", unit=" file"):
#         task_name = file_path.stem
#         records = load_jsonl(str(file_path))
#         # {model: {'metric': value, ...}, ...}
#         list_of_scores = [
#             {pr.get('model', 'default'): pr['result']['scores']
#              for pr in r['process_results'] if 'result' in pr and 'scores' in pr['result']}
#             for r in records
#         ]
#         model_scores = defaultdict(lambda: defaultdict(list))
#         # Collect scores
#         for entry in list_of_scores:
#             for model, scores in entry.items():
#                 for metric, value in scores.items():
#                     model_scores[model][metric].append(value)
#         # Compute mean scores
#         mean_scores = {
#             model: {
#                 metric: round(np.mean(values), 2)
#                 for metric, values in metrics.items()
#             }
#             for model, metrics in model_scores.items()
#         }
#         # Sort mean_scores by model name (alphabetically)
#         sorted_mean_scores = {model: dict(sorted(metrics.items()))
#                               for model, metrics in sorted(mean_scores.items())}
#         aggregated_results[task_name] = sorted_mean_scores
#     # Sort the aggregated results by task name (alphabetically)
#     sorted_aggregated_results = dict(sorted(aggregated_results.items()))
#     return sorted_aggregated_results

def aggregate_results(model_dir: Path) -> Dict:
    """Aggregate results from JSONL files in the specified directory."""
    aggregated_results = {}

    for file_path in tqdm(model_dir.glob("*.jsonl"), desc=f"Processing {model_dir.name}", unit=" file"):
        task_name = file_path.stem
        records = load_jsonl(str(file_path))

        # Extract model scores
        list_of_scores = [
            {pr.get('model', 'default'): pr['result']['scores']
             for pr in r['process_results'] if 'result' in pr and 'scores' in pr['result']}
            for r in records
        ]

        model_scores = defaultdict(lambda: defaultdict(list))
        for entry in list_of_scores:
            for model, scores in entry.items():
                for metric, value in scores.items():
                    model_scores[model][metric].append(value)

        # Compute mean scores
        mean_scores = {
            model: {
                metric: round(np.mean(values), 2)
                for metric, values in metrics.items()
            }
            for model, metrics in model_scores.items()
        }

        # Sort mean_scores by model name (alphabetically)
        sorted_mean_scores = {
            model: dict(sorted(metrics.items()))
            for model, metrics in sorted(mean_scores.items())
        }

        # Extract and aggregate judges_agreement if present
        judges_agreement_scores = defaultdict(list)
        for r in records:
            if 'judges_agreement' in r:
                for metric, value in r['judges_agreement'].items():
                    if isinstance(value, (int, float)):
                        judges_agreement_scores[metric].append(value)

        # Average judges_agreement scores if available
        aggregated_result = {
            "mean_scores": sorted_mean_scores
        }

        if len(model_scores) > 1 and judges_agreement_scores:
            averaged_agreement = {
                metric: round(np.mean(values), 2)
                for metric, values in judges_agreement_scores.items()
            }
            aggregated_result["judges_agreement"] =  {'default': averaged_agreement}

        aggregated_results[task_name] = aggregated_result

    return dict(sorted(aggregated_results.items()))


def aggregate_results_directory(config: MainConfig, use_cache: bool = True) -> None:
    """Aggregate results across all subdirectories based on a configuration object."""
    results_dir = config.paths.results_directory
    output_dir = results_dir / 'json'
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == 'json':
            continue

        output_file = output_dir / f"{model_dir.name}.json"
        if output_file.exists() and use_cache:
            print(f"Skipping existing file: {output_file}")
            continue

        aggregated = aggregate_results(model_dir)   
        save_json(aggregated, str(output_file))