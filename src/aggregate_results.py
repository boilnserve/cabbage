from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from tqdm import tqdm
from loguru import logger

from utils import load_yaml, load_jsonl, save_json
import numpy as np
from collections import defaultdict

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


def aggregate_results(model_dir: Path) -> Dict[str, Any]:
    """Aggregate results from JSONL files in the specified directory."""
    #results_path = Path(model_dir)
    aggregated_results = {}

    for file_path in tqdm(model_dir.glob("*.jsonl"), desc=f"Processing {model_dir.name}", unit="file"):
        task_name = file_path.stem
        records = load_jsonl(str(file_path))

        list_of_scores = [{pr.get('model', 'default'): pr['result']['scores'] for pr in r['process_results']} for r in records]
        
        # Initialize nested dict to accumulate scores
        model_scores = defaultdict(lambda: defaultdict(list))
        # Collect scores
        for entry in list_of_scores:  # assuming your list is stored in `data`
            for model, scores in entry.items():
                for metric, value in scores.items():
                    model_scores[model][metric].append(value)

        # Compute mean scores
        mean_scores = {
            model: {
                metric: round(np.mean(values),2)
                for metric, values in metrics.items()
            }
            for model, metrics in model_scores.items()
        }
        # Sort mean_scores by model name (alphabetically)
        sorted_mean_scores = {model: dict(sorted(metrics.items())) for model, metrics in sorted(mean_scores.items())}

        aggregated_results[task_name] = sorted_mean_scores
        
    # Sort the aggregated results by task name (alphabetically)
    sorted_aggregated_results = dict(sorted(aggregated_results.items()))

    return sorted_aggregated_results


def aggregate_results_directory(config_path: Path, use_cache: bool = True) -> None:
    """Aggregate results across all subdirectories based on a configuration YAML file."""
    
    config = load_yaml(config_path)
    results_dir = Path(config['models']['results_directory'])
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
