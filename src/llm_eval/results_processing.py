from loguru import logger

from llm_eval.utils.configuration import MainConfig, load_experiments_config
from llm_eval.evaluators import get_evaluator


def process_inference_results(config: MainConfig, use_cache: bool = True) -> None:
    """Process inference results for all models and experiments, applying the appropriate evaluator and saving results. Args: config: MainConfig object. use_cache: Whether to use cache for processing results."""
    experiments_dir = config.paths.experiments_directory
    results_dir = config.paths.results_directory

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
            experiment_config = experiments_config[experiment_name]

            if not experiment_config:
                logger.warning(f"No config found for experiment: {experiment_name}")
                continue
            
            metric = experiment_config.process_results.metric
            logger.info(f"Processing {model_name}/{experiment_name} with metric: {metric}")
            evaluator = get_evaluator(metric, config, experiment_config, use_cache)
            evaluator.evaluate_and_save(experiment_file)
