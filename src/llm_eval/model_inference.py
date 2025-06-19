import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from sglang.utils import terminate_process
from loguru import logger

from llm_eval.request_generation import Experiment
from llm_eval.llm_client import LLMClient
from llm_eval.utils.file_io import save_jsonl, save_json, load_json
from llm_eval.utils.process_utils import launch_server, wait_for_server
from llm_eval.utils.configuration import MainConfig
from llm_eval.request_generation import generate_experiment_inputs

class ModelInferenceRunner:
    def __init__(self, config: MainConfig, use_cache: bool) -> None:
        self.config = config
        self.models = self.config.models.providers
        self.results_dir = self.config.paths.results_directory
        self.metadata_path = self.results_dir / 'metadata.json'
        self.metadata = load_json(self.metadata_path) if self.metadata_path.exists() else {}
        self.use_cache = use_cache

    def run(self, experiments: List[Experiment]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        for model in self.models:
            model_name = model.model
            model_safe = model_name.replace('/', '_')
            model_path = self.results_dir / model_safe
            if self.use_cache:
                experiments_to_run = self._filter_experiments(experiments, model_path)
            else:
                experiments_to_run = experiments
            if not experiments_to_run:
                logger.info(f"Skipping {model_name}, all experiments already processed.")
                continue

            logger.info(f"Running inference for {model_name}...")
            model_path.mkdir(parents=True, exist_ok=True)
            # Here, model is a ProviderConfig object
            client = LLMClient(
                model_name=model.model,
                base_url=model.base_url or "",
                api_key=os.getenv(model.api_key, '') if model.api_key else '',
                timeout=model.timeout or 120
            )

            server_process = None
            if model.server_args:
                server_process = launch_server(model_name, model.server_args)
                wait_for_server(model.server_args.get('port', 8000), timeout=300)

            start = time.time()
            for experiment in experiments_to_run:
                logger.info(f"\tProcessing {experiment.name}")
                results = client.generate_until(experiment.requests)
                self._save_experiment_results(model_path, experiment, results)

            self.metadata[model_safe] = {
                'inference_time': round(time.time() - start, 2),
                'date': datetime.now().strftime("%d-%m-%Y"),
                'limit': self.config.experiments.limit
            }
            save_json(self.metadata, self.metadata_path)

            if server_process:
                terminate_process(server_process)

    def _filter_experiments(self, experiments: List[Experiment], model_path: Path) -> List[Experiment]:
        existing = {p.stem for p in model_path.glob("*.jsonl")}
        experiments_to_run = [e for e in experiments if e.name not in existing]
        return experiments_to_run

    def _save_experiment_results(self, model_path: Path, experiment: Experiment, results: List[Dict[str,str]]) -> None:
        docs = [
            {
                'original_doc': {
                    **{k: v for k, v in req.doc.items() if k != 'images'}, # 'images' field removed to ensure JSON serializable
                    'correct_letter': req.correct_letter,
                    }, 
                'inference_result': {
                    'input_prompt': req.input,
                    **res,
                },
            }
            for req, res in zip(experiment.requests, results)
        ]
        save_jsonl(model_path / f"{experiment.name}.jsonl", docs)

def run_model_inference(config: MainConfig, experiments: List[Experiment], use_cache: bool = True) -> None:
    experiments = generate_experiment_inputs(config)
    runner = ModelInferenceRunner(config, use_cache)
    runner.run(experiments)