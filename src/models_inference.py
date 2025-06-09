import os
import time
import yaml
import torch
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from openai_client import OpenAIClient
from utils import save_jsonl, save_json, load_json
from sglang.utils import terminate_process
from generate_requests import Experiment
from loguru import logger

def load_yaml(path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
class InferenceRunner:
    def __init__(self, config_path: Path):
        self.config = load_yaml(config_path)
        self.models = self.config['models']['providers']
        self.results_dir = Path(self.config['models']['results_directory'])
        self.metadata_path = self.results_dir / 'metadata.json'
        self.metadata = load_json(self.metadata_path) if self.metadata_path.exists() else {}

    def run(self, experiments: List[Experiment]):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        for model in self.models:
            model_name = model['model']
            model_safe = model_name.replace('/', '_')
            model_path = self.results_dir / model_safe

            experiments_to_run = self._filter_experiments(experiments, model_path)
            if not experiments_to_run:
                logger.info(f"Skipping {model_name}, all experiments already processed.")
                continue

            logger.info(f"Running inference for {model_name}...")
            model_path.mkdir(parents=True, exist_ok=True)
            client = OpenAIClient(
                model_name=model_name,
                base_url=model['base_url'],
                api_key=os.getenv(model['api_key'], ''),
                timeout=model.get('timeout', 120)
            )

            server_process = None
            if server_args := model.get('server_args'):
                server_process = self._launch_server(model_name, server_args)
                self._wait_for_server(server_args.get('port', 8000), timeout=300)

            start = time.time()
            for experiment in experiments_to_run:
                logger.info(f"\tProcessing {experiment.name}")
                results = client.generate_until(experiment.requests)
                self._save_experiment_results(model_path, experiment, results)

            self.metadata[model_safe] = {
                'inference_time': round(time.time() - start, 2),
                'date': datetime.now().strftime("%d-%m-%Y"),
                'limit': self.config['experiments'].get('limit', 0)
            }
            save_json(self.metadata, self.metadata_path)

            if server_process:
                terminate_process(server_process)

    def _filter_experiments(self, experiments: List[Experiment], model_path: Path) -> List[Experiment]:
        existing = {p.stem for p in model_path.glob("*.jsonl")}
        experiments_to_run = [e for e in experiments if e.name not in existing]
        return experiments_to_run

    def _save_experiment_results(self, model_path: Path, experiment: Experiment, results: List[Dict[str,str]]):
        docs = [
            {
                'original_doc': {**{k: v for k, v in dict(doc).items() if k != 'images'}}, # 'images' field removed to ensure the document is JSON serializable
                'inference_result': {
                    'input_prompt': req.input,
                    **res,
                    #'reasoning': res['reasoning'],
                    #'model_answer': res['model_answer'],
                    'correct_letter': req.correct_letter,
                },
            }
            for doc, req, res in zip(experiment.dataset, experiment.requests, results)
        ]

        save_jsonl(model_path / f"{experiment.name}.jsonl", docs)

    def _launch_server(self, model_name: str, args: dict) -> subprocess.Popen:
        gpus = torch.cuda.device_count()
        command = (
            f"python -m sglang_router.launch_server"
            f" --model-path {model_name}"
            f" --chat-template {args['chat_template']}"
            f" --dp-size {min(gpus, args['dp_size'])}"
            f" --tp-size {args['tp_size']}"
            f" --mem-fraction-static 0.9"
            f" --router-policy round_robin"
            f" --max-running-requests 100"
            f" --port {args.get('port', 8000)}"
        )
        return subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _wait_for_server(self, port: int, timeout: int):
        base_url = f"http://localhost:{port}"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{base_url}/v1/models", headers={"Authorization": "Bearer None"})
                if response.status_code == 200:
                    logger.info("Server is ready.")
                    time.sleep(5)
                    return
            except requests.RequestException:
                time.sleep(1)
        raise TimeoutError("Server did not become ready in time.")

def models_inference(config_path: Path, experiments: List[Experiment]):
    runner = InferenceRunner(config_path)
    runner.run(experiments)
