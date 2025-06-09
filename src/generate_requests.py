import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datasets import Dataset, load_dataset
from tqdm import tqdm
from typing import cast
import random
# ---------- Utility Functions ----------
def load_yaml(path: Path) -> dict:
    with path.open('r') as f:
        return yaml.safe_load(f)

def format_options(options_list: List[str]) -> str:
    return "\n".join([f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options_list)])

def get_options_list(doc, input_config) -> Optional[List[str]]:
    if doc.get('options') and input_config.get('question_type') == 'multiple_choice':
        options = doc['options']
        difficulty = input_config.get("difficulty")
        if difficulty:
            options_list = options.get(difficulty)
        elif options.get('default'):
            options_list = options['default']
        else:
            filtered_options = {k: v for k, v in options.items() if v}
            if filtered_options:
                options_list = random.choice(list(filtered_options.values()))
            else:
                raise ValueError("No valid options available in the provided dictionary.")
        if not options_list:
            raise ValueError(f"No options found for difficulty level {difficulty}.")
        return options_list

def get_correct_letter(options: Optional[List[str]], answer: str) -> Optional[str]:
    if not options or answer not in options:
        return None
    
    correct_index = options.index(answer)
    return chr(ord('A') + correct_index)

def resize_image(image, size: Tuple[int, int]):
    image = image.convert("RGB")
    image.thumbnail(size)
    return image

def extract_visuals(doc: dict) -> List[Any]:
        visuals = [resize_image(image, (512, 512)) for image in doc.get('images',[])]
        if len(visuals) > 1:
            visuals = [resize_image(img, (256, 256)) for img in visuals]
        return visuals

def format_prompt(doc: dict, options_list: Optional[List[str]], input_config: Dict[str, Any]) -> str:
    prompt_parts = [input_config.get("pre_prompt", ""), f"Question: {doc['question']}"]

    if options_list:
        prompt_parts.append(f"\nOptions:\n{format_options(options_list)}")

    prompt_parts.append(input_config.get("post_prompt", ""))
    return "\n".join(part for part in prompt_parts if part.strip())

def filter_dataset(ds, filters):
    if 'difficulty' in filters:
        difficulty=filters['difficulty']
        ds = ds.filter(lambda row: bool(row.get('options',{}).get(difficulty)))
    elif 'images_num' in filters:
        images_num = filters['images_num']
        ds = ds.filter(lambda row: len(row.get('images', [])) == images_num)
    return ds

def prepare_dataset(experiment_config, general_config):        
    ds_raw = load_dataset(experiment_config['dataset']['path'], name=experiment_config['dataset']['subset'], split=experiment_config['dataset']['split'])
    ds = cast(Dataset, ds_raw)
    
    if not isinstance(ds, Dataset):
        raise ValueError(f"Expected `Dataset`, got {type(ds)}")
    filters = experiment_config['dataset'].get('filters',{})
    if filters:
        ds = filter_dataset(ds, filters)
    limit = general_config['experiments'].get('limit')
    return  ds.select(list(range(limit))) if limit is not None else ds
# ---------- Data Structures ----------

@dataclass
class ExperimentRequest:
    visuals: List[Any]
    input: str
    gen_kwargs: Dict[str, Any]
    correct_letter: Optional[str]
    question_type: str

@dataclass
class Experiment:
    name: str
    dataset: Dataset
    experiment_config: Dict[str, Any]
    general_config: Dict[str, Any]
    requests: List[ExperimentRequest] = field(default_factory=list)

    @classmethod
    def from_config(cls, experiment_path: Path, general_config: Dict[str, Any]) -> "Experiment":
        experiment_config = load_yaml(experiment_path)
        experiment = cls(
            name=experiment_config['dataset']['name'],
            experiment_config=experiment_config,
            general_config=general_config,
            dataset=prepare_dataset(experiment_config, general_config)
        )
        experiment._create_requests()
        return experiment
        
    def _create_requests(self):
        if not self.dataset:
            raise ValueError(f"Dataset not loaded for experiment '{self.name}'")
        for doc in tqdm(self.dataset, desc=f"Preparing {self.name}"):
            options_list = get_options_list(dict(doc), self.experiment_config['model_input'])
            self.requests.append(ExperimentRequest(
                visuals = extract_visuals(dict(doc)),
                input = format_prompt(dict(doc), options_list, self.experiment_config.get("model_input", {})),
                correct_letter = get_correct_letter(options_list, dict(doc)['answer']),
                gen_kwargs=self.general_config['models'].get('default_parameters', {}),
                question_type=self.experiment_config['model_input'].get("question_type"),
            ))

# ---------- experiment Loading + Pipeline ----------

def load_experiments(config_path: Path) -> List[Experiment]:
    general_config = load_yaml(config_path)
    experiments_dir = Path(general_config['experiments']['base_dir'])

    use_all = general_config['experiments'].get('use_all', False)
    allowed_datasets = None if use_all else general_config['experiments'].get('datasets')
    experiments = []

    for experiment_path in experiments_dir.glob("*.yaml"):
        experiment_config = load_yaml(experiment_path)
        experiment_name = experiment_config.get('dataset',{}).get('name')

        # Skip if whitelist is set and experiment is not included
        if allowed_datasets and experiment_name not in allowed_datasets:
            continue

        experiments.append(Experiment.from_config(experiment_path, general_config))

    return experiments

def generate_requests(config_path: Path) -> List[Experiment]:
    try:
        experiments = load_experiments(config_path)
        if not experiments:
            raise ValueError(f"No valid experiments found in '{config_path}'.")
        return experiments
    except Exception as e:
        raise RuntimeError(f"Fatal error during experiment generation from '{config_path}': {e}") from e