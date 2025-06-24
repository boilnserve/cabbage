import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast
from datasets import Dataset, load_dataset
from tqdm import tqdm

from llm_eval.utils.formatting import format_prompt, get_answer_letter
from llm_eval.utils.image_utils import extract_visuals
from llm_eval.utils.configuration import MainConfig, ExperimentConfig, load_experiment_config

# ---------- Utility Functions ----------

def filter_dataset(ds: Dataset, filters: Dict) -> Dataset:
    """Filter a HuggingFace Dataset according to the provided filters. Args: ds: The input Dataset. filters: Dictionary of filter criteria. Returns: Filtered Dataset."""
    if not filters:
        return ds
    if 'difficulty' in filters:
        difficulty = filters['difficulty']
        ds = ds.filter(lambda row: bool(row.get('options', {}).get(difficulty)))
    elif 'images_num' in filters:
        images_num = filters['images_num']
        ds = ds.filter(lambda row: len(row.get('images', [])) == images_num)
    return ds

def prepare_dataset(
    exp_conf: ExperimentConfig, 
    global_conf: MainConfig
) -> Dataset:
    """Load and filter the dataset according to experiment and general config. Args: exp_conf: ExperimentConfig object. global_conf: MainConfig object. Returns: Filtered Dataset."""
    ds_raw = load_dataset(str(exp_conf.dataset.path), name=exp_conf.dataset.subset,split=exp_conf.dataset.split)
    ds = cast(Dataset, ds_raw)
    if not isinstance(ds, Dataset):
        raise ValueError(f"Expected `Dataset`, got {type(ds)}")
    # Add filters support if you add to DatasetConfig
    filters = exp_conf.dataset.filters
    if filters:
        ds = filter_dataset(ds, filters)
    limit = global_conf.experiments.limit
    if limit:
        return ds.select(list(range(limit)))
    return ds

# ---------- Data Structures ----------

@dataclass
class ExperimentInput:
    """Data structure for a single experiment input, including doc, visuals, prompt, and generation kwargs."""
    doc: Dict
    visuals: List[Any]
    input: str
    gen_kwargs: Dict[str, Any]
    correct_letter: Optional[str]
    question_type: str

@dataclass
class Experiment:
    """Data structure for an experiment, including dataset, config, and requests."""
    name: str
    dataset: Dataset
    experiment_config: ExperimentConfig
    general_config: MainConfig
    requests: List[ExperimentInput] = field(default_factory=list)

    @classmethod
    def from_config(cls, experiment_path: Path, general_config: MainConfig) -> "Experiment":
        """Create an Experiment instance from a config file and general config. Args: experiment_path: Path to the experiment YAML. general_config: MainConfig object. Returns: Experiment instance."""
        exp_conf = load_experiment_config(experiment_path)
        instance = cls(
            name=exp_conf.dataset.name,
            experiment_config=exp_conf,
            general_config=general_config,
            dataset=prepare_dataset(exp_conf, general_config),
        )
        instance._create_requests()
        return instance

    def _create_requests(self) -> None:
        """Populate the requests list for this experiment from the dataset."""
        if not self.dataset:
            raise ValueError(f"Dataset not loaded for experiment '{self.name}'")
        for doc in tqdm(self.dataset, desc=f"Preparing {self.name}"):
            doc_dict = dict(doc)  # Convert for utils
            self.requests.append(
                ExperimentInput(
                    doc=doc_dict,
                    visuals=extract_visuals(doc_dict),
                    input=format_prompt(doc_dict, self.experiment_config.model_input),
                    correct_letter=get_answer_letter(doc_dict, self.experiment_config.model_input),
                    gen_kwargs=self.general_config.models.default_parameters.model_dump(),
                    question_type=self.experiment_config.model_input.question_type,
                )
            )

# ---------- experiment Loading + Pipeline ----------

def load_experiments(config: MainConfig) -> List[Experiment]:
    """Load all experiment YAMLs from the directory in config, filtered by datasets or use_all. Args: config: MainConfig object. Returns: List of Experiment objects."""
    experiments_dir = config.paths.experiments_directory
    use_all = config.experiments.use_all
    allowed_datasets = None if use_all else set(config.experiments.datasets)
    experiments = []

    for experiment_path in experiments_dir.glob("*.yaml"):
        exp_conf = load_experiment_config(experiment_path)
        exp_name = exp_conf.dataset.name
        if allowed_datasets and exp_name not in allowed_datasets:
            continue
        experiments.append(Experiment.from_config(experiment_path, config))

    if not experiments:
        raise ValueError(f"No valid experiments found in '{experiments_dir}'.")

    return experiments

def generate_experiment_inputs(config: MainConfig) -> List[Experiment]:
    """Load all relevant experiments as Experiment dataclass instances using the validated MainConfig. Args: config: MainConfig object. Returns: List of Experiment objects."""
    try:
        experiments = load_experiments(config)
        if not experiments:
            raise ValueError(f"No valid experiments found in {str(config.paths.experiments_directory)}.")
        return experiments
    except Exception as e:
        raise RuntimeError(
            f"Fatal error during experiment generation from {str(config.paths.experiments_directory)}: {e}"
        ) from e