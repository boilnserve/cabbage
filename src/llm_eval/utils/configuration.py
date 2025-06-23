from pydantic import BaseModel, field_validator, ValidationError
from typing import Optional, List, Dict
from pathlib import Path
from llm_eval.utils.file_io import load_yaml
from loguru import logger
import os


def validate_api_key_env(v: Optional[str]) -> Optional[str]:
    if v:
        value = os.environ.get(v)
        if value is None or value.strip() == "":
            raise ValueError(f"Environment variable '{v}' is missing or empty (modify the .env file).")
        return value
    return v

# Add a new handler, print only the message (plus optional coloring)
class ProviderConfig(BaseModel):
    name: str
    model: str
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    timeout: Optional[int] = None
    server_args: Optional[dict] = None
    
    @field_validator("api_key_env_var")
    @classmethod
    def validate_api_key_env_var(cls, v):
        return validate_api_key_env(v)

class EvaluatorConfig(BaseModel):
    name: str
    model: str
    base_url: str
    api_key_env_var: str
    
    @field_validator("api_key_env_var")
    @classmethod
    def validate_api_key_env_var(cls, v):
        return validate_api_key_env(v)

class DefaultParameters(BaseModel):
    max_new_tokens: int
    temperature: float

# ---- PATHS ----
class PathsConfig(BaseModel):
    experiments_directory: Path
    results_directory: Path
    evaluation_prompts: Path

   # Individual field validators
    @field_validator('experiments_directory')
    def check_experiments_directory_exists(cls, v):
        if not Path(v).is_dir():
            raise ValueError(f"ERROR: experiments_directory '{v}' does not exist or is not a directory.")
        return v

    @field_validator('evaluation_prompts')
    def check_evaluation_prompts_exists(cls, v):
        if not Path(v).is_file():
            raise ValueError(f"ERROR: evaluation_prompts '{v}' does not exist or is not a file.")
        return v

    @field_validator('results_directory')
    @classmethod
    def check_results_dir_exists_or_create(cls, v):
        v = Path(v)
        if v.exists():
            print(f"[INFO] results_directory '{v}' exists.")
        else:
            v.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] results_directory '{v}' did not exist and was created.")
        return v

# ---- TOP-LEVEL EXPERIMENTS & MODELS ----
class ExperimentsConfig(BaseModel):
    limit: Optional[int] = 0
    use_all: Optional[bool] = False
    datasets: List[str]

class ModelBlockConfig(BaseModel):
    default_parameters: DefaultParameters
    providers: List[ProviderConfig]

class EvaluatorsBlockConfig(BaseModel):
    default_parameters: DefaultParameters
    providers: List[EvaluatorConfig]
    prompts_dict: Dict[str,str] = {}

class MainConfig(BaseModel):
    paths: PathsConfig
    experiments: ExperimentsConfig
    models: ModelBlockConfig
    evaluators: EvaluatorsBlockConfig

    @field_validator("paths", mode="after")
    @classmethod
    def check_paths(cls, v):
        return v

# ---- NEW: DATASET CONFIG ----
class DatasetConfig(BaseModel):
    name: str
    path: Optional[str] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    filters: Optional[Dict] = None

# ---- EXPERIMENT FILE SCHEMA ----
class ModelInputConfig(BaseModel):
    question_type: str
    pre_prompt: Optional[str] = ""
    post_prompt: Optional[str] = ""
    difficulty: Optional[str] = None

class ProcessResultsConfig(BaseModel):
    metric: str
    evaluation_type: Optional[str] = None

class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    model_input: ModelInputConfig
    process_results: ProcessResultsConfig
    
    class Config:
        protected_namespaces = ()

# ---- YAML LOADERS ----

def load_main_config(path: Path) -> MainConfig | None:
    try:
        raw = load_yaml(path)
        config = MainConfig(**raw)
        # Load the prompts and attach as an attribute
        config.evaluators.prompts_dict = load_yaml(config.paths.evaluation_prompts)
        return config
    except ValidationError as e:
        print("\nCONFIGURATION ERROR in main config file:", path)
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err['loc'])
            msg = err['msg']
            type_ = err.get("type", "unknown")
            print(f"  - [{loc}] {msg} (type: {type_})")
        print("\nPlease check your configuration file and correct the missing or invalid fields.\n")
        return None


def load_experiment_config(path: Path) -> ExperimentConfig:
    try:
        raw = load_yaml(path)
        return ExperimentConfig(**raw)
    except ValidationError as e:
        print(f"\nCONFIGURATION ERROR in experiment config file: {path}")
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err['loc'])
            msg = err['msg']
            type_ = err.get("type", "unknown")
            print(f"  - [{loc}] {msg} (type: {type_})")
        print("\nPlease fix the above issue(s) in:", path.name)
        raise ValueError(f"Failed to load configuration from {path}")

def load_experiments_config(experiments_dir: Path) -> Dict[str, ExperimentConfig]:
    config_dict = {}
    for file in experiments_dir.glob("*.yaml"):
        try:
            config = load_experiment_config(file)
            if config:
                name = config.dataset.name
                if name:
                    config_dict[name] = config
                else:
                    logger.warning(f"No dataset name in {file.name}, skipping.")
            else:
                logger.warning(f"Problem in the configuration loading for file: {file}.")
        except Exception as e:
            logger.warning(f"Failed loading {file}: {e}")
    return config_dict
