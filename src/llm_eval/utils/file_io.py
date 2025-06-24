import json
from datasets import load_dataset, concatenate_datasets
import yaml

def load_yaml(path) -> dict:
    """Load a YAML file from the given path. Args: path: Path to the YAML file. Returns: Parsed YAML as a dictionary."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_json(filename):
    """Load a JSON file from the given filename. Args: filename: Path to the JSON file. Returns: Parsed JSON as a Python object."""
    with open(filename, 'r') as f:
        return json.load(f)    

def save_json(data, filename):
    """Save a Python object as a JSON file. Args: data: Data to save. filename: Path to the output JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_jsonl(file_path):
    """Load a JSONL (JSON Lines) file. Args: file_path: Path to the JSONL file. Returns: List of parsed JSON objects, one per line."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data
def save_jsonl(file_path, data):
    """Save data as a JSONL file, filtering out non-serializable objects. Args: file_path: Path to the output JSONL file. data: List of serializable objects to write."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')