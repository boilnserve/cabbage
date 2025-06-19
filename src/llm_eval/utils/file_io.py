import json
from datasets import load_dataset, concatenate_datasets
import yaml

def load_yaml(path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)    

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data
def save_jsonl(file_path, data):
    """Save data as a JSONL file, filtering out non-serializable objects."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')