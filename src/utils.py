import json
from datasets import load_dataset, concatenate_datasets
import yaml

def load_yaml(path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

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

# def remove_non_serializable(obj):
#     """Recursively remove non-serializable objects from dictionaries and lists."""
#     if isinstance(obj, dict):
#         # For dictionaries, only include serializable key-value pairs
#         cleaned_dict = {}
#         for k, v in obj.items():
#             if is_serializable(v):
#                 cleaned_dict[k] = remove_non_serializable(v)
#         return cleaned_dict
#     elif isinstance(obj, list):
#         # For lists, include only serializable items
#         return [remove_non_serializable(item) for item in obj if is_serializable(item)]
#     else:
#         # Return the object if it's not a dict or list
#         return obj

# def is_serializable(obj):
#     """Check if the object is serializable by attempting to dump it as JSON."""
#     if isinstance(obj,dict):
#         return True
#     try:
#         json.dumps(obj)
#         return True
#     except (TypeError, OverflowError):
#         return False
 
# def load_dataset_dict(folder_path, concat=False):
#     dataset_name = "parquet"
#     data_files = {
#         "dev": f"{folder_path}/dev-0*.parquet",
#         "test": f"{folder_path}/test-*.parquet",
#         "validation": f"{folder_path}/validation-*.parquet"
#     }
#     try:
#         dataset = load_dataset(dataset_name, data_files=data_files)
#     except Exception as e:
#         print(f"Error loading dataset on path {folder_path}: {e}")
#         return None
#     if concat:
#         return concatenate_datasets([dataset['dev'], dataset['test'], dataset['validation']])
#     return dataset