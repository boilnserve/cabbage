# 🥬 CABBAGE  **Comprehensive Agricultural Benchmark Backed by AI-Guided Evaluation**

*Cabbage* is a modular framework for running, evaluating, and aggregating results from Large Language Models (LLMs) using configurable experiment YAML files. It supports various evaluators, model types, and pipelines—all from a flexible command-line interface.

---

🥬✅ **Code Now Available!** ✅🥬  
The CABBAGE benchmark is now ripe for exploration — our code and evaluation pipelines are live and ready for use! 🌱💻

Dive into the repo and start benchmarking: everything you need to grow your agri-AI models is right here.

You can also explore the **CABBAGE dataset** on Hugging Face:  
👉 [boilnserve/cabbage](https://huggingface.co/datasets/boilnserve/cabbage)


## 🚀 Installation

1. Clone the repository and enter the directory:
    ```sh
    git clone https://github.com/boilnserve/cabbage.git
    cd cabbage
    ```

2. Install the package in editable mode:
    ```sh
    pip install -e .
    ```

## ⚙️ Configuration

### 1. API Keys

Set your API keys and other secrets in the `.env` file at the project root.  
Edit `.env` and fill in the appropriate values. Example:
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...



### 2. Experiment Settings

- All experiment parameters (models, evaluation pipeline, etc.) are defined in `configs/all_experiments.yaml`.
- Edit this file to:
    - Change experiment names, dataset or output paths, etc.
    - Customize which models to evaluate and judge.
    - Modify any other experiment settings.
- You can also create or modify additional YAML files within the `configs/` and `experiment_configs/` directories as needed for custom experiments.



## 🏃 Usage

After setup and configuration, run your experiment pipeline with:
```sh
llm_eval all -c configs/all_experiments.yaml
```

Or run individual steps:
```sh
llm_eval inference -c configs/all_experiments.yaml   # Only inference
llm_eval process -c configs/all_experiments.yaml     # Only process results
llm_eval aggregate -c configs/all_experiments.yaml   # Only aggregate results
```

- The `-c`/`--config` flag specifies the main YAML config file.
- All experiment and model settings are controlled via YAML files in `configs/` and `experiment_configs/`.


## 🧩 Extending Cabbage

- **Add new experiments**: Place new YAML files in `experiment_configs/` and reference them in your main config.
- **Add new models**: Edit the `models` section in your config YAML and provide API keys in `.env` as needed.
- **Custom evaluators**: Implement a new evaluator class in `evaluators.py` and register it in `EVALUATOR_REGISTRY`.
- **Custom pipelines**: Modify or extend `run_pipeline.py` for new workflows.

---

We welcome feedback, contributions, and curious minds.  
If you find it useful, give us a ⭐️ or keep an 👀 on updates!

🥬🌾 Happy harvesting!
