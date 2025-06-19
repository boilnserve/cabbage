# 🥬 CABBAGE  **Comprehensive Agricultural Benchmark Backed by AI-Guided Evaluation**

*Cabbage* is a modular framework for running, evaluating, and aggregating results from Large Language Models (LLMs) using configurable experiment YAML files. It supports various evaluators, model types, and pipelines—all from a flexible command-line interface.

---

🥬✅ **Code Now Available!** ✅🥬  
The CABBAGE benchmark is now ripe for exploration — our code and evaluation pipelines are live and ready for use! 🌱💻

Dive into the repo and start benchmarking: everything you need to grow your agri-AI models is right here.

You can also explore the **CABBAGE dataset** on Hugging Face:  
👉 [boilnserve/cabbage](https://huggingface.co/datasets/boilnserve/cabbage)

---

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
---

## ⚙️ Configuration

### 1. API Keys

Set your API keys and other secrets in the `.env` file at the project root.  
Edit `.env` and fill in the appropriate values. Example:
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...


---

### 2. Experiment Settings

- All experiment parameters (models, evaluation pipeline, etc.) are defined in `docs/all_experiments.yaml`.
- Edit this file to:
    - Change experiment names, dataset or output paths, etc.
    - Customize which models to evaluate and judge.
    - Modify any other experiment settings.

You can also create or modify additional YAML files within the `docs/` and `experiment_configs/` directories as needed for custom experiments.

---

## 🏃 Usage

After setup and configuration, run your experiment with:
```sh
llm_eval -c docs/all_experiments.yaml
```

This will execute the pipeline as defined in your YAML configuration.

---

We welcome feedback, contributions, and curious minds.  
If you find it useful, give us a ⭐️ or keep an 👀 on updates!

🥬🌾 Happy harvesting!
