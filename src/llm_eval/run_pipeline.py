#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import llm_eval.request_generation
import llm_eval.model_inference
import llm_eval.results_processing
import llm_eval.results_aggregation
from llm_eval.utils.configuration import load_main_config

# Robustly search for .env file in several locations
possible_dotenv_paths = [
    Path.cwd() / ".env",
    Path(__file__).parent.parent.parent / ".env",  # project root
    Path.home() / "cabbage" / ".env"
]
dotenv_loaded = False
for dotenv_path in possible_dotenv_paths:
    if dotenv_path.exists():
        print(f"Loading .env from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
        dotenv_loaded = True
        break
if not dotenv_loaded:
    print("Warning: No .env file found in common locations.")

def run_inference(config_path: Path):
    """Run only the model inference step."""
    config = load_main_config(config_path)
    if not config:
        return
    print("\n######### MODELS INFERENCE #########\n")
    llm_eval.model_inference.run_model_inference(config, use_cache=False)

def run_process(config_path: Path):
    """Run only the results processing step."""
    config = load_main_config(config_path)
    if not config:
        return
    print("\n######### PROCESSING RESULTS #########\n")
    llm_eval.results_processing.process_inference_results(config, use_cache=False)

def run_aggregate(config_path: Path):
    """Run only the results aggregation step."""
    config = load_main_config(config_path)
    if not config:
        return
    print("\n######### AGGREGATING RESULTS #########\n")
    llm_eval.results_aggregation.aggregate_results_directory(config, use_cache=False)

def run_pipeline(config_path: Path):
    """Main function to run the model evaluation pipeline (all steps)."""
    run_inference(config_path)
    run_process(config_path)
    run_aggregate(config_path)

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline step to run")

    # Shared config argument
    def add_config_arg(p):
        p.add_argument(
            "-c", "--config",
            type=Path,
            required=True,
            help="Path to the configuration YAML file."
        )

    # Subcommands
    parser_all = subparsers.add_parser("all", help="Run the full pipeline (inference, process, aggregate)")
    add_config_arg(parser_all)
    parser_infer = subparsers.add_parser("inference", help="Run only the model inference step")
    add_config_arg(parser_infer)
    parser_process = subparsers.add_parser("process", help="Run only the results processing step")
    add_config_arg(parser_process)
    parser_agg = subparsers.add_parser("aggregate", help="Run only the results aggregation step")
    add_config_arg(parser_agg)

    args = parser.parse_args()

    if args.command == "all":
        run_pipeline(args.config)
    elif args.command == "inference":
        run_inference(args.config)
    elif args.command == "process":
        run_process(args.config)
    elif args.command == "aggregate":
        run_aggregate(args.config)
    else:
        parser.print_help()
