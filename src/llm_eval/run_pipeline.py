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
# General TODO
#   1. Add docstring to each function
#   2. caching in different files for each passage
#   3. create a command for each step separately


# Explicitly specify the path to your .env file
dotenv_path = Path.home() / "cabbage_eval" / ".env"
print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

def run_pipeline(config_path: Path):
    """Main function to run the model evaluation pipeline."""
    
    config = load_main_config(config_path)
    if not config:
        return

    print("\n######### MODELS INFERENCE #########\n")
    # Run models inference
    llm_eval.model_inference.run_model_inference(config, use_cache=False)

    print("\n######### PROCESSING RESULTS #########\n")
    # Process results
    llm_eval.results_processing.process_inference_results(config, use_cache=False)

    print("\n######### AGGREGATING RESULTS #########\n")
    # Aggregate results (without using cache)
    llm_eval.results_aggregation.aggregate_results_directory(config, use_cache=False)

def main():
    # Argument parser for the command-line interface
    parser = argparse.ArgumentParser(description="Evaluate models over a dataset using a specified config file.")
    parser.add_argument(
        "-c", "--config", 
        type=Path, 
        required=True,
        help="Path to the configuration YAML file."
    )

    # Parse arguments
    args = parser.parse_args()

    
    # Run the main pipeline with the provided config path
    run_pipeline(args.config)
