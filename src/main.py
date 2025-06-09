#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv
from pathlib import Path

import generate_requests
import models_inference
import process_results
import aggregate_results

import logging

# Set the logging level to WARNING to reduce verbosity (INFO and DEBUG messages will be suppressed)
logging.basicConfig(level=logging.WARNING)

for logger_name in [
    "generate_requests",
    "models_inference",
    "process_results",
    "aggregate_results"
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
    
# Optionally silence third-party libraries like `rich` or `tqdm`
logging.getLogger("tqdm").setLevel(logging.WARNING)


def main(config_path: Path):
    """Main function to run the model evaluation pipeline."""
    
    
    # Load environment variables
    load_dotenv()
    
    print("\n######### DATASET PREPARATION ##############\n")
    # Generate requests based on the config file
    experiments = generate_requests.generate_requests(config_path)

    print("\n######### MODELS INFERENCE ##############\n")
    # Run models inference
    models_inference.models_inference(config_path, experiments)

    print("\n######### PROCESSING RESULTS ##############\n")
    # Process results
    process_results.process_results(config_path)

    print("\n######### AGGREGATING RESULTS ##############\n")
    # Aggregate results (without using cache)
    aggregate_results.aggregate_results_directory(config_path, use_cache=False)

if __name__ == "__main__":
    # Argument parser for the command-line interface
    parser = argparse.ArgumentParser(description="Evaluate models over a dataset using a specified config file.")
    parser.add_argument(
        "-c", "--config", 
        type=Path, 
        help="Path to the configuration YAML file."
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.config:
        args.config = Path(os.path.join(os.path.dirname(__file__), '..', 'docs', 'config.yaml'))
    
    # Run the main pipeline with the provided config path
    main(args.config)
