"""
This module provides utility functions for the BioEmb project, including:
- Parsing configuration files.
- A custom Hugging Face Trainer callback for detailed CSV logging.
- A helper function for setting random seeds for reproducibility.
"""

import os
import csv
import argparse
import yaml
import random
import numpy as np
import torch
from transformers import TrainerCallback
from typing import Dict, Any

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_config() -> Dict[str, Any]:
    """
    Loads a YAML config file and overrides its values with command-line arguments.

    1.  Parses the --config argument to find the YAML file path.
    2.  Loads the YAML file.
    3.  Dynamically adds arguments to the parser based on the YAML keys.
    4.  Parses all arguments again, allowing command-line args to override YAML values.

    Returns:
        Dict[str, Any]: The final configuration dictionary.
    """
    parser = argparse.ArgumentParser(description="Train a model with YAML config and overrides.")

    # Step 1: Add the --config argument first
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bbbp.yaml",
        help="Path to the YAML configuration file for the experiment.",
    )

    # Step 2: Parse only the known arguments (i.e., --config)
    # The 'remaining_argv' will hold any other arguments like --dataset
    args, remaining_argv = parser.parse_known_args()

    # Step 3: Load the YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Step 4: Dynamically add arguments to the parser based on YAML keys
    for key, value in config.items():
        # Infer the type from the default value
        arg_type = type(value) if value is not None else str
        # Special handling for boolean flags
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true', default=value)
        else:
            parser.add_argument(f'--{key}', type=arg_type, default=value)

    # Step 5: Parse all arguments again. This time, it will recognize the new ones.
    # The values from 'remaining_argv' will override the defaults set from the YAML file.
    final_args = parser.parse_args(remaining_argv)

    # Convert the final argparse namespace to a dictionary
    final_config = vars(final_args)

    # Don't forget to include the original config path if you need it
    final_config['config_path'] = args.config

    return final_config


class EvalLoggingCallback(TrainerCallback):
    """
    A Hugging Face TrainerCallback to log evaluation metrics to a CSV file.

    This callback ensures that evaluation results from each evaluation step
    are saved in a structured format, making it easy to track model performance
    over time. It handles dynamic addition of new metric columns.
    """

    def __init__(self, output_dir: str, filename: str = "evaluation_log.csv"):
        """
        Initializes the callback.

        Args:
            output_dir (str): Directory where the CSV log file will be saved.
            filename (str): The name of the CSV file.
        """
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, filename)
        self._headers_written = False

    def on_evaluate(self, args, state, control, metrics: Dict[str, float] = None, **kwargs):
        """
        Logs metrics to the CSV file after an evaluation is completed.
        """
        if metrics is None:
            return

        log_metrics = metrics.copy()
        log_metrics['step'] = state.global_step

        os.makedirs(self.output_dir, exist_ok=True)

        # Write headers if the file doesn't exist, otherwise append.
        write_mode = 'w' if not os.path.exists(self.csv_path) else 'a'
        
        with open(self.csv_path, write_mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(log_metrics.keys()))
            if write_mode == 'w' or not self._headers_written:
                writer.writeheader()
                self._headers_written = True
            writer.writerow(log_metrics)

        print(f"Logged evaluation metrics at step {state.global_step} to {self.csv_path}")

