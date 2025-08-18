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
    Parses command-line arguments to load a YAML configuration file.

    The path to the configuration file is expected as a command-line
    argument `--config`.

    Returns:
        Dict[str, Any]: A dictionary containing the experiment configuration.
    """
    parser = argparse.ArgumentParser(description="Train BioEmb or baseline models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bbbp.yaml",
        help="Path to the YAML configuration file for the experiment.",
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config


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

