#!/usr/bin/env python3
"""
Main entry point for the weather prediction project.
This script allows running experiments with different models and configurations.
"""

import argparse
import sys
import os
from pathlib import Path

# Make sure we can import from other modules
sys.path.append(str(Path(__file__).parent))

from experiments.run_experiment import load_config, run_linear_experiment, run_lstm_experiment


def main():
    parser = argparse.ArgumentParser(description="Weather Prediction Model Training and Evaluation")
    parser.add_argument(
        "--model", 
        type=str,
        choices=['linear', 'lstm', 'all'],
        default='all',
        help="Model type to train (linear, lstm, or all)"
    )
    parser.add_argument(
        "--config_dir", 
        type=str, 
        default="experiments/configs",
        help="Directory containing model configuration files"
    )
    
    args = parser.parse_args()
    
    # Get configuration file paths
    config_dir = Path(args.config_dir)
    
    if args.model.lower() == 'linear' or args.model.lower() == 'all':
        linear_config_path = config_dir / "linear_config.yaml"
        if linear_config_path.exists():
            print(f"\n{'='*50}\nRunning Linear Model Experiment\n{'='*50}")
            linear_config = load_config(linear_config_path)
            run_linear_experiment(linear_config)
        else:
            print(f"Error: Linear model config not found at {linear_config_path}")
    
    if args.model.lower() == 'lstm' or args.model.lower() == 'all':
        lstm_config_path = config_dir / "lstm_config.yaml"
        if lstm_config_path.exists():
            print(f"\n{'='*50}\nRunning LSTM Model Experiment\n{'='*50}")
            lstm_config = load_config(lstm_config_path)
            run_lstm_experiment(lstm_config)
        else:
            print(f"Error: LSTM model config not found at {lstm_config_path}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures/linear", exist_ok=True)
    os.makedirs("results/figures/lstm", exist_ok=True)
    
    # Move combined_data.csv to data directory if it's not there
    data_dir = Path("data")
    data_file = Path("combined_data.csv")
    if data_file.exists() and not (data_dir / data_file).exists():
        os.makedirs(data_dir, exist_ok=True)
        import shutil
        shutil.copy(data_file, data_dir / data_file)
        print(f"Copied {data_file} to {data_dir / data_file}")
    
    main()