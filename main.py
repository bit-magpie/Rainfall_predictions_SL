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
from experiments.run_experiment import run_gru_experiment, run_cnn_lstm_experiment, run_transformer_experiment
# Import functions for multi-output models
from experiments.run_experiment import run_gru_multi_output_experiment, run_lstm_multi_output_experiment
from experiments.run_experiment import run_cnn_lstm_multi_output_experiment, run_transformer_multi_output_experiment


def main():
    parser = argparse.ArgumentParser(description="Weather Prediction Model Training and Evaluation")
    parser.add_argument(
        "--model", 
        type=str,
        choices=[
            'linear', 'lstm', 'gru', 'cnn_lstm', 'transformer', 
            'gru_multi_output', 'lstm_multi_output', 'cnn_lstm_multi_output', 'transformer_multi_output',
            'all', 'all_single', 'all_multi'
        ],
        default='all',
        help="Model type to train. Use 'all_single' for all single-output models, 'all_multi' for all multi-output models, or 'all' for all models."
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
    
    # Define model groups
    single_output_models = ['linear', 'lstm', 'gru', 'cnn_lstm', 'transformer']
    multi_output_models = ['gru_multi_output', 'lstm_multi_output', 'cnn_lstm_multi_output', 'transformer_multi_output']
    
    # Determine which models to run
    models_to_run = []
    if args.model.lower() == 'all':
        models_to_run = single_output_models + multi_output_models
    elif args.model.lower() == 'all_single':
        models_to_run = single_output_models
    elif args.model.lower() == 'all_multi':
        models_to_run = multi_output_models
    else:
        models_to_run = [args.model.lower()]
    
    # Run Linear model experiment if requested
    if 'linear' in models_to_run:
        linear_config_path = config_dir / "linear_config.yaml"
        if linear_config_path.exists():
            print(f"\n{'='*50}\nRunning Linear Model Experiment\n{'='*50}")
            linear_config = load_config(linear_config_path)
            run_linear_experiment(linear_config)
        else:
            print(f"Error: Linear model config not found at {linear_config_path}")
    
    # Run LSTM model experiment if requested
    if 'lstm' in models_to_run:
        lstm_config_path = config_dir / "lstm_config.yaml"
        if lstm_config_path.exists():
            print(f"\n{'='*50}\nRunning LSTM Model Experiment\n{'='*50}")
            lstm_config = load_config(lstm_config_path)
            run_lstm_experiment(lstm_config)
        else:
            print(f"Error: LSTM model config not found at {lstm_config_path}")
    
    # Run GRU model experiment if requested
    if 'gru' in models_to_run:
        gru_config_path = config_dir / "gru_config.yaml"
        if gru_config_path.exists():
            print(f"\n{'='*50}\nRunning GRU Model Experiment\n{'='*50}")
            gru_config = load_config(gru_config_path)
            run_gru_experiment(gru_config)
        else:
            print(f"Error: GRU model config not found at {gru_config_path}")
    
    # Run CNN-LSTM model experiment if requested
    if 'cnn_lstm' in models_to_run:
        cnn_lstm_config_path = config_dir / "cnn_lstm_config.yaml"
        if cnn_lstm_config_path.exists():
            print(f"\n{'='*50}\nRunning CNN-LSTM Model Experiment\n{'='*50}")
            cnn_lstm_config = load_config(cnn_lstm_config_path)
            run_cnn_lstm_experiment(cnn_lstm_config)
        else:
            print(f"Error: CNN-LSTM model config not found at {cnn_lstm_config_path}")
    
    # Run Transformer model experiment if requested
    if 'transformer' in models_to_run:
        transformer_config_path = config_dir / "transformer_config.yaml"
        if transformer_config_path.exists():
            print(f"\n{'='*50}\nRunning Transformer Model Experiment\n{'='*50}")
            transformer_config = load_config(transformer_config_path)
            run_transformer_experiment(transformer_config)
        else:
            print(f"Error: Transformer model config not found at {transformer_config_path}")
    
    # Run GRU Multi-Output model experiment if requested
    if 'gru_multi_output' in models_to_run:
        gru_multi_output_config_path = config_dir / "gru_multi_output_config.yaml"
        if gru_multi_output_config_path.exists():
            print(f"\n{'='*50}\nRunning GRU Multi-Output Model Experiment\n{'='*50}")
            gru_multi_output_config = load_config(gru_multi_output_config_path)
            run_gru_multi_output_experiment(gru_multi_output_config)
        else:
            print(f"Error: GRU Multi-Output model config not found at {gru_multi_output_config_path}")
    
    # Run LSTM Multi-Output model experiment if requested
    if 'lstm_multi_output' in models_to_run:
        lstm_multi_output_config_path = config_dir / "lstm_multi_output_config.yaml"
        if lstm_multi_output_config_path.exists():
            print(f"\n{'='*50}\nRunning LSTM Multi-Output Model Experiment\n{'='*50}")
            lstm_multi_output_config = load_config(lstm_multi_output_config_path)
            run_lstm_multi_output_experiment(lstm_multi_output_config)
        else:
            print(f"Error: LSTM Multi-Output model config not found at {lstm_multi_output_config_path}")
    
    # Run CNN-LSTM Multi-Output model experiment if requested
    if 'cnn_lstm_multi_output' in models_to_run:
        cnn_lstm_multi_output_config_path = config_dir / "cnn_lstm_multi_output_config.yaml"
        if cnn_lstm_multi_output_config_path.exists():
            print(f"\n{'='*50}\nRunning CNN-LSTM Multi-Output Model Experiment\n{'='*50}")
            cnn_lstm_multi_output_config = load_config(cnn_lstm_multi_output_config_path)
            run_cnn_lstm_multi_output_experiment(cnn_lstm_multi_output_config)
        else:
            print(f"Error: CNN-LSTM Multi-Output model config not found at {cnn_lstm_multi_output_config_path}")
    
    # Run Transformer Multi-Output model experiment if requested
    if 'transformer_multi_output' in models_to_run:
        transformer_multi_output_config_path = config_dir / "transformer_multi_output_config.yaml"
        if transformer_multi_output_config_path.exists():
            print(f"\n{'='*50}\nRunning Transformer Multi-Output Model Experiment\n{'='*50}")
            transformer_multi_output_config = load_config(transformer_multi_output_config_path)
            run_transformer_multi_output_experiment(transformer_multi_output_config)
        else:
            print(f"Error: Transformer Multi-Output model config not found at {transformer_multi_output_config_path}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    
    # Create directories for each model's results
    model_types = [
        'linear', 'lstm', 'gru', 'cnn_lstm', 'transformer', 
        'gru_multi_output', 'lstm_multi_output', 'cnn_lstm_multi_output', 'transformer_multi_output'
    ]
    for model_type in model_types:
        os.makedirs(f"results/figures/{model_type}", exist_ok=True)
    
    # Move combined_data.csv to data directory if it's not there
    data_dir = Path("data")
    data_file = Path("combined_data.csv")
    if data_file.exists() and not (data_dir / data_file).exists():
        os.makedirs(data_dir, exist_ok=True)
        import shutil
        shutil.copy(data_file, data_dir / data_file)
        print(f"Copied {data_file} to {data_dir / data_file}")
    
    main()