import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.linear.model import LinearRegressionModel, train as train_linear, predict as predict_linear, get_feature_importance
from models.lstm.model import LSTMModel, train as train_lstm, predict as predict_lstm, save_model
from models.gru.model import GRUModel, train as train_gru, predict as predict_gru
from models.cnn_lstm.model import CNNLSTMModel, train as train_cnn_lstm, predict as predict_cnn_lstm
from models.transformer.model import TransformerModel, train as train_transformer, predict as predict_transformer
from utils.data_utils import prepare_data, inverse_transform_target
from utils.metrics import calculate_metrics, save_metrics
from utils.plot_utils import plot_training_history, plot_predictions, plot_scatter, plot_feature_importance


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(data_dict, batch_size=32):
    """Create PyTorch DataLoaders from data dictionary."""
    # Training data
    train_dataset = TensorDataset(
        torch.FloatTensor(data_dict['X_train']),
        torch.FloatTensor(data_dict['y_train'])
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation data
    val_dataset = TensorDataset(
        torch.FloatTensor(data_dict['X_val']),
        torch.FloatTensor(data_dict['y_val'])
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Test data
    test_dataset = TensorDataset(
        torch.FloatTensor(data_dict['X_test']),
        torch.FloatTensor(data_dict['y_test'])
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def run_linear_experiment(config):
    """Run experiment with linear model."""
    print("Running linear model experiment...")
    
    # Prepare data
    data_dict = prepare_data(
        file_path=config['data']['file_path'],
        target_col=config['data']['target_col'],
        seq_length=config['data']['seq_length'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict, batch_size=config['data']['batch_size']
    )
    
    # Get input dimensions
    num_features = data_dict['X_train'].shape[2]
    seq_length = data_dict['X_train'].shape[1]
    input_dim = num_features * seq_length
    
    # Create model
    model = LinearRegressionModel(
        input_dim=input_dim,
        output_dim=config['model']['params']['output_dim'],
        hidden_dims=config['model']['params'].get('hidden_dims', None)
    )
    
    # Train model
    history = train_linear(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Evaluate model
    y_true, y_pred = predict_linear(model, test_loader)
    
    # Inverse transform predictions and true values
    y_true_original = inverse_transform_target(y_true, data_dict['scaler_y'])
    y_pred_original = inverse_transform_target(y_pred, data_dict['scaler_y'])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_original, y_pred_original)
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(config['paths']['model_save_path']), exist_ok=True)
    os.makedirs(config['paths']['figures_path'], exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), config['paths']['model_save_path'])
    print(f"Model saved to {config['paths']['model_save_path']}")
    
    # Save metrics
    save_metrics(
        metrics_dict=metrics,
        model_name="linear",
        config=config['model']['params'],
        file_path=config['paths']['metrics_save_path']
    )
    
    # Plot and save training history
    plot_training_history(
        history,
        title="Linear Model Training History",
        save_path=os.path.join(config['paths']['figures_path'], "training_history.png")
    )
    
    # Plot and save predictions
    plot_predictions(
        y_true_original, y_pred_original,
        title="Linear Model Predictions",
        save_path=os.path.join(config['paths']['figures_path'], "predictions.png")
    )
    
    # Plot and save scatter
    plot_scatter(
        y_true_original, y_pred_original,
        title="Linear Model: Predicted vs Actual",
        save_path=os.path.join(config['paths']['figures_path'], "scatter.png")
    )
    
    # Plot feature importance if it's a simple linear model
    try:
        importance = get_feature_importance(model)
        # Replicate feature names for each time step
        feature_names = data_dict['feature_names']
        all_features = []
        for i in range(seq_length):
            all_features.extend([f"{name}_{i+1}" for name in feature_names])
            
        plot_feature_importance(
            all_features, importance,
            title="Linear Model Feature Importance",
            save_path=os.path.join(config['paths']['figures_path'], "feature_importance.png")
        )
    except ValueError as e:
        print(f"Warning: {e}")


def run_lstm_experiment(config):
    """Run experiment with LSTM model."""
    print("Running LSTM model experiment...")
    
    # Prepare data
    data_dict = prepare_data(
        file_path=config['data']['file_path'],
        target_col=config['data']['target_col'],
        seq_length=config['data']['seq_length'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict, batch_size=config['data']['batch_size']
    )
    
    # Get input size
    input_size = data_dict['X_train'].shape[2]  # Number of features
    
    # Create model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=config['model']['params']['hidden_size'],
        num_layers=config['model']['params']['num_layers'],
        output_size=config['model']['params']['output_size'],
        dropout=config['model']['params']['dropout']
    )
    
    # Train model
    history = train_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        clip_grad=config['training']['clip_grad']
    )
    
    # Evaluate model
    y_true, y_pred = predict_lstm(model, test_loader)
    
    # Inverse transform predictions and true values
    y_true_original = inverse_transform_target(y_true, data_dict['scaler_y'])
    y_pred_original = inverse_transform_target(y_pred, data_dict['scaler_y'])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_original, y_pred_original)
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(config['paths']['model_save_path']), exist_ok=True)
    os.makedirs(config['paths']['figures_path'], exist_ok=True)
    
    # Save model
    save_model(model, config['paths']['model_save_path'])
    print(f"Model saved to {config['paths']['model_save_path']}")
    
    # Save metrics
    save_metrics(
        metrics_dict=metrics,
        model_name="lstm",
        config=config['model']['params'],
        file_path=config['paths']['metrics_save_path']
    )
    
    # Plot and save training history
    plot_training_history(
        history,
        title="LSTM Model Training History",
        save_path=os.path.join(config['paths']['figures_path'], "training_history.png")
    )
    
    # Plot and save predictions
    plot_predictions(
        y_true_original, y_pred_original,
        title="LSTM Model Predictions",
        save_path=os.path.join(config['paths']['figures_path'], "predictions.png")
    )
    
    # Plot and save scatter
    plot_scatter(
        y_true_original, y_pred_original,
        title="LSTM Model: Predicted vs Actual",
        save_path=os.path.join(config['paths']['figures_path'], "scatter.png")
    )


def run_gru_experiment(config):
    """Run experiment with GRU model."""
    print("Running GRU model experiment...")
    
    # Prepare data
    data_dict = prepare_data(
        file_path=config['data']['file_path'],
        target_col=config['data']['target_col'],
        seq_length=config['data']['seq_length'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict, batch_size=config['data']['batch_size']
    )
    
    # Get input size
    input_size = data_dict['X_train'].shape[2]  # Number of features
    
    # Create model
    model = GRUModel(
        input_size=input_size,
        hidden_size=config['model']['params']['hidden_size'],
        num_layers=config['model']['params']['num_layers'],
        output_size=config['model']['params']['output_size'],
        dropout=config['model']['params']['dropout'],
        bidirectional=config['model']['params'].get('bidirectional', False)
    )
    
    # Train model
    history = train_gru(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        clip_grad=config['training']['clip_grad']
    )
    
    # Evaluate model
    y_true, y_pred = predict_gru(model, test_loader)
    
    # Inverse transform predictions and true values
    y_true_original = inverse_transform_target(y_true, data_dict['scaler_y'])
    y_pred_original = inverse_transform_target(y_pred, data_dict['scaler_y'])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_original, y_pred_original)
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(config['paths']['model_save_path']), exist_ok=True)
    os.makedirs(config['paths']['figures_path'], exist_ok=True)
    
    # Save model
    save_model(model, config['paths']['model_save_path'])
    print(f"Model saved to {config['paths']['model_save_path']}")
    
    # Save metrics
    save_metrics(
        metrics_dict=metrics,
        model_name="gru",
        config=config['model']['params'],
        file_path=config['paths']['metrics_save_path']
    )
    
    # Plot and save training history
    plot_training_history(
        history,
        title="GRU Model Training History",
        save_path=os.path.join(config['paths']['figures_path'], "training_history.png")
    )
    
    # Plot and save predictions
    plot_predictions(
        y_true_original, y_pred_original,
        title="GRU Model Predictions",
        save_path=os.path.join(config['paths']['figures_path'], "predictions.png")
    )
    
    # Plot and save scatter
    plot_scatter(
        y_true_original, y_pred_original,
        title="GRU Model: Predicted vs Actual",
        save_path=os.path.join(config['paths']['figures_path'], "scatter.png")
    )


def run_cnn_lstm_experiment(config):
    """Run experiment with CNN-LSTM model."""
    print("Running CNN-LSTM model experiment...")
    
    # Prepare data
    data_dict = prepare_data(
        file_path=config['data']['file_path'],
        target_col=config['data']['target_col'],
        seq_length=config['data']['seq_length'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict, batch_size=config['data']['batch_size']
    )
    
    # Get input dimensions
    input_size = data_dict['X_train'].shape[2]  # Number of features
    seq_length = data_dict['X_train'].shape[1]  # Sequence length
    
    # Create model
    model = CNNLSTMModel(
        input_size=input_size,
        seq_length=seq_length,
        cnn_filters=config['model']['params']['cnn_filters'],
        kernel_size=config['model']['params']['kernel_size'],
        lstm_hidden=config['model']['params']['lstm_hidden'],
        lstm_layers=config['model']['params']['lstm_layers'],
        output_size=config['model']['params']['output_size'],
        dropout=config['model']['params']['dropout']
    )
    
    # Train model
    history = train_cnn_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        clip_grad=config['training']['clip_grad']
    )
    
    # Evaluate model
    y_true, y_pred = predict_cnn_lstm(model, test_loader)
    
    # Inverse transform predictions and true values
    y_true_original = inverse_transform_target(y_true, data_dict['scaler_y'])
    y_pred_original = inverse_transform_target(y_pred, data_dict['scaler_y'])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_original, y_pred_original)
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(config['paths']['model_save_path']), exist_ok=True)
    os.makedirs(config['paths']['figures_path'], exist_ok=True)
    
    # Save model
    save_model(model, config['paths']['model_save_path'])
    print(f"Model saved to {config['paths']['model_save_path']}")
    
    # Save metrics
    save_metrics(
        metrics_dict=metrics,
        model_name="cnn_lstm",
        config=config['model']['params'],
        file_path=config['paths']['metrics_save_path']
    )
    
    # Plot and save training history
    plot_training_history(
        history,
        title="CNN-LSTM Model Training History",
        save_path=os.path.join(config['paths']['figures_path'], "training_history.png")
    )
    
    # Plot and save predictions
    plot_predictions(
        y_true_original, y_pred_original,
        title="CNN-LSTM Model Predictions",
        save_path=os.path.join(config['paths']['figures_path'], "predictions.png")
    )
    
    # Plot and save scatter
    plot_scatter(
        y_true_original, y_pred_original,
        title="CNN-LSTM Model: Predicted vs Actual",
        save_path=os.path.join(config['paths']['figures_path'], "scatter.png")
    )


def run_transformer_experiment(config):
    """Run experiment with Transformer model."""
    print("Running Transformer model experiment...")
    
    # Prepare data
    data_dict = prepare_data(
        file_path=config['data']['file_path'],
        target_col=config['data']['target_col'],
        seq_length=config['data']['seq_length'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict, batch_size=config['data']['batch_size']
    )
    
    # Get input size
    input_size = data_dict['X_train'].shape[2]  # Number of features
    
    # Create model
    model = TransformerModel(
        input_size=input_size,
        d_model=config['model']['params']['d_model'],
        nhead=config['model']['params']['nhead'],
        num_layers=config['model']['params']['num_layers'],
        dim_feedforward=config['model']['params']['dim_feedforward'],
        output_size=config['model']['params']['output_size'],
        dropout=config['model']['params']['dropout']
    )
    
    # Train model
    history = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm']
    )
    
    # Evaluate model
    y_true, y_pred = predict_transformer(model, test_loader)
    
    # Inverse transform predictions and true values
    y_true_original = inverse_transform_target(y_true, data_dict['scaler_y'])
    y_pred_original = inverse_transform_target(y_pred, data_dict['scaler_y'])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_original, y_pred_original)
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(config['paths']['model_save_path']), exist_ok=True)
    os.makedirs(config['paths']['figures_path'], exist_ok=True)
    
    # Save model
    save_model(model, config['paths']['model_save_path'])
    print(f"Model saved to {config['paths']['model_save_path']}")
    
    # Save metrics
    save_metrics(
        metrics_dict=metrics,
        model_name="transformer",
        config=config['model']['params'],
        file_path=config['paths']['metrics_save_path']
    )
    
    # Plot and save training history
    plot_training_history(
        history,
        title="Transformer Model Training History",
        save_path=os.path.join(config['paths']['figures_path'], "training_history.png")
    )
    
    # Plot and save predictions
    plot_predictions(
        y_true_original, y_pred_original,
        title="Transformer Model Predictions",
        save_path=os.path.join(config['paths']['figures_path'], "predictions.png")
    )
    
    # Plot and save scatter
    plot_scatter(
        y_true_original, y_pred_original,
        title="Transformer Model: Predicted vs Actual",
        save_path=os.path.join(config['paths']['figures_path'], "scatter.png")
    )


def main():
    parser = argparse.ArgumentParser(description="Run a rainfall prediction experiment")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to experiment configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run experiment based on model type
    if config['model']['name'].lower() == 'linear':
        run_linear_experiment(config)
    elif config['model']['name'].lower() == 'lstm':
        run_lstm_experiment(config)
    elif config['model']['name'].lower() == 'gru':
        run_gru_experiment(config)
    elif config['model']['name'].lower() == 'cnn_lstm':
        run_cnn_lstm_experiment(config)
    elif config['model']['name'].lower() == 'transformer':
        run_transformer_experiment(config)
    else:
        raise ValueError(f"Unsupported model: {config['model']['name']}")


if __name__ == "__main__":
    main()