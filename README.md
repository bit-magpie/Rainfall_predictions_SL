# Rainfall Prediction Project

A machine learning project for predicting rainfall, minimum temperature, and maximum temperature using various model architectures. The project supports multiple models and provides tools for experiment tracking, visualization, and model evaluation.

## Project Overview

This project implements a structured framework for training and evaluating machine learning models to predict weather variables based on the `combined_data.csv` dataset. It currently supports:

- Linear models with configurable hidden layers
- LSTM (Long Short-Term Memory) models for time series forecasting
- GRU (Gated Recurrent Unit) models for time series forecasting
- CNN-LSTM hybrid models combining convolutional and recurrent layers
- Transformer models with self-attention mechanisms
- Random Forest models for multi-output regression
- LSTM models with attention mechanisms for improved focus on relevant features

Models can be run in either single-output mode (predicting rainfall only) or multi-output mode (simultaneously predicting rainfall, minimum temperature, and maximum temperature). Each model can be easily configured through YAML configuration files, and the framework provides comprehensive visualization and evaluation tools.

## Project Structure

```
prediction_model/
├── data/                   # Data storage
│   └── combined_data.csv   # Dataset
├── models/                 # Model implementations
│   ├── linear/             # Linear model
│   ├── lstm/               # LSTM model
│   ├── gru/                # GRU model
│   ├── cnn_lstm/           # CNN-LSTM hybrid model
│   ├── transformer/        # Transformer model
│   ├── random_forest/      # Random Forest model
│   └── lstm_attention/     # LSTM with Attention model
├── utils/                  # Utility functions
├── experiments/            # Experiment management
│   ├── run_experiment.py   # Experiment runner
│   └── configs/            # Model configurations
├── results/                # Results storage
│   ├── figures/            # Visualizations
│   ├── metrics/            # Evaluation metrics
│   └── models/             # Saved models
├── main.py                 # Main entry point
└── requirements.txt        # Dependencies
```

## Setting Up the Environment

1. Clone the repository (or navigate to the project directory)

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Scripts

### Basic Usage

Run the main script to train and evaluate models:

```bash
python main.py --model [model_type]
```

### Available Models

#### Single-Output Models
- `linear`: Linear regression model
- `lstm`: LSTM model 
- `gru`: GRU model
- `cnn_lstm`: CNN-LSTM hybrid model
- `transformer`: Transformer model

#### Multi-Output Models
- `lstm_multi_output`: LSTM model predicting multiple weather variables
- `gru_multi_output`: GRU model predicting multiple weather variables
- `cnn_lstm_multi_output`: CNN-LSTM model predicting multiple weather variables
- `transformer_multi_output`: Transformer model predicting multiple weather variables
- `random_forest_multi_output`: Random Forest model predicting multiple weather variables
- `lstm_attention_multi_output`: LSTM with Attention model predicting multiple weather variables

#### Model Groups
- `all`: Run all available models (default)
- `all_single`: Run all single-output models
- `all_multi`: Run all multi-output models

### Command-Line Arguments

#### Model Selection
```bash
--model MODEL_TYPE          # Select which model to train (default: all)
--config_dir CONFIG_DIR     # Directory containing model configuration files (default: experiments/configs)
```

#### Data Processing Options
```bash
--interpolate METHOD        # Method for data interpolation (linear, time, spline, polynomial, nearest, pad)
--interpolation_limit LIMIT # Maximum number of consecutive NaN values to fill (default: 5)
--resample FREQUENCY        # Frequency to resample data to increase density (e.g., '12h', '6h', '1D')
```

#### Hardware Acceleration
```bash
--use_gpu                   # Use GPU for training if available (flag)
```

### Example Commands

Train and evaluate the linear model:
```bash
python main.py --model linear
```

Train the LSTM with Attention multi-output model using spline interpolation and GPU acceleration:
```bash
python main.py --model lstm_attention_multi_output --interpolate spline --interpolation_limit 5 --resample 12h --use_gpu
```

Train all multi-output models:
```bash
python main.py --model all_multi
```

## Model Descriptions

### Linear Model
A simple linear regression model that can optionally include hidden layers to create a multi-layer perceptron.

### LSTM (Long Short-Term Memory)
Recurrent neural network architecture designed for time series forecasting with the ability to capture long-term dependencies.

### GRU (Gated Recurrent Unit)
A simpler recurrent neural network architecture similar to LSTM but with fewer parameters.

### CNN-LSTM
Hybrid model that combines convolutional layers for feature extraction with LSTM layers for temporal modeling.

### Transformer
Model based on the self-attention mechanism from the "Attention Is All You Need" paper, effective for capturing complex relationships in sequential data.

### Random Forest
Ensemble learning method that constructs multiple decision trees and outputs the average prediction of individual trees.

### LSTM with Attention
LSTM model enhanced with an attention mechanism that allows the model to focus on the most relevant parts of the input sequence.

## Customizing Experiments

You can modify the models' hyperparameters by editing the configuration files in the `experiments/configs/` directory. Each model has its own YAML configuration file:

- `linear_config.yaml`: Configuration for linear models
- `lstm_config.yaml`: Configuration for LSTM models
- `gru_config.yaml`: Configuration for GRU models
- `cnn_lstm_config.yaml`: Configuration for CNN-LSTM models
- `transformer_config.yaml`: Configuration for Transformer models
- `lstm_multi_output_config.yaml`: Configuration for multi-output LSTM models
- `gru_multi_output_config.yaml`: Configuration for multi-output GRU models
- `cnn_lstm_multi_output_config.yaml`: Configuration for multi-output CNN-LSTM models
- `transformer_multi_output_config.yaml`: Configuration for multi-output Transformer models
- `random_forest_multi_output_config.yaml`: Configuration for multi-output Random Forest models
- `lstm_attention_multi_output_config.yaml`: Configuration for multi-output LSTM with Attention models

Example customizations:

- Change sequence length to use more/fewer past data points
- Adjust hidden layers and their dimensions
- Modify learning rates and regularization parameters
- Configure training epochs and batch sizes
- Enable/disable bidirectional recurrent layers
- Adjust attention mechanisms

## Data Preprocessing Features

### Interpolation
The system supports various interpolation methods to handle missing values:
- `linear`: Linear interpolation between points
- `time`: Time-based interpolation considering time index
- `spline`: Cubic spline interpolation for smooth curves
- `polynomial`: Polynomial interpolation
- `nearest`: Nearest value interpolation
- `pad`: Padding with last valid observation

### Data Enhancement
To increase data density and potentially improve model accuracy, you can resample the data to a higher frequency:
- `--resample 12h`: Resample to 12-hour intervals
- `--resample 6h`: Resample to 6-hour intervals
- `--resample 1D`: Resample to daily intervals

## Results and Evaluation

After running experiments, you can find:

- Model checkpoint files in `results/models/`
- Evaluation metrics in `results/metrics/`
- Visualizations in `results/figures/`:
  - Training history plots
  - Prediction vs. actual value plots
  - Scatter plots of predicted vs. actual values
  - Feature importance plots (for applicable models)
  - Target component analysis (for multi-output models)

## Metrics

The system evaluates models using multiple metrics:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

For multi-output models, metrics are calculated both overall and for each target variable.