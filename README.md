# Rainfall Prediction Project

A machine learning project for predicting rainfall using different model architectures. The project supports multiple models and provides tools for experiment tracking, visualization, and model evaluation.

## Project Overview

This project implements a structured framework for training and evaluating machine learning models to predict rainfall based on the `combined_data.csv` dataset. It currently supports:

- Linear models with configurable hidden layers
- LSTM models for time series forecasting

Each model can be easily configured through YAML configuration files, and the framework provides comprehensive visualization and evaluation tools.

## Project Structure

```
prediction_model/
├── data/                   # Data storage
│   └── combined_data.csv   # Dataset
├── models/                 # Model implementations
│   ├── linear/             # Linear model
│   └── lstm/               # LSTM model
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

Where `[model_type]` can be:
- `linear`: Run only the linear model
- `lstm`: Run only the LSTM model
- `all`: Run both models (default)

### Example Commands

Train and evaluate the linear model:
```bash
python main.py --model linear
```

Train and evaluate the LSTM model:
```bash
python main.py --model lstm
```

Train and evaluate both models:
```bash
python main.py
```

## Customizing Experiments

You can modify the models' hyperparameters by editing the configuration files in the `experiments/configs/` directory:

- `linear_config.yaml`: Configuration for linear models
- `lstm_config.yaml`: Configuration for LSTM models

Example customizations:

- Change sequence length to use more/fewer past data points
- Adjust hidden layers and their dimensions
- Modify learning rates and regularization parameters
- Configure training epochs and batch sizes

## Results and Evaluation

After running experiments, you can find:

- Model checkpoint files in `results/models/`
- Evaluation metrics in `results/metrics/`
- Visualizations in `results/figures/`:
  - Training history plots
  - Prediction vs. actual value plots
  - Scatter plots of predicted vs. actual values

## Extending the Project

To add new model types:
1. Create a new directory in `models/`
2. Implement the model class and training functions
3. Create a configuration file in `experiments/configs/`
4. Extend the experiment runner to support your new model