import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics for model evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    Dictionary of evaluation metrics
    """
    # Ensure arrays are flattened for metric calculation
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # Adding small epsilon to avoid division by zero
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def save_metrics(metrics_dict, model_name, config=None, file_path=None):
    """
    Save metrics to CSV file.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metric names and values
    model_name : str
        Name of the model
    config : dict, optional
        Model configuration parameters
    file_path : str, optional
        Path to save the metrics
        
    Returns:
    --------
    Path to saved metrics file
    """
    if file_path is None:
        file_path = f"results/metrics/{model_name}_metrics.csv"
    
    # Create a flat dictionary for the DataFrame
    data = {'model': model_name}
    
    # Add metrics
    for metric_name, metric_value in metrics_dict.items():
        data[metric_name] = metric_value
    
    # Add config parameters if provided
    if config is not None:
        for param_name, param_value in config.items():
            data[f"param_{param_name}"] = param_value
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame([data])
    
    try:
        existing_df = pd.read_csv(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass
        
    df.to_csv(file_path, index=False)
    
    return file_path