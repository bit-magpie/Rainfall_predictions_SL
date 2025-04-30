import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

def plot_training_history(history, title='Training History', save_path=None):
    """
    Plot training and validation loss.
    
    Parameters:
    -----------
    history : dict or keras History object
        Training history containing loss values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    if hasattr(history, 'history'):
        history = history.history
    
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_predictions(y_true, y_pred, title='Model Predictions', save_path=None):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Flatten arrays for plotting
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Time steps for x-axis
    x = np.arange(len(y_true))
    
    plt.plot(x, y_true, label='Actual', marker='o', markersize=2, linestyle='-')
    plt.plot(x, y_pred, label='Predicted', marker='x', markersize=2, linestyle='--')
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Rainfall')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_scatter(y_true, y_pred, title='Predicted vs Actual', save_path=None):
    """
    Create a scatter plot of predicted vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(8, 8))
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Perfect prediction line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_names, importance_values, title='Feature Importance', save_path=None):
    """
    Plot feature importance for linear models.
    
    Parameters:
    -----------
    feature_names : list
        Names of features
    importance_values : array-like
        Importance values for each feature
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    # Create DataFrame for better visualization
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_values})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()