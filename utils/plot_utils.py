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


def plot_multi_target_predictions(y_true, y_pred, target_cols, title='Multi-Target Predictions', save_path=None):
    """
    Plot actual vs predicted values for multiple targets.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values with shape (n_samples, n_targets)
    y_pred : array-like
        Predicted target values with shape (n_samples, n_targets)
    target_cols : list
        List of target column names
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets), sharex=True)
    
    # Time steps for x-axis
    x = np.arange(len(y_true))
    
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        ax.plot(x, y_true[:, i], label=f'Actual {col}', marker='o', markersize=2, linestyle='-')
        ax.plot(x, y_pred[:, i], label=f'Predicted {col}', marker='x', markersize=2, linestyle='--')
        
        ax.set_title(f'{col} Predictions')
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend()
    
    axes[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_multi_target_scatter(y_true, y_pred, target_cols, title='Multi-Target Scatter Plot', save_path=None):
    """
    Create a grid of scatter plots for multiple targets.
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values with shape (n_samples, n_targets)
    y_pred : array-like
        Predicted target values with shape (n_samples, n_targets)
    target_cols : list
        List of target column names
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 5))
    
    if n_targets == 1:
        axes = [axes]
    
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        # Perfect prediction line
        max_val = max(np.max(y_true[:, i]), np.max(y_pred[:, i]))
        min_val = min(np.min(y_true[:, i]), np.min(y_pred[:, i]))
        
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(f'{col}')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.grid(True)
    
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_target_components(y_true, y_pred, target_cols, title='Target Components Analysis', save_path=None):
    """
    Plot detailed comparison of the target components (error distribution, correlation, etc.).
    
    Parameters:
    -----------
    y_true : array-like
        Actual target values with shape (n_samples, n_targets)
    y_pred : array-like
        Predicted target values with shape (n_samples, n_targets)
    target_cols : list
        List of target column names
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(n_targets, 2, figsize=(12, 4*n_targets))
    
    for i, col in enumerate(target_cols):
        # Error histogram
        errors = y_pred[:, i] - y_true[:, i]
        axes[i, 0].hist(errors, bins=30, alpha=0.7, color='blue')
        axes[i, 0].axvline(0, color='red', linestyle='--')
        axes[i, 0].set_title(f'{col} - Error Distribution')
        axes[i, 0].set_xlabel('Prediction Error')
        axes[i, 0].set_ylabel('Frequency')
        
        # Prediction vs Actual correlation
        correlation = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        axes[i, 1].scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        axes[i, 1].set_title(f'{col} - Correlation: {correlation:.4f}')
        axes[i, 1].set_xlabel('Actual')
        axes[i, 1].set_ylabel('Predicted')
        
        # Add 45-degree line
        max_val = max(np.max(y_true[:, i]), np.max(y_pred[:, i]))
        min_val = min(np.min(y_true[:, i]), np.min(y_pred[:, i]))
        axes[i, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[i, 1].grid(True)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()