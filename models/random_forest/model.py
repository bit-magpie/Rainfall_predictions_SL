import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize a Random Forest model for weather prediction.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            The number of trees in the forest.
        max_depth : int, default=None
            The maximum depth of the trees.
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        # Create a multi-output random forest regressor
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1  # Use all available processors
            )
        )
        self.feature_importances_ = None
    
    def reshape_for_sklearn(self, X):
        """
        Reshape 3D tensor data to 2D format for sklearn.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data with shape (samples, sequence_length, features)
            
        Returns:
        --------
        numpy.ndarray
            Reshaped data with shape (samples, sequence_length * features)
        """
        batch_size, seq_length, n_features = X.shape
        return X.reshape(batch_size, seq_length * n_features)
    
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data with shape (samples, sequence_length, features)
        y : numpy.ndarray
            Target data with shape (samples, n_outputs)
            
        Returns:
        --------
        self
        """
        # Reshape X for sklearn
        X_reshaped = self.reshape_for_sklearn(X)
        self.model.fit(X_reshaped, y)
        
        # Store feature importances if available
        if hasattr(self.model.estimators_[0], 'feature_importances_'):
            self.feature_importances_ = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        
        return self
    
    def predict(self, X):
        """
        Generate predictions from input data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data with shape (samples, sequence_length, features)
            
        Returns:
        --------
        numpy.ndarray
            Predictions with shape (samples, n_outputs)
        """
        X_reshaped = self.reshape_for_sklearn(X)
        return self.model.predict(X_reshaped)
    
    def get_feature_importance(self):
        """
        Get feature importance from the model.
        
        Returns:
        --------
        numpy.ndarray
            Feature importance scores
        """
        if self.feature_importances_ is not None:
            return self.feature_importances_
        else:
            raise ValueError("Feature importances not available. Model may not be trained yet.")

    def save(self, path):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load(path):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
            
        Returns:
        --------
        RandomForestModel
            Loaded model
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model


def train(model, X_train, y_train, X_val, y_val):
    """
    Train the Random Forest model and return training history.
    
    Parameters:
    -----------
    model : RandomForestModel
        The model to train
    X_train : numpy.ndarray
        Training data with shape (samples, sequence_length, features)
    y_train : numpy.ndarray
        Training targets with shape (samples, n_outputs)
    X_val : numpy.ndarray
        Validation data with shape (samples, sequence_length, features)
    y_val : numpy.ndarray
        Validation targets with shape (samples, n_outputs)
    
    Returns:
    --------
    dict
        Training history with validation scores (may be None for RF as training is one-shot)
    """
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Random Forest training is one-shot, so we don't have a training history
    # But we can compute validation metrics
    y_val_pred = model.predict(X_val)
    
    # Calculate MSE on validation set
    val_mse = np.mean((y_val - y_val_pred) ** 2)
    print(f"Validation MSE: {val_mse:.4f}")
    
    # No actual training history to return for Random Forest, but we can return the validation metrics
    # This is mainly for API compatibility with other models
    return None


def predict(model, X_test, y_test=None):
    """
    Generate predictions from the model.
    
    Parameters:
    -----------
    model : RandomForestModel
        Trained model
    X_test : numpy.ndarray
        Test data with shape (samples, sequence_length, features)
    y_test : numpy.ndarray, optional
        Test targets, passed through if provided
    
    Returns:
    --------
    tuple
        (y_test, y_pred) if y_test is provided, otherwise just y_pred
    """
    y_pred = model.predict(X_test)
    
    if y_test is not None:
        return y_test, y_pred
    else:
        return y_pred


def save_model(model, path):
    """
    Save the model to a file.
    
    Parameters:
    -----------
    model : RandomForestModel
        The model to save
    path : str
        Path to save the model
    """
    model.save(path)