import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load and preprocess the weather data."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Handle missing values
    df = df.interpolate(method='linear')
    
    return df

def create_sequences_multi_target(data, target_cols, seq_length=12, pred_length=1):
    """
    Create input sequences for multi-target time series model.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The input data including features and targets
    target_cols : list
        List of column names of the targets to predict
    seq_length : int
        Length of input sequence
    pred_length : int
        Prediction horizon
        
    Returns:
    --------
    X, y : numpy arrays
        Input features and target values
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length:i+seq_length+pred_length][target_cols].values.flatten())
        
    return np.array(X), np.array(y)

def create_sequences(data, target_col, seq_length=12, pred_length=1):
    """
    Create input sequences for single-target time series model.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The input data including features and target
    target_col : str
        Column name of the target to predict
    seq_length : int
        Length of input sequence
    pred_length : int
        Prediction horizon
        
    Returns:
    --------
    X, y : numpy arrays
        Input features and target values
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length:i+seq_length+pred_length][target_col].values)
        
    return np.array(X), np.array(y)

def prepare_data_multi_target(file_path, target_cols=['Rainfall', 'MinTemp', 'MaxTemp'], 
                      seq_length=12, test_size=0.2, val_size=0.2, random_state=42):
    """
    Prepare data for multi-target model training.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV data file
    target_cols : list
        List of column names of the target variables
    seq_length : int
        Length of input sequence
    test_size : float
        Proportion of data for testing
    val_size : float
        Proportion of training data for validation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dictionary containing train, validation, and test data along with scalers
    """
    # Load and clean data
    df = load_data(file_path)
    
    # Create feature and target columns
    features = df.columns.tolist()
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = {}
    
    df_scaled = pd.DataFrame(scaler_X.fit_transform(df), columns=features, index=df.index)
    
    # Store original target values for inverse transformation
    for col in target_cols:
        scaler_y[col] = StandardScaler()
        scaler_y[col].fit(df[col].values.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences_multi_target(df_scaled, target_cols, seq_length=seq_length)
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, shuffle=False
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val, 
        'y_test': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': features,
        'target_cols': target_cols
    }

def prepare_data(file_path, target_col='Rainfall', seq_length=12, test_size=0.2, val_size=0.2, random_state=42):
    """
    Prepare data for single-target model training.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV data file
    target_col : str
        Column name of the target variable
    seq_length : int
        Length of input sequence
    test_size : float
        Proportion of data for testing
    val_size : float
        Proportion of training data for validation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dictionary containing train, validation, and test data along with scalers
    """
    # Load and clean data
    df = load_data(file_path)
    
    # Create feature and target columns
    features = df.columns.tolist()
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    df_scaled = pd.DataFrame(scaler_X.fit_transform(df), columns=features, index=df.index)
    
    # Store original target values for inverse transformation
    original_target = df[target_col].values.reshape(-1, 1)
    scaler_y.fit(original_target)
    
    # Create sequences
    X, y = create_sequences(df_scaled, target_col, seq_length=seq_length)
    y = y.reshape(-1, 1)
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, shuffle=False
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val, 
        'y_test': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': features
    }

def inverse_transform_target(y_scaled, scaler):
    """Transform scaled predictions back to original scale."""
    if len(y_scaled.shape) == 1:
        y_scaled = y_scaled.reshape(-1, 1)
    return scaler.inverse_transform(y_scaled)

def inverse_transform_multi_target(y_scaled, scalers, target_cols):
    """
    Transform scaled predictions back to original scale for multi-target models.
    
    Parameters:
    -----------
    y_scaled : numpy array
        Scaled predictions with shape (n_samples, n_targets)
    scalers : dict
        Dictionary of scalers for each target
    target_cols : list
        List of target column names
        
    Returns:
    --------
    numpy array of unscaled predictions with shape (n_samples, n_targets)
    """
    n_samples = y_scaled.shape[0]
    n_targets = len(target_cols)
    y_original = np.zeros((n_samples, n_targets))
    
    for i, col in enumerate(target_cols):
        # Extract the predictions for this target
        y_target = y_scaled[:, i].reshape(-1, 1)
        # Inverse transform
        y_original[:, i] = scalers[col].inverse_transform(y_target).flatten()
    
    return y_original