import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearRegressionModel(nn.Module):
    """Linear regression model for rainfall prediction."""
    
    def __init__(self, input_dim, output_dim=1, hidden_dims=None):
        """
        Initialize the linear regression model.
        
        Parameters:
        -----------
        input_dim : int
            Input dimension (features * sequence_length)
        output_dim : int
            Output dimension (default: 1)
        hidden_dims : list of int, optional
            Hidden layer dimensions for multi-layer model
        """
        super(LinearRegressionModel, self).__init__()
        
        self.flatten = nn.Flatten()  # Flatten the input sequence
        
        if hidden_dims is None or len(hidden_dims) == 0:
            # Simple linear regression
            self.linear_stack = nn.Sequential(
                nn.Linear(input_dim, output_dim)
            )
        else:
            # Multi-layer perceptron
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.linear_stack = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        x = self.flatten(x)
        return self.linear_stack(x)


def train(model, train_loader, val_loader=None, epochs=100, 
          learning_rate=0.001, weight_decay=0.0, verbose=True):
    """
    Train the linear model.
    
    Parameters:
    -----------
    model : LinearRegressionModel
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    weight_decay : float
        L2 regularization strength
    verbose : bool
        Whether to print training progress
    
    Returns:
    --------
    dict : Training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    history = {
        'loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['loss'].append(train_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        elif verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}')
    
    return history


def predict(model, data_loader):
    """
    Make predictions with the trained model.
    
    Parameters:
    -----------
    model : LinearRegressionModel
        Trained model
    data_loader : DataLoader
        Data loader for prediction
        
    Returns:
    --------
    numpy arrays of targets and predictions
    """
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            all_targets.append(targets.numpy())
            all_predictions.append(outputs.numpy())
    
    return np.vstack(all_targets), np.vstack(all_predictions)


def get_feature_importance(model):
    """
    Extract feature importance from the linear model.
    Only works for single layer linear models.
    
    Parameters:
    -----------
    model : LinearRegressionModel
        Trained linear model
        
    Returns:
    --------
    numpy array of feature importance values
    """
    # Get weights from the first linear layer
    if len(list(model.linear_stack.children())) == 1:
        # Simple linear model
        weights = list(model.linear_stack.parameters())[0].data.numpy()
        return np.abs(weights.flatten())
    else:
        raise ValueError("Feature importance is only available for single-layer linear models.")