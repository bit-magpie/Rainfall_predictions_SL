import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMModel(nn.Module):
    """LSTM model for rainfall prediction."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of features in input
        hidden_size : int
            Size of LSTM hidden layers
        num_layers : int
            Number of stacked LSTM layers
        output_size : int
            Number of output features (default: 1)
        dropout : float
            Dropout probability
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use only the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(lstm_out)
        
        # Final output layer
        out = self.fc(out)
        
        return out


def train(model, train_loader, val_loader=None, epochs=100, 
          learning_rate=0.001, weight_decay=0.0, clip_grad=5.0, verbose=True):
    """
    Train the LSTM model.
    
    Parameters:
    -----------
    model : LSTMModel
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
    clip_grad : float
        Gradient clipping value
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
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
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
    model : LSTMModel
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


def save_model(model, path):
    """Save the trained model."""
    torch.save(model.state_dict(), path)


def load_model(model_class, path, **kwargs):
    """Load a trained model."""
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model