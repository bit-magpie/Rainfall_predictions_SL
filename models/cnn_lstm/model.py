import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for rainfall prediction.
    Uses CNN layers for feature extraction followed by LSTM for temporal modeling.
    """
    
    def __init__(self, input_size, seq_length, cnn_filters=[32, 64], 
                 kernel_size=3, lstm_hidden=64, lstm_layers=1, 
                 output_size=1, dropout=0.2):
        """
        Initialize the CNN-LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of features in input
        seq_length : int
            Length of input sequence
        cnn_filters : list
            Number of filters in each convolutional layer
        kernel_size : int
            Size of convolutional kernels
        lstm_hidden : int
            Size of LSTM hidden layers
        lstm_layers : int
            Number of LSTM layers
        output_size : int
            Number of output features
        dropout : float
            Dropout probability
        """
        super(CNNLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length
        
        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = 1  # Start with 1 channel (will reshape input)
        
        for out_channels in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output dimensions
        cnn_output_size = self._get_cnn_output_size(input_size, seq_length, cnn_filters, kernel_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(lstm_hidden, output_size)
    
    def _get_cnn_output_size(self, input_size, seq_length, cnn_filters, kernel_size):
        """Calculate CNN output dimensions after pooling"""
        # We're using 1D convolutions with padding=(kernel_size//2), so the feature dimension
        # stays the same after each conv layer, but is halved after each pooling layer
        num_pool_layers = len(cnn_filters)
        feature_size = input_size
        seq_len = seq_length // (2 ** num_pool_layers)  # Each pooling reduces by half
        
        return cnn_filters[-1]  # Return the number of output channels
    
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
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch, channels, sequence, features) -> (batch, channels, sequence * features)
        x = x.view(batch_size, 1, -1)
        
        # Apply CNN
        cnn_out = self.cnn(x)
        
        # Reshape for LSTM: (batch, channels, features) -> (batch, sequence, channels)
        # We treat the CNN output as a sequence where each time step has 'channels' features
        lstm_in = cnn_out.permute(0, 2, 1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(lstm_in)
        
        # Use the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(lstm_out)
        
        # Final output layer
        out = self.fc(out)
        
        return out


def train(model, train_loader, val_loader=None, epochs=100, 
          learning_rate=0.001, weight_decay=0.0, clip_grad=5.0, verbose=True):
    """
    Train the CNN-LSTM model.
    
    Parameters:
    -----------
    model : CNNLSTMModel
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
    model : CNNLSTMModel
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