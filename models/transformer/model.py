import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    Adds information about the relative or absolute position of tokens in the sequence.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer model for rainfall prediction."""
    
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, output_size=1, dropout=0.1):
        """
        Initialize the Transformer model.
        
        Parameters:
        -----------
        input_size : int
            Number of features in input
        d_model : int
            Dimension of model (embedding dimension)
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer encoder layers
        dim_feedforward : int
            Dimension of feedforward network in transformer
        output_size : int
            Number of output features (default: 1)
        dropout : float
            Dropout probability
        """
        super(TransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
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
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(x)
        
        # Use the last sequence position for prediction
        output = output[:, -1, :]
        
        # Project to output size
        output = self.output_layer(output)
        
        return output


def create_src_mask(src, pad_idx=0):
    """Create a mask to hide padding and future words."""
    src_mask = (src != pad_idx)
    return src_mask


def train(model, train_loader, val_loader=None, epochs=100, 
          learning_rate=0.0005, weight_decay=0.0001, 
          warmup_steps=4000, max_grad_norm=5.0, verbose=True):
    """
    Train the Transformer model.
    
    Parameters:
    -----------
    model : TransformerModel
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    epochs : int
        Number of training epochs
    learning_rate : float
        Maximum learning rate
    weight_decay : float
        L2 regularization strength
    warmup_steps : int
        Number of warmup steps for learning rate scheduler
    max_grad_norm : float
        Maximum gradient norm for gradient clipping
    verbose : bool
        Whether to print training progress
    
    Returns:
    --------
    dict : Training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
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
                print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        elif verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    
    return history


def predict(model, data_loader):
    """
    Make predictions with the trained model.
    
    Parameters:
    -----------
    model : TransformerModel
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