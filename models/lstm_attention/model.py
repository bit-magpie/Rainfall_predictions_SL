import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Attention(nn.Module):
    """
    Attention mechanism for LSTM model.
    Allows the model to focus on different parts of the input sequence.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
        # Defining the attention layers
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        Forward pass.
        
        Parameters:
        -----------
        hidden : torch.Tensor
            Hidden state of the LSTM, shape (batch_size, hidden_size)
        encoder_outputs : torch.Tensor
            All hidden states of the input sequence, shape (batch_size, seq_len, hidden_size)
            
        Returns:
        --------
        torch.Tensor : Context vector
        torch.Tensor : Attention weights
        """
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        
        # Repeat hidden state seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate hidden state with encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Calculate attention weights
        attention = self.v(energy).squeeze(2)
        attention_weights = torch.softmax(attention, dim=1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights


class LSTMWithAttentionModel(nn.Module):
    """LSTM with Attention model for multi-output weather prediction."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, 
                 output_size=1, dropout=0.3, bidirectional=False):
        """
        Initialize the LSTM with Attention model.
        
        Parameters:
        -----------
        input_size : int
            Number of features in input
        hidden_size : int
            Number of hidden units in LSTM
        num_layers : int
            Number of LSTM layers
        output_size : int
            Number of output features (default: 1)
        dropout : float
            Dropout probability
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(LSTMWithAttentionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = Attention(hidden_size * self.directions)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * self.directions * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
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
        # LSTM forward pass
        outputs, (hidden, cell) = self.lstm(x)
        
        # Get the last hidden state (for each direction if bidirectional)
        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Apply attention
        context, attention_weights = self.attention(last_hidden, outputs)
        
        # Concatenate context vector with the last hidden state
        combined = torch.cat([context, last_hidden], dim=1)
        
        # Feed through fully connected layer
        out = self.fc(combined)
        out = torch.relu(out)
        out = self.dropout(out)
        
        # Output layer
        out = self.output_layer(out)
        
        return out


def train(model, train_loader, val_loader=None, epochs=100, 
          learning_rate=0.001, weight_decay=0.0001, clip_grad=1.0, verbose=True):
    """
    Train the LSTM with Attention model.
    
    Parameters:
    -----------
    model : LSTMWithAttentionModel
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        L2 regularization strength
    clip_grad : float
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=verbose)
    
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
            if clip_grad > 0:
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
            
            # Update learning rate
            scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        elif verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return history


def predict(model, data_loader):
    """
    Make predictions with the trained model.
    
    Parameters:
    -----------
    model : LSTMWithAttentionModel
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