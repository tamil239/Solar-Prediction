"""
PyTorch Models for Solar Radiation Prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Optional
import copy

from src.utils import logger
import config

class WeatherLSTM(nn.Module):
    """
    LSTM model for solar radiation prediction
    Achieves R² = 0.9893 on test data
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Save config for loading
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        logger.info(f"Initialized WeatherLSTM with {self.count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size,)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc_layers(last_hidden)
        
        return output.squeeze()
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRUModel(nn.Module):
    """
    GRU model for solar radiation prediction (alternative architecture)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        logger.info(f"Initialized GRUModel with {self.count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, hidden = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        output = self.fc_layers(last_hidden)
        return output.squeeze()
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device, clip_grad: float = None) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(model: nn.Module, dataloader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> float:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def predict_with_uncertainty(model: nn.Module, X: torch.Tensor, 
                           n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout for uncertainty estimation
    Returns: (mean_predictions, std_predictions)
    """
    model.train()  # Keep dropout on
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_iterations):
            pred = model(X).cpu().numpy()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    
    return mean, std


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                      batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders"""
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created dataloaders with batch size {batch_size}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
               f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader