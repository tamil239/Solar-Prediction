"""
Configuration file for Solar Radiation Prediction Project
"""

import torch
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
APP_DIR = BASE_DIR / 'app'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR, APP_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data settings
DATA_FILE = DATA_DIR / 'SolarPrediction.csv'
TARGET_COLUMN = 'Radiation'
WINDOW_SIZE = 24  # Use past 24 hours to predict next hour

# Feature settings
FEATURE_COLUMNS = [
    'Temperature',
    'Pressure',
    'Humidity',
    'Speed',
    'WindDirection(Degrees)'
]

# Model hyperparameters (optimized from research)
MODEL_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 1.5e-4,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'epochs': 100,
    'patience': 30,
    'gradient_clip': 1.0
}

# Training settings
TRAIN_CONFIG = {
    'train_ratio': 0.64,
    'val_ratio': 0.21,
    'test_ratio': 0.15,
    'random_seed': 42
}

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")