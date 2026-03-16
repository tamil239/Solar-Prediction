"""
Utility functions for Solar Radiation Prediction Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import sys

# Setup logging
def setup_logging(name: str = 'solar_prediction', log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_dir / f'{name}_{datetime.now():%Y%m%d_%H%M%S}.log')
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

# Create global logger
logger = setup_logging()

def save_model(model: torch.nn.Module, 
               scalers: Dict, 
               feature_names: List[str], 
               model_config: Dict,
               filepath: Path):
    """Save model and associated objects"""
    model_data = {
        'model_state_dict': model.state_dict(),
        'scalers': scalers,
        'feature_names': feature_names,
        'model_config': model_config
    }
    torch.save(model_data, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model(filepath: Path, model_class, device: torch.device):
    """Load model and associated objects"""
    checkpoint = torch.load(filepath, map_location=device)
    model = model_class(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Model loaded from {filepath}")
    return model, checkpoint['scalers'], checkpoint['feature_names']

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate comprehensive evaluation metrics"""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        explained_variance_score
    )
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    
    # Calculate MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # Calculate Max Error
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'R2': float(r2),
        'ExplainedVariance': float(evs),
        'MAPE': float(mape),
        'MaxError': float(max_error)
    }

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     title: str = "Predictions vs Actual", 
                     save_path: Optional[Path] = None):
    """Plot predictions against actual values"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # Scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values (W/m²)')
    axes[0, 0].set_ylabel('Predicted Values (W/m²)')
    axes[0, 0].set_title('Predictions vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values (W/m²)')
    axes[0, 1].set_ylabel('Residuals (W/m²)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution of residuals
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residual (W/m²)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time series comparison (first 200 points)
    n_points = min(200, len(y_true))
    axes[1, 1].plot(y_true[:n_points], label='Actual', alpha=0.7, linewidth=1)
    axes[1, 1].plot(y_pred[:n_points], label='Predicted', alpha=0.7, linewidth=1)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Radiation (W/m²)')
    axes[1, 1].set_title('Time Series Comparison (first 200 points)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def save_results(results: Dict, filepath: Path):
    """Save results to JSON"""
    # Convert numpy types to Python types for JSON serialization
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            clean_results[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64, np.int64)):
            clean_results[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            clean_results[k] = int(v)
        elif isinstance(v, dict):
            clean_results[k] = {kk: float(vv) if isinstance(vv, (np.float32, np.float64)) else vv 
                               for kk, vv in v.items()}
        else:
            clean_results[k] = v
    
    with open(filepath, 'w') as f:
        json.dump(clean_results, f, indent=4)
    logger.info(f"Results saved to {filepath}")

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")