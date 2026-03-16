"""
Prediction module for real-time inference
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import Union, List

from src.models import WeatherLSTM
from src.utils import logger
import config


class SolarPredictor:
    """
    Real-time predictor for solar radiation
    """
    
    def __init__(self, model_path: Path = None):
        self.device = config.DEVICE
        
        if model_path is None:
            model_path = config.MODELS_DIR / 'best_model.pth'
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: Path):
        """Load trained model and scalers"""
        logger.info(f"Loading predictor from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = WeatherLSTM(**checkpoint['model_config']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.feature_scaler = joblib.load(config.MODELS_DIR / 'feature_scaler.pkl')
        self.target_scaler = joblib.load(config.MODELS_DIR / 'target_scaler.pkl')
        
        logger.info("Predictor loaded successfully")
    
    def preprocess_input(self, data: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """
        Preprocess input data for prediction
        Args:
            data: Raw input data (should match training features)
        Returns:
            Preprocessed data ready for model
        """
        if isinstance(data, pd.DataFrame):
            # Ensure correct feature order
            # This should match the feature_names from training
            expected_features = ['Temperature', 'Pressure', 'Humidity', 'Speed', 
                               'WindDirection(Degrees)', 'Hour', 'Minute', 
                               'TimeSin', 'TimeCos', 'DaylightMinutes', 
                               'TimeSinceSunrise', 'IsDaytime', 'Temp_Humidity', 
                               'Pressure_Temp']
            
            # Check if all features are present
            missing = set(expected_features) - set(data.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")
            
            X = data[expected_features].values
        else:
            X = np.array(data)
        
        # Ensure correct shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Create sequence (need window_size - 1 previous points)
        # For simplicity, we'll assume the input already has the sequence
        # In practice, you'd need to maintain a buffer of previous values
        
        return X_scaled
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False):
        """
        Make predictions
        Args:
            X: Input data (n_samples, seq_len, n_features) or (seq_len, n_features)
            return_uncertainty: Whether to return uncertainty estimates
        Returns:
            Predictions (and uncertainties if requested)
        """
        # Handle 1D array (single sample with 14 features from Streamlit)
        if X.ndim == 1:
            X = X.reshape(1, 1, -1)  # (batch=1, seq_len=1, features=14)
        # Handle 2D array (single sequence)
        elif X.ndim == 2:
            X = X.reshape(1, *X.shape)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            if return_uncertainty:
                # Monte Carlo Dropout
                self.model.train()
                predictions = []
                for _ in range(100):
                    pred = self.model(X_tensor).cpu().numpy()
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                mean_pred = predictions.mean(axis=0)
                std_pred = predictions.std(axis=0)
                
                # Convert to original scale
                mean_pred_original = self.target_scaler.inverse_transform(
                    mean_pred.reshape(-1, 1)
                ).flatten()
                std_pred_original = self.target_scaler.inverse_transform(
                    std_pred.reshape(-1, 1)
                ).flatten()
                
                # Return scalar for single prediction
                if len(mean_pred_original) == 1:
                    return float(mean_pred_original[0]), float(std_pred_original[0])
                return mean_pred_original, std_pred_original
            else:
                pred_scaled = self.model(X_tensor).cpu().numpy()
                pred_original = self.target_scaler.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).flatten()
                
                # Return scalar for single prediction
                if len(pred_original) == 1:
                    return float(pred_original[0])
                return pred_original
    
    def predict_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Predict on a batch of sequences"""
        return self.predict(X_batch, return_uncertainty=False)
    
    def predict_with_confidence(self, X: np.ndarray, confidence: float = 0.95):
        """
        Predict with confidence intervals
        Args:
            X: Input data
            confidence: Confidence level (e.g., 0.95 for 95% CI)
        Returns:
            predictions, lower_bound, upper_bound
        """
        from scipy import stats
        
        mean_pred, std_pred = self.predict(X, return_uncertainty=True)
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        return mean_pred, lower_bound, upper_bound


# Example usage
if __name__ == "__main__":
    predictor = SolarPredictor()
    
    # Example: predict for a single sample
    # You need to provide a sequence of 24 hours of data
    sample_input = np.random.randn(1, 24, 14)  # (batch, seq_len, features)
    
    pred = predictor.predict(sample_input)
    print(f"Predicted radiation: {pred[0]:.2f} W/m²")
    
    # With uncertainty
    mean, std = predictor.predict(sample_input, return_uncertainty=True)
    print(f"Prediction with uncertainty: {mean[0]:.2f} ± {std[0]:.2f} W/m²")
    
    # With confidence interval
    mean, lower, upper = predictor.predict_with_confidence(sample_input, confidence=0.95)
    print(f"95% CI: [{lower[0]:.2f}, {upper[0]:.2f}] W/m²")