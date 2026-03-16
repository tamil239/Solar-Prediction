"""
Model Evaluation Module for Solar Radiation Prediction
Provides comprehensive model evaluation and analysis
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from typing import Dict, Tuple, Optional

from src.models import WeatherLSTM, predict_with_uncertainty
from src.utils import logger, calculate_metrics, plot_predictions, save_results
import config


class ModelEvaluator:
    """
    Comprehensive model evaluator for solar radiation prediction
    """
    
    def __init__(self, model_path: Path = None):
        self.device = config.DEVICE
        
        if model_path is None:
            model_path = config.MODELS_DIR / 'best_model.pth'
        
        self.model_path = model_path
        self.model = None
        self.scalers = {}
        self.history = None
        
        if self.model_path.exists():
            self._load_model()
        else:
            logger.warning(f"Model not found at {self.model_path}")
    
    def _load_model(self):
        """Load trained model and associated objects"""
        logger.info(f"Loading model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load model
        self.model = WeatherLSTM(**checkpoint['model_config']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        # Load scalers
        feature_scaler_path = config.MODELS_DIR / 'feature_scaler.pkl'
        target_scaler_path = config.MODELS_DIR / 'target_scaler.pkl'
        
        if feature_scaler_path.exists():
            self.scalers['feature_scaler'] = joblib.load(feature_scaler_path)
        if target_scaler_path.exists():
            self.scalers['target_scaler'] = joblib.load(target_scaler_path)
        
        logger.info("Model loaded successfully")
    
    def load_test_data(self, data_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data for evaluation
        Args:
            data_path: Path to test data CSV (optional)
        Returns:
            X_test, y_test arrays
        """
        from src.data_preprocessing import SolarDataPreprocessor
        
        if data_path is None:
            data_path = config.DATA_FILE
        
        logger.info("Loading test data...")
        
        preprocessor = SolarDataPreprocessor(str(data_path))
        
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         _, scalers) = preprocessor.run_pipeline(
            remove_outliers=True,
            window_size=config.WINDOW_SIZE
        )
        
        self.scalers = scalers
        
        return X_test, y_test
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation on test data
        Args:
            X_test: Test features
            y_test: Test targets
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        logger.info("="*50)
        logger.info("MODEL EVALUATION")
        logger.info("="*50)
        
        self.model.eval()
        
        # Get predictions
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Calculate scaled metrics
        scaled_metrics = calculate_metrics(y_test, predictions_scaled)
        
        # Inverse transform to original scale
        if 'target_scaler' in self.scalers:
            target_scaler = self.scalers['target_scaler']
            predictions_original = target_scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            targets_original = target_scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()
        else:
            predictions_original = predictions_scaled
            targets_original = y_test
        
        # Calculate original scale metrics
        original_metrics = calculate_metrics(targets_original, predictions_original)
        
        # Calculate correlation
        correlation = np.corrcoef(targets_original, predictions_original)[0, 1]
        
        # Compile results
        results = {
            'scaled_metrics': scaled_metrics,
            'original_metrics': original_metrics,
            'correlation': float(correlation),
            'predictions': predictions_original.tolist(),
            'targets': targets_original.tolist()
        }
        
        # Log metrics
        logger.info("\n" + "="*50)
        logger.info("SCALED METRICS:")
        logger.info("="*50)
        for metric, value in scaled_metrics.items():
            logger.info(f"{metric:20s}: {value:.6f}")
        
        logger.info("\n" + "="*50)
        logger.info("ORIGINAL SCALE METRICS:")
        logger.info("="*50)
        for metric, value in original_metrics.items():
            if metric in ['MAE', 'RMSE', 'MSE', 'MaxError']:
                logger.info(f"{metric:20s}: {value:.2f}")
            elif metric == 'MAPE':
                logger.info(f"{metric:20s}: {value:.2f}%")
            else:
                logger.info(f"{metric:20s}: {value:.4f}")
        
        logger.info(f"{'Correlation':20s}: {correlation:.4f}")
        
        return results
    
    def evaluate_with_uncertainty(self, X_test: np.ndarray, y_test: np.ndarray, 
                                  n_iterations: int = 100) -> Dict:
        """
        Evaluate model with uncertainty estimation using Monte Carlo Dropout
        Args:
            X_test: Test features
            y_test: Test targets
            n_iterations: Number of MC dropout iterations
        Returns:
            Dictionary including uncertainty metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        logger.info("Evaluating with Monte Carlo Dropout...")
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Get predictions with uncertainty
        mean_pred, std_pred = predict_with_uncertainty(
            self.model, X_tensor, n_iterations=n_iterations
        )
        
        # Convert to original scale
        if 'target_scaler' in self.scalers:
            target_scaler = self.scalers['target_scaler']
            mean_original = target_scaler.inverse_transform(
                mean_pred.reshape(-1, 1)
            ).flatten()
            std_original = target_scaler.inverse_transform(
                std_pred.reshape(-1, 1)
            ).flatten()
            targets_original = target_scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()
        else:
            mean_original = mean_pred
            std_original = std_pred
            targets_original = y_test
        
        # Calculate metrics
        metrics = calculate_metrics(targets_original, mean_original)
        
        # Calculate calibration
        calibration_errors = []
        for i in range(len(targets_original)):
            if std_original[i] > 0:
                z = abs(targets_original[i] - mean_original[i]) / std_original[i]
                calibration_errors.append(z)
        
        avg_calibration_error = np.mean(calibration_errors)
        
        results = {
            'metrics': metrics,
            'mean_predictions': mean_original.tolist(),
            'std_predictions': std_original.tolist(),
            'targets': targets_original.tolist(),
            'avg_uncertainty': float(np.mean(std_original)),
            'calibration_error': float(avg_calibration_error)
        }
        
        logger.info(f"Average prediction uncertainty: ±{np.mean(std_original):.2f}")
        logger.info(f"Average calibration error: {avg_calibration_error:.2f}σ")
        
        return results
    
    def plot_results(self, predictions: np.ndarray, targets: np.ndarray, 
                    save_path: Path = None):
        """
        Plot prediction results
        Args:
            predictions: Model predictions
            targets: Actual values
            save_path: Path to save plot
        """
        if save_path is None:
            save_path = config.RESULTS_DIR / 'evaluation_plot.png'
        
        plot_predictions(
            targets,
            predictions,
            title="Solar Radiation Prediction - Evaluation",
            save_path=save_path
        )
    
    def save_results(self, results: Dict, save_path: Path = None):
        """
        Save evaluation results to JSON
        Args:
            results: Results dictionary
            save_path: Path to save results
        """
        if save_path is None:
            save_path = config.RESULTS_DIR / 'evaluation_results.json'
        
        save_results(results, save_path)


def evaluate_model(model_path: str = None, test_data_path: str = None) -> Dict:
    """
    Convenience function to evaluate the model
    Args:
        model_path: Path to model file
        test_data_path: Path to test data
    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator(model_path=model_path)
    
    X_test, y_test = evaluator.load_test_data(test_data_path)
    
    results = evaluator.evaluate(X_test, y_test)
    
    # Save results
    evaluator.save_results(results)
    
    # Plot results
    if 'predictions' in results and 'targets' in results:
        evaluator.plot_results(
            np.array(results['predictions']),
            np.array(results['targets'])
        )
    
    return results


if __name__ == "__main__":
    try:
        evaluator = ModelEvaluator()
        X_test, y_test = evaluator.load_test_data()
        results = evaluator.evaluate(X_test, y_test)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETE")
        print("="*50)
        print(f"R² Score: {results['original_metrics']['R2']:.4f}")
        print(f"RMSE: {results['original_metrics']['RMSE']:.2f}")
        print(f"MAE: {results['original_metrics']['MAE']:.2f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python run.py --mode train")
