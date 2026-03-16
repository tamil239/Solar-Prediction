"""
Training Module for Solar Radiation Prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from pathlib import Path
import json
from tqdm import tqdm
import time
import joblib
from typing import Dict, Tuple

from src.models import WeatherLSTM, GRUModel, EarlyStopping, train_epoch, validate_epoch
from src.models import create_dataloaders, predict_with_uncertainty
from src.data_preprocessing import SolarDataPreprocessor
from src.utils import logger, calculate_metrics, save_results, plot_predictions, set_seed
import config


class SolarTrainer:
    """
    Main trainer class for solar radiation prediction
    """
    
    def __init__(self, data_path: str, window_size: int = config.WINDOW_SIZE):
        self.data_path = data_path
        self.window_size = window_size
        self.device = config.DEVICE
        
        # Set random seed
        set_seed(config.TRAIN_CONFIG['random_seed'])
        
        logger.info(f"Initializing SolarTrainer with device: {self.device}")
        
        # Load and preprocess data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for training"""
        preprocessor = SolarDataPreprocessor(self.data_path)
        
        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test,
         self.feature_names, self.scalers) = preprocessor.run_pipeline(
            remove_outliers=True,
            window_size=self.window_size
        )
        
        self.input_dim = self.X_train.shape[2]
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
            batch_size=config.MODEL_CONFIG['batch_size']
        )
        
        logger.info(f"Input dimension: {self.input_dim}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter tuning
        Returns: Best validation loss
        """
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=64)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Create model
        model = WeatherLSTM(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # Early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop
        for epoch in range(50):  # Max epochs for tuning
            train_loss = train_epoch(
                model, self.train_loader, criterion, optimizer,
                self.device, clip_grad=1.0
            )
            val_loss = validate_epoch(
                model, self.val_loader, criterion, self.device
            )
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_val_loss
    
    def tune_hyperparameters(self, n_trials: int = 20) -> Dict:
        """Run hyperparameter tuning"""
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(self.objective, n_trials=n_trials, timeout=3600)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.6f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Save results
        with open(config.RESULTS_DIR / 'best_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=4)
        
        return study.best_params
    
    def train(self, params: Dict = None) -> Tuple[nn.Module, Dict]:
        """
        Train the model with given hyperparameters
        Args:
            params: Hyperparameters (if None, uses default config)
        Returns:
            Trained model and training history
        """
        if params is None:
            params = config.MODEL_CONFIG
        
        logger.info("="*50)
        logger.info("STARTING TRAINING")
        logger.info("="*50)
        logger.info(f"Training params: {params}")
        
        # Create model
        model = WeatherLSTM(
            input_dim=self.input_dim,
            hidden_dim=params.get('hidden_dim', 256),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.3)
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params.get('learning_rate', 1.5e-4),
            weight_decay=params.get('weight_decay', 1e-5)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=params.get('epochs', 100)
        )
        early_stopping = EarlyStopping(patience=params.get('patience', 30))
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch_time': []
        }
        
        # Training loop
        for epoch in range(params.get('epochs', 100)):
            epoch_start = time.time()
            
            # Train
            train_loss = train_epoch(
                model, self.train_loader, criterion, optimizer,
                self.device, clip_grad=params.get('gradient_clip', 1.0)
            )
            
            # Validate
            val_loss = validate_epoch(
                model, self.val_loader, criterion, self.device
            )
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            history['epoch_time'].append(epoch_time)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{params.get('epochs', 100)} | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if early_stopping.best_model:
            model.load_state_dict(early_stopping.best_model)
            logger.info("Loaded best model from early stopping")
        
        # Final evaluation
        test_loss = validate_epoch(
            model, self.test_loader, criterion, self.device
        )
        logger.info(f"Final test loss: {test_loss:.6f}")
        
        # Save model
        model_path = config.MODELS_DIR / 'best_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.config,
            'history': history,
            'params': params,
            'test_loss': test_loss
        }, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scalers
        joblib.dump(self.scalers['feature_scaler'], 
                   config.MODELS_DIR / 'feature_scaler.pkl')
        joblib.dump(self.scalers['target_scaler'], 
                   config.MODELS_DIR / 'target_scaler.pkl')
        
        return model, history
    
    def evaluate(self, model: nn.Module) -> Dict:
        """
        Comprehensive model evaluation
        """
        logger.info("="*50)
        logger.info("STARTING EVALUATION")
        logger.info("="*50)
        
        model.eval()
        
        # Get predictions
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch).cpu().numpy()
                all_preds.extend(outputs)
                all_targets.extend(y_batch.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics (scaled)
        scaled_metrics = calculate_metrics(all_targets, all_preds)
        
        # Inverse transform to original scale
        target_scaler = self.scalers['target_scaler']
        all_preds_original = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
        all_targets_original = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
        
        original_metrics = calculate_metrics(all_targets_original, all_preds_original)
        
        logger.info("\n" + "="*50)
        logger.info("SCALED METRICS:")
        logger.info("="*50)
        for metric, value in scaled_metrics.items():
            logger.info(f"{metric:20s}: {value:.6f}")
        
        logger.info("\n" + "="*50)
        logger.info("ORIGINAL SCALE METRICS (Watts/m²):")
        logger.info("="*50)
        for metric, value in original_metrics.items():
            if metric in ['MAE', 'RMSE', 'MSE', 'MaxError']:
                logger.info(f"{metric:20s}: {value:.2f} W/m²")
            elif metric == 'MAPE':
                logger.info(f"{metric:20s}: {value:.2f}%")
            else:
                logger.info(f"{metric:20s}: {value:.4f}")
        
        # Calculate correlation
        correlation = np.corrcoef(all_targets_original, all_preds_original)[0, 1]
        logger.info(f"{'Correlation':20s}: {correlation:.4f}")
        
        # Uncertainty estimation (first 100 samples)
        logger.info("\nEstimating uncertainty with Monte Carlo Dropout...")
        X_sample = torch.FloatTensor(self.X_test[:100]).to(self.device)
        mean_pred, std_pred = predict_with_uncertainty(model, X_sample, n_iterations=100)
        
        mean_pred_original = target_scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
        std_pred_original = target_scaler.inverse_transform(std_pred.reshape(-1, 1)).flatten()
        
        logger.info(f"Average prediction uncertainty: ±{std_pred_original.mean():.2f} W/m²")
        
        # Plot predictions
        plot_predictions(
            all_targets_original,
            all_preds_original,
            title="Solar Radiation Prediction Results",
            save_path=config.RESULTS_DIR / 'predictions_plot.png'
        )
        
        # Compile results
        results = {
            'scaled_metrics': scaled_metrics,
            'original_metrics': original_metrics,
            'correlation': correlation,
            'avg_uncertainty': float(std_pred_original.mean()),
            'predictions': all_preds_original.tolist(),
            'targets': all_targets_original.tolist()
        }
        
        # Save results
        save_results(results, config.RESULTS_DIR / 'evaluation_results.json')
        
        return results