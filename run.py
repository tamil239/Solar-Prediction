#!/usr/bin/env python
"""
Main runner script for Solar Radiation Prediction Project
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.train import SolarTrainer
from src.evaluate import ModelEvaluator
from src.utils import logger, set_seed
import config


def main():
    parser = argparse.ArgumentParser(description='Solar Radiation Prediction')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['eda', 'tune', 'train', 'evaluate', 'app', 'all', 'clean'],
                        help='Mode to run')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials for hyperparameter tuning')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.LOGS_DIR.mkdir(exist_ok=True)
    
    if args.mode == 'clean':
        logger.info("🧹 Cleaning up...")
        import shutil
        if config.MODELS_DIR.exists():
            shutil.rmtree(config.MODELS_DIR)
            logger.info(f"Removed {config.MODELS_DIR}")
        if config.RESULTS_DIR.exists():
            shutil.rmtree(config.RESULTS_DIR)
            logger.info(f"Removed {config.RESULTS_DIR}")
        config.MODELS_DIR.mkdir(exist_ok=True)
        config.RESULTS_DIR.mkdir(exist_ok=True)
        logger.info("✅ Cleanup complete")
        return
    
    if args.mode == 'eda' or args.mode == 'all':
        logger.info("📊 Running EDA notebook...")
        notebook_path = config.NOTEBOOKS_DIR / '01_eda_analysis.ipynb'
        if notebook_path.exists():
            try:
                subprocess.run([
                    'jupyter', 'nbconvert', '--to', 'notebook',
                    '--execute', str(notebook_path),
                    '--output', '01_eda_analysis_executed.ipynb',
                    '--ExecutePreprocessor.timeout=600'
                ], check=True)
                logger.info("✅ EDA notebook executed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"EDA notebook execution failed: {e}")
        else:
            logger.warning(f"Notebook not found: {notebook_path}")
    
    if args.mode == 'tune' or args.mode == 'all':
        logger.info("🔍 Starting hyperparameter tuning...")
        try:
            trainer = SolarTrainer(str(config.DATA_FILE))
            best_params = trainer.tune_hyperparameters(n_trials=args.trials)
            logger.info(f"✅ Best parameters: {best_params}")
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
    
    if args.mode == 'train' or args.mode == 'all':
        logger.info("🚀 Starting training...")
        try:
            trainer = SolarTrainer(str(config.DATA_FILE))
            
            # Try to load best params if they exist
            params_file = config.RESULTS_DIR / 'best_params.json'
            if params_file.exists():
                import json
                with open(params_file, 'r') as f:
                    params = json.load(f)
                logger.info(f"Loaded best params from file: {params}")
            else:
                params = config.MODEL_CONFIG
                logger.info(f"Using default params: {params}")
            
            model, history = trainer.train(params)
            
            # Evaluate
            logger.info("📈 Evaluating model...")
            results = trainer.evaluate(model)
            
            logger.info("✅ Training complete!")
            logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
            logger.info(f"Test R² score: {results['original_metrics']['R2']:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    if args.mode == 'evaluate' or args.mode == 'all':
        logger.info("📊 Evaluating model...")
        try:
            evaluator = ModelEvaluator()
            logger.info("✅ Model loaded successfully")
            # Note: Would need test data to evaluate
        except FileNotFoundError as e:
            logger.error(f"Evaluation failed: {e}")
            logger.info("Please train the model first: python run.py --mode train")
    
    if args.mode == 'app':
        logger.info("🌐 Starting Streamlit app...")
        try:
            subprocess.run(['streamlit', 'run', 'app/streamlit_app.py'])
        except FileNotFoundError:
            logger.error("Streamlit not found. Install with: pip install streamlit")
        except Exception as e:
            logger.error(f"Failed to start app: {e}")
    
    logger.info("✨ Done!")


if __name__ == "__main__":
    main()