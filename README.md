# ☀️ Solar Radiation Prediction with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This project implements a state-of-the-art LSTM model for solar radiation prediction, achieving **R² scores up to 0.9893**. The model helps integrate solar energy into power grids by providing accurate forecasts with uncertainty estimates.

## 📊 Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² Score | 0.989 | Explains 98.9% of variance |
| RMSE | 12.4 W/m² | Typical error magnitude |
| MAE | 8.2 W/m² | Average absolute error |
| Correlation | 0.994 | Near-perfect prediction correlation |

## 🎯 Features

- **Deep Learning Model**: 2-layer LSTM with 256 hidden units
- **Uncertainty Estimation**: Monte Carlo Dropout for confidence intervals
- **Web Interface**: Streamlit app for easy predictions
- **Hyperparameter Tuning**: Optuna integration
- **Comprehensive EDA**: Detailed exploratory data analysis
- **Production Ready**: Modular code with logging and error handling

## 📁 Project Structure
