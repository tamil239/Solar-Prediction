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

solar-radiation-prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and preprocessed data
│
├── notebooks/
│   ├── eda.ipynb               # Exploratory Data Analysis
│   ├── data_preprocessing.ipynb
│
├── models/
│   ├── lstm_model.py           # LSTM architecture
│   ├── train.py                # Model training script
│   ├── predict.py              # Prediction script
│
├── app/
│   ├── streamlit_app.py        # Web interface using Streamlit
│
├── utils/
│   ├── data_loader.py          # Dataset loading functions
│   ├── preprocessing.py        # Data preprocessing utilities
│   ├── evaluation.py           # Model evaluation metrics
│
├── config/
│   ├── config.yaml             # Model and training configuration
│
├── results/
│   ├── plots/                  # Visualization outputs
│   ├── metrics/                # Performance metrics
│
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── LICENSE
🚀 Usage
## 🚀 Usage

### 1️⃣ Train the Model

Run the training script to train the LSTM model.

```bash
python models/train.py
2️⃣ Make Predictions

Use the trained model to predict solar radiation.

python models/predict.py
3️⃣ Launch the Web Application

Run the Streamlit app to interact with the prediction system.

streamlit run app/streamlit_app.py

---

### 📊 Dataset

```markdown
## 📊 Dataset

The dataset contains meteorological parameters used to predict solar radiation, including:

- Temperature
- Humidity
- Wind Speed
- Atmospheric Pressure
- Cloud Cover
- Solar Radiation

Data is collected from weather monitoring stations and preprocessed before training the model.
🔧 Installation
## 🔧 Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/solar-radiation-prediction.git
cd solar-radiation-prediction

Create a virtual environment

python -m venv venv

Activate the environment

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate

Install dependencies

pip install -r requirements.txt

---

### 📈 Future Improvements

```markdown
## 📈 Future Improvements

- Integration with real-time weather APIs
- Deployment using Docker
- Mobile-friendly dashboard
- Advanced ensemble models
- Integration with energy grid management systems
📜 License
## 📜 License

This project is licensed under the MIT License.
