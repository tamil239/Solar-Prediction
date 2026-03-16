"""
Data Preprocessing Module for Solar Radiation Prediction
Handles data loading, cleaning, feature engineering, and sequence creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.utils import logger


class SolarDataPreprocessor:
    """
    Comprehensive data preprocessor for solar radiation prediction
    Handles all preprocessing steps from raw data to model-ready sequences
    """
    
    def __init__(self, data_path: str, window_size: int = 24):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the raw data CSV file
            window_size: Number of time steps to use for prediction (default: 24 hours)
        """
        self.data_path = Path(data_path)
        self.window_size = window_size
        self.df = None
        self.feature_scaler = None
        self.target_scaler = None
        self.scalers = {}
        
        logger.info(f"Initialized SolarDataPreprocessor with window_size={window_size}")
    
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        logger.info(f"Loading data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Rename columns to remove special characters
        df.columns = df.columns.str.strip()
        
        # Parse datetime
        df['Data'] = pd.to_datetime(df['Data'], format='%m/%d/%Y %I:%M:%S %p')
        
        # Sort by datetime
        df = df.sort_values('Data').reset_index(drop=True)
        
        self.df = df
        logger.info(f"Loaded {len(df):,} records from {df['Data'].min()} to {df['Data'].max()}")
        
        return df
    
    def explore_data(self) -> Dict:
        """Basic data exploration"""
        if self.df is None:
            self.load_data()
        
        exploration = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'statistics': self.df.describe().to_dict()
        }
        
        logger.info(f"Data shape: {exploration['shape']}")
        logger.info(f"Missing values: {sum(exploration['missing_values'].values())}")
        
        return exploration
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer time-based and interaction features
        """
        if self.df is None:
            self.load_data()
        
        df = self.df.copy()
        
        # Time-based features
        df['Hour'] = df['Data'].dt.hour
        df['Minute'] = df['Data'].dt.minute
        df['Day'] = df['Data'].dt.day
        df['Month'] = df['Data'].dt.month
        df['DayOfWeek'] = df['Data'].dt.dayofweek
        
        # Parse sunrise/sunset times
        def parse_time_to_minutes(time_str):
            """Convert time string to minutes from midnight"""
            try:
                parts = str(time_str).split(':')
                return int(parts[0]) * 60 + int(parts[1])
            except:
                return 0
        
        sunrise_minutes = df['TimeSunRise'].apply(parse_time_to_minutes)
        sunset_minutes = df['TimeSunSet'].apply(parse_time_to_minutes)
        
        # Current time in minutes
        current_time_minutes = df['Hour'] * 60 + df['Minute']
        
        # Daylight features
        df['DaylightMinutes'] = sunset_minutes - sunrise_minutes
        
        # Time since sunrise (can be negative before sunrise)
        df['TimeSinceSunrise'] = current_time_minutes - sunrise_minutes
        
        # Cyclical time encoding
        # Hour encoding (24-hour cycle)
        df['TimeSin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['TimeCos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        # Minute encoding (60-minute cycle)
        minute_fraction = (df['Hour'] * 60 + df['Minute']) / (24 * 60)
        df['MinuteSin'] = np.sin(2 * np.pi * minute_fraction)
        df['MinuteCos'] = np.cos(2 * np.pi * minute_fraction)
        
        # Month encoding (12-month cycle)
        df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Day/night indicator
        df['IsDaytime'] = ((current_time_minutes >= sunrise_minutes) & 
                          (current_time_minutes <= sunset_minutes)).astype(int)
        
        # Time until/after sunset
        df['TimeUntilSunset'] = sunset_minutes - current_time_minutes
        
        # Interaction features
        df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
        df['Pressure_Temp'] = df['Pressure'] * df['Temperature']
        df['Temp_Speed'] = df['Temperature'] * df['Speed']
        df['Humidity_Speed'] = df['Humidity'] * df['Speed']
        
        # Radiation ratio features (for time encoding)
        solar_noon = (sunrise_minutes + sunset_minutes) / 2
        df['TimeFromSolarNoon'] = np.abs(current_time_minutes - solar_noon)
        
        # Fill any missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        self.df = df
        logger.info(f"Engineered features. New shape: {df.shape}")
        
        return df
    
    def remove_outliers(self, columns: List[str], n_std: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using z-score method
        
        Args:
            columns: Columns to check for outliers
            n_std: Number of standard deviations to use as threshold
        """
        if self.df is None:
            self.load_data()
        
        df = self.df.copy()
        initial_len = len(df)
        
        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[np.abs(df[col] - mean) <= n_std * std]
        
        removed = initial_len - len(df)
        logger.info(f"Removed {removed} outliers ({removed/initial_len*100:.2f}%)")
        
        self.df = df
        return df
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input using sliding window
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            X_seq: Sequences of shape (n_samples, window_size, n_features)
            y_seq: Target values of shape (n_samples,)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.window_size):
            X_seq.append(X[i:i + self.window_size])
            y_seq.append(y[i + self.window_size])
        
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_features(self, feature_columns: List[str]) -> np.ndarray:
        """Extract and scale feature columns"""
        if self.df is None:
            self.engineer_features()
        
        X = self.df[feature_columns].values
        
        # Scale features
        self.feature_scaler = MinMaxScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        self.scalers['feature_scaler'] = self.feature_scaler
        
        logger.info(f"Prepared {X.shape[1]} features: {feature_columns}")
        
        return X_scaled
    
    def prepare_target(self, target_column: str = 'Radiation') -> np.ndarray:
        """Prepare and scale target variable"""
        if self.df is None:
            self.engineer_features()
        
        y = self.df[target_column].values.reshape(-1, 1)
        
        # Scale target
        self.target_scaler = MinMaxScaler()
        y_scaled = self.target_scaler.fit_transform(y).flatten()
        
        self.scalers['target_scaler'] = self.target_scaler
        
        return y_scaled
    
    def run_pipeline(self, 
                    remove_outliers: bool = True,
                    window_size: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray, np.ndarray,
                                                        List[str], Dict]:
        """
        Full preprocessing pipeline
        
        Args:
            remove_outliers: Whether to remove outliers
            window_size: Window size for sequences (overrides instance window_size)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scalers)
        """
        if window_size is not None:
            self.window_size = window_size
        
        # Load and process data
        self.load_data()
        self.engineer_features()
        
        # Define feature columns
        feature_columns = [
            'Temperature', 'Pressure', 'Humidity', 'Speed', 'WindDirection(Degrees)',
            'Hour', 'Minute', 'TimeSin', 'TimeCos', 'DaylightMinutes',
            'TimeSinceSunrise', 'IsDaytime', 'Temp_Humidity', 'Pressure_Temp'
        ]
        
        # Remove outliers if requested
        if remove_outliers:
            outlier_cols = ['Temperature', 'Pressure', 'Humidity', 'Speed', 'Radiation']
            self.remove_outliers(outlier_cols)
        
        # Prepare features and target
        X = self.prepare_features(feature_columns)
        y = self.prepare_target('Radiation')
        
        # Ensure same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        # Train/val/test split (64/21/15)
        n = len(X)
        train_end = int(n * 0.64)
        val_end = int(n * 0.85)  # 64 + 21 = 85%
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
        
        logger.info(f"Data split: Train={len(X_train_seq)}, Val={len(X_val_seq)}, Test={len(X_test_seq)}")
        
        feature_names = feature_columns
        
        return (X_train_seq, X_val_seq, X_test_seq, 
                y_train_seq, y_val_seq, y_test_seq,
                feature_names, self.scalers)
    
    def get_processed_data(self) -> pd.DataFrame:
        """Return the processed dataframe"""
        if self.df is None:
            self.load_data()
            self.engineer_features()
        return self.df


def preprocess_raw_data(data_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Convenience function to preprocess raw data
    
    Args:
        data_path: Path to raw data
        output_path: Optional path to save processed data
        
    Returns:
        Processed dataframe
    """
    preprocessor = SolarDataPreprocessor(data_path)
    preprocessor.load_data()
    preprocessor.engineer_features()
    df = preprocessor.get_processed_data()
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    
    return df
