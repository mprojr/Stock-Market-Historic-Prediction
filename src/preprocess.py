#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing and feature engineering for stock market prediction.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw stock market data from CSV file.
    
    Args:
        file_path (str): Path to the raw data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data by handling missing values, duplicates, and outliers.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Cleaning data...")
    
    # Make a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Convert date column to datetime if exists
    if 'Date' in clean_df.columns:
        clean_df['Date'] = pd.to_datetime(clean_df['Date'])
    
    # Handle missing values
    initial_missing = clean_df.isnull().sum().sum()
    if initial_missing > 0:
        print(f"Found {initial_missing} missing values")
        # Forward fill for time series data
        clean_df = clean_df.fillna(method='ffill')
        # If still have missing values, use backward fill
        clean_df = clean_df.fillna(method='bfill')
        remaining_missing = clean_df.isnull().sum().sum()
        print(f"Remaining missing values: {remaining_missing}")
    
    # Remove duplicates if any
    initial_rows = len(clean_df)
    clean_df = clean_df.drop_duplicates()
    if len(clean_df) < initial_rows:
        print(f"Removed {initial_rows - len(clean_df)} duplicate rows")
    
    return clean_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for stock market prediction.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        pd.DataFrame: Dataframe with additional engineered features
    """
    print("Engineering features...")
    
    # Make a copy to avoid modifying the original
    feature_df = df.copy()
    
    # Ensure we have the necessary columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in feature_df.columns for col in required_cols):
        print("Warning: Not all required columns (Open, High, Low, Close, Volume) are present")
        return feature_df
    
    # Calculate basic technical indicators
    
    # Returns: Daily returns
    feature_df['Return'] = feature_df['Close'].pct_change()
    
    # Volatility: Rolling standard deviation of returns
    feature_df['Volatility_5d'] = feature_df['Return'].rolling(window=5).std()
    feature_df['Volatility_20d'] = feature_df['Return'].rolling(window=20).std()
    
    # Moving Averages
    feature_df['MA_5d'] = feature_df['Close'].rolling(window=5).mean()
    feature_df['MA_20d'] = feature_df['Close'].rolling(window=20).mean()
    
    # MACD components
    feature_df['EMA_12d'] = feature_df['Close'].ewm(span=12, adjust=False).mean()
    feature_df['EMA_26d'] = feature_df['Close'].ewm(span=26, adjust=False).mean()
    feature_df['MACD'] = feature_df['EMA_12d'] - feature_df['EMA_26d']
    feature_df['MACD_Signal'] = feature_df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Relative Strength Index (RSI) - 14 days
    # Calculate daily price changes
    delta = feature_df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over 14 days
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate relative strength (RS) and RSI
    rs = avg_gain / avg_loss
    feature_df['RSI_14d'] = 100 - (100 / (1 + rs))
    
    # Drop rows with NaN values created by rolling calculations
    feature_df = feature_df.dropna()
    
    print(f"Created dataframe with {len(feature_df.columns)} features")
    return feature_df


def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str = 'Close',
    test_size: float = 0.2,
    forecast_horizon: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for training and testing by shifting the target column
    to create a forecast horizon and splitting the data.
    
    Args:
        df (pd.DataFrame): Feature dataframe
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        forecast_horizon (int): Number of periods ahead to forecast
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    print(f"Preparing train/test data with forecast horizon: {forecast_horizon}")
    
    # Create the target variable by shifting the close price
    df = df.copy()
    df[f'Target_{forecast_horizon}d'] = df[target_col].shift(-forecast_horizon)
    
    # Drop rows with NaN in the target
    df = df.dropna()
    
    # Define features and target
    features = df.drop([f'Target_{forecast_horizon}d', 'Date'] if 'Date' in df.columns else [f'Target_{forecast_horizon}d'], axis=1)
    target = df[f'Target_{forecast_horizon}d']
    
    # Split data chronologically
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def process_data(
    input_file: str,
    output_file: Optional[str] = None,
    forecast_horizon: int = 1,
    save_processed: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete data processing pipeline from raw data to train/test splits.
    
    Args:
        input_file (str): Path to raw data file
        output_file (Optional[str]): Path to save processed data (if None, derives from input_file)
        forecast_horizon (int): Number of periods ahead to forecast
        save_processed (bool): Whether to save the processed data
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    # Default output file
    if output_file is None and save_processed:
        dir_name = os.path.dirname(input_file)
        base_name = os.path.basename(input_file).split('.')[0]
        output_file = os.path.join(dir_name.replace('raw', 'processed'), f"{base_name}_processed.csv")
    
    # Load and process data
    raw_df = load_raw_data(input_file)
    clean_df = clean_data(raw_df)
    feature_df = engineer_features(clean_df)
    
    # Save processed data if requested
    if save_processed and output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        feature_df.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")
    
    # Prepare train/test split
    return prepare_train_test_data(feature_df, forecast_horizon=forecast_horizon)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process stock market data")
    parser.add_argument("--input", required=True, help="Path to raw data file")
    parser.add_argument("--output", help="Path to save processed data")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (days)")
    parser.add_argument("--no-save", action="store_true", help="Do not save processed data")
    
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test = process_data(
        args.input,
        args.output,
        forecast_horizon=args.horizon,
        save_processed=not args.no_save
    )
    
    print("Data processing complete!") 