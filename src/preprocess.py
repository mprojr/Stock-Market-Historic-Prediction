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


def calculate_stochastic_rsi(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, rsi_period: int = 14) -> pd.DataFrame:
    """
    Calculate Stochastic RSI, a second-derivative oscillator of price.
    
    Args:
        df (pd.DataFrame): DataFrame containing price data
        k_period (int): Period for %K line
        d_period (int): Period for %D line (signal)
        rsi_period (int): Period for RSI calculation
        
    Returns:
        pd.DataFrame: DataFrame with Stochastic RSI features
    """
    # Calculate RSI
    delta = df['Close'].diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    down = abs(down)
    
    avg_gain = up.rolling(window=rsi_period).mean()
    avg_loss = down.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI
    stoch_rsi = (rsi - rsi.rolling(k_period).min()) / (rsi.rolling(k_period).max() - rsi.rolling(k_period).min())
    k = stoch_rsi.rolling(k_period).mean() * 100  # %K
    d = k.rolling(d_period).mean()  # %D (signal line)
    
    return pd.DataFrame({
        'StochRSI_K': k,
        'StochRSI_D': d,
        'StochRSI_Raw': stoch_rsi * 100
    })


def calculate_volume_profile(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """
    Calculate a simple volume profile by price zones
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        bins (int): Number of price zones to divide the range into
        
    Returns:
        pd.DataFrame: DataFrame with volume profile features
    """
    # Create price bins
    price_range = df['Close'].max() - df['Close'].min()
    bin_size = price_range / bins
    
    # Calculate which bin each price falls into
    df['price_bin'] = ((df['Close'] - df['Close'].min()) / bin_size).astype(int)
    df.loc[df['price_bin'] == bins, 'price_bin'] = bins - 1  # Handle edge case
    
    # Calculate volume per bin
    volume_profile = df.groupby('price_bin')['Volume'].sum().reset_index()
    
    # Create features indicating how far price is from high volume zones
    bin_centers = df['Close'].min() + (volume_profile['price_bin'] + 0.5) * bin_size
    volume_profile['bin_center_price'] = bin_centers
    
    # Sort by volume to find highest volume areas
    high_vol_zones = volume_profile.sort_values('Volume', ascending=False)['bin_center_price'].values
    
    # Calculate distance to nearest high volume zone
    result = pd.DataFrame(index=df.index)
    for i, zone in enumerate(high_vol_zones[:3]):  # Top 3 volume zones
        result[f'Dist_HighVol_Zone_{i+1}'] = (df['Close'] - zone).abs()
    
    return result


def engineer_features(df: pd.DataFrame, include_volume_profile: bool = True) -> pd.DataFrame:
    """
    Create new features for stock market prediction.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        include_volume_profile (bool): Whether to include volume profile analysis
        
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
    
    # Moving Averages - include 5, 20, and now 50 day
    feature_df['MA_5d'] = feature_df['Close'].rolling(window=5).mean()
    feature_df['MA_20d'] = feature_df['Close'].rolling(window=20).mean()
    feature_df['MA_50d'] = feature_df['Close'].rolling(window=50).mean()
    
    # EMA - include 20 and 50 day as requested
    feature_df['EMA_12d'] = feature_df['Close'].ewm(span=12, adjust=False).mean()
    feature_df['EMA_20d'] = feature_df['Close'].ewm(span=20, adjust=False).mean()
    feature_df['EMA_26d'] = feature_df['Close'].ewm(span=26, adjust=False).mean()
    feature_df['EMA_50d'] = feature_df['Close'].ewm(span=50, adjust=False).mean()
    
    # Price distance from key EMAs
    feature_df['Dist_EMA20'] = (feature_df['Close'] - feature_df['EMA_20d']) / feature_df['Close']
    feature_df['Dist_EMA50'] = (feature_df['Close'] - feature_df['EMA_50d']) / feature_df['Close']
    
    # EMA crossovers (potential trade signals)
    feature_df['EMA20_cross_EMA50'] = feature_df['EMA_20d'] - feature_df['EMA_50d']
    
    # MACD components
    feature_df['MACD'] = feature_df['EMA_12d'] - feature_df['EMA_26d']
    feature_df['MACD_Signal'] = feature_df['MACD'].ewm(span=9, adjust=False).mean()
    feature_df['MACD_Histogram'] = feature_df['MACD'] - feature_df['MACD_Signal']
    
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
    
    # Add Stochastic RSI
    stoch_rsi_df = calculate_stochastic_rsi(feature_df)
    feature_df = pd.concat([feature_df, stoch_rsi_df], axis=1)
    
    # Volume-based indicators
    feature_df['Volume_Change'] = feature_df['Volume'].pct_change()
    feature_df['Vol_MA_5d'] = feature_df['Volume'].rolling(window=5).mean()
    feature_df['Vol_MA_20d'] = feature_df['Volume'].rolling(window=20).mean()
    feature_df['Vol_Ratio'] = feature_df['Volume'] / feature_df['Vol_MA_20d']
    
    # Volume profile analysis
    if include_volume_profile:
        try:
            vol_profile = calculate_volume_profile(feature_df)
            feature_df = pd.concat([feature_df, vol_profile], axis=1)
        except Exception as e:
            print(f"Skipping volume profile analysis due to error: {e}")
    
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
    save_processed: bool = True,
    include_volume_profile: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete data processing pipeline from raw data to train/test splits.
    
    Args:
        input_file (str): Path to raw data file
        output_file (Optional[str]): Path to save processed data (if None, derives from input_file)
        forecast_horizon (int): Number of periods ahead to forecast
        save_processed (bool): Whether to save the processed data
        include_volume_profile (bool): Whether to include volume profile analysis
        
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
    feature_df = engineer_features(clean_df, include_volume_profile=include_volume_profile)
    
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
    parser.add_argument("--no-volume-profile", action="store_true", help="Disable volume profile analysis")
    
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test = process_data(
        args.input,
        args.output,
        forecast_horizon=args.horizon,
        save_processed=not args.no_save,
        include_volume_profile=not args.no_volume_profile
    )
    
    print("Data processing complete!") 