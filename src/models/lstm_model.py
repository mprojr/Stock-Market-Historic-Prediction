#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from src.train import visualize_technical_indicators



def create_lstm_dataset(X, y, time_steps=1):
    """
    Create sequences for LSTM.
    Each input is a sequence of `time_steps` rows, and the output is the next target value.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def train_lstm_model(csv_file: str, time_steps: int = 1, epochs: int = 30, batch_size: int = 32):
    """
    Train an LSTM model using engineered features.

    Args:
        csv_file (str): Path to the cleaned stock CSV file
        time_steps (int): How many past steps to look at to predict next step
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    # Load raw data
    df = pd.read_csv(csv_file)

    # Engineer features with volume profile
    from src.preprocess import engineer_features
    data = engineer_features(df, include_volume_profile=True)

    # Selected features used in traditional models
    selected_features = [
        'EMA_20d', 'EMA_50d',
        'RSI_14d', 'StochRSI_K', 'StochRSI_D',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'Volume', 'Vol_MA_20d', 'Vol_Ratio',
        'Dist_EMA20', 'Dist_EMA50', 'EMA20_cross_EMA50'
    ]

    # Filter only available features
    available_features = [col for col in selected_features if col in data.columns]
    missing_features = set(selected_features) - set(available_features)
    if missing_features:
        print(f"⚠️ Warning: Missing features (skipped): {missing_features}")

    X = data[available_features]
    y = data['Close']

    # Normalize
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Create LSTM datasets
    X_seq, y_seq = create_lstm_dataset(X_scaled, y_scaled, time_steps)

    # Train-test split
    split_index = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Predict
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Evaluation
    mse = mean_squared_error(y_test_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_true, y_pred)
    r2 = r2_score(y_test_true, y_pred)
    y_true_dir = np.sign(np.diff(np.append([y_test_true[0]], y_test_true)))
    y_pred_dir = np.sign(np.diff(np.append([y_test_true[0]], y_pred)))
    direction_accuracy = np.mean(y_true_dir == y_pred_dir)
    
        # Visualization (rich indicators)
    try:
        data_with_index = data.set_index(pd.to_datetime(df['Date'][-len(data):]))

        # Get last N rows matching predictions
        test_data = data_with_index.iloc[-len(y_pred):]


        visualize_technical_indicators(
            df=test_data,
            predictions=y_pred,
            forecast_horizon=1,
            fig_path="report/technical_indicators_lstm.png"
        )

    except Exception as e:
        print(f"⚠️ Skipping technical indicator plot due to error: {e}")


    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Direction Accuracy: {direction_accuracy:.4f}")
    df['Date'] = pd.to_datetime(df['Date'])  # ensure datetime
    date_index = df['Date'].iloc[-len(y_pred):]

    # Plot
    plot_actual_vs_predicted(y_test_true, y_pred, date_index=date_index, save_path="report/actual_vs_predicted.png")


    return model, history


def plot_training_history(history, save_path="report/loss_plot.png"):
    """
    Plot training and validation loss over epochs and save the plot.
    """
    # Create report folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    print(f"✅ Loss plot saved to {save_path}")

    # Also show the plot (optional)
    plt.show()
    
def plot_actual_vs_predicted(y_true, y_pred, date_index=None, save_path="report/actual_vs_predicted.png"):
    """
    Plot Actual vs Predicted stock prices and save the plot.
    
    Args:
        y_true (array-like): True stock prices
        y_pred (array-like): Predicted stock prices
        date_index (array-like, optional): Optional datetime index for x-axis
        save_path (str): File path to save the figure
    """
    plt.figure(figsize=(12, 6))

    x = date_index if date_index is not None else range(len(y_true))

    plt.plot(x, y_true, label='Actual')
    plt.plot(x, y_pred, label='Predicted')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date' if date_index is not None else 'Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Rotate x-axis labels if it's datetime
    if date_index is not None:
        plt.xticks(rotation=45)

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path)
    print(f"✅ Actual vs Predicted plot saved to {save_path}")

    # Show the plot
    plt.show()



# Example usage (you can comment this out if importing)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction")
    parser.add_argument('--csv', type=str, required=True, help='Path to the stock CSV file')
    parser.add_argument('--time_steps', type=int, default=10, help='Number of past steps to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    args = parser.parse_args()

    model, history = train_lstm_model(
        csv_file=args.csv,
        time_steps=args.time_steps,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    plot_training_history(history, save_path="report/loss_plot.png")   # 👈 Now it saves automatically
    