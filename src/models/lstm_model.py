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
    Train an LSTM model on stock data.

    Args:
        csv_file (str): Path to the cleaned stock CSV file
        time_steps (int): How many past steps to look at to predict next step
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Engineer features (assuming you already have engineer_features implemented)
    from src.preprocess import engineer_features
    data = engineer_features(df)

    # Features and target
    feature_cols = data.columns.difference(['Date', 'Close', 'Adj Close'])
    X = data[feature_cols]
    y = data['Close']

    # Normalize features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Create LSTM datasets
    X_seq, y_seq = create_lstm_dataset(X_scaled, y_scaled, time_steps)

    # Train-test split (chronological)
    split_index = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train model
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

    # Evaluate
    mse = mean_squared_error(y_test_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_true, y_pred)
    r2 = r2_score(y_test_true, y_pred)

    # Direction Accuracy
    y_true_direction = np.sign(np.diff(np.append([y_test_true[0]], y_test_true)))
    y_pred_direction = np.sign(np.diff(np.append([y_test_true[0]], y_pred)))
    direction_accuracy = np.mean(y_true_direction == y_pred_direction)

    # Print evaluation
    print("\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")

    plot_actual_vs_predicted(y_test_true, y_pred, save_path="report/actual_vs_predicted.png")

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
    
def plot_actual_vs_predicted(y_true, y_pred, save_path="report/actual_vs_predicted.png"):
    """
    Plot Actual vs Predicted stock prices and save the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
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
    