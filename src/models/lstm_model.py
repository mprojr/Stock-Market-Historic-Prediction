#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from src.train import visualize_technical_indicators
from typing import Dict, Any



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


def analyze_lstm_fitting(history, y_test, y_pred, threshold_ratio=1.5):
    """
    Analyze LSTM model for overfitting and underfitting.
    
    Args:
        history: Training history containing loss values
        y_test: Actual test values
        y_pred: Predicted values
        threshold_ratio: Threshold for overfitting detection
        
    Returns:
        Dict containing analysis and recommendations
    """
    result = {
        "has_overfitting": False,
        "has_underfitting": False,
        "analysis": "",
        "recommendations": []
    }
    
    # Get final loss values
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    # Calculate test R2 score
    test_r2 = r2_score(y_test, y_pred)
    
    # Detect overfitting
    loss_ratio = final_val_loss / final_train_loss
    
    if loss_ratio > threshold_ratio:
        result["has_overfitting"] = True
        result["analysis"] += "Overfitting detected: Model performs significantly better on training data than validation data. "
        result["recommendations"].extend([
            "Add dropout layers (e.g., try 0.2-0.5 dropout rate)",
            "Reduce model complexity (decrease number of LSTM units)",
            "Add L1/L2 regularization",
            "Increase batch size",
            "Use early stopping with patience"
        ])
    
    # Detect underfitting
    if test_r2 < 0.5 and final_train_loss > 0.1:
        result["has_underfitting"] = True
        result["analysis"] += "Underfitting detected: Model fails to capture the underlying patterns in the data. "
        result["recommendations"].extend([
            "Increase model complexity (add more LSTM layers or units)",
            "Train for more epochs",
            "Decrease batch size",
            "Add more features or engineer new features",
            "Consider increasing the time steps"
        ])
    
    if not result["has_overfitting"] and not result["has_underfitting"]:
        result["analysis"] = "No significant overfitting or underfitting detected. Model appears to be well-balanced."
    
    return result


def analyze_lstm_fitting(history, y_test, y_pred, threshold_ratio=1.5):
    """
    Analyze LSTM model for overfitting and underfitting.
    
    Args:
        history: Training history containing loss values
        y_test: Actual test values
        y_pred: Predicted values
        threshold_ratio: Threshold for overfitting detection
        
    Returns:
        Dict containing analysis and recommendations
    """
    result = {
        "has_overfitting": False,
        "has_underfitting": False,
        "analysis": "",
        "recommendations": []
    }
    
    # Get final loss values
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    # Calculate test R2 score
    test_r2 = r2_score(y_test, y_pred)
    
    # Detect overfitting
    loss_ratio = final_val_loss / final_train_loss
    
    if loss_ratio > threshold_ratio:
        result["has_overfitting"] = True
        result["analysis"] += "Overfitting detected: Model performs significantly better on training data than validation data. "
        result["recommendations"].extend([
            "Add dropout layers (e.g., try 0.2-0.5 dropout rate)",
            "Reduce model complexity (decrease number of LSTM units)",
            "Add L1/L2 regularization",
            "Increase batch size",
            "Use early stopping with patience"
        ])
    
    # Detect underfitting
    if test_r2 < 0.5 and final_train_loss > 0.1:
        result["has_underfitting"] = True
        result["analysis"] += "Underfitting detected: Model fails to capture the underlying patterns in the data. "
        result["recommendations"].extend([
            "Increase model complexity (add more LSTM layers or units)",
            "Train for more epochs",
            "Decrease batch size",
            "Add more features or engineer new features",
            "Consider increasing the time steps"
        ])
    
    if not result["has_overfitting"] and not result["has_underfitting"]:
        result["analysis"] = "No significant overfitting or underfitting detected. Model appears to be well-balanced."
    
    return result


def train_lstm_model(csv_file: str, time_steps: int = 1, epochs: int = 30, batch_size: int = 32, lstm_units: int = 64):
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
        LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    import time
    start_time = time.time()
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    end_time = time.time()
    
    # Analyze computational complexity
    complexity_analysis = analyze_lstm_complexity(
        model=model,
        X_train=X_train,
        start_time=start_time,
        end_time=end_time
    )
    
    print("\nComputational Complexity Analysis:")
    print(complexity_analysis["analysis"])
    print("Model-specific parameters:")
    for k, v in complexity_analysis["model_params"].items():
        print(f"  {k}: {v}")
    
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

     # After model.fit, add:
    print("\nModel Fitting Analysis:")
    fitting_analysis = analyze_lstm_fitting(history, y_test_true, y_pred)
    print(fitting_analysis["analysis"])
    if fitting_analysis["recommendations"]:
        print("\nRecommendations:")
        for rec in fitting_analysis["recommendations"]:
            print(f"- {rec}")
    
    # Save the analysis to a JSON file
    analysis_path = "report/lstm_fitting_analysis.json"
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    with open(analysis_path, 'w') as f:
        json.dump(fitting_analysis, f, indent=2)
    print(f"\n✅ Fitting analysis saved to {analysis_path}")
    
    # Plot training history with clear overfitting/underfitting visualization
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.axhline(y=history.history['loss'][-1], color='blue', linestyle='--', alpha=0.3)
    plt.axhline(y=history.history['val_loss'][-1], color='red', linestyle='--', alpha=0.3)
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the enhanced loss plot
    loss_plot_path = "report/lstm_loss_analysis.png"
    plt.savefig(loss_plot_path)
    print(f"✅ Enhanced loss analysis plot saved to {loss_plot_path}")

    metrics = (mse, rmse, mae, r2, direction_accuracy)
    return model, history, metrics



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

def run_lstm_ablation_study(csv_file: str, time_steps=10, epochs=50, batch_size=32, unit_list=[32, 64, 128]):
    print("Running Ablation Study on LSTM Width (Units)")
    results = []

    for units in unit_list:
        print(f"\n🔧 Training with LSTM Units = {units}...")
        


        # You must capture metrics from train_lstm_model, so let's assume you return them
        # If not, we can extract them from a modified return
        # Here, we assume train_lstm_model prints them and we collect manually
        # Instead, let’s return them from the function

        # Return like: return model, history, (mse, rmse, mae, r2, direction_accuracy)
        _, _, metrics = train_lstm_model(
            csv_file=csv_file,
            time_steps=time_steps,
            epochs=epochs,
            batch_size=batch_size,
            lstm_units=units
        )

        results.append({
            'LSTM Units': units,
            'MSE': metrics[0],
            'RMSE': metrics[1],
            'MAE': metrics[2],
            'R2': metrics[3],
            'Direction Accuracy': metrics[4]
        })

    # Print final table
    print("\n📊 Ablation Results:")
    print(f"{'Units':<10}{'MSE':<10}{'RMSE':<10}{'MAE':<10}{'R2':<10}{'DirAcc':<10}")
    for r in results:
        print(f"{r['LSTM Units']:<10}{r['MSE']:<10.4f}{r['RMSE']:<10.4f}{r['MAE']:<10.4f}{r['R2']:<10.4f}{r['Direction Accuracy']:<10.4f}")

def analyze_lstm_complexity(
    model: tf.keras.Model,
    X_train: np.ndarray,
    start_time: float,
    end_time: float
) -> Dict[str, Any]:
    """
    Analyze computational complexity of the LSTM model.
    
    Args:
        model: Trained LSTM model
        X_train: Training features
        start_time: Training start time
        end_time: Training end time
        
    Returns:
        Dict containing complexity analysis
    """
    import psutil
    import os
    
    # Calculate trainable parameters
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_variables])
    
    complexity = {
        "training_time": end_time - start_time,
        "model_size_bytes": model.count_params() * 4,  # 4 bytes per float32 parameter
        "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
        "input_shape": X_train.shape,
        "trainable_parameters": trainable_params,
        "model_params": {
            "lstm_units": [layer.units for layer in model.layers if isinstance(layer, LSTM)],
            "sequence_length": X_train.shape[1],
            "n_features": X_train.shape[2],
            "theoretical_complexity": f"O(sequence_length * n_features * lstm_units^2)"
        },
        "analysis": ""
    }
    
    # Generate analysis text
    complexity["analysis"] = (
        f"LSTM model trained in {complexity['training_time']:.2f} seconds using {complexity['memory_usage_mb']:.2f}MB memory. "
        f"Model has {trainable_params:,} trainable parameters. "
        f"Model size: {complexity['model_size_bytes']/1024/1024:.2f}MB. "
    )
    
    return complexity

def analyze_lstm_complexity(
    model: tf.keras.Model,
    X_train: np.ndarray,
    start_time: float,
    end_time: float
) -> Dict[str, Any]:
    """
    Analyze computational complexity of the LSTM model.
    
    Args:
        model: Trained LSTM model
        X_train: Training features
        start_time: Training start time
        end_time: Training end time
        
    Returns:
        Dict containing complexity analysis
    """
    import psutil
    import os
    
    # Calculate trainable parameters
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    
    complexity = {
        "training_time": end_time - start_time,
        "model_size_bytes": model.count_params() * 4,  # 4 bytes per float32 parameter
        "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
        "input_shape": X_train.shape,
        "trainable_parameters": trainable_params,
        "model_params": {
            "lstm_units": [layer.units for layer in model.layers if isinstance(layer, LSTM)],
            "sequence_length": X_train.shape[1],
            "n_features": X_train.shape[2],
            "theoretical_complexity": f"O(sequence_length * n_features * lstm_units^2)"
        },
        "analysis": ""
    }
    
    # Generate analysis text
    complexity["analysis"] = (
        f"LSTM model trained in {complexity['training_time']:.2f} seconds using {complexity['memory_usage_mb']:.2f}MB memory. "
        f"Model has {trainable_params:,} trainable parameters. "
        f"Model size: {complexity['model_size_bytes']/1024/1024:.2f}MB. "
    )
    
    return complexity


# Example usage (you can comment this out if importing)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM model for stock prediction or run ablation study")
    parser.add_argument('--csv', type=str, required=True, help='Path to the stock CSV file')
    parser.add_argument('--time_steps', type=int, default=10, help='Number of past steps to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study instead of single training')
    args = parser.parse_args()

    if args.ablation:
        run_lstm_ablation_study(
            csv_file=args.csv,
            time_steps=args.time_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unit_list=[32, 64, 128]  # or any list you want to test
        )
    else:
        model, history, _ = train_lstm_model(
            csv_file=args.csv,
            time_steps=args.time_steps,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        plot_training_history(history, save_path="report/loss_plot.png")
