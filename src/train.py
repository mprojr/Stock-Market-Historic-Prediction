#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training and evaluation scripts for stock market prediction models.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Add src to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import process_data
from models.models import get_model


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'rf',
    model_params: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model on the given data.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        model_type (str): Type of model to train
        model_params (Optional[Dict[str, Any]]): Parameters for the model
        save_path (Optional[str]): Path to save the trained model
        
    Returns:
        Tuple[Any, Dict[str, float]]: Trained model and training metrics
    """
    print(f"Training {model_type} model...")
    
    # Set default parameters if not provided
    if model_params is None:
        model_params = {}
    
    # Create and train the model
    model = get_model(model_type, **model_params)
    model.fit(X_train, y_train)
    
    # Evaluate on training data
    train_metrics = model.evaluate(X_train, y_train)
    print("Training metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save the model if requested
    if save_path:
        import joblib
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")
    
    return model, train_metrics


def visualize_technical_indicators(
    df: pd.DataFrame, 
    predictions: np.ndarray = None,
    forecast_horizon: int = 1,
    fig_path: Optional[str] = None
) -> None:
    """
    Create visualizations of technical indicators and model predictions.
    
    Args:
        df (pd.DataFrame): DataFrame with price data and indicators
        predictions (np.ndarray, optional): Model predictions if available
        forecast_horizon (int): Number of periods ahead forecasted
        fig_path (Optional[str]): Path to save the figure
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define the layout
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # 1. Price chart with key EMAs
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.plot(df.index, df['EMA_20d'], label='EMA 20', alpha=0.8)
    ax1.plot(df.index, df['EMA_50d'], label='EMA 50', alpha=0.8)
    
    # Add predictions if available
    if predictions is not None:
        # Create a shifted index to properly align predictions
        pred_index = df.index[-len(predictions):]
        ax1.plot(pred_index, predictions, 'r--', label=f'Predicted (t+{forecast_horizon})')
    
    ax1.set_title('Price with 20 & 50 EMA')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Stochastic RSI
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df.index, df['StochRSI_K'], label='%K')
    ax2.plot(df.index, df['StochRSI_D'], label='%D', alpha=0.8)
    ax2.axhline(80, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(20, color='g', linestyle='--', alpha=0.3)
    ax2.set_title('Stochastic RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend()
    
    # 3. Regular RSI
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df.index, df['RSI_14d'])
    ax3.axhline(70, color='r', linestyle='--', alpha=0.3)
    ax3.axhline(30, color='g', linestyle='--', alpha=0.3)
    ax3.set_title('RSI (14)')
    ax3.set_ylim(0, 100)
    ax3.grid(True)
    
    # 4. Volume with 20-day average
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(df.index, df['Volume'], alpha=0.6, label='Volume')
    ax4.plot(df.index, df['Vol_MA_20d'], color='r', label='20-day Avg Volume')
    ax4.set_title('Volume')
    ax4.legend()
    ax4.grid(True)
    
    # 5. MACD
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(df.index, df['MACD'], label='MACD')
    ax5.plot(df.index, df['MACD_Signal'], label='Signal Line')
    ax5.bar(df.index, df['MACD_Histogram'], alpha=0.3, label='Histogram')
    ax5.axhline(0, color='k', linestyle='-', alpha=0.2)
    ax5.set_title('MACD')
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    
    if fig_path:
        plt.savefig(fig_path)
        print(f"Technical indicator visualization saved to {fig_path}")
    else:
        plt.show()


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_data: Optional[pd.DataFrame] = None,
    save_results: bool = False,
    results_path: Optional[str] = None,
    visualize: bool = False,
    fig_path: Optional[str] = None,
    tech_indicators_fig_path: Optional[str] = None,
    forecast_horizon: int = 1
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model (Any): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        raw_data (Optional[pd.DataFrame]): Original data with technical indicators
        save_results (bool): Whether to save evaluation results
        results_path (Optional[str]): Path to save evaluation results
        visualize (bool): Whether to visualize predictions
        fig_path (Optional[str]): Path to save visualization figures
        tech_indicators_fig_path (Optional[str]): Path to save technical indicator figures
        forecast_horizon (int): Number of periods ahead forecasted
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    print("Evaluating model on test data...")
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print("Test metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get feature importances if available
    importances = model.feature_importance()
    if importances:
        print("Top 10 feature importances:")
        for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feature}: {importance:.4f}")
    
    # Save results if requested
    if save_results and results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results = {
            'metrics': metrics,
            'feature_importances': importances
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    # Visualize predictions if requested
    if visualize:
        # Basic visualization of predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('Actual vs Predicted Stock Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        if fig_path:
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            plt.savefig(fig_path)
            print(f"Figure saved to {fig_path}")
        else:
            plt.show()
        
        # Technical indicators visualization if raw data is provided
        if raw_data is not None and tech_indicators_fig_path:
            # Get the portion of raw_data that corresponds to X_test
            if len(raw_data) > len(X_test):
                # Align the raw data with test data
                test_data = raw_data.iloc[-len(X_test):]
                visualize_technical_indicators(
                    test_data, 
                    predictions=y_pred,
                    forecast_horizon=forecast_horizon,
                    fig_path=tech_indicators_fig_path
                )
    
    return metrics


def train_evaluate_pipeline(
    input_file: str,
    model_type: str = 'rf',
    model_params: Optional[Dict[str, Any]] = None,
    output_dir: str = '../models',
    forecast_horizon: int = 1,
    visualize: bool = True,
    include_volume_profile: bool = True,
    save_model: bool = True,
    save_results: bool = True
) -> None:
    """
    Complete pipeline for training and evaluating a stock prediction model.
    
    Args:
        input_file (str): Path to the raw data file
        model_type (str): Type of model to train
        model_params (Optional[Dict[str, Any]]): Parameters for the model
        output_dir (str): Directory to save outputs
        forecast_horizon (int): Number of periods ahead to forecast
        visualize (bool): Whether to visualize results
        include_volume_profile (bool): Whether to include volume profile analysis
        save_model (bool): Whether to save the trained model
        save_results (bool): Whether to save evaluation results
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up output paths
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_type}_model_{timestamp}.pkl") if save_model else None
    results_path = os.path.join(output_dir, f"{model_type}_results_{timestamp}.json") if save_results else None
    fig_path = os.path.join(output_dir, f"{model_type}_prediction_{timestamp}.png") if visualize else None
    tech_indicators_fig_path = os.path.join(output_dir, f"{model_type}_indicators_{timestamp}.png") if visualize else None
    
    # Process data
    raw_data_path = os.path.join(os.path.dirname(output_dir), 'processed', f"{os.path.basename(input_file).split('.')[0]}_processed.csv")
    
    X_train, X_test, y_train, y_test = process_data(
        input_file=input_file,
        output_file=raw_data_path,
        forecast_horizon=forecast_horizon,
        include_volume_profile=include_volume_profile
    )
    
    # Load saved raw data with indicators for visualization
    raw_data = pd.read_csv(raw_data_path) if os.path.exists(raw_data_path) else None
    
    # Train model
    model, _ = train_model(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        model_params=model_params,
        save_path=model_path
    )
    
    # Evaluate model
    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        raw_data=raw_data,
        save_results=save_results,
        results_path=results_path,
        visualize=visualize,
        fig_path=fig_path,
        tech_indicators_fig_path=tech_indicators_fig_path,
        forecast_horizon=forecast_horizon
    )
    
    print("Training and evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate stock market prediction models")
    parser.add_argument("--input", required=True, help="Path to the raw data file")
    parser.add_argument("--model-type", default="rf", choices=["linear", "ridge", "lasso", "elasticnet", "rf", "gb", "svm"],
                        help="Type of model to train")
    parser.add_argument("--output-dir", default="report", help="Directory to save outputs")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (days)")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--no-save-model", action="store_true", help="Disable model saving")
    parser.add_argument("--no-save-results", action="store_true", help="Disable results saving")
    parser.add_argument("--no-volume-profile", action="store_true", help="Disable volume profile analysis")
    
    args = parser.parse_args()
    
    train_evaluate_pipeline(
        input_file=args.input,
        model_type=args.model_type,
        output_dir=args.output_dir,
        forecast_horizon=args.horizon,
        visualize=not args.no_visualize,
        include_volume_profile=not args.no_volume_profile,
        save_model=not args.no_save_model,
        save_results=not args.no_save_results
    ) 