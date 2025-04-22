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
from typing import Dict, Any, Optional, Tuple

# Add src to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import process_data
from src.models import get_model


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


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_results: bool = False,
    results_path: Optional[str] = None,
    visualize: bool = False,
    fig_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model (Any): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        save_results (bool): Whether to save evaluation results
        results_path (Optional[str]): Path to save evaluation results
        visualize (bool): Whether to visualize predictions
        fig_path (Optional[str]): Path to save visualization figures
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    print("Evaluating model on test data...")
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print("Test metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
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
        y_pred = model.predict(X_test)
        
        # Plot actual vs predicted
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
    
    return metrics


def train_evaluate_pipeline(
    input_file: str,
    model_type: str = 'rf',
    model_params: Optional[Dict[str, Any]] = None,
    output_dir: str = '../models',
    forecast_horizon: int = 1,
    visualize: bool = True,
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
    
    # Process data
    X_train, X_test, y_train, y_test = process_data(
        input_file=input_file,
        forecast_horizon=forecast_horizon
    )
    
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
        save_results=save_results,
        results_path=results_path,
        visualize=visualize,
        fig_path=fig_path
    )
    
    print("Training and evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate stock market prediction models")
    parser.add_argument("--input", required=True, help="Path to the raw data file")
    parser.add_argument("--model-type", default="rf", choices=["linear", "ridge", "lasso", "elasticnet", "rf", "gb", "svm"],
                        help="Type of model to train")
    parser.add_argument("--output-dir", default="../models", help="Directory to save outputs")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (days)")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--no-save-model", action="store_true", help="Disable model saving")
    parser.add_argument("--no-save-results", action="store_true", help="Disable results saving")
    
    args = parser.parse_args()
    
    train_evaluate_pipeline(
        input_file=args.input,
        model_type=args.model_type,
        output_dir=args.output_dir,
        forecast_horizon=args.horizon,
        visualize=not args.no_visualize,
        save_model=not args.no_save_model,
        save_results=not args.no_save_results
    ) 