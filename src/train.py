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
from src.models.models import get_model


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
    
    import time
    start_time = time.time()
    
    # Set default parameters if not provided
    if model_params is None:
        model_params = {}
    
    # Create and train the model
    model = get_model(model_type, **model_params)
    model.fit(X_train, y_train)
    
    end_time = time.time()
    
    # Analyze computational complexity
    complexity_analysis = analyze_computational_complexity(
        model=model,
        X_train=X_train,
        start_time=start_time,
        end_time=end_time,
        model_type=model_type
    )
    
    print("\nComputational Complexity Analysis:")
    print(complexity_analysis["analysis"])
    if "model_params" in complexity_analysis:
        print("Model-specific parameters:")
        for k, v in complexity_analysis["model_params"].items():
            print(f"  {k}: {v}")
    
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
    forecast_horizon: int = 1,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    model_type: str = 'rf'
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
        X_train (Optional[pd.DataFrame]): Training features for overfitting detection
        y_train (Optional[pd.Series]): Training target for overfitting detection
        model_type (str): Type of model being evaluated
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    print("Evaluating model on test data...")
    
    # Evaluate the model on test data
    test_metrics = model.evaluate(X_test, y_test)
    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get feature importances if available
    importances = model.feature_importance()
    if importances:
        print("Top 10 feature importances:")
        for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feature}: {importance:.4f}")
    
    # Detect overfitting/underfitting if training data is provided
    fitting_analysis = None
    if X_train is not None and y_train is not None:
        # Calculate training metrics
        train_metrics = model.evaluate(X_train, y_train)
        
        # Analyze for overfitting/underfitting
        fitting_analysis = detect_model_fitting_issues(train_metrics, test_metrics, model_type)
        
        # Print analysis
        print("\nModel Fitting Analysis:")
        print(fitting_analysis["analysis"])
        if fitting_analysis["recommendations"]:
            print("Recommendations:")
            for rec in fitting_analysis["recommendations"]:
                print(f"  - {rec}")
    
    # Save results if requested
    if save_results and results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results = {
            'metrics': test_metrics,
            'feature_importances': importances
        }
        
        # Add fitting analysis if available
        if fitting_analysis:
            results['fitting_analysis'] = fitting_analysis
            
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
    
    return test_metrics


def detect_model_fitting_issues(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    model_type: str
) -> Dict[str, Any]:
    """
    Detect overfitting or underfitting issues in the model.
    
    Args:
        train_metrics (Dict[str, float]): Metrics from training data
        test_metrics (Dict[str, float]): Metrics from test data
        model_type (str): Type of model being evaluated
        
    Returns:
        Dict[str, Any]: Dictionary with analysis and recommendations
    """
    result = {
        "has_overfitting": False,
        "has_underfitting": False,
        "analysis": "",
        "recommendations": []
    }
    
    # Get key metrics
    train_rmse = train_metrics.get('rmse', float('inf'))
    test_rmse = test_metrics.get('rmse', float('inf'))
    train_r2 = train_metrics.get('r2', 0)
    test_r2 = test_metrics.get('r2', 0)
    
    # Detect overfitting
    rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')
    r2_diff = train_r2 - test_r2
    
    if rmse_ratio > 1.5 or r2_diff > 0.3:
        result["has_overfitting"] = True
        result["analysis"] += "Overfitting detected: Model performs significantly better on training data than test data. "
        
        # Model-specific recommendations for overfitting
        if model_type in ['rf', 'gb']:
            result["recommendations"].extend([
                "Reduce model complexity (decrease max_depth, increase min_samples_split)",
                "Add regularization (decrease n_estimators)",
                "Consider feature selection to reduce dimensionality"
            ])
        elif model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
            result["recommendations"].extend([
                "Increase regularization strength (alpha parameter)",
                "Reduce feature count through feature selection"
            ])
        elif model_type == 'svm':
            result["recommendations"].extend([
                "Increase regularization (decrease C parameter)",
                "Try a different kernel"
            ])
        
    # Detect underfitting
    if test_r2 < 0.5 and train_r2 < 0.6:
        result["has_underfitting"] = True
        result["analysis"] += "Underfitting detected: Model fails to capture the underlying patterns in the data. "
        
        # Model-specific recommendations for underfitting
        if model_type in ['rf', 'gb']:
            result["recommendations"].extend([
                "Increase model complexity (increase max_depth, n_estimators)",
                "Add more features or engineer new features",
                "If using GB, decrease learning rate and increase n_estimators"
            ])
        elif model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
            result["recommendations"].extend([
                "Try non-linear models as the relationship may be non-linear",
                "Engineer additional features that might capture non-linear relationships",
                "Decrease regularization strength"
            ])
        elif model_type == 'svm':
            result["recommendations"].extend([
                "Try a different kernel (e.g., 'rbf' instead of 'linear')",
                "Decrease regularization (increase C parameter)",
                "Adjust gamma parameter if using rbf kernel"
            ])
    
    # If no issues detected
    if not result["has_overfitting"] and not result["has_underfitting"]:
        result["analysis"] = "No significant overfitting or underfitting detected. Model appears to be well-balanced."
        
    return result


def analyze_computational_complexity(
    model: Any,
    X_train: pd.DataFrame,
    start_time: float,
    end_time: float,
    model_type: str
) -> Dict[str, Any]:
    """
    Analyze computational complexity of the model.
    
    Args:
        model: Trained model
        X_train: Training features
        start_time: Training start time
        end_time: Training end time
        model_type: Type of model
        
    Returns:
        Dict containing complexity analysis
    """
    import psutil
    import os
    from sys import getsizeof
    
    complexity = {
        "training_time": end_time - start_time,
        "model_size_bytes": getsizeof(model),
        "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
        "input_shape": X_train.shape,
        "analysis": ""
    }
    
    # Model specific analysis
    if model_type in ['rf', 'gb']:
        n_trees = model.model.n_estimators
        max_depth = model.model.max_depth if model.model.max_depth else "unlimited"
        complexity["model_params"] = {
            "n_trees": n_trees,
            "max_depth": max_depth,
            "theoretical_complexity": f"O(n_trees * n_samples * log(n_samples)) ≈ O({n_trees} * {X_train.shape[0]} * log({X_train.shape[0]}))"
        }
    elif model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
        complexity["model_params"] = {
            "n_features": X_train.shape[1],
            "theoretical_complexity": f"O(n_samples * n_features) ≈ O({X_train.shape[0]} * {X_train.shape[1]})"
        }
    elif model_type == 'svm':
        complexity["model_params"] = {
            "n_support_vectors": len(model.model.support_vectors_) if hasattr(model.model, 'support_vectors_') else "N/A",
            "theoretical_complexity": f"O(n_samples^2 * n_features) ≈ O({X_train.shape[0]}^2 * {X_train.shape[1]})"
        }
    
    # Generate analysis text
    complexity["analysis"] = f"Model trained in {complexity['training_time']:.2f} seconds using {complexity['memory_usage_mb']:.2f}MB memory. "
    complexity["analysis"] += f"Model size: {complexity['model_size_bytes']/1024/1024:.2f}MB. "
    
    return complexity


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
    model, train_metrics = train_model(
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
        forecast_horizon=forecast_horizon,
        X_train=X_train,
        y_train=y_train,
        model_type=model_type
    )
    
    print("Training and evaluation complete!")
    
def run_traditional_model_ablation(
    input_file: str,
    model_type: str,
    param_grid: List[Dict[str, Any]],
    forecast_horizon: int = 1,
    include_volume_profile: bool = True,
    output_dir: str = "report/ablation",
) -> None:
    import pandas as pd
    from src.preprocess import process_data
    from src.train import train_model, evaluate_model
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the data once
    raw_data_path = os.path.join(os.path.dirname(output_dir), 'processed', f"{os.path.basename(input_file).split('.')[0]}_processed.csv")
    X_train, X_test, y_train, y_test = process_data(
        input_file=input_file,
        output_file=raw_data_path,
        forecast_horizon=forecast_horizon,
        include_volume_profile=include_volume_profile
    )
    raw_data = pd.read_csv(raw_data_path) if os.path.exists(raw_data_path) else None

    results = []

    for params in param_grid:
        print(f"\n🔍 Testing {model_type} with params: {params}")
        model, _ = train_model(
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
            model_params=params
        )

        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            raw_data=raw_data,
            save_results=False,
            visualize=False
        )

        result_entry = {
            "Model": model_type,
            "Params": json.dumps(params),
            "MSE": metrics.get("mse"),
            "RMSE": metrics.get("rmse"),
            "MAE": metrics.get("mae"),
            "R2": metrics.get("r2"),
            "Direction Accuracy": metrics.get("direction_accuracy")
        }
        results.append(result_entry)

    df_results = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, f"ablation_{model_type}_{timestamp}.csv")
    df_results.to_csv(output_csv, index=False)
    print(f"\n✅ Ablation results saved to {output_csv}")


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
    
    # Tree-based model parameters
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth for tree-based models")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum samples required to split a node")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study for traditional models")
    args = parser.parse_args()

    model_params = {}
    if args.model_type in ['rf', 'gb']:
        model_params = {
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "n_estimators": args.n_estimators
        }
        # Remove None values
        model_params = {k: v for k, v in model_params.items() if v is not None}
    if args.ablation:
        if args.model_type in ["rf", "gb"]:
            param_grid = [
                {"n_estimators": 50},
                {"n_estimators": 100},
                {"n_estimators": 200}
            ]
        elif args.model_type in ["ridge", "lasso", "elasticnet"]:
            param_grid = [
                {"alpha": 0.01},
                {"alpha": 0.1},
                {"alpha": 1.0}
            ]
        elif args.model_type == "linear":
            param_grid = [
                {},  # no hyperparameters for LinearRegression
            ]
        elif args.model_type == "svm":
            param_grid = [
                {"C": 0.1},
                {"C": 1.0},
                {"C": 10.0}
            ]
        else:
            raise ValueError("Ablation grid not defined for this model type.")

        run_traditional_model_ablation(
            input_file=args.input,
            model_type=args.model_type,
            param_grid=param_grid,
            forecast_horizon=args.horizon,
            include_volume_profile=not args.no_volume_profile,
            output_dir=args.output_dir
        )
    else:
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