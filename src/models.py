#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine learning model definitions for stock market prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_linear_model(alpha: float = 0.1, l1_ratio: float = 0.5, model_type: str = 'linear') -> BaseEstimator:
    """
    Create a linear model for stock price prediction.
    
    Args:
        alpha (float): Regularization strength
        l1_ratio (float): The ElasticNet mixing parameter (0 <= l1_ratio <= 1)
        model_type (str): Type of linear model ('linear', 'ridge', 'lasso', or 'elasticnet')
        
    Returns:
        BaseEstimator: Scikit-learn estimator
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha)
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create a pipeline with standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    return pipeline


def create_tree_model(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    model_type: str = 'rf'
) -> BaseEstimator:
    """
    Create a tree-based model for stock price prediction.
    
    Args:
        n_estimators (int): Number of trees in the ensemble
        max_depth (int): Maximum depth of the trees
        min_samples_split (int): Minimum samples required to split a node
        model_type (str): Type of tree model ('rf' for Random Forest or 'gb' for Gradient Boosting)
        
    Returns:
        BaseEstimator: Scikit-learn estimator
    """
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    elif model_type == 'gb':
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def create_svm_model(C: float = 1.0, epsilon: float = 0.1, kernel: str = 'rbf') -> BaseEstimator:
    """
    Create an SVM model for stock price prediction.
    
    Args:
        C (float): Regularization parameter
        epsilon (float): Epsilon in the epsilon-SVR model
        kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        
    Returns:
        BaseEstimator: Scikit-learn estimator
    """
    model = SVR(C=C, epsilon=epsilon, kernel=kernel)
    
    # Create a pipeline with standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    return pipeline


class TimeSeriesModel:
    """
    Wrapper class for time series models with specialized evaluation and forecasting.
    """
    
    def __init__(self, model: BaseEstimator, forecast_horizon: int = 1):
        """
        Initialize the time series model.
        
        Args:
            model (BaseEstimator): Scikit-learn estimator
            forecast_horizon (int): Number of periods ahead to forecast
        """
        self.model = model
        self.forecast_horizon = forecast_horizon
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'TimeSeriesModel':
        """
        Fit the model to the training data.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Training features
            y (Union[pd.Series, np.ndarray]): Training target
            
        Returns:
            TimeSeriesModel: Fitted model
        """
        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the fitted model.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted, call fit before predict")
        
        return self.model.predict(X)
    
    def evaluate(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Test features
            y (Union[pd.Series, np.ndarray]): Test target
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted, call fit before evaluate")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Additional metrics for financial time series
        # Direction accuracy (DA): percentage of times the direction of price change is correctly predicted
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_true = y.values
        else:
            y_true = y
            
        # Calculate direction changes (actual vs. predicted)
        y_true_direction = np.sign(np.diff(np.append([y_true[0]], y_true)))
        y_pred_direction = np.sign(np.diff(np.append([y_true[0]], y_pred)))
        
        # Direction accuracy
        da = np.mean(y_true_direction == y_pred_direction)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': da
        }
    
    def feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importances if the model supports it.
        
        Returns:
            Optional[Dict[str, float]]: Dictionary mapping feature names to importance scores, 
                                       or None if not supported
        """
        if not self.is_fitted:
            return None
        
        # Check if model has feature_importances_ attribute (tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models have coef_ attribute
            importances = np.abs(self.model.coef_)
        elif hasattr(self.model, 'steps'):
            # Check if it's a pipeline with a model that has feature importances
            final_step = self.model.steps[-1][1]
            if hasattr(final_step, 'feature_importances_'):
                importances = final_step.feature_importances_
            elif hasattr(final_step, 'coef_'):
                importances = np.abs(final_step.coef_)
            else:
                return None
        else:
            return None
        
        # Return feature importances with names if available
        if self.feature_names is not None and len(self.feature_names) == len(importances):
            return dict(zip(self.feature_names, importances))
        else:
            return dict(zip([f"feature_{i}" for i in range(len(importances))], importances))


def get_model(model_type: str, **kwargs) -> TimeSeriesModel:
    """
    Factory function to create a model of the specified type.
    
    Args:
        model_type (str): Type of model to create
        **kwargs: Additional parameters for the model
        
    Returns:
        TimeSeriesModel: Model instance
    """
    forecast_horizon = kwargs.pop('forecast_horizon', 1)
    
    if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
        model = create_linear_model(model_type=model_type, **kwargs)
    elif model_type in ['rf', 'gb']:
        model = create_tree_model(model_type=model_type, **kwargs)
    elif model_type == 'svm':
        model = create_svm_model(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return TimeSeriesModel(model, forecast_horizon=forecast_horizon)


# Example usage when run as a script
if __name__ == "__main__":
    import argparse
    from sklearn.datasets import make_regression
    
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Create and fit a model
    model = get_model('rf', n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print("Model evaluation:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}") 