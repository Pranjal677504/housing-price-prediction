"""
Model training and evaluation utilities
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os


def train_models(X_train, y_train):
    """
    Train multiple regression models.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        
    Returns:
        dict: Dictionary of trained models
    """
    models = {}
    
    # Linear Regression
    models['Linear Regression'] = LinearRegression()
    models['Linear Regression'].fit(X_train, y_train)
    
    # Ridge Regression
    models['Ridge Regression'] = Ridge(alpha=1.0)
    models['Ridge Regression'].fit(X_train, y_train)
    
    # Random Forest
    models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['Random Forest'].fit(X_train, y_train)
    
    # Gradient Boosting
    models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
    models['Gradient Boosting'].fit(X_train, y_train)
    
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate models and return performance metrics.
    
    Args:
        models (dict): Dictionary of trained models
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        
    Returns:
        dict: Evaluation metrics for each model
    """
    results = {}
    
    for name, model in models.items():
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        results[name] = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
        }
    
    return results


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Loaded model object
    """
    return joblib.load(filepath)
